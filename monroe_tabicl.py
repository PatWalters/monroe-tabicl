#!/usr/bin/env python
"""Build and apply a Monroe + TabICLv2 regression model from a CSV of SMILES.

Monroe is a frozen graph-transformer encoder that turns each molecule into one
720-d vector. TabICLv2 reads those vectors in context: it is handed the training
embeddings with their labels and predicts new molecules in one forward pass, with
no weights updated. So "training" here means embedding the training set, and a
saved model is those embeddings, their labels and the settings used to make them.
Predicting embeds the new molecules and hands both sets to TabICL.

    # train, report 5-fold CV, and save the model
    python monroe_tabicl.py train expansion_train.csv SMILES "HLM CLint" --log --cv 5 --save hlm.npz

    # predict new molecules with the saved model
    python monroe_tabicl.py predict expansion_test.csv SMILES --model hlm.npz -o hlm_pred.csv

The Monroe checkout (with `git lfs pull` run in it) is found through MONROE_HOME,
default ~/software/monroe. TabICL's checkpoint is downloaded from Hugging Face on
first use.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# The in-context head: TabICLv2, through the authors' `tabicl` package. Named
# explicitly rather than left to the package default, so a later tabicl release
# cannot change which model a saved run used.
TABICL_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"

MONROE_HOME = Path(
    os.environ.get("MONROE_HOME", Path.home() / "software" / "monroe")
).expanduser().resolve()
CHECKPOINT_DIR = MONROE_HOME / "checkpoint"

# The width of Monroe's graph-level embedding. Checked after embedding, because a
# different checkpoint would still produce vectors and still fit.
EMBEDDING_DIM = 720
MODEL_FORMAT = 1


def monroe_modules():
    """Import Monroe from its checkout, failing loudly if the weights are absent."""
    if not (CHECKPOINT_DIR / "weights.pt").exists():
        raise SystemExit(
            f"no Monroe checkpoint at {CHECKPOINT_DIR}\n"
            f"set MONROE_HOME to the monroe checkout (currently {MONROE_HOME}) and "
            "run `git lfs pull` in it"
        )
    if str(MONROE_HOME) not in sys.path:
        sys.path.insert(0, str(MONROE_HOME))
    # Not monroe.eval.tabpfn: it imports tabpfn, which this script does not need.
    import monroe.eval.embed as embed
    from monroe.model import ckpt

    def load_ckpt(path: str, device: str):
        """Monroe's load_ckpt, but mapping the weights onto `device`.

        The released weights were saved from CUDA, and load_ckpt passes no
        map_location, so on a CPU-only machine it fails before building anything.
        """
        hp_dict, state_dict = ckpt._load_config_and_weights(ckpt._resolve_ckpt_dir(path),
                                                            device=device)
        encoder = ckpt._build_encoder(hp_dict)
        # load_ckpt wraps the encoder so the saved "encoder." keys line up.
        wrapper = torch.nn.Module()
        wrapper.encoder = encoder
        wrapper.load_state_dict(state_dict, strict=False)
        return encoder

    import torch

    return embed, load_ckpt


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class SeededDistGeom:
    """rdDistGeom, except that ETKDGv3() comes with a fixed random seed.

    Monroe's predict_structure builds its conformer from ETKDGv3 with
    useRandomCoords and no randomSeed, so each molecule's conformer depends on
    the state of whichever pool worker happens to pick it up -- and that depends
    on timing. On 300 molecules, 60-90 came out different between two runs.
    With the seed fixed, the conformer depends on the molecule alone.
    """

    def __init__(self, module, seed: int):
        self._module, self._seed = module, seed

    def ETKDGv3(self):
        params = self._module.ETKDGv3()
        params.randomSeed = self._seed
        return params

    def __getattr__(self, name):
        return getattr(self._module, name)


CONFORMER_SEED = 42


def embed_with_progress(embed_mod, smiles: list[str], encoder, device: str, batch_size: int,
                        workers: int) -> dict[str, np.ndarray]:
    """Monroe's embed_smiles, step for step, with a progress bar on each stage.

    Featurizing (an RDKit conformer per molecule) is most of the time and runs
    across a process pool; encoding is batched. Monroe's own function reports
    neither, so this rebuilds it from its pieces: the same featurizer, the same
    order, the same batches. The one change is the fixed conformer seed (see
    SeededDistGeom), which makes the vectors identical from run to run and
    independent of the number of workers.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from torch_geometric.data import Batch
    from tqdm import tqdm

    from monroe.model import featurizer

    graphs_mod = embed_mod._graphs
    original_write = graphs_mod.tqdm.write
    graphs_mod.tqdm.write = lambda *args, **kwargs: None  # one line per molecule is too noisy
    # Set before the pool forks, so every worker inherits it.
    original_dist_geom = featurizer.rdDistGeom
    featurizer.rdDistGeom = SeededDistGeom(original_dist_geom, CONFORMER_SEED)
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("fork")) as pool:
            built = [graph for graph in tqdm(
                pool.map(embed_mod._featurize_one, smiles, chunksize=8),
                total=len(smiles), desc="featurizing", unit="mol", dynamic_ncols=True)
                if graph is not None]
    finally:
        graphs_mod.tqdm.write = original_write
        featurizer.rdDistGeom = original_dist_geom
    if not built:
        return {}

    graphs = [embed_mod.to_pyg(graph) for _, graph in built]
    vectors = []
    with torch.no_grad(), tqdm(total=len(graphs), desc="encoding", unit="mol",
                               dynamic_ncols=True) as bar:
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
            graph_embedding, _ = encoder(batch)
            vectors.append(graph_embedding.cpu().numpy())
            bar.update(batch.num_graphs)

    return dict(zip([smi for smi, _ in built], np.concatenate(vectors)))


def embed(smiles: list[str], batch_size: int, workers: int | None) -> tuple[np.ndarray, np.ndarray]:
    """One Monroe embedding per input SMILES, in input order.

    Molecules the featurizer cannot handle come back as NaN rows with `ok` false,
    so the caller decides what to do with them.
    """
    embed_mod, load_ckpt = monroe_modules()
    device = device_name()
    encoder = load_ckpt(str(CHECKPOINT_DIR), device).to(device).eval()

    unique = list(pd.unique(pd.Series(smiles)))
    print(f"embedding {len(unique)} unique molecules on {device}", flush=True)
    start = time.time()
    # Monroe defaults the worker count with os.sched_getaffinity, which macOS lacks.
    lookup = embed_with_progress(embed_mod, unique, encoder, device, batch_size,
                                 workers or os.cpu_count())
    print(f"embedded {len(lookup)}/{len(unique)} in {time.time() - start:.0f}s", flush=True)

    X = np.full((len(smiles), EMBEDDING_DIM), np.nan, dtype=np.float32)
    ok = np.zeros(len(smiles), dtype=bool)
    for row, smi in enumerate(smiles):
        vector = lookup.get(smi)
        if vector is not None:
            if len(vector) != EMBEDDING_DIM:
                raise SystemExit(f"expected {EMBEDDING_DIM}-d embeddings, got {len(vector)}")
            X[row] = vector
            ok[row] = True
    return X, ok


# --- memory safety ----------------------------------------------------------
#
# On CPU, TabICL holds a 128-number vector (and wider intermediates) for every
# cell of the table it is working on, rows x features, all at once. With its
# defaults the full ExpansionRx LogD job (7,321 rows x 720 features) ran a 48 GB
# Mac out of memory and took it down: macOS does not kill the process, the whole
# machine stalls. So TabICL runs in a child process that the parent watches and
# stops before it can get there, and a job that cannot fit is refused up front.
#
# Peak memory, measured on an Apple-silicon Mac, is roughly a fixed ~6 GB plus a
# few KB per row per feature: 3.6 KB at 360 features (12.6 GB for LogD's 5,039
# training rows), over 4.3 KB at 720. The estimate uses 5 KB to stay on the safe
# side. With kv_cache="repr" the training rows are worked through once in fit()
# and only `PREDICT_CHUNK` new rows at a time in predict(), so the rows that
# count are the larger of the two.
MEMORY_BASE_GB = 6.0
MEMORY_PER_CELL_KB = 5.0
PREDICT_CHUNK = 500
# With no --max-memory, TabICL may use what is free now, less this much for
# macOS and whatever else is running.
RESERVE_GB = 6.0
# And it is stopped at once if the system's free memory falls below this.
FLOOR_GB = 3.0
# What this process still holds after embedding (Monroe and PyTorch), measured.
EMBEDDING_HELD_GB = 3.0
# How many Monroe features TabICL sees: the highest-variance ones in the
# training set. On ExpansionRx LogD, 360 scored better than all 720 (R2 0.754 vs
# ~0.72) at under three quarters of the memory; see README.md.
DEFAULT_FEATURES = 360


def estimate_gb(rows: int, features: int) -> float:
    return MEMORY_BASE_GB + MEMORY_PER_CELL_KB * rows * features / 2**20


def memory_budget(max_memory: float | None) -> tuple[float, float]:
    """The memory TabICL may use, and the memory free right now, in GB."""
    import psutil

    free = psutil.virtual_memory().available / 2**30
    return (max_memory if max_memory else free - RESERVE_GB), free


def check_memory(fit_rows: int, features: int, max_memory: float | None,
                 before_embedding: bool = False) -> float:
    """Refuse a TabICL job that will not fit; otherwise return its memory budget.

    Called before any embedding is done as well as just before TabICL starts,
    so a job that cannot fit is turned down at once rather than minutes in.
    Before embedding, it also allows for the ~2.5 GB this process keeps after
    embedding, which is no longer free by the time TabICL starts.
    """
    need = estimate_gb(max(fit_rows, PREDICT_CHUNK), features)
    budget, free = memory_budget(max_memory)
    if before_embedding and not max_memory:
        budget -= EMBEDDING_HELD_GB
    if need > budget:
        fits = int((budget - MEMORY_BASE_GB) * 2**20 / (MEMORY_PER_CELL_KB * fit_rows))
        raise SystemExit(
            f"not starting TabICL: {fit_rows} training rows x {features} features needs "
            f"about {need:.1f} GB, and it may use {budget:.1f} GB "
            + ("(set by --max-memory). " if max_memory else
               f"({free:.1f} GB free now, less {RESERVE_GB:.0f} GB kept for the system"
               + (f" and {EMBEDDING_HELD_GB:.0f} GB for embedding" if before_embedding else "")
               + "). ")
            + "Close other applications, set --max-memory, or train with fewer features"
            + (f" (--features {fits} or fewer would fit)" if fits >= 10 else ""))
    return budget


def select_features(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-variance columns of X, in their original order."""
    if not k or k >= X.shape[1]:
        return np.arange(X.shape[1])
    return np.sort(np.argsort(X.var(axis=0))[::-1][:k])


def _tabicl_worker(in_path: str, out_path: str) -> None:
    """Fit and predict in the child process, reading and writing .npy/.npz files.

    batch_size=1 runs the 8 ensemble members one at a time rather than together,
    and kv_cache="repr" lets predict() take the new rows in chunks. Both change
    the predictions by rounding only (< 1e-5 on LogD).

    TabICL reports no progress of its own, so the bars count its passes: with
    batch_size=1 it calls predict_stats_with_cache once per ensemble member,
    in fit() to cache the training rows and in predict() for each chunk.
    """
    from tabicl import TabICLRegressor
    from tabicl._model.tabicl import TabICL
    from tqdm import tqdm

    data = np.load(in_path)
    X_query, output_type = data["X_query"], str(data["output_type"])
    model = TabICLRegressor(checkpoint_version=TABICL_CHECKPOINT,
                            random_state=int(data["seed"]), device=str(data["device"]),
                            batch_size=1, kv_cache="repr")
    bars = {"fit": tqdm(total=model.n_estimators, desc="TabICL fit", unit="pass",
                        dynamic_ncols=True)}
    one_pass = TabICL.predict_stats_with_cache

    def counted_pass(self, *args, **kwargs):
        out = one_pass(self, *args, **kwargs)
        # fit() stores the training-row cache; predict() reads it.
        bars["fit" if kwargs.get("store_cache") else "predict"].update(1)
        return out

    TabICL.predict_stats_with_cache = counted_pass  # this child process only
    model.fit(data["X_fit"], data["y_fit"])
    bars["fit"].close()
    n_chunks = -(-len(X_query) // PREDICT_CHUNK)
    bars["predict"] = tqdm(total=model.n_estimators * n_chunks, desc="TabICL predict",
                           unit="pass", dynamic_ncols=True)
    pred = np.concatenate([model.predict(X_query[start:start + PREDICT_CHUNK],
                                         output_type=output_type)
                           for start in range(0, len(X_query), PREDICT_CHUNK)])
    bars["predict"].close()
    np.save(out_path, pred)


def predict_tabicl(X_fit, y_fit, X_query, seed: int, output_type: str,
                   max_memory: float | None = None) -> np.ndarray:
    """TabICLv2 predictions for X_query given the labelled context, memory-guarded."""
    import multiprocessing as mp
    import tempfile

    import psutil

    budget = check_memory(len(X_fit), X_fit.shape[1], max_memory)

    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = f"{tmp}/in.npz", f"{tmp}/out.npy"
        np.savez(in_path, X_fit=X_fit, y_fit=y_fit, X_query=X_query, seed=seed,
                 device=device_name(), output_type=output_type)
        # spawn, not fork: a clean interpreter, and no copy of the parent's memory.
        child = mp.get_context("spawn").Process(target=_tabicl_worker, args=(in_path, out_path))
        child.start()
        watched = psutil.Process(child.pid)
        peak = 0.0
        while child.is_alive():
            try:
                used = sum(p.memory_info().rss
                           for p in [watched, *watched.children(recursive=True)]) / 2**30
            except psutil.Error:
                used = 0.0
            available = psutil.virtual_memory().available / 2**30
            peak = max(peak, used)
            if used > budget or available < FLOOR_GB:
                for p in [*watched.children(recursive=True), watched]:
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                child.join()
                why = (f"it reached {used:.1f} GB, over its {budget:.1f} GB budget"
                       if used > budget else
                       f"the system was down to {available:.1f} GB free")
                raise SystemExit(f"stopped TabICL to protect the machine: {why}. Close other "
                                 "applications, or train with fewer --features")
            time.sleep(0.05)
        child.join()
        if child.exitcode != 0:
            raise SystemExit(f"TabICL failed (exit code {child.exitcode}); see the error above")
        print(f"TabICL peak memory {peak:.1f} GB (budget {budget:.1f} GB)", flush=True)
        return np.load(out_path).astype(float)


def read_table(path: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: no column(s) {', '.join(map(repr, missing))}; "
                         f"found {', '.join(map(repr, df.columns))}")
    return df


ZERO_STRATEGIES = ["drop", "floor", "offset", "error"]
# Above this share of zeros, --zeros drop warns that it is discarding a lot.
DROP_WARN_FRACTION = 0.10


def log_scale(y_raw: np.ndarray, zeros: str, floor: float | None = None,
              offset: float | None = None, name: str = "target"):
    """log10 of the target, with zeros handled by `zeros`.

    In an assay readout a zero almost always means "below what the assay can
    quantify", not a true zero, so the value is censored rather than missing:

      drop    leave zero rows out of the training set. The default: see below.
      floor   replace zeros with `floor` before the log, by default half the
              smallest positive training value -- the usual LOQ/2 substitution.
      offset  log10(x + offset), by default offset 1. Keeps every row, but the
              scale is nearly linear below `offset`, and the right offset
              depends on the units.
      error   refuse to run if there are any zeros.

    Why drop. On ExpansionRx HLM and MLM CLint (3-4% zeros), trained on the
    training file and scored on the test file -- with TabPFN as the head, before
    this script moved to TabICL, and not yet repeated -- all four rank the test set
    equally well (Spearman within 0.01) and pick out its measured zeros equally
    well, but on the molecules with a measured value dropping has the lowest
    log-scale error and the floor the highest (log MAE 0.291 vs 0.315 on HLM,
    0.351 vs 0.371 on MLM). A floor puts the zeros in a cluster ~2.5 log units
    below the rest, which drags TabPFN's mean prediction down for any molecule
    that might belong to it; +1 narrows that gap and lands in between. None of
    the strategies predicts the zeros themselves as low. When zeros are a large
    share of the data, dropping them costs real information, hence the warning.

    Negative values are always an error: a quantity that can be negative is
    not one to log scale (it is usually a log already, like LogD).

    Returns the scaled values, the rows kept, and a dict describing the
    transform, which is stored with the model so `unscale` can invert it.
    """
    if (y_raw < 0).any():
        raise SystemExit(f"--log needs non-negative values, but {int((y_raw < 0).sum())} rows "
                         f"of {name!r} are negative -- is it already on a log scale?")
    zero = y_raw == 0
    keep = np.ones(len(y_raw), dtype=bool)
    positive = y_raw[~zero]
    if not len(positive):
        raise SystemExit(f"--log: {name!r} has no positive values")

    if zeros == "offset":
        offset = 1.0 if offset is None else offset
        if offset <= 0:
            raise SystemExit("--offset must be positive")
        transform = {"zeros": "offset", "offset": float(offset)}
        y = np.log10(y_raw + offset)
    else:
        if zeros == "error" and zero.any():
            raise SystemExit(f"--log: {int(zero.sum())} rows of {name!r} are zero; "
                             "choose how to handle them with --zeros")
        if zeros == "drop":
            keep = ~zero
            if zero.mean() > DROP_WARN_FRACTION:
                print(f"warning: {zero.mean():.0%} of {name!r} is zero. Dropping that many "
                      "leaves the model blind to the low end of the range; consider "
                      "--zeros offset or --zeros floor")
        if zeros == "floor":
            floor = positive.min() / 2 if floor is None else floor
            if not 0 < floor <= positive.min():
                raise SystemExit(f"--floor must be > 0 and no larger than the smallest "
                                 f"positive value ({positive.min():g})")
        transform = {"zeros": zeros, "floor": float(floor) if zeros == "floor" else None}
        y = np.log10(np.where(zero, floor if zeros == "floor" else 1.0, y_raw))
    if zero.any():
        what = (f"set to the floor {floor:g}" if zeros == "floor" else
                f"kept by the offset {offset:g}" if zeros == "offset" else "dropped")
        print(f"{int(zero.sum())} zero values of {name!r} {what}")
    return y[keep], keep, transform


def unscale(pred: np.ndarray, transform: dict) -> np.ndarray:
    """Back from log10 to the original units."""
    if transform["zeros"] == "offset":
        # Anything the model puts below log10(offset) means a value below zero,
        # which the data never had; report it as zero.
        return np.clip(10 ** pred - transform["offset"], 0, None)
    return 10 ** pred


def cross_validate(X, y, groups, n_folds: int, seed: int, output_type: str,
                   max_memory: float | None) -> None:
    """K-fold CV over the training set, grouped so duplicate SMILES never straddle folds."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    unique, group_idx = np.unique(groups, return_inverse=True)
    fold_of_group = np.random.default_rng(seed).permutation(len(unique)) % n_folds
    fold_of = fold_of_group[group_idx]

    pred = np.empty_like(y)
    for k in range(n_folds):
        fit, test = fold_of != k, fold_of == k
        pred[test] = predict_tabicl(X[fit], y[fit], X[test], seed, output_type, max_memory)
        print(f"  fold {k + 1}/{n_folds} done", flush=True)
    print(f"{n_folds}-fold CV (modelled scale): "
          f"R2 {r2_score(y, pred):.3f}  "
          f"Spearman {spearmanr(y, pred).statistic:.3f}  "
          f"MAE {mean_absolute_error(y, pred):.3f}")


def train(args) -> None:
    if args.features < 0:
        raise SystemExit("--features must be 0 (all) or a positive number")
    df = read_table(args.file, [args.smiles_col, args.target_col])
    y_raw = pd.to_numeric(df[args.target_col], errors="coerce")
    keep = y_raw.notna() & df[args.smiles_col].notna()
    if (~keep).sum():
        print(f"skipping {int((~keep).sum())} rows with no SMILES or no numeric "
              f"{args.target_col!r}")
    df, y_raw = df.loc[keep].reset_index(drop=True), y_raw[keep].to_numpy(float)

    transform = None
    if args.log:
        if args.floor is not None and args.zeros != "floor":
            raise SystemExit("--floor only applies with --zeros floor")
        if args.offset is not None and args.zeros != "offset":
            raise SystemExit("--offset only applies with --zeros offset")
        y, kept, transform = log_scale(y_raw, args.zeros, args.floor, args.offset,
                                       args.target_col)
        df = df.loc[kept].reset_index(drop=True)
    else:
        if args.floor is not None or args.offset is not None:
            raise SystemExit("--floor and --offset only apply with --log")
        y = y_raw

    n_features = min(args.features or EMBEDDING_DIM, EMBEDDING_DIM)
    if args.cv:
        # Each fold fits on (cv - 1) / cv of the rows.
        check_memory(len(df) * (args.cv - 1) // args.cv, n_features, args.max_memory,
                     before_embedding=True)

    smiles = df[args.smiles_col].astype(str).tolist()
    X, ok = embed(smiles, args.batch_size, args.workers)
    if (~ok).sum():
        print(f"dropping {int((~ok).sum())} molecules Monroe could not featurize")
    X, y, smiles = X[ok], y[ok], [s for s, good in zip(smiles, ok) if good]
    print(f"training set: {len(y)} molecules"
          + (" (target log10 scaled)" if args.log else ""))
    # Chosen on the training embeddings alone (no labels), and stored with the
    # model so predict uses the same ones.
    cols = select_features(X, args.features)
    print(f"using {len(cols)} of {X.shape[1]} Monroe features"
          + (" (highest variance)" if len(cols) < X.shape[1] else ""))

    if args.cv:
        cross_validate(X[:, cols], y, np.array(smiles), args.cv, args.seed, args.output_type,
                       args.max_memory)

    if args.save:
        import importlib.metadata

        meta = {
            "format": MODEL_FORMAT,
            "smiles_col": args.smiles_col,
            "target_col": args.target_col,
            "log10": bool(args.log),
            "log_transform": transform,
            "seed": args.seed,
            "conformer_seed": CONFORMER_SEED,
            "output_type": args.output_type,
            "head": "tabicl",
            # tabicl carries no __version__, so ask the installed distribution.
            "tabicl_version": importlib.metadata.version("tabicl"),
            "tabicl_checkpoint": TABICL_CHECKPOINT,
            "features": int(len(cols)),
            # The name only: a full path would carry the local directory layout
            # into every model file that gets shared.
            "training_file": Path(args.file).name,
        }
        # X keeps all 720 features; `cols` says which ones TabICL reads.
        np.savez_compressed(args.save, X=X, y=y, cols=cols, smiles=np.array(smiles),
                            meta=np.array(json.dumps(meta)))
        print(f"saved model to {args.save}; predicting with it needs about "
              f"{estimate_gb(len(y), len(cols)):.0f} GB of memory")
    elif not args.cv:
        print("nothing to do: pass --save to keep the model, or --cv N to evaluate it")


def load_model(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    try:
        with np.load(path, allow_pickle=False) as saved:
            X, y, meta = saved["X"], saved["y"], json.loads(str(saved["meta"]))
            # Models saved before --features existed use every feature.
            cols = saved["cols"] if "cols" in saved.files else np.arange(X.shape[1])
    except FileNotFoundError:
        raise SystemExit(f"{path}: no such model file")
    except (OSError, ValueError, KeyError):
        raise SystemExit(f"{path}: not a model saved by this script (train --save)")
    if meta.get("format") != MODEL_FORMAT:
        raise SystemExit(f"{path}: not a model saved by this script (train --save)")
    return X, y, cols, meta


def predict(args) -> None:
    X_fit, y_fit, cols, meta = load_model(args.model)
    import importlib.metadata

    installed = importlib.metadata.version("tabicl")
    if meta.get("head") != "tabicl":
        # A model is only embeddings, values and settings, so one saved by the
        # TabPFN version of this script works as it is.
        print("note: this model was saved by the TabPFN version of this script; "
              "its training set is used as it is, with TabICLv2 predicting")
    elif meta["tabicl_checkpoint"] != TABICL_CHECKPOINT:
        print(f"warning: model saved with TabICL checkpoint {meta['tabicl_checkpoint']}, "
              f"predicting with {TABICL_CHECKPOINT}")
    elif meta["tabicl_version"] != installed:
        print(f"warning: model saved with tabicl {meta['tabicl_version']}, running "
              f"{installed}; predictions may differ slightly")
    if meta.get("conformer_seed") != CONFORMER_SEED:
        print("warning: this model was trained before conformers were seeded, so its "
              "training embeddings are not exactly reproducible; retrain it to fix that")

    check_memory(len(X_fit), len(cols), args.max_memory, before_embedding=True)

    df = read_table(args.file, [args.smiles_col])
    smiles = df[args.smiles_col].fillna("").astype(str).tolist()
    X, ok = embed(smiles, args.batch_size, args.workers)
    if (~ok).sum():
        print(f"{int((~ok).sum())} molecules could not be featurized; their predictions are empty")

    pred = np.full(len(df), np.nan)
    if ok.any():
        pred[ok] = predict_tabicl(X_fit[:, cols], y_fit, X[ok][:, cols], meta["seed"],
                                  meta["output_type"], args.max_memory)

    name = meta["target_col"]
    if meta["log10"]:
        # Models saved before --zeros existed refused zeros outright.
        transform = meta.get("log_transform") or {"zeros": "error"}
        log_name = (f"log10_{name}_plus{transform['offset']:g}_pred"
                    if transform["zeros"] == "offset" else f"log10_{name}_pred")
        df[log_name] = pred
        df[f"{name}_pred"] = unscale(pred, transform)
    else:
        df[f"{name}_pred"] = pred

    out = args.output or f"{Path(args.file).stem}_pred.csv"
    df.to_csv(out, index=False)
    print(f"wrote {int(ok.sum())} predictions to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--batch-size", type=int, default=32,
                        help="molecules per encoder forward pass (default: 32)")
    common.add_argument("--workers", type=int, default=None,
                        help="conformer-generation processes (default: every core)")
    common.add_argument("--max-memory", type=float, default=None, metavar="GB",
                        help="memory TabICL may use before it is stopped (default: what is "
                             f"free when it starts, less {RESERVE_GB:.0f} GB)")

    p = sub.add_parser("train", parents=[common], help="build a model from SMILES and data")
    p.add_argument("file", help="CSV with SMILES and the values to model")
    p.add_argument("smiles_col", help="name of the SMILES column")
    p.add_argument("target_col", help="name of the column to predict")
    p.add_argument("--log", action="store_true",
                   help="model log10 of the target; predictions are reported in both scales")
    p.add_argument("--zeros", default="drop", choices=ZERO_STRATEGIES,
                   help="with --log, how to treat zero values (default: drop). drop: "
                        "leave them out of training; floor: replace them with --floor; "
                        "offset: log10(x + --offset); error: refuse")
    p.add_argument("--floor", type=float, default=None,
                   help="value zeros are replaced with (default: half the smallest positive "
                        "value). Set it to the assay's limit of quantitation if you know it")
    p.add_argument("--offset", type=float, default=None,
                   help="the constant added before the log with --zeros offset (default: 1)")
    p.add_argument("--features", type=int, default=DEFAULT_FEATURES, metavar="K",
                   help=f"give TabICL the K highest-variance Monroe features (default: "
                        f"{DEFAULT_FEATURES}; 0 for all {EMBEDDING_DIM}, which needs about "
                        "twice the memory)")
    p.add_argument("--save", metavar="MODEL.npz", help="write the model to this file")
    p.add_argument("--cv", type=int, default=0, metavar="N",
                   help="report N-fold cross-validated R2/Spearman/MAE (default: off)")
    p.add_argument("--seed", type=int, default=42, help="TabICL's random seed (default: 42)")
    p.add_argument("--output-type", default="mean", choices=["mean", "median"],
                   help="TabICL's point estimate (default: mean)")
    p.set_defaults(func=train)

    p = sub.add_parser("predict", parents=[common], help="predict new SMILES with a saved model")
    p.add_argument("file", help="CSV with the SMILES to predict")
    p.add_argument("smiles_col", help="name of the SMILES column")
    p.add_argument("--model", required=True, metavar="MODEL.npz",
                   help="model file written by train --save")
    p.add_argument("-o", "--output", help="output CSV (default: <file>_pred.csv)")
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
