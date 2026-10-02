# Monroe + TabICL

`monroe_tabicl.py` builds a regression or classification model from a CSV of
SMILES and measured values, and uses it to predict new molecules.

It pairs two pretrained models. [Monroe](https://github.com/blazejba/monroe) is a
frozen graph-transformer encoder that turns each molecule (via an RDKit 3D
conformer) into a 720-number vector. [TabICLv2](https://github.com/soda-inria/tabicl)
reads those vectors *in context*: it is handed the training vectors with their
values (or class labels) and predicts new molecules in one forward pass. No
weights are trained, and there are no hyperparameters to tune. TabICLv2 has a
regression head and a classification head, and the script can use either.

So "training" means embedding the training set, and a saved model is those
embeddings, their values and the settings used to make them. Predicting embeds the
new molecules and gives both sets to TabICL.

TabICL is given the 360 of Monroe's 720 features with the highest variance in the
training set, not all of them. On this data that predicts as well or better, and
it is what lets a full-sized job run on a laptop at all: see
[Memory](#memory).

## Installation

### What you need

- conda ([Miniforge](https://github.com/conda-forge/miniforge) is fine)
- `git` and [Git LFS](https://git-lfs.com) (`brew install git-lfs` on a Mac),
  for Monroe's weights
- about 5 GB of disk: 3.4 GB for the Monroe checkout, 1.7 GB for the
  environment, and 0.1 GB for the TabICL checkpoint, downloaded on first use
- enough free memory for the jobs you run: about 13 GB for a 5,000-molecule
  training set (see [Memory](#memory))

A GPU is optional. Everything runs on CPU.

### 1. Get Monroe and its weights

```bash
git lfs install            # once per user, if you have not used Git LFS before
git clone https://github.com/blazejba/monroe ~/software/monroe
cd ~/software/monroe && git lfs pull
```

Monroe is not installed with pip. The script imports it straight from this
checkout, which is also where the encoder weights live. It looks in
`~/software/monroe`; if you put the checkout somewhere else, point `MONROE_HOME`
at it (for example `export MONROE_HOME=/path/to/monroe` in your shell profile).
The script was tested at Monroe commit `57238ed`.

Check that the weights really downloaded: `checkpoint/weights.pt` should be about
300 MB. If it is a few hundred bytes, it is a Git LFS placeholder, and `git lfs
pull` needs to be run again.

### 2. Create the environment

```bash
conda create -n monroe python=3.12
conda activate monroe
pip install -r requirements.txt
```

`requirements.txt` pins the versions the results below came from, including
`tabicl==2.1.1`, the release the validation study used. The script names the
TabICLv2 checkpoints itself (`tabicl-regressor-v2-20260212.ckpt`, and
`tabicl-classifier-v2-20260212.ckpt` for classification), so a newer
`tabicl` would not silently load a different model.

On a Linux machine with a GPU, install the PyTorch build for your CUDA version
first, then the requirements:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install -r requirements.txt
```

TabICL is BSD-licensed and its checkpoint is public, so there is no account or
licence step; it is downloaded from Hugging Face the first time it is used.

### 3. Check the installation

From this directory, with the environment active:

```bash
python - <<'EOF'
import os, pathlib, importlib.metadata, torch
home = pathlib.Path(os.environ.get("MONROE_HOME", "~/software/monroe")).expanduser()
weights = home / "checkpoint" / "weights.pt"
print("tabicl", importlib.metadata.version("tabicl"))
print("Monroe weights:", f"{weights.stat().st_size / 1e6:.0f} MB" if weights.exists() else f"missing at {weights}")
print("device:", "cuda" if torch.cuda.is_available() else "cpu")
EOF
```

which should print

```
tabicl 2.1.1
Monroe weights: 300 MB
device: cpu
```

(`cuda` on a GPU machine). Then run the whole pipeline on the first 500
molecules of the training set:

```bash
head -501 expansion_train.csv > smoke.csv
python monroe_tabicl.py train smoke.csv SMILES "MLM CLint" --log --cv 5
```

On a Mac with no GPU this takes 1–2 minutes and ends with

```
5-fold CV (modelled scale): R2 0.518  Spearman 0.708  MAE 0.279
```

To check the classification head as well, split the same column at 100:

```bash
python monroe_tabicl.py train smoke.csv SMILES "MLM CLint" --task classification --threshold 100 --cv 5
```

which takes about the same time (the TabICL classifier checkpoint, 0.1 GB, is
downloaded the first time) and ends with

```
5-fold CV: accuracy 0.900  balanced accuracy 0.653  MCC 0.426  ROC AUC 0.875  log loss 0.256
  class       n  recall  precision
  <100       58   0.328      0.679
  >=100     423   0.979      0.914
```

On a GPU the numbers may differ in the last decimal place. Delete `smoke.csv`
afterwards.

## Usage

```bash
conda activate monroe

# train on LogD and save the model
python monroe_tabicl.py train expansion_train.csv SMILES LogD --save logd_tabicl.npz

# predict new molecules with it
python monroe_tabicl.py predict expansion_test.csv SMILES --model logd_tabicl.npz -o logd_tabicl_pred.csv
```

Both modes take the data file and the SMILES column first. `train` also takes the
column to model.

A column with a wide range, such as a clearance, is usually better modelled on a
log scale. Add `--log`, and `--cv 5` for a cross-validated estimate of how well it
works:

```bash
python monroe_tabicl.py train expansion_train.csv SMILES "HLM CLint" --log --cv 5 --save hlm.npz
```

### Classification

Add `--task classification` to predict a class instead of a number. Each distinct
value of the target column is a class (text such as `active`/`inactive`, or whole
numbers such as `0`/`1`), up to 10 classes. To classify a numeric column, add
`--threshold T`, which splits it into `<T` and `>=T`:

```bash
python monroe_tabicl.py train expansion_train.csv SMILES LogD --task classification \
    --threshold 2 --cv 5 --save logd_cls.npz
python monroe_tabicl.py predict expansion_test.csv SMILES --model logd_cls.npz -o logd_cls_pred.csv
```

This uses TabICLv2's classifier checkpoint in place of the regressor; everything
else (embedding, feature selection, the memory guard) is the same. `--log`,
`--zeros`, `--floor`, `--offset` and `--output-type` are regression-only, and
`--threshold` is classification-only; mixing them is an error.

How the target column becomes classes:

- **Labels.** Text values are used as they are. A numeric column whose values are
  all whole numbers gets labels `0`, `1`, … rather than `0.0`, `1.0`. Classes are
  ordered alphabetically, which sets the order of the probability columns.
- **Threshold.** With `--threshold T` the column must be numeric, and the two
  classes are named `<T` and `>=T` (for example `<2` and `>=2`).
- **Limits.** At least 2 and at most 10 classes. TabICLv2 handles 10 natively;
  more needs a mode that cannot use the cache the [memory guard](#memory) relies
  on. A column with more than 10 distinct values is refused with a suggestion to
  use `--threshold` or regression, since it is probably continuous.
- **Small classes.** `train` prints the size of each class. It warns if the
  smallest class has fewer molecules than there are CV folds.

With `--cv`, folds keep repeated SMILES together as in regression, and are also
stratified by class, so a rare class is spread across them. The report gives
accuracy, balanced accuracy (the mean of the per-class recalls), the Matthews
correlation coefficient (MCC: from -1 to 1, with 0 no better than chance, using
every cell of the confusion matrix; the multiclass form for more than two
classes), ROC AUC (one-vs-rest, macro-averaged, for more than two classes) and
log loss, then each class's count, recall and precision. With imbalanced classes,
read balanced accuracy, MCC and the per-class recall rather than accuracy. In the
smoke test above, 88% of molecules are in one class: accuracy is 0.90, but MCC is
0.43 and only a third of the minority class is found.

`predict` writes the most probable class and a probability for every class. For
a decision threshold other than 0.5, or to rank molecules, use the probability
columns.

So far classification has only been run on small samples: the smoke test above,
and a 300-molecule sample of LogD split at 2 (3-fold CV accuracy 0.81, ROC AUC
0.90). There is no full train/test result yet.

### `train FILE SMILES_COL TARGET_COL`

| Option | |
| --- | --- |
| `--save MODEL.npz` | write the model to this file |
| `--task {regression,classification}` | what to predict (default `regression`; see [Classification](#classification)) |
| `--threshold T` | with `--task classification`, classify a numeric target as `<T` or `>=T` |
| `--features K` | give TabICL the K highest-variance Monroe features (default 360; `0` for all 720, which needs much more memory) |
| `--log` | model log10 of the target (see [Log scaling and zeros](#log-scaling-and-zeros)) |
| `--zeros {drop,floor,offset,error}` | with `--log`, how to treat zero values (default `drop`) |
| `--floor VALUE` | with `--zeros floor`, the value zeros become (default: half the smallest positive value) |
| `--offset VALUE` | with `--zeros offset`, the constant added before the log (default 1) |
| `--cv N` | report N-fold cross-validated R², Spearman and MAE, on the modelled scale (classification: accuracy, balanced accuracy, MCC, ROC AUC, log loss) |
| `--seed` | TabICL's random seed (default 42) |
| `--output-type {mean,median}` | TabICL's point estimate, for regression (default `mean`) |

Rows with no SMILES or a non-numeric value (for classification, an empty label)
are skipped, and so are molecules Monroe cannot featurize; each is reported with a count. Cross-validation keeps
repeated SMILES in the same fold. `--save` reports how much memory predicting with
the saved model will need.

### `predict FILE SMILES_COL --model MODEL.npz`

| Option | |
| --- | --- |
| `--model MODEL.npz` | a model written by `train --save` (required) |
| `-o, --output` | output CSV (default `<file>_pred.csv`) |

The output is the input file with every original column kept and predictions
added:

| Model trained with | Columns added |
| --- | --- |
| no `--log` | `<target>_pred` |
| `--log` | `log10_<target>_pred` and `<target>_pred` (original units) |
| `--log --zeros offset` | `log10_<target>_plus1_pred` and `<target>_pred` |
| `--task classification` | `<target>_pred` (the most probable class) and `<target>_prob_<class>` for each class |

For example, a model trained with `LogD --task classification --threshold 2`
adds `LogD_pred` (`<2` or `>=2`), `LogD_prob_<2` and `LogD_prob_>=2`. Each row's
probabilities sum to 1.

Molecules Monroe cannot featurize keep their row with an empty prediction.

Both modes show progress bars: two while embedding (`featurizing`, the RDKit
conformers, which is most of the time, then `encoding`), and two for TabICL (`TabICL
fit`, one pass per ensemble member over the training set, then `TabICL predict`,
one pass per member for each 500 new molecules). On LogD, fit passes take about 10
seconds each and predict passes about 1.

Both modes also take:

| Option | |
| --- | --- |
| `--max-memory GB` | the memory TabICL may use (default: what is free when it starts, less 6 GB) |
| `--workers N` | conformer-generation processes (default: every core) |
| `--batch-size N` | molecules per Monroe encoder pass (default 32) |

## Results

### LogD

Trained on the 5,039 molecules in `expansion_train.csv` with a LogD value, and
scored on the 2,270 in `expansion_test.csv` that have one:

| | R² | Spearman | MAE |
| --- | --- | --- | --- |
| **This script (TabICL, 360 features)** | **0.754** | **0.882** | **0.379** |
| TabICL, 180 features | 0.734 | 0.871 | 0.400 |
| TabPFN 3, all 720 features (the previous version of this script) | 0.722 | 0.860 | 0.403 |
| Validation study, TabICL, all 720 | 0.725 ± 0.014 | 0.861 ± 0.009 | 0.397 ± 0.011 |
| Validation study, TabPFN 3, all 720 | 0.718 ± 0.017 | 0.858 ± 0.011 | 0.402 ± 0.015 |

The study's figures are the mean of 25 cross-validation folds, each trained on
four fifths of the training set, from the `expansion-ml-comparison` study in
[model-validation-central](https://github.com/PatWalters/model-validation-central);
that study ran TabICL on a GPU.

On a 48 GB Apple-silicon Mac (CPU only), training took about 2 minutes, almost all of it
embedding. Predicting the test set took about 3: a minute to embed it and two for
TabICL, which peaked at 12.6 GB. Predicting again gives the same numbers to
within 0.00004 (see [Reproducibility](#reproducibility)).

### Choosing 360 features

Whether to cut the features was decided on LogD and then checked on the two
clearance endpoints, log10 scaled with zeros dropped and scored on the test
molecules with a measured value:

| | HLM R² / Spearman / MAE | MLM R² / Spearman / MAE |
| --- | --- | --- |
| **TabICL, 360 features** | **0.317 / 0.614 / 0.291** | **0.254 / 0.530 / 0.355** |
| TabICL, 180 features | 0.291 / 0.589 / 0.294 | 0.127 / 0.476 / 0.389 |
| TabPFN 3, all 720 features | 0.306 / 0.618 / 0.295 | 0.271 / 0.540 / 0.351 * |
| TabICL, all 720 features | does not fit (stopped at 18 GB) | does not fit (stopped at 18 GB) |

\* From an earlier run, with embeddings made before conformers were seeded; the
same run on seeded embeddings was stopped at 18 GB.

360 features is level with all 720 under TabPFN on the two clearances and ahead
on LogD. 180 falls away on MLM. The features are chosen by variance alone, on the
training set, without looking at the values being modelled.

Reducing the embedding with PCA instead does not work: on LogD, PCA to 256 or
128 dimensions kept 99% of the variance but dropped R² to 0.679 and 0.687.
TabICL appears to use Monroe's individual features, which PCA mixes together.

All of this is one train/test split per endpoint, without error bars.

## Memory

On CPU, TabICL holds a 128-number vector, and wider intermediates, for every cell
of the table it is working on: every row times every feature, all at once. That
makes it memory-hungry. With its default settings, the full LogD job (7,321 rows
× 720 features) needed far more than the 20–25 GB free on a 48 GB Mac with
ordinary applications open, and it took the machine down: macOS does not kill
the process that is using too much memory, the whole system stalls.

The script guards against that in three ways.

**It uses less memory to begin with.** TabICL runs its 8 ensemble members one at
a time rather than together, and with `kv_cache="repr"` it works through the
training rows once and then predicts new molecules 500 at a time. Neither changes
the predictions beyond rounding (less than 10⁻⁵ on LogD). And it reads 360
features rather than 720.

**It refuses a job that will not fit, before starting.** It estimates the peak
from the training rows and features (about 6 GB plus 5 KB per row per feature,
from measurements on that Mac, rounded up), and compares that with the memory it
may use: what is free when it starts, less 6 GB for the system, or `--max-memory`.
The check runs before any embedding, so a refusal is immediate, and it says what
would fit:

```
not starting TabICL: 4031 training rows x 720 features needs about 19.8 GB, and it may use 17.4 GB (26.4 GB free now, less 6 GB kept for the system and 3 GB for embedding). Close other applications, set --max-memory, or train with fewer features (--features 593 or fewer would fit)
```

**It stops TabICL if it goes over anyway.** TabICL runs in a separate process,
and the script checks its memory 20 times a second. If it passes its budget, or
the machine's free memory falls below 3 GB, the script kills it and exits:

```
stopped TabICL to protect the machine: it reached 18.5 GB, over its 18.5 GB budget. Close other applications, or train with fewer --features
```

Each run reports TabICL's peak memory, and `train --save` reports what predicting
with the saved model will need. As a guide, TabICL's measured peaks on that Mac:

| Job | Training molecules | 360 features | 720 features |
| --- | --- | --- | --- |
| the smoke test, per CV fold | about 385 | 3.3 GB | — |
| HLM CLint | 3,595 | 10.1 GB | stopped at 18 GB |
| MLM CLint | 4,375 | 12.5 GB | stopped at 18 GB |
| LogD | 5,039 | 12.6 GB | stopped at 20 GB |

The HLM and MLM figures come from the feature comparison above, which ran TabICL
without the script's chunked prediction, so the script itself uses a little less.

The classifier is the same network with a different output layer, so the
script uses the same estimate for it. It has only been measured on small jobs
so far: 2.9 GB per fold at 200 training molecules and 3.4 GB at 385, close to
the regressor.

On a GPU machine, TabICL batches its work on the GPU and none of this is an
issue, so `--features 0` (all 720) is practical there.

## Log scaling and zeros

`--log` refuses negative values, since a quantity that can be negative is usually
a log already (LogD is). It also needs a decision about zeros, which `log10`
cannot take. In assay data a zero almost always means "below what the assay can
quantify". `--zeros` chooses what to do with them:

- **`drop`** (default) leaves them out of training. If more than 10% of the values
  are zero, it warns that the model will not see the low end of the range.
- **`floor`** replaces them with `--floor`, by default half the smallest positive
  value (the usual LOQ/2 substitution). Set it to the assay's limit of
  quantitation if you know it.
- **`offset`** uses `log10(x + offset)`, as the validation study did with `+1`. The
  scale is nearly linear below the offset, and the right offset depends on the
  units. Back-transformed predictions below zero are reported as 0.
- **`error`** stops if there are any zeros.

`drop` is the default because it did best when the four were compared on the two
endpoints with the most zeros, HLM and MLM CLint, trained on the training file and
scored on the test file. That comparison was made with TabPFN as the head, before
the move to TabICL, and has not been repeated:

| | HLM log MAE | MLM log MAE | HLM Spearman | MLM Spearman |
| --- | --- | --- | --- | --- |
| floor, half the smallest value | 0.315 | 0.371 | 0.639 | 0.543 |
| floor, the smallest value | 0.312 | 0.368 | 0.635 | 0.543 |
| `log10(x + 1)` | 0.299 | 0.358 | 0.632 | 0.542 |
| **drop** | **0.291** | **0.351** | 0.641 | 0.540 |

Log MAE is over the test molecules with a measured non-zero value, in log10 units
of the original values, so the four are on the same scale.

All four rank molecules equally well. On error, the floor does worst, probably
because it puts the zeros in a cluster about 2.5 log units below everything else,
which pulls the average prediction down for any molecule that might belong to it.
None of them predicts measured zeros as low (typical prediction 5–12).

This is one train/test split with no error bars, the differences are small (0.02–
0.03 log units), and the test sets have only 15 and 9 zeros. Treat it as a
reasonable default, not a settled result.

## Reproducibility

Running the same command twice gives the same embeddings, bit for bit, whatever
`--workers` is set to, and predictions that agree to within about 10⁻⁵.

The embeddings took one fix. Monroe builds each molecule's conformer with RDKit's ETKDG from
random starting coordinates and never sets a seed, so a molecule's conformer
depended on which worker process happened to handle it: 60–90 of 300 molecules
changed between runs. The script now gives ETKDG a fixed seed (42) for every
molecule, without modifying the Monroe checkout. A model saved before this fix
still works, but `predict` warns that it should be retrained.

TabICL is seeded by `--seed`, but its arithmetic on CPU is not exactly
repeatable: from the same embeddings, runs in one session sometimes match bit for
bit and sometimes differ by up to 5 × 10⁻⁷, and runs in different sessions by up
to 3 × 10⁻⁵ (LogD predictions, which span about 7 units). That is floating-point
rounding, not a different model. This has been tested on CPU only; on a CUDA GPU,
expect differences of a similar size.

## Running without a GPU

Monroe's checkpoint loader assumes CUDA, because the weights were saved on a GPU,
so the script loads them onto the CPU itself. TabICL runs on CPU as it is, within
the memory limits above. The Mac's GPU (MPS) is not used.

## Models from the TabPFN version

This script replaced `monroe_tabpfn.py`, which used TabPFN 3 as the head. A model
saved by that version holds only embeddings, values and settings, so it still
loads, and TabICL predicts from it. But it uses all 720 features, so a large one
will not fit in memory on a laptop: one trained on the 5,039 LogD molecules is
refused with a message saying so. Retrain it to use 360 features.

## Files

| File | |
| --- | --- |
| `monroe_tabicl.py` | the script |
| `requirements.txt` | the Python environment |
| `expansion_train.csv`, `expansion_test.csv` | ExpansionRx ADME data, train/test split |
| `logd_tabicl.npz` | LogD model trained on `expansion_train.csv` (360 features) |
| `logd_tabicl_pred.csv` | its predictions for `expansion_test.csv` |

## Data

`expansion_train.csv` and `expansion_test.csv` are the training and test sets of
the [OpenADMET-ExpansionRx Blind
Challenge](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data):
ADME measurements on molecules from real drug discovery campaigns on RNA-mediated
diseases, contributed by Expansion Therapeutics and released by OpenADMET under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are included here
so the examples and results above can be reproduced.

## The saved model format

A model is a compressed NumPy `.npz` file, loaded without pickle, so it is safe to
open from an untrusted source. It holds the training embeddings (`X`, all 720
features), which features TabICL reads (`cols`), the modelled values (`y`, after
any log transform, or class indices for classification), the training SMILES, and
a JSON `meta` record: the column names, the task (`regression` or
`classification`), the class labels and any threshold, the log transform and its
parameters, both seeds, the TabICL point estimate, the `tabicl` version and
checkpoint, the number of features, and the training file. `predict` warns if the
installed `tabicl` or the checkpoint differs from the one the model was saved
with.

Adding classification moved the format from 1 to 2. Format 1 models (all of them
regression, including `logd_tabicl.npz`) still load as they are. An older copy of
the script refuses a format 2 file rather than misreading a classification model
as regression.
