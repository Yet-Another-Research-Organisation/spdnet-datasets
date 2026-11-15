# MOABB adapter

[MOABB](https://moabb.neurotechx.com) (Mother of All BCI Benchmarks) gathers
public brain-computer interface (BCI) datasets, mostly electroencephalography
(EEG), along with paradigms and evaluation protocols. The paradigms include
motor imagery, P300 (an event-related potential, ERP) and SSVEP (steady-state
visually evoked potentials). `spdnet_datasets.moabb` turns its epochs into symmetric positive
definite (SPD) covariance matrices for SPDNet models. It reuses MOABB for
everything else: the datasets, their download, the epoching, the paradigms and
the evaluations.

```{note}
EEG is a biosignal, and biosignal datasets are outside the main selection of
`spdnet-datasets`. The adapter therefore lives on the dedicated `moabb` branch.
```

## Install

```bash
pip install "spdnet-datasets[moabb]"
```

The `moabb` extra pulls in `moabb>=1.7.2`, and with it MNE-Python and
pyriemann.

MOABB downloads the recordings into the MNE data directory. Set it before the
first run:

```bash
export MNE_DATA=/path/to/mne_data
```

## What the adapter adds

| Object | Use it for |
|---|---|
| `Covariances` | First step of a scikit-learn pipeline passed to a MOABB evaluation |
| `MOABBCovariances` | PyTorch dataset of the covariances, with groups and MOABB splits, for SPDNet training |
| `moabb_splits(y, metadata)` | Train/test indices of the three MOABB protocols |

`Covariances` and `MOABBCovariances` estimate each covariance with
`spdnet_datasets.core.estimators.estimate_covariance`:

- `estimator` is one of `"scm"`, `"ledoit_wolf"`, `"tyler"` or `"student"`;
  `tyler` and `student` need the `robust` extra;
- an epoch with `n_times < 2 * n_channels` raises `InsufficientSamplesError`;
- a matrix that is not SPD raises `NotSPDError`.

`Covariances` is not `pyriemann.estimation.Covariances`. Both map
`(n_trials, n_channels, n_times)` to `(n_trials, n_channels, n_channels)`, but
this one runs the checks above.

## Use it in a MOABB evaluation

```python
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from spdnet_datasets.moabb import Covariances

dataset = BNCI2014_001()
dataset.subject_list = [1, 2]
pipelines = {
    "LW+TS+LR": make_pipeline(Covariances("ledoit_wolf"), TangentSpace(),
                              LogisticRegression(max_iter=1000)),
}
evaluation = CrossSessionEvaluation(paradigm=LeftRightImagery(), datasets=[dataset])
results = evaluation.process(pipelines)   # one row per (subject, session)
```

With `normalize=True`, `Covariances.fit` learns the geometric scale (see
`core.normalize`) on the training fold only. `transform` then applies that
scale to the test fold.

## Train an SPDNet on MOABB data

```python
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

from spdnet_datasets.base import create_dataloaders
from spdnet_datasets.moabb import MOABBCovariances

ds = MOABBCovariances(BNCI2014_001(), LeftRightImagery(), subjects=[1, 2, 3],
                      estimator="scm", split="cross_subject_0")
train, val, test = create_dataloaders(ds, batch_size=32, num_workers=0,
                                      split="cross_subject_0")
x, y = next(iter(train))   # x: (32, 22, 22) float64 SPD matrices, y: (32,) labels
```

`MOABBCovariances` calls `paradigm.get_data(dataset=..., subjects=...)` once.
It stores the result in a cache, then estimates one covariance per epoch.

### Output contract

Each item is `(x, y)`, where `y` indexes `ds.classes`. The value of `x`
depends on the `return_*` flags:

| Flags | `x` |
|---|---|
| default | `cov`, shape `(n_channels, n_channels)`, float64 |
| `return_mean=True` | `(cov, mean)`, with `mean` of shape `(n_channels,)`, float64 |
| `return_raw=True` | `(cov, raw)`, with the epoch `raw` of shape `(n_times, n_channels)`, float32 |
| both | `(cov, mean, raw)` |

### Groups and splits

- `ds.metadata` is MOABB's metadata, one row per trial. It holds `subject`,
  `session` and `run`.
- `ds.subjects` and `ds.sessions` give each trial's subject and session.
- `ds.groups` is `"<subject>-<session>"`.
- `ds.splits` holds the folds produced by MOABB's own splitters, so they follow
  the same protocols as the MOABB evaluations. The within-session folds are
  identical to those of `WithinSessionEvaluation(random_state=seed)` when both
  use the same seed:

| Split names | Protocol | Matches |
|---|---|---|
| `within_session_<k>` | 5-fold stratified cross-validation inside each (subject, session) | `WithinSessionEvaluation` |
| `cross_session_<k>` | leave one session out, per subject (≥ 2 sessions) | `CrossSessionEvaluation` |
| `cross_subject_<k>` | leave one subject out (≥ 2 subjects) | `CrossSubjectEvaluation` |
| `recommended` | first fold of the strictest available protocol | |

Every split is `{"train", "val", "test"}`, and `val` is empty. MOABB
protocols have no validation set: carve one out of `train` yourself, for
example with `core.splits.group_split`.

### Normalization

The covariances can be multiplied by a geometric scale, so that the geometric
mean of the eigenvalues over the training covariances is 1. That scale is only
computed when you pass `split=`, and it is fitted on that split's training
indices. Without `split`, the covariances are left unscaled.

Pass the same split name to `MOABBCovariances(split=...)` and to
`create_dataloaders(split=...)`. With two different names, the scale is fitted
on trials that may be in the test set of the other split.

### Cache and `keep`

MOABB and MNE manage the raw recordings in `MNE_DATA`. The adapter never
deletes them. Under `SPDNET_CACHE_DIR/moabb_<dataset code>/`, it caches:

- `processed/<hash>/`: the epochs, labels and metadata;
- `covariance/<hash>/`: the unscaled covariances, the channel means, the
  labels and the metadata.

The hash covers the dataset code, the subjects, the paradigm class with its
public settings, and the estimator. A second run with the same settings reads
the cache and does not call `get_data` again. `force=True` recomputes.

| `keep` | Cache content |
|---|---|
| `keep_full` (default) or `keep_processed` | epochs and covariances |
| `keep_covariance` | covariances only; `return_raw` is then unavailable |

## Tested versions

| Package | Version |
|---|---|
| moabb | 1.7.2 |
| mne | 1.13.2 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| torch | 2.14.1 |
| pyriemann | 0.12 |
| Python | 3.11 |
