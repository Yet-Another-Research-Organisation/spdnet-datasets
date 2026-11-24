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

```{dataset-preview} moabb
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
| `MOABBDataset` | The same dataset built from class names, registered as `"moabb"` for spdnet-training configs |
| `moabb_splits(y, metadata)` | Train/test indices of the three MOABB protocols |
| `stratified_split`, `subject_fold_splits` | Within- and cross-subject splits with a validation set |

`Covariances` and `MOABBCovariances` estimate each covariance with
`spdnet_datasets.core.estimators.estimate_covariance`:

- `estimator` is one of `"scm"`, `"ledoit_wolf"`, `"tyler"` or `"student"`;
  `tyler` and `student` need the `robust` extra;
- in `Covariances` (a scikit-learn transformer, which cannot drop trials), an
  epoch with `n_times < 2 * n_channels` raises `InsufficientSamplesError` and a
  matrix that is not SPD (condition number above `1e12` included) raises
  `NotSPDError`;
- in `MOABBCovariances`, such trials are excluded and counted
  (`dataset.n_excluded`); labels and metadata stay aligned, and an error is
  raised only if no trial is left. Typical causes are a common average
  reference (rank `n_channels - 1`) or a flat channel;
- every kept trial has the same channels, in the same order
  (`dataset.channels`). Subjects whose channel list differs from the most
  common one in the dataset, or differs between their own sessions or runs, are
  excluded with a warning and listed in `dataset.excluded_subjects`.

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

`MOABBCovariances` calls `paradigm.get_data(dataset=..., subjects=[s],
return_epochs=True)` once per subject, keeps the subjects with the same
channels, and stores the epochs in a cache. It then estimates one covariance
per epoch.

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
  use the same seed. Each protocol is built on its own: one that cannot be
  built (for instance fewer trials per class and session than `n_folds`, or a
  single subject for cross-subject) is left out with a warning, and the others
  remain available:

| Split names | Protocol | Matches |
|---|---|---|
| `within_session_<k>` | 5-fold stratified cross-validation inside each (subject, session) | `WithinSessionEvaluation` |
| `cross_session_<k>` | leave one session out, per subject (subjects with a single session are skipped) | `CrossSessionEvaluation` |
| `cross_subject_<k>` | leave one subject out (≥ 2 subjects) | `CrossSubjectEvaluation` |
| `recommended` | first fold of the strictest available protocol | |
| `stratified` | trials split into train / val / test, stratified by class (only with `holdout=(val_size, test_size)`) | |
| `subject_fold_<k>` | cross-subject folds with held-out validation subjects and a calibration pool (only with `subject_folds=n`) | |

Every split has `train`, `val` and `test` (the `subject_fold_<k>` splits also
`calib_val` and `calib`). In the MOABB protocols `val` is empty, since MOABB has
no validation set: carve one out of `train` yourself, for example with
`core.splits.group_split`. The `stratified` split is within-subject: it draws
trials, stratified by class, so the subjects are mixed across the three parts.
It and `subject_fold_<k>` are the splits to use with `spdnet-train`, whose early
stopping monitors the validation set.

The `subject_fold_<k>` splits (`subject_folds=n`, {func}`~spdnet_datasets.moabb.subject_fold_splits`)
are the cross-subject counterpart, built like `stratified_fold_<k>` of the
nilearn datasets:

- the subjects are cut into `n` groups of nearly equal size, drawn with
  `fold_seed` (default 0, kept apart from `seed` so that a subject stays in the
  same fold across runs; excluding a subject changes every group). Fold `k` tests group `k`, validates on the next
  `val_groups` groups (default 1) and trains on the others. Every subject is
  tested once over the `n` folds;
- each session of a validation or test subject is cut in recording order: its
  first trials (`calib_fraction`, default one half, rounded down) form an unlabeled
  calibration pool (`calib_val`, `calib`), the others are evaluated (`val`,
  `test`). An adaptation to the session only sees trials recorded before the
  evaluated ones, in the same session, so the folds stay purely cross-subject
  for datasets with several sessions. The first `K` trials of a pool are the
  calibration of size `K`, and `val` and `test` do not depend on `K`. The
  loader must keep the recording order. MOABB keeps it for BNCI2014_001,
  Weibo2014 and Schirrmeister2017, and stores Cho2017 class by class: the labels
  are only used to raise an error when a part of a session misses a class.

Leakage risks and measures:

| Risk | Measure | Residual |
|---|---|---|
| same subject in training and evaluation | subject groups, checked by `core.splits.assert_no_leakage` | none |
| calibration recorded after the evaluated trials | first trials of each session in recording order | loaders that do not keep the recording order: detected when they store the trials class by class (Cho2017) |
| geometric scale fitted on held-out subjects | fitted on `train` only | none |

### Normalization

The covariances can be multiplied by a geometric scale, so that the geometric
mean of the eigenvalues over the training covariances is 1. That scale is only
computed when you pass `split=`, and it is fitted on that split's training
indices. Without `split`, the covariances are left unscaled. With the scale `s`,
the channel means (`return_mean`) and the epochs (`return_raw`) are multiplied
by `sqrt(s)`, so that the covariance of the returned epoch is still the returned
covariance.

Pass the same split name to `MOABBCovariances(split=...)` and to
`create_dataloaders(split=...)`. With two different names, the scale is fitted
on trials that may be in the test set of the other split.

### From a configuration file

`MOABBDataset` builds the same dataset from class names. It is registered as
`moabb`, so `DatasetManager` and the Hydra configs of `spdnet-training` can
select it:

```yaml
# configs/dataset/bnci2014_001.yaml
name: moabb
dataset: BNCI2014_001          # class of moabb.datasets
paradigm: MotorImagery         # class of moabb.paradigms
paradigm_params: {n_classes: 4, events: [left_hand, right_hand, feet, tongue],
                  fmin: 8.0, fmax: 32.0, resample: 128}
dataset_params: {}             # e.g. {sessions: [1, 2]}
exclude_subjects: []
input_dim: 22                  # checked against the number of channels
split_strategy: stratified
holdout: [0.1, 0.15]           # (val_size, test_size) of the stratified split
seed: 42
batch_size: 32
num_workers: 0
```

```bash
spdnet-train --config-dir configs dataset=bnci2014_001 model.hidden_layers_size=[16]
```

The BiMap sizes must not exceed the number of channels (22 here). The
`scripts/eeg/` folder of spdnet-training holds such configs for several
motor-imagery datasets, with one experiment preset per dataset.


### Cache and `keep`

MOABB and MNE manage the raw recordings in `MNE_DATA`. The adapter never
deletes them. Under `SPDNET_CACHE_DIR/moabb_<dataset code>/`, it caches:

- `processed/<hash>/`: the epochs, labels and metadata;
- `covariance/<hash>/`: the unscaled covariances, the channel means, the
  labels and the metadata.

The `processed` hash covers the dataset code and every public setting of the
MOABB dataset object (for instance `PhysionetMI(imagined=True)` and
`PhysionetMI(executed=True)` get different entries), the subjects, and the
paradigm class with its public settings. Settings that are objects are keyed
by their configuration (`get_params()` or their public attributes), never by
their `repr`; a setting that cannot be fingerprinted raises `TypeError` rather
than risking two configurations sharing one cache entry. The
`covariance` hash adds the estimator only: changing the estimator recomputes
the covariances from the cached epochs without calling `get_data` again. A
second run with the same settings reads the cache. `force=True` recomputes.

MOABB needs no account: it downloads the recordings itself (from the
datasets' official servers or NEMAR) into `MNE_DATA`.

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

<!-- covariance-series: generated section, edit write_sections.py -->
## Covariance series

{class}`~spdnet_datasets.series.CovarianceSeries` cuts each EEG epoch into a series of
covariance tokens (see {doc}`/guide/series` and {doc}`series_overview`). The modes and
their defaults come from the registry; the first one is the default.

```{series-table} moabb
```

`MOABBCovariances` is built from a MOABB dataset and paradigm, so the series
wraps an instance (no loader name). The sampling rate is read from it (`sfreq`: the
paradigm's `resample`, otherwise `(n_times - 1)` over the epoch duration). The filter bank
needs epochs that keep 4-40 Hz: the default `LeftRightImagery` filters 8-32 Hz, which leaves
the outer bands nearly empty and gives a warning; build the paradigm with
`LeftRightImagery(fmin=1, fmax=45)`, as below. `temporal` uses windows of 4C samples when that
gives at least 4 windows, otherwise the window that gives 4 (never below 2C); an epoch too
short for 2 windows raises `ValueError`. Measured on BNCI2014_001 (C = 22, 250 Hz, 9
subjects) and Cho2017 (C = 64, 512 Hz, 5 subjects).

```{dataset-figure} moabb/series_filter_bank
```

```{dataset-figure} moabb/series_temporal
```

Measured on the real data (BERTHA, November 2026, default parameters of each mode,
`describe()`): number of samples, tokens per series, observations per token among the
unmasked tokens, fraction of SPD (unmasked) tokens and condition number κ of the unmasked
tokens.

| Mode | Samples | S (max / median) | p | n_t (median, min) | n_t / p | SPD tokens | κ (5 / 50 / 95 %) |
|---|---|---|---|---|---|---|---|
| `filter_bank` (BNCI2014_001, 9 subjects) | 2 592 | 9 / 9 | 22 | 1 001 (1 001) | 45.5 | 100.00 % | 1.3e+04 / 5.3e+04 / 2.5e+05 |
| `temporal` (BNCI2014_001, 9 subjects) | 2 592 | 22 / 22 | 22 | 88 (88) | 4.0 | 99.99 % | 5.3e+04 / 1.6e+05 / 5.8e+05 |
| `filter_bank` (Cho2017, 5 subjects) | 1 000 | 9 / 9 | 64 | 1 537 (1 537) | 24.0 | 99.31 % | 1.0e+10 / 5.6e+10 / 3.2e+11 |
| `temporal` (Cho2017, 5 subjects) | 1 000 | 12 / 12 | 64 | 256 (256) | 4.0 | 100.00 % | 1.8e+07 / 6.1e+07 / 4.8e+08 |

The adapter is built from objects, so the series wraps an instance; every default
can be overridden:

```python
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from spdnet_datasets.moabb import MOABBCovariances
from spdnet_datasets.series import CovarianceSeries, collate_series

# a pass band that covers the filter bank (4-40 Hz)
epochs = MOABBCovariances(BNCI2014_001(), LeftRightImagery(fmin=1, fmax=45), subjects=[1, 2])
series = CovarianceSeries(epochs)  # filter_bank: 9 tokens of 22 x 22
series = CovarianceSeries(epochs, "temporal", window=125)  # windows of 0.5 s at 250 Hz
(tokens, mask), label = series[0]
```
<!-- /covariance-series -->

## How to cite

````{admonition} If you use this adapter in a publication, cite
:class: tip

```{include} _cite/moabb.md
```
````
