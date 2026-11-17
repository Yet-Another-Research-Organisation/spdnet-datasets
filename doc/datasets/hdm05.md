# HDM05

```{list-table}
:stub-columns: 1

* - Domain
  - skeleton (motion capture)
* - Task(s)
  - fine classification (104 classes with `c93`, 117 with `c31`)
* - Covariance
  - provided 93×93 (31 joints × 3 coordinates), n = number of frames of the sequence (186 to 901 after the n ≥ 2p filter); option 31×31
* - Outputs
  - cov only (`OUTPUTS = ("cov",)`: no raw data, no mean)
* - Samples
  - `c93`: 1 482 sequences (604 of 2 086 excluded, T < 186) · `c31`: 2 086
* - Splits
  - `stratified` = `recommended`: 5 stratified folds (`c93`, fold 1: 888 / 297 / 297; `c31`: 1 251 / 417 / 418). No actor split possible.
* - Module
  - `spdnet_datasets.skeleton.hdm05` (`fetch_hdm05`, `HDM05Dataset`, registry name `hdm05`)
```

```{warning}
**Actor leakage.** HDM05 was recorded with 5 actors, but the copy has no actor
or take identifier: segments of one take and one actor fall in train and
test, which inflates the scores compared with a cross-actor protocol. The literature on
SPDNet (a network for symmetric positive definite, SPD, matrices) uses 10 random 50/50 splits, which has the same
bias. `HDM05Dataset` emits a `UserWarning`.
```

```{dataset-preview} hdm05
```

## Description and source

HDM05 is a motion capture database of the University of Bonn: actions
performed by 5 actors, recorded with an optical Vicon system at 120 Hz. The
copy is the version used by SPDNet: one covariance matrix of the 31 joints × 3
coordinates per motion segment, in 117 action classes.

- Reference: M. Müller, T. Röder, M. Clausen, B. Eberhardt, B. Krüger and
  A. Weber, "Documentation Mocap Database HDM05", Technical report
  CG-2007-2, Universität Bonn, 2007. Source:
  <http://resources.mpi-inf.mpg.de/HDM05/>. Licence: free for research (site
  conditions).
- Covariance protocol: Z. Huang and L. Van Gool, "A Riemannian Network for SPD
  Matrix Learning", *AAAI (Association for the Advancement of Artificial Intelligence)
  Conference* 2017.
- Local copy: `<data root>/SKELETON/HDM05`, 2 086 files
  `{id}_{frames}_{class}.npy`, each a symmetric 93×93 float64 matrix.

## Getting the data

- Default (`source="local"`), **manual**: the 2 086 `{id}_{frames}_{class}.npy`
  matrices are an export without public source. Without a local copy the fetch
  opens <https://resources.mpi-inf.mpg.de/HDM05/> and raises
  `ManualDownloadRequired`; give the export with `fetch_hdm05(archive=...)`.
- `source="eth"`, direct: `fetch_hdm05(source="eth")` downloads
  `HDM05_SPDData.zip` (603 MB) of the SPDNet authors (ETH Zurich), checks its
  sha256 (computed on 2026-11-15, none is published) and reads the `.mat`
  matrices directly. These are **other data**: 19 491 matrices of 130 actions
  (segments, grouped by sequence in the split), not the 2 086 local ones; only
  `features="c93"`.

## Raw data

The skeletons, the estimator used to build the matrices, the centring and the
actor and take identifiers are **not** in the copy: only the matrices are, so
there is no `raw_samples` figure and no `return_raw` / `return_mean`. The file
name gives the number of frames T of the sequence (94 to 901, median 251) and
the class (0 to 116). The order of the 93 variables is not documented; the
atlas infers a coordinate-major order (x of the 31 joints, then y, then z) from
the block structure of the variances.

## Classes

Classes are named `"000"` to `"116"` (index in the SPDNet release; the action
names are not in the copy). The class sizes are uneven: with `c93`, 1 to 31
sequences per class; with `c31`, 4 to 52.

```{dataset-figure} hdm05/class_hist
```

Without a mean output, the profile figure shows the mean log10 variance of the
93 variables per class (first 8 classes): the three blocks of 31 values are the
three coordinates.

```{dataset-figure} hdm05/class_means
```

## Covariance construction

The matrices are provided and cannot be re-estimated, so only
`estimator="scm"` is accepted (any other value raises `ValueError`).

- `features="c93"` (default): the provided matrices, p = 93. The n ≥ 2p rule
  needs T ≥ 186 frames: the **604** sequences with T < 186 are excluded and
  listed in `dataset.excluded`, and 13 classes disappear (1 482 sequences,
  104 classes).
- `features="c31"`: 31×31 joint covariance, sum of the three per-coordinate
  blocks of C93, `C31[j, k] = Σ_d C93[31d + j, 31d + k]`. 2p = 62 ≤ 94 keeps
  all 2 086 sequences and the 117 classes. It relies on the inferred
  coordinate-major order.

Every kept matrix is checked to be symmetric positive definite (SPD) and
symmetrised. With `normalize=True` (default), the covariances are multiplied
by the scalar fitted on the training part (geometric mean of the eigenvalues
= 1).

```{dataset-figure} hdm05/cov_samples
```

```{dataset-figure} hdm05/eigen_spectrum
```

## Splits

There is no official split and no actor information.
`split="stratified"` (alias `"recommended"`): 5 folds stratified by class
(`StratifiedKFold`, seed 0); test = fold `fold`, validation = the next fold,
train = the three others. With `c93` some classes have fewer than 5
sequences, so they are missing from the validation or test part of some
folds. Groups are the sequences themselves (one group per sequence), so the
right panel of the figure is trivial.

```{dataset-figure} hdm05/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| same actor and same take in train and test | none possible (no identifier); `UserWarning` | **high**: scores optimistic compared with a cross-actor protocol |
| consecutive ids of one class (116 class changes over 2 085 transitions) | the order does not give the takes; not used | documented |
| short sequences (T < 2p) give ill-conditioned matrices | excluded with `c93`; `c31` keeps them with n ≥ 2p | none |

## Supported tasks

- `task="fine"`: 104 (`c93`) or 117 (`c31`) classes. No coarse grouping is
  defined (the action names are not in the copy).
- `classes`, `max_classes` and `max_samples_per_class` are honoured;
  `drop_classes`, `merge_classes`; `balance="undersample" | "oversample" |
  "weights"` (training part only).

## Usage

```python
from spdnet_datasets.skeleton.hdm05 import HDM05Dataset

train = HDM05Dataset(subset="train", fold=1)                    # 93x93, warns about actors
all31 = HDM05Dataset(subset=None, features="c31", max_classes=20)
```

## Migration notes

- `scaling_factor` (the former loader multiplied the matrices by a constant,
  e.g. 190) is replaced by `normalize=True`. The legacy class
  `spdnet_datasets.real.hdm05` keeps `scaling_factor`, `subset=None` and
  `normalize=False`, but the registry name `"hdm05"` now points to the new
  class, which **ignores** `scaling_factor` (it is swallowed by the extra
  keyword arguments): a configuration `{"name": "hdm05", "scaling_factor": 190}`
  now gives normalised covariances.
- Sequences with T < 186 are no longer kept with `c93`; use `c31` to keep all
  2 086 sequences.
