# GSOFF

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), airborne, tree species
* - Task(s)
  - fine classification (12 classes, `code 1` to `code 12`)
* - Covariance
  - 63×63 (one band in two of 125), n = 225 pixels of a 15×15 window (n / 2p = 1.79)
* - Outputs
  - cov · mean · raw (15 × 15 × 63)
* - Samples
  - 1 249 windows, none discarded
* - Splits
  - `spatial` = `recommended` (96-pixel blocks, fold 0: 739 / 250 / 255)
* - Module
  - `spdnet_datasets.hyperspectral.gsoff` (`fetch_gsoff`, `GSOFFDataset`, registry name `gsoff`)
```

## Description and source

GSOFF (Gaofeng State Owned Forest Farm) is an airborne
hyperspectral scene with a ground truth (GT) of tree species. The local copy
comes from the code repository of a 3D convolutional network for tree species
classification:
<https://github.com/Bin-Zh/Three-dimensional-convolutional-neural-network-model-for-tree-species-classification-using-airborne->.
The repository's README links to a Baidu Pan download. The GT raster was
produced from the polygons `tree_species.shp` by the repository's script
`1vector2raster.py`.

```{warning}
**Provenance and licence unknown.** No reference paper, licence, sensor
description, wavelengths or class names could be found. The attribute table of
the shapefile (`.dbf`) is missing, so classes are only known as codes 1 to 12.
The acquisition date is unknown as well. Do not redistribute the data, and
cite the repository above if you publish results.
```

- Scene: `source/data/testarea2_new/sg_quac_test2.tif`, 572 × 906 pixels × 125
  bands, float32. QUAC stands for QUick Atmospheric Correction. The scene is in
  a local metric frame with no EPSG (European Petroleum Survey Group) code. The atlas fiche says 570 × 906; the
  file is 572 × 906.
- GT: `sg_quac_test2_gt.tif`, 572 × 906 uint8, codes 1 to 12, 0 = unlabelled
  (213 700 labelled pixels).
- The dataset folder also holds 1 106 prepared 16 × 16 windows
  (`gaofeng_windows_*.npy`). This loader does not use them: it recomputes
  15 × 15 windows from the scene.

## Raw data

The raw output is the 15 × 15 × 63 window that the covariance is computed on.
The figure shows the band-mean image of four windows.

```{raw} html
<iframe src="../_static/figures/gsoff/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

Twelve codes, highly imbalanced: `code 1` has 355 windows and `code 12` has 5.
The counts are `code 1` 355, `code 2` 42, `code 3` 44, `code 4` 87, `code 5`
229, `code 6` 99, `code 7` 29, `code 8` 67, `code 9` 77, `code 10` 122,
`code 11` 93 and `code 12` 5. The first figure gives the counts per part of
the spatial split. The second figure gives the mean spectrum of the first 8
classes (in alphabetical order of their names).

```{raw} html
<iframe src="../_static/figures/gsoff/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/gsoff/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

The construction follows the original preprocessing script; the window size is
15 instead of 16:

1. **Windows**: for each class, every grid of step 15 with an offset in
   [0, 15)² is scanned. A window is *pure* for a class when at least 80 % of
   its labelled pixels belong to that class (`threshold=0.8`). The offset that
   yields the most pure windows of the class is kept. Windows of different
   classes may therefore overlap. With `window=16` the same function gives
   back exactly the 1 106 prepared windows, at the same positions (tested).
2. **Bands**: one in two (`band_step=2`), so p = 63. With `band_step=1`
   (p = 125), a 15 × 15 window has n = 225 < 2p = 250, and the loader raises
   `InsufficientSamplesError` before computing anything.
3. **Observations**: the non-zero, finite pixels of the window. Windows with
   fewer than 2p of them would be discarded and counted
   (`info["excluded_n_lt_2p"]`): none are, because every window has 225 pixels.
4. **Estimator**: `estimator="scm"` by default, the sample covariance matrix
   (SCM) of the centred pixels. `"ledoit_wolf"`, `"tyler"` and `"student"` are
   also available. With n / p ≈ 3.6, the SCM is poorly conditioned; consider
   Ledoit–Wolf shrinkage.
5. **Normalisation**: with `normalize=True` (the default), the covariances are
   multiplied by a scalar fitted on the training part of `split`, so that the
   geometric mean of their eigenvalues is 1. The cache keeps the unscaled
   matrices. Means and raw windows are multiplied by √s.

```{raw} html
<iframe src="../_static/figures/gsoff/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/gsoff/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

No paper is known, so there is no official split. The loader proposes a
spatial split, as recommended in the dataset atlas:

- The 96 × 96-pixel blocks of the scene (50 non-empty blocks) are the groups
  of a `StratifiedGroupKFold(5)` (seed 0). Test = fold `fold`, validation =
  fold `fold + 1` and train = the three other folds.
- Validation and test windows that overlap a training window are removed, and
  so are test windows that overlap a validation window. Overlap means a
  Chebyshev distance < 15 between top-left corners.
- With `fold=0`: 739 train, 250 validation and 255 test windows, with 5
  windows removed by the buffer. `code 12` is absent from validation and test,
  and `code 7` is absent from validation.

```{raw} html
<iframe src="../_static/figures/gsoff/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| overlapping windows (classes of the same location can overlap) | windows of val/test overlapping train (or test overlapping val) are removed; checked by the real-data test | none |
| spatial autocorrelation between neighbouring windows | groups = 96-pixel blocks, never split | windows on both sides of a block border remain close |
| random split of windows | not offered | none |
| rare classes | stratification over blocks | `code 12` (5 windows) absent from val and test at fold 0 |

## Supported tasks

- `task="fine"`: the 12 codes. There is no coarse map (`task="coarse"`
  raises an error), because the class names are unknown.
- `drop_classes=["code 12"]`, `merge_classes={...}`, `classes=[...]` and
  `max_classes` restrict or regroup the classes.
- `balance="undersample"`, `"oversample"` or `"weights"` acts on the training
  part only.

## Usage

```python
from spdnet_datasets.hyperspectral.gsoff import GSOFFDataset, fetch_gsoff

ds = GSOFFDataset()                               # all 1 249 windows, ds.splits["spatial"]
train = GSOFFDataset(subset="train", estimator="ledoit_wolf")
bunch = fetch_gsoff(keep="keep_covariance")       # arrays only (covariances, labels, splits)
```

Through the manager: `{"name": "gsoff", "split": "recommended", ...}`.

## Migration notes

- The 15 × 15 windows recomputed from the scene replace the prepared 16 × 16
  windows: 1 249 windows instead of 1 106 (new reference counts, which differ
  from the atlas fiche).
- `GSOffDataset` is kept as an alias of `GSOFFDataset`.
- `split=` now names a split (`"recommended"`, `"spatial"`); the former
  `split="train"` raises an error. Use `subset="train"` instead. `subset=None`
  (the default) keeps every sample.
- The estimator is chosen with `estimator=`. A former keyword such as
  `cov_method="ledoit_wolf"` is silently absorbed by `**kwargs` and gives the
  SCM.
