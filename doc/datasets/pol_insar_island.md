# Pol-InSAR-Island

```{list-table}
:stub-columns: 1

* - Domain
  - SAR (synthetic aperture radar), multi-frequency PolInSAR land cover
* - Task(s)
  - fine classification (12 classes)
* - Covariance
  - 12×12 real (`logpower`, default; n = labelled pixels of a 32 m cell, ≥ 512) · 24×24 block-diagonal (`t6`)
* - Outputs
  - cov · mean · raw (32, 32, 12) (`logpower` only)
* - Splits
  - `spatial512_fold0` … `fold4` (`recommended` = fold 0; 2 700 / 982 / 1 107) · `fp1_to_fp2` · `official` checkerboard (optimistic)
* - Module
  - {mod}`spdnet_datasets.sar.pol_insar_island` (`fetch_pol_insar_island`, `PolInSARIslandDataset`)
```

## Description and source

Pol-InSAR-Island ("PolINSAR Iceland" in the mission text) is a benchmark for
land-cover classification from PolInSAR (polarimetric SAR interferometry).
The F-SAR airborne sensor of the DLR (German Aerospace Center) flew over
Baltrum, an island of the German Wadden Sea, in L band and S band. The
acquisitions are repeat-pass along two flight paths, FP1 and FP2. Both paths
are on a 1 m UTM 32N grid (Universal Transverse Mercator, zone 32 North):

- FP1: 2 502 × 3 616 px;
- FP2: 2 540 × 3 616 px;
- the two paths overlap on 728 m.

For every pixel and band, the dataset gives the 6 × 6 PolInSAR coherency
matrix T6, with official labels in 12 classes.

- Hochstuhl S. M. et al., *Pol-InSAR-Island – A Benchmark Dataset for
  Multi-frequency Pol-InSAR Data Land Cover Classification*, ISPRS Open
  Journal of Photogrammetry and Remote Sensing, 2023.
- Data: KITopenData, version 2, doi:10.35097/1700,
  <https://radar.kit.edu/radar/en/dataset/IAzBEMXnbTndvIZG>. Licence:
  CC BY-NC-SA 4.0. Download the BagIt archive by hand and place it under
  `<root>/SAR/Pol-InSAR-Island`.

## Raw data

```text
data/<FP>/<band>/T6         ENVI rasters: 6 real diagonals + 15 complex upper terms (36 real images)
label/<FP>/label_{train,test}.bin   12 classes, 0 = unlabelled
```

The georeferencing comes from the label headers, because the FP1 L-band
headers are wrong (zone 30, origin 0). The default raw output is the
(32, 32, 12) array of log-powers log T11 … log T66 of the L and S bands for
every pixel of the cell, unlabelled pixels included. The figure shows the mean
over the 12 channels.

```{raw} html
<iframe src="../_static/figures/pol_insar_island/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

The 12 classes are tidal flat, water, coastal shrub, dense high vegetation,
white dune, peat bog, grey dune, couch grass, upper saltmarsh, lower
saltmarsh, sand and settlement. Peat bog is rare and spatially concentrated:
in fold 0, all its 53 cells fall in the training part.

```{raw} html
<iframe src="../_static/figures/pol_insar_island/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/pol_insar_island/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

**Sample: one cell.** A cell is a non-overlapping 32 × 32 px (32 m) square of
one flight path. It is labelled by its majority class when two conditions
hold:

- at least `min_labelled` of its pixels are labelled: 512 for the default
  union cells, 256 for the official per-split cells;
- the majority class holds at least `purity=0.75` of them, the atlas
  definition.

The union cells, built on the train and test labels merged, number 5 033 in
52 blocks of 512 m.

**Representations** (`representation=`):

- `"logpower"` (default): p = 6 × number of bands = 12 real features, the
  logs of the six diagonal terms T11 … T66 of the L and S bands. The
  observations are the labelled pixels of the cell, so n ≥ 512 ≥ 2p. Every
  estimator of `core.estimators` applies. This is the only construction with
  correlations between the L and S bands, but it ignores the phases. The
  median condition number of the SCM covariances is about 20.
- `"t6"`: for each band, the mean T6 of the labelled pixels of the cell. That
  is a 6 × 6 Hermitian matrix: each pixel T6 has rank one, so the mean is the
  sample covariance of the scattering vectors. It is mapped to the real SPD
  (symmetric positive definite) matrix `[[Re T, −Im T], [Im T, Re T]]`, which
  is 12 × 12. The map is injective, commutes with products, inverses and
  logarithms, and doubles every eigenvalue. The two bands are placed on a
  block diagonal, giving 24 × 24. The L–S cross terms cannot be computed from
  per-band T6. Only `estimator="scm"` applies, and `return_raw` is not
  available.

The mean output is the per-channel mean over the labelled pixels. For
`logpower`, these are the p log-powers. For `t6`, they are the 36 real T6
parameters per band: the diagonals, then the real and imaginary parts of the
upper terms. At load time, the geometric scale of `core.normalize` is fitted
on the training part.

```{raw} html
<iframe src="../_static/figures/pol_insar_island/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/pol_insar_island/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

| Split | train / val / test | Definition |
|---|---|---|
| `spatial512_fold{k}` (`recommended` = fold 0) | 2 700 / 982 / 1 107 (fold 0) | `StratifiedGroupKFold(5)` on 512 m UTM blocks shared by FP1 and FP2; validation is a grouped fold of the rest; guard band |
| `fp1_to_fp2` | 1 476 / 0 / 1 964 | train on FP1, test on FP2, both outside the 728 m overlap (shift in acquisition geometry) |
| `official` (`cells="official"`) | 3 315 / 0 / 3 403 | cells built separately on `label_train` and `label_test` |

**Guard band.** In `spatial512_fold{k}`, training and validation cells that
touch a test cell are removed. Two cells touch when their centres are closer
than two cells (Chebyshev distance, in UTM). In fold 0 this removes 244 of the
5 033 cells. The other folds have 2 859–2 945 training and 930–1 064 test
cells.

**Official split.** The official train and test labels form a 256 × 256 px
checkerboard, so neighbouring cells sit on both sides. The loader raises a
`UserWarning`, and the scores are optimistic: the atlas measures +0.03 to
+0.08 of balanced accuracy. The official cells follow the atlas definition
(≥ 256 labelled pixels), which reproduces its 3 315 / 3 403 cells; at ≥ 512
pixels there would be only 2 380 / 2 529.

```{raw} html
<iframe src="../_static/figures/pol_insar_island/split_map.html" class="plotly-figure" style="height:620px" loading="lazy"></iframe>
<iframe src="../_static/figures/pol_insar_island/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Neighbouring cells in train and test (official checkerboard) | `recommended` uses 512 m blocks and a guard band; `official` raises a `UserWarning` | spatial correlation beyond one cell |
| The same ground seen from both flight paths (728 m overlap) | the blocks are shared by FP1 and FP2, so an overlapping area falls on one side; `fp1_to_fp2` excludes the overlap | none |
| Normalisation fitted on test data | geometric scale fitted on the training part only | none |
| Classes absent from a fold (peat bog) | stratified group folds | per-class scores undefined in some folds; report macro scores over the classes present |

## Supported tasks

- `task="fine"`: 12 classes.
- `drop_classes` / `merge_classes` take class names, for example merging the
  two saltmarsh classes or the two dune classes.
- `balance` acts on the training part only.
- `bands` (L, S or both), `cell`, `purity` and `min_labelled` change the
  construction.

## Usage

```python
from spdnet_datasets.sar.pol_insar_island import PolInSARIslandDataset

ds = PolInSARIslandDataset(split="recommended")                 # logpower, 12×12
t6 = PolInSARIslandDataset(representation="t6")                 # 24×24, scm only
off = PolInSARIslandDataset(cells="official", split="official")  # optimistic, warns
```
