# Rices_90

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), laboratory, seed varieties
* - Task(s)
  - fine classification (90 rice varieties)
* - Covariance
  - 64×64 (256 bands averaged by groups of 4), n = 141 to 914 pixels of a grain (median 399; n / 2p ≥ 1.10)
* - Outputs
  - cov · mean · raw (n_pixels × 64, one grain)
* - Samples
  - 8 803 grains from 180 cubes (none discarded)
* - Splits
  - `bundle` = `recommended` (bundle 1 → 2: 3 518 / 880 / 4 405) · `bundle_reverse` (3 524 / 881 / 4 398) · `cross_session` (test restricted to 17 varieties: 846)
* - Module
  - `spdnet_datasets.hyperspectral.rices90` (`fetch_rices90`, `Rices90Dataset`, registry name `rices90`)
```

## Description and source

Rices_90 contains VNIR (visible and near-infrared) cubes of 90 Vietnamese
rice varieties, acquired by a push-broom laboratory imager with 256 bands
from 383 to 1 006 nm. Each variety has two bundles of about 48 grains laid
out on a grid, which gives 180 cubes. The acquisitions span 9 sessions,
whose dates are not provided (only the session folders).

- Reference: S. D. Fabiyi *et al.*, "Varietal classification of rice seeds
  using RGB and hyperspectral images", *IEEE Access* 8, 2020. This citation
  comes from the bibliography of the dataset atlas and was not re-verified.
- Licence: unknown (ask the authors). No public mirror was identified; the
  loader reads a local copy.

## Getting the data

Direct, no account: Zenodo record [3241923](https://zenodo.org/records/3241923)
(CC BY 4.0). Without a local copy, `fetch_rices90()` downloads
`RGB and VIS-NIR HSI Data for 90 Rice Seed Varieties.zip` (17.3 GB, md5 of the
record checked) into `<cache>/Rices_90/downloads/` and extracts it (about 35 GB
in total). The zip has the layout read here; the CRC-32 of `index.csv` and of
sampled cubes equal the local copy.

```python
fetch_rices90(archive="~/Downloads/RGB and VIS-NIR HSI Data for 90 Rice Seed Varieties.zip")
```

## Raw data

The cubes are ENVI files (ENVI: Environment for Visualizing Images), stored
as BIL (band interleaved by line) uint16 with 12 useful bits. Each cube has
336 columns, 921 to 932 lines and 256 bands. `index.csv` gives the variety,
bundle, session folder and file. Each session folder has a dark reference
`black.raw`, and each cube holds a white reference panel below the grains.

The raw output of a sample is the **matrix of the pixels of one grain**
(n_pixels × 64 binned bands), not an image. The figure shows four grains as
band × pixel matrices.

```{raw} html
<iframe src="../_static/figures/rices90/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

90 varieties of 96 to 106 grains each: the classes are almost balanced. The
mean-spectrum figure shows only the first 8 varieties in alphabetical order.

```{raw} html
<iframe src="../_static/figures/rices90/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/rices90/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

The construction follows the atlas recipe, as did the former `series_rices`:

1. **Calibration**: the session's dark reference is subtracted. The
   reflectance is then relative to the white panel, using only the rows above
   the panel.
2. **Grains**: the grains are the connected components of an Otsu mask of
   the mean reflectance, after a morphological opening. Only components
   whose area lies in [0.4, 2] × the median area are kept. The image is
   cropped above the white panel first, because the panel biases the Otsu
   threshold otherwise.
3. **Bands**: the bands are **averaged by groups of 4** (`band_bin=4`), so
   p = 64. With all 256 bands, the median grain has n / p ≈ 1.7 < 2; with 64
   bands, the smallest grain (141 pixels) still has n ≥ 2p = 128. Grains
   with fewer than 2p pixels would be discarded and counted: none are.
   Averaging was preferred to keeping one band in four: it is the atlas
   recipe, and it reduces noise.
4. **Estimator**: SCM (sample covariance matrix) by default;
   `"ledoit_wolf"`, `"tyler"` or `"student"` are also available.
5. **Normalisation**: covariances are scaled on the training part of `split`
   (geometric mean of the eigenvalues = 1). Means and raw pixels are
   multiplied by √s.

The segmentation gives 8 803 grains. The atlas fiche gives 8 796 (48 to 54
per cube): the small difference comes from the segmentation details.

```{raw} html
<iframe src="../_static/figures/rices90/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/rices90/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

The groups (`dataset.groups`) are the cubes, that is, one bundle of one
variety. The grains of a cube are never split between train and test.

- **`bundle` (recommended)**: train on bundle 1 and test on bundle 2. 20 % of
  the bundle-1 grains, drawn at random (`val_fraction=0.2`, seed 0), form the
  validation set. This gives 3 518 / 880 / 4 405. **The validation grains
  come from the same cubes as the training grains**, so validation scores
  are optimistic; only the test is grouped.
- **`bundle_reverse`**: the other way round, 3 524 / 881 / 4 398.
- **`cross_session`**: as `bundle`, but the test keeps only the 17 varieties
  whose two bundles were acquired in two different sessions (846 grains).

```{raw} html
<iframe src="../_static/figures/rices90/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| grains of one cube on both sides | train and test come from different bundles; checked by the real-data test | none for test |
| **variety confounded with session**: for 73 of 90 varieties, both bundles share their session | `cross_session` test (17 varieties) | `bundle` scores mix variety and session. The atlas measures a TS-LR (tangent-space logistic regression) OA (overall accuracy) of 0.516 on same-session test grains, but 0.016 on the 17 cross-session varieties (chance: 0.011) |
| validation drawn from the training cubes | documented | validation is optimistic; do not use it to report results |

## Supported tasks

- `task="fine"`: 90 varieties. There is no coarse map. The varieties could
  be grouped by type (for example glutinous `Nep*` varieties), with
  `merge_classes`.
- `classes=[...]` or `max_classes` select a subset of varieties.
  `balance=...` is rarely needed.

## Usage

```python
from spdnet_datasets.hyperspectral.rices90 import Rices90Dataset

ds = Rices90Dataset()                                 # ds.splits: bundle, bundle_reverse, cross_session
cross = Rices90Dataset(split="cross_session", subset="test")
lw = Rices90Dataset(estimator="ledoit_wolf")
```

## Migration notes

- `band_bin=4` replaces the former band subsampling. The grains are
  re-segmented from the cubes; the former `Rices_90_cov/` files are not read.
- `split="train"` raises an error: use `subset="train"`. `subset=None` is the
  default.
- Use `estimator=`. `cov_method=...` is silently absorbed by `**kwargs` and
  gives the SCM.
