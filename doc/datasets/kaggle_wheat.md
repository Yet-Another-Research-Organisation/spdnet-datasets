# Wheat disease (ICPR 2026, Kaggle)

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI) from an unmanned aerial vehicle (UAV), crop disease
* - Task(s)
  - fine classification (3 classes: Health, Other, Rust)
* - Covariance
  - 107×107 (bands 9 to 115 of 125), n = 778 to 1 024 non-zero pixels of a 32×32 patch (n / 2p ≥ 3.6)
* - Outputs
  - cov · mean · raw (32 × 32 × 107)
* - Samples
  - 575 labelled patches (600 minus 23 entirely zero and 2 with fewer than 2p non-zero pixels)
* - Splits
  - `stratified` = `recommended` (5 folds, fold 0: 344 / 116 / 115)
* - Module
  - `spdnet_datasets.hyperspectral.kaggle_wheat` (`fetch_kaggle_wheat`, `KaggleWheatDataset`, registry name `kaggle_wheat`)
```

## Description and source

These are the data of task 1 of the ICPR 2026 competition "Beyond Visible
Spectrum: AI for Agriculture" (ICPR: International Conference on Pattern
Recognition), distributed on Kaggle. Each sample is a wheat patch seen by a
snapshot hyperspectral camera on a UAV: a 32 × 32 hyperspectral (HS) patch with 125 or 126
bands, plus a multispectral (MS) patch of 64 × 64 × 5 and an RGB (red, green,
blue) patch of 64 × 64. Only the HS patches are used here.

- Reference: L. Han, W. Huang, X. Zhang, T. Sobeih, Y. Shi and Y. Dong,
  ICPR 2026 Competition on Beyond Visible Spectrum: AI for Agriculture,
  task 1, <https://han-research.gitlab.io/Agvision/>. This reference was
  found by a web search and is not verified.
- Licence: competition terms (not verified). Kaggle account required; manual
  download into `<data root>/HYPERSPECTRAL/Wheat_Disease_ICPR26`.
- Train: 600 labelled patches, 200 per class, with the class in the file
  name. Validation: 300 unlabelled patches (the competition submission).
  Wavelengths and acquisition dates are not provided.

## Raw data

`{train,val}/HS/<stem>.tif` holds uint16 HS patches of 32 × 32 × 125
(489 train patches) or × 126 (111 train patches). The 126th band, of unknown
position, is dropped. 23 train patches (Health 9, Other 14) and 11 validation
patches are entirely zero. The raw output is the 32 × 32 × 107 patch; the
figure shows its band-mean image.

```{raw} html
<iframe src="../_static/figures/kaggle_wheat/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

After the exclusions, the counts are Rust 200, Health 191 and Other 184. The
figure gives the counts per part of the stratified split, then the mean
spectrum per class.

```{raw} html
<iframe src="../_static/figures/kaggle_wheat/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/kaggle_wheat/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

1. **Bands**: the 126th band is dropped when present (125 bands). The first
   `skip_first=8` and last `skip_last=10` noisy bands are then removed, so
   p = 107 (bands 9 to 115). This is the band window of the former
   precomputation.
2. **Observations**: the non-zero pixels of the patch (at most 1 024).
   Entirely zero patches are **excluded** (`info["excluded_all_zero"]`: 23),
   and so are patches with fewer than 2p = 214 non-zero pixels
   (`info["excluded_n_lt_2p"]`: 2). The former precomputation kept them as
   zero matrices, which are not SPD (symmetric positive definite).
3. **Estimator**: SCM (sample covariance matrix) of the centred pixels by
   default; `"ledoit_wolf"`, `"tyler"` or `"student"` are also available.
4. **Normalisation**: covariances are scaled on the training part of `split`
   (geometric mean of the eigenvalues = 1). Means and raw patches are not
   scaled.

The unlabelled validation set (300 patches: 11 zero and 1 short, so 288
covariances) is available through `fetch_kaggle_wheat(part="val")`. Its labels
are `None` and it has no split; the dataset class does not expose it.

```{raw} html
<iframe src="../_static/figures/kaggle_wheat/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/kaggle_wheat/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

The official split (600 labelled / 300 unlabelled) cannot be evaluated
locally. The labelled patches are therefore split with a stratified 5-fold
split, implemented as a `StratifiedGroupKFold` with one group per patch
(seed 0). Test = fold `fold` and validation = fold `fold + 1`. With
`fold=0`: 344 train, 116 validation and 115 test patches.

No plot, flight or position information is available, so the split cannot
be grouped. In the figure, the "samples per group" panel is trivial (one patch
per group).

```{raw} html
<iframe src="../_static/figures/kaggle_wheat/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| neighbouring patches of one plot on both sides | none possible (no plot, flight or position metadata) | **not measurable**: scores may be optimistic |
| duplicated patches | the only MD5 (Message Digest 5) checksum duplicates are the zero patches, which are excluded | none |
| zero / short patches | excluded and counted | none |

## Supported tasks

- `task="fine"`: 3 classes. There is no coarse map. `merge_classes={"Rust":
  "Disease", "Other": "Disease"}` gives a binary health-versus-disease task.
- `balance="undersample"`, `"oversample"` or `"weights"` acts on the training
  part only. The classes are almost balanced.

## Usage

```python
from spdnet_datasets.hyperspectral.kaggle_wheat import KaggleWheatDataset, fetch_kaggle_wheat

ds = KaggleWheatDataset(fold=0)                        # 575 patches, ds.splits["stratified"]
train = KaggleWheatDataset(subset="train", estimator="ledoit_wolf")
val_kaggle = fetch_kaggle_wheat(part="val")             # unlabelled competition set
```

## Migration notes

- Zero patches are no longer returned as zero matrices: 575 samples instead
  of 600.
- `split="train"` raises an error: use `subset="train"`. `subset=None` is the
  default.
- Use `estimator=`. `cov_method=...` is silently absorbed by `**kwargs` and
  gives the SCM.
