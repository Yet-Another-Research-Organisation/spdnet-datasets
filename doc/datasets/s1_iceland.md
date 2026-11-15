# S1 Iceland SAR Dataset

```{list-table}
:stub-columns: 1

* - Domain
  - SAR (synthetic aperture radar) time series, land cover
* - Task(s)
  - fine classification (7 classes) · coarse (5 groups)
* - Covariance
  - 30×30 = VV and VH amplitude at 15 summer dates, n = valid pixels of a 32 × 32 sub-patch (≥ 922)
* - Outputs
  - cov · mean (30,) · raw (32, 32, 30)
* - Splits
  - `official` (= `recommended`): train / val 2019, test 2025 (39 790 / 11 366 / 6 791)
* - Module
  - {mod}`spdnet_datasets.sar.s1_iceland` (`fetch_s1_iceland`, `S1IcelandDataset`)
```

```{note}
This dataset is built at LISTIC and **not published yet**: the module lives on
the `local` branch only. `FILES` is empty, and the fetch only checks a local
copy against `MANIFEST_NAS.tsv`. The distribution licence is still to be
defined. The Sentinel-1 data and the CLC+ labels are free Copernicus products.
```

## Description and source

The S1 Iceland SAR Dataset contains Sentinel-1 (S1) time series over Iceland.
The source images are IW SLC (interferometric wide swath, single look complex)
products in dual polarisation, VV and VH: vertical emission, with vertical or
horizontal reception. The series cover the summer (June to September) of two
years:

- 2019: S1A and S1B, 6-day revisit;
- 2025: S1A and S1C.

The images are resampled to a 10 m latitude/longitude grid. The labels come
from CLC+ (the Copernicus Land Cover plus Backbone), merged into seven
classes.

Two products are ready for learning: segmentation (ascending orbit) and change
(descending orbit). The loader uses the segmentation product:

- train: 779 patches of 256 × 256 px in 2019;
- validation: 213 patches in 2019;
- test: 135 patches in 2025, on a disjoint band of 10 km blocks.

Companion stores hold the γ⁰ amplitude, with radiometric terrain correction
and thermal-noise removal.

- Reference: LISTIC project, draft IGARSS paper. The dataset is not published.
- Local copy: `<root>/SAR/ICELAND_SAR`.

## Raw data

```text
segmentation_amp_dataset.zarr/{train,val,test}/X_amp   (N, T, 2, 256, 256) uint16 = γ⁰ amplitude × 6000 (VV, VH)
segmentation_dataset.zarr/{train,val,test}/{X,y,r0,c0}  complex X (validity only), dense labels, patch position
splits/segmentation_blocks.shp                         10 km blocks of the official split
```

The amplitude is `X_amp / amp_scale` with `amp_scale=6000`. Amplitudes below
1/6000 are stored as 0, which affects about 7 % of the VV and 27 % of the VH
values.

A pixel is valid when VV and VH are non-zero at every retained date. The
`nodata` mask is not enough: it misses the swath gaps of single 2025 dates,
where the amplitude is 0, so validity is recomputed from the complex `X`.

The raw output of a sample is its (32, 32, 30) block of amplitudes, with
invalid pixels set to 0.

```{raw} html
<iframe src="../_static/figures/s1_iceland/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

| Class | train | val | test |
|---|---|---|---|
| periodic_herbaceous | 23 080 | 3 991 | 4 237 |
| water | 6 322 | 3 823 | 552 |
| sparse_vegetated | 3 820 | 596 | 1 181 |
| snow_ice | 2 055 | 173 | 350 |
| other_vegetation | 1 721 | 233 | 421 |
| open_spaces | 1 602 | 2 445 | **0** |
| sealed | 1 190 | 105 | 50 |

`open_spaces` does not occur in the test set. Macro-F1 computed by scikit-learn
over all predicted labels counts that class with a score of 0. Use balanced
accuracy, or restrict macro-F1 to the classes present in the test set.

```{raw} html
<iframe src="../_static/figures/s1_iceland/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/s1_iceland/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

**Sample.** Each 256 × 256 patch is cut into non-overlapping 32 × 32
sub-patches. A sub-patch is kept with the rules of the dataset benchmark, the
atlas fiche `iceland_sar`:

- at least 50 % of its pixels are labelled (`min_labeled`);
- at least 90 % of its pixels are valid (`min_valid`);
- its majority class covers at least 60 % of the labelled valid pixels
  (`min_purity`).

The covariance is estimated on the valid pixels, so n ≥ 922 ≥ 2p = 60.

**Channels.** p = 30 channels, in the order VV(t₁ … t₁₅), VH(t₁ … t₁₅): the
amplitude at the 15 dates of one summer.

- 2019 has 15 dates in the segmentation product, every 6 days from
  2019-06-18 to 2019-09-10.
- 2025 has 16 dates, from 2025-06-16 to 2025-09-14. `select_dates` keeps the
  contiguous window of 15 dates whose days of year are closest to the 2019
  dates. This drops **2025-09-14**, the last date, and leaves every 2025 date
  two days before its 2019 counterpart. Dropping the first date instead would
  give a four-day shift.

The retained dates are in `dataset.dates`.

**Exclusions.** A sub-patch whose covariance is not SPD (symmetric positive
definite) is excluded and counted in `n_excluded`, for example when a channel
is constant over the window. No sub-patch is excluded on the real data, with
the validity rule above.

**Normalisation.** The cache keeps unscaled covariances. At load time, the
geometric scale of `core.normalize` is fitted on the training part. The median
condition number of the SCM covariances is about 66.

```{raw} html
<iframe src="../_static/figures/s1_iceland/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/s1_iceland/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

`official` (alias `recommended`) is the split shipped with the dataset:

- train: the 2019 training patches, 39 790 sub-patches;
- val: the 2019 validation patches, 11 366 sub-patches;
- test: the 2025 test patches, 6 791 sub-patches, on a disjoint band of 10 km
  blocks.

`groups` is the 10 km block that holds the sub-patch centre, a 1 000 × 1 000
px cell of the common 10 m grid. The sub-patches fall in 127 blocks.

The training count equals the atlas fiche exactly. The test count is 6 791
instead of 6 728: the fiche tests the 16 dates, including the dropped one.
The fiche gives no validation count.

```{raw} html
<iframe src="../_static/figures/s1_iceland/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Same pixels in two splits | windows never overlap across splits; 10 km blocks and years disjoint (checked) | none |
| Spatial proximity between splits | none added: the dataset documentation announces a buffer of at least 5 km, but **this was not verified**. On the common 10 m grid, train and val patches are **adjacent** (0 px gap, no overlap), and train and test are 128 px = 1.28 km apart | validation is spatially correlated with training, so validation scores are optimistic; the test set is a domain shift (region, year, S1C, 2-day date offset) |
| Overlapping 256 px patches within one split | none | some training sub-patches overlap each other (no effect across splits) |
| Normalisation fitted on test data | geometric scale fitted on the training part only | none |
| Date selection using test labels | `select_dates` uses only dates (days of year) | none |

## Supported tasks

- `task="fine"`: 7 classes.
- `task="coarse"`: `COARSE_MAP` merges other_vegetation, periodic_herbaceous
  and sparse_vegetated into `vegetation`, which leaves 5 groups.
- `drop_classes` / `merge_classes` take class names. Dropping `open_spaces`
  aligns train and test.
- `balance` acts on the training part only.
- `block`, `min_labeled`, `min_valid` and `min_purity` change the sample
  definition.

## Usage

```python
from spdnet_datasets.sar.s1_iceland import S1IcelandDataset

ds = S1IcelandDataset(split="official")
cov, y = ds[0]                       # (30, 30)
ds.dates["test"]                     # the 15 retained 2025 dates
test = S1IcelandDataset(subset="test", drop_classes=["open_spaces"])
```
