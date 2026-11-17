# PASTIS-R

```{list-table}
:stub-columns: 1

* - Domain
  - SITS (satellite image time series), optical and radar
* - Task(s)
  - fine classification (18 crop classes) · coarse (6 groups)
* - Covariance
  - p×p with p = 10, 12, 16, 30 or 32 (scenarios A–E), n = pixels of the parcel (n ≥ 2p)
* - Outputs
  - cov · mean · raw (n, p) pixels of the parcel
* - Splits
  - `official` (folds 1-3 / 4 / 5) · `cv1` … `cv5`
* - Module
  - {mod}`spdnet_datasets.sits.pastis_r` (`fetch_pastis_r`, `PastisRDataset`)
```

```{dataset-preview} pastis_r
```

## Description and source

PASTIS-R contains 2 433 patches of 128 × 128 pixels at 10 m. The patches come
from four Sentinel-2 tiles in France (`t30uxv`, `t31tfj`, `t31tfm`, `t32ulu`) and
cover September 2018 to October 2019. Every patch has three image time series:

- Sentinel-2 (S2), the optical multispectral sensor, with 38 to 61 dates;
- Sentinel-1 (S1) synthetic aperture radar (SAR) in ascending orbit (S1A);
- S1 in descending orbit (S1D), about 70 dates per orbit.

The annotations give a semantic map with 20 classes, including background and
void, plus parcel identifiers. The French Land Parcel Identification System
provides the crop types.

- Sainte Fare Garnot V., Landrieu L., *Panoptic segmentation of satellite image
  time series with convolutional temporal attention networks*, ICCV (International Conference on Computer Vision) 2021.
- Sainte Fare Garnot V., Landrieu L., Chehata N., *Multi-modal temporal
  attention models for crop mapping from satellite time series*, ISPRS (International Society for Photogrammetry and
  Remote Sensing) Journal of Photogrammetry and Remote Sensing, 2022 (the radar extension, "-R").
- Data and documentation: <https://github.com/VSainteuf/pastis-benchmark>
  (Zenodo archive, CC BY 4.0 according to the authors' page).

The loader reads a local copy laid out as `<root>/SITS/PASTIS-R/PASTIS-R/`
(`DATA_S2`, `DATA_S1A`, `DATA_S1D`, `ANNOTATIONS`, `metadata.geojson`). The local
copy is checked against `MANIFEST_NAS.tsv` (otherwise, see *Getting the data*).

## Getting the data

Direct access, no account: Zenodo record 5735646 (CC BY 4.0). Without a local copy,
`fetch_pastis_r()` downloads `PASTIS-R.zip` (53.7 GB), checks its sha256 and
extracts it into `<cache>/pastis_r/downloads/`; `fetch_pastis_r(archive=...)` uses
a downloaded zip (checked, kept) or its extracted folder. PASTIS without radar
(Zenodo 5012942, 28.8 GB) is not registered.

## Raw data

| Array | Shape | Content |
|---|---|---|
| `DATA_S2/S2_<patch>.npy` | (T, 10, 128, 128) int16 | reflectance × 10⁴, bands B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 |
| `DATA_S1A/S1A_<patch>.npy`, `DATA_S1D/…` | (T, 3, 128, 128) float16 | backscatter in dB: VV, VH, VV − VH (VV and VH: vertical emission, vertical or horizontal reception) |
| `ANNOTATIONS/TARGET_<patch>.npy` | (3, 128, 128) | semantic class in channel 0 (0 background, 19 void) |
| `ANNOTATIONS/ParcelIDs_<patch>.npy` | (128, 128) | parcel identifier of each pixel |
| `metadata.geojson` | 2 433 features | tile, fold, acquisition dates of each series |

The `ParcelIDs` files are in
`ANNOTATIONS/`, not in `INSTANCE_ANNOTATIONS/` as some descriptions say.
35 patches have images but no entry in `metadata.geojson`, so they have no
label and no fold. The loader ignores them.

The figure shows four parcels of scenario C as returned by `return_raw=True`.
Each column is one pixel and each row one of the 16 standardised channels.
Parcels have irregular shapes, so `n` changes from one parcel to the next.

```{dataset-figure} pastis_r/raw_samples
```

## Classes

The label of a parcel is the majority class of `TARGET[0]` over its pixels.
Background (0) and void (19) are dropped, which leaves 18 crop classes. Meadow
is by far the largest class and potatoes the smallest. The figures below are
for scenario C (`official` split).

```{dataset-figure} pastis_r/class_hist
```

```{dataset-figure} pastis_r/class_means
```

## Covariance construction

**Sample**: one parcel of one patch. Its observations are the parcel pixels
that are valid on every channel. A parcel cut by a patch border appears in each
patch it touches (1 024 identifiers). Each copy is a separate sample, and the
copies always stay in the same fold.

**Scenarios** (`scenario=`):

| Scenario | Channels | p | min. pixels (2p) | Parcels | train / val / test |
|---|---|---|---|---|---|
| A | S2, 10 bands, 1 date | 10 | 30 | 82 112 | 49 412 / 16 694 / 16 006 |
| B | A + NDVI (normalised difference vegetation index) + NDWI (normalised difference water index) | 12 | 30 | 82 102 | 49 407 / 16 690 / 16 005 |
| C | S2 10 + S1A (VV, VH, VV/VH) + S1D (idem), 1 date | 16 | 32 | 79 684 | 47 825 / 16 363 / 15 496 |
| D | S2 10 bands, 3 consecutive dates | 30 | 60 | 63 670 | 38 132 / 12 974 / 12 564 |
| E | (S2 10 + S1A 3 + S1D 3) × 2 consecutive S2 dates | 32 | 64 | 62 867 | 37 769 / 12 895 / 12 203 |

Counts are measured on the local copy with the default options and
`estimator="scm"`, the SCM (sample covariance matrix).
The minimum pixel count follows the n ≥ 2p rule.

**Indices.** NDVI = (B08 − B04) / (B08 + B04). NDWI follows McFeeters: NDWI = (B03 − B08) / (B03 + B08). A pixel whose denominator
is not positive is invalid.

**Radar ratio.** PASTIS-R stores S1 in dB, and its third channel is exactly
VV − VH in dB. That channel is a linear combination of the other two, so the
covariance of `[VV, VH, VV/VH]` in dB is singular. With the default
`s1_domain="db"`, the loader keeps VV and VH in dB and recomputes the ratio in
linear scale, 10^((VV − VH)/10). With `s1_domain="linear"`, all three channels
are linear.

**Dates** (scenarios C and E pair S2 with S1; A, B and D use the same S2 date
selection without S1). All the patches of one tile share their acquisition
dates. For each patch, the loader keeps the same dates for all its parcels:

1. Candidate windows are the k consecutive S2 acquisitions (k = 1, 1, 1, 3, 2
   for A–E) whose mean date lies within `date_window=15` days of
   `ref_date="2019-07-15"`. A summer date is chosen because it has few clouds.
   `date_index` selects a fixed position in the series instead.
2. Every S2 date of the window needs an S1A and an S1D acquisition within
   `max_gap_days=3` days. Distinct S2 dates use distinct S1 images. D and E
   windows span at most `max_span_days=20` days.
3. PASTIS has no cloud mask. The loader uses the fraction of pixels with
   B02 > 1500 (reflectance 0.15) as a cloud proxy, averaged over the window,
   and requires it to stay at or below `max_cloud_fraction=0.1`.
4. Among the valid windows, the loader keeps the one with the smallest maximal
   S1 gap, then the smallest offset to `ref_date`, then the fewest clouds.
5. **Homogeneity.** The gap of a covariance is its largest absolute S2–S1 gap.
   A patch is dropped when its gap differs from the median gap over all
   covariances by more than `gap_tolerance=1` day.

The default `max_gap_days=3` comes from the measured gap distribution: the 90th
percentile over all S2 dates of all patches. The distribution has a median of
2 days, a 95th percentile of 5 days and a maximum of 15 days for S1A and
19 days for S1D. The retained gaps are:

- C: 1 day for 57 799 parcels and 2 days for 21 885; 40 patches (1 482 parcels)
  are dropped by the homogeneity rule.
- E: 2 days for 44 919 parcels and 3 days for 17 948; no patch is dropped.

With the defaults, the first retained S2 date of a patch falls between
29 June and 29 July 2019. The dates and the
signed gaps (S1 − S2, in days) of each parcel are in `dataset.meta`
(`s2_dates`, `s1a_dates`, `gap_s1a`, `s1d_dates`, `gap_s1d`, `max_abs_gap`). The
full tables are written to the processed cache (`date_gaps.csv`,
`gap_histogram.csv`).

```{dataset-figure} pastis_r/date_gaps
```

```{dataset-figure} pastis_r/date_timeline
```

**Rejections** (default options). Of the 2 433 patches, 15 are cloudy for
scenarios A–C and 118 for D and E, which need several dates. Parcels below the
pixel threshold are dropped: 3 399 for A, 4 288 for C, 19 466 for E. Estimates
that fail the SPD (symmetric positive definite) check of `core.check_spd` are
excluded and counted in `dataset.report["covariance"]`. The check requires
symmetry, a positive smallest eigenvalue and a condition number of at most
10¹². It excludes 36 parcels for A, 41 for B, 99 for C, 272 for D and 206
for E. `report` gives every count by reason, tile and fold.

**Normalisation.** The channels have different units: reflectance × 10⁴,
indices in [−1, 1], dB and linear ratios. Before estimation, each channel is
standardised (mean 0, standard deviation 1). The statistics are pooled over the
pixels of the **training folds of the chosen split** only, so Ledoit-Wolf
shrinkage is not dominated by the S2 bands and no test statistic is used. The
covariance cache therefore depends on `split`. At load time, the scalar
geometric scale s of `core.normalize` is fitted on the training covariances
(means and raw pixels are multiplied by √s). The
`NORM_*.json` files shipped with PASTIS are not used, because they do not cover
NDVI, NDWI or the radar ratio.

**Conditioning.** The median condition number κ of the delivered SCM
covariances is about 2.2·10³ for A, 6.3·10⁴ for B, 6.2·10³ for C,
4.3·10⁴ for D and 3.2·10⁴ for E. In B, NDVI and NDWI are nearly collinear with
B03, B04 and B08, which explains the high value. Numerically singular estimates (κ > 10¹²)
are excluded, as counted above. `estimator="ledoit_wolf"` gives better
conditioned matrices.

```{dataset-figure} pastis_r/cov_samples
```

```{dataset-figure} pastis_r/eigen_spectrum
```

## Splits

PASTIS-R has five official folds, made of spatially separated groups of
patches. The loader applies them by patch:

- `official` (= `cv5` = `recommended`): folds 1–3 train, fold 4 validation,
  fold 5 test;
- `cv1` … `cv5`: fold k is the test set, fold k − 1 (fold 5 for k = 1) the
  validation set, and the other three folds the training set.

The group of a sample is its patch. `make_splits` checks with
`assert_no_leakage` that train, validation and test never share a patch.

```{dataset-figure} pastis_r/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Same patch in train and test | split by patch on the official folds, checked by `assert_no_leakage` | none |
| Neighbouring parcels on both sides | official folds are spatially separated groups of patches | parcels at a fold border can be close |
| One parcel cut by a patch border | its copies are separate samples of the same fold; of the 1 024 repeated identifiers, the 6 that cross folds are all void (dropped) | none |
| Normalisation statistics from the test set | standardisation and geometric scale fitted on the training folds only | none |
| Date selection tuned on test data | the dates are chosen per patch from the S2 / S1 metadata and the cloud proxy only, without labels | none |

## Supported tasks

- `task="fine"`: 18 crop classes.
- `task="coarse"`: 6 groups from `COARSE_MAP`: winter cereals, summer cereals,
  oilseeds, grassland and fodder, permanent crops, root crops and vegetables.
- `drop_classes`, `merge_classes` and `classes` take class names.
- `balance="undersample"`, `"oversample"` or `"weights"` acts on the training
  part only.
- There is no regression target.

## Usage

```python
from spdnet_datasets.sits.pastis_r import PastisRDataset

ds = PastisRDataset(scenario="C", split="official", subset="train")
cov, y = ds[0]                       # (16, 16) float64, class index
ds.meta[["s2_dates", "gap_s1a", "gap_s1d"]].head()

# raw pixels: n varies, so use batch_size=1 or a custom collate_fn
ds_raw = PastisRDataset(scenario="E", return_raw=True)
```
