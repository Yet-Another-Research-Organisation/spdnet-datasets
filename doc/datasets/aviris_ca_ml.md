# AVIRIS-C California multilabel

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), airborne, time series, land cover
* - Task(s)
  - multilabel classification, 13 classes (default, `task="multilabel"`) · regression of the 13 CDL proportions (`task="regression"`) · dominant class (`task="fine"`)
* - Covariance
  - 155×155 (155 cleaned bands of 224), one per (window, date), n = the valid pixels of the 32 × 32 window (821 to 1 024, median 1 024; n / 2p ≥ 2.6); Ledoit-Wolf by default
* - Outputs
  - `mode="date"`: cov · mean (155,) · raw (32, 32, 155) · `mode="series"`: (tokens (14, 155, 155), mask (14,)) · mean (14, 155) · raw (14, 32, 32, 155)
* - Samples
  - `mode="date"`: 24 211 window-dates (the QC-ok dates of the 2 034 windows) · `mode="series"`: 2 031 windows (3 have no QC-ok date)
* - Splits
  - `fold0` … `fold4` (5 spatial folds of 10 km blocks) · `recommended` = `train_val_test` = `fold4` (14 331 / 4 872 / 5 008 window-dates, 1 218 / 397 / 416 windows) · `random` (sample-level 70 / 10 / 20, ignores the groups, see {doc}`/guide/splits`)
* - Module
  - `spdnet_datasets.hyperspectral.aviris_ca_ml` (`fetch_aviris_ca_ml`, `AVIRISCAMultilabelDataset`, registry name `aviris_ca_ml`)
```

```{dataset-preview} aviris_ca_ml
```

*One window with vineyards at four of its quality-controlled dates, in false
colour (860 / 650 / 550 nm, one stretch for the four dates).*

## Description and source

AVIRIS (Airborne Visible/InfraRed Imaging Spectrometer) Classic is the
imaging spectrometer of NASA and JPL (Jet Propulsion Laboratory), flown on
ER-2 aircraft. It records 224 bands from 366 to 2 496 nm, about 10 nm apart.
The dataset uses the L2 "calibrated reflectance" flight lines distributed by
the ORNL DAAC (Oak Ridge National Laboratory Distributed Active Archive
Center; NASA CMR collection `C2711871294-ORNL_CLOUD`): 943 flight lines over
California, from 2008-06-11 to 2020-10-13, most of them in 2013–2015 (the
HyspIRI preparatory campaign).

One sample of the release is a fixed **window of 600 m × 600 m** on a grid
aligned with the 30 m grid of the CDL (EPSG:5070, CONUS Albers): 32 × 32 pixels
of 18.75 m, the native pixels (14.8–17 m, sometimes rotated by 20–44°) being
area-averaged onto that grid. Each window is observed at **14 dates**, chosen
among its acquisitions to spread over time; 171 windows have fewer dates and
are NaN-padded. The labels come from the USDA NASS Cropland Data Layer (CDL,
2008–2021), the 30 m land-cover map of the National Agricultural Statistics
Service.

- Release: Hugging Face dataset
  [`musmb/aviris-ca-multilabel`](https://huggingface.co/datasets/musmb/aviris-ca-multilabel),
  revision `0cc7afc9`, 7.1 GB in 10 Parquet shards. The dataset card
  (`README.md`) and `metadata/checks.json` describe the construction and its
  checks.
- Licence: **CC BY 4.0**. Any use must also credit the original data
  providers: NASA/JPL AVIRIS-Classic L2 reflectance distributed by the ORNL
  DAAC, and the USDA NASS Cropland Data Layer (see "How to cite"). The labels
  are derived from the CDL by the dataset authors; they are not an official
  USDA product.

## Getting the data

The Hugging Face repository is **private** for now: an account with access
and a read token are needed. The token is looked up as for the other
credentialed sources ({doc}`/getting_started/installation`): `HF_TOKEN`, the
Hugging Face token cache (`~/.cache/huggingface/token`, written by
`huggingface-cli login`), the system keyring, then a hidden prompt.

Without a local copy, `fetch_aviris_ca_ml()` downloads `bands.csv` and the
shards of the requested folds from the pinned revision
`0cc7afc9d49c18faa37622683e3839d4e30a24b9` into
`<cache_dir>/aviris_ca_ml/downloads/<revision>/` (sizes and sha256 checked;
the sha256 are the LFS identifiers of the Hub). This folder is outside the
`keep` policy and is never deleted by the loader. `folds=[0]` downloads only
the two shards of fold 0 (1.4 GB).

A local copy of the release (a `git clone` of the repository or a
`huggingface-cli download` folder) is read in place, after checking the file
sizes against the pinned revision (`verify="size"`; `"sha256"` hashes the
files, `None` skips the check):

```python
from spdnet_datasets.hyperspectral import AVIRISCAMultilabelDataset

ds = AVIRISCAMultilabelDataset(data_dir="~/aviris-ca-multilabel")  # or archive=<folder>
```

The default location is `<SPDNET_DATA_ROOT>/HYPERSPECTRAL/AVIRIS-CA-Multilabel`.
Reading the shards needs `pyarrow` (`pip install 'spdnet-datasets[aviris]'`).

## Raw data

One row of a shard is one window: `cube` (14, 155, 32, 32) float16
reflectance (0–1, NaN = no data, padded dates all NaN), `dates`, `acquired`,
`qc_ok`, `flight_ids`, `nan_frac` (14 values each), the per-date labels
`labels_date` (14, 13) (−1 on padded dates) and proportions `props_date`
(14, 13), the window labels `labels_window` (13,) and proportions
`props_window` (13,), the spatial fold, the 10 km block, the zone and the
coordinates of the window.

Band mean of four windows at one date (`mode="date"`, `return_raw=True`):

```{dataset-figure} aviris_ca_ml/raw_samples
```

### Bands

The 224 bands of `bands.csv` (median wavelength and FWHM over the flight-line
headers; wavelengths differ by at most 1.7 nm between flights and are not
resampled; column `removed_by` gives the reason of each removal):

| Step | Removed | Kept |
|---|---|---|
| base mask: water-vapour absorption and detector edges (366–414, 1 303–1 473, 1 782–2 028, 2 447–2 496 nm) | 58 | 166 |
| isolated-band spikes (rate above 20 × the median rate): 424, 938, 947, 1 483, 1 742, 2 038, 2 048, 2 407, 2 417 nm | 9 | 157 |
| wings of the 940 nm water-vapour feature: 928, 957 nm | 2 | **155** |

The 155 kept bands cover 434–2 437 nm (`dataset.wavelengths`, nm;
`dataset.bands` are their indices after `band_step`).

## Classes

The CDL codes are grouped into 40 thematic groups, then into 12 fine classes
and `other` (all the remaining mapped groups: crops other than almonds and
grapes, other orchards, deciduous forest, wetlands, snow). Background, clouds
and non-agricultural or undefined codes are left out of the denominator; every
(window, year) used has at least 320 of its 400 CDL pixels mapped.

| id | class | CDL codes | windows (fold 0 / 1 / 2 / 3 / 4) | windows | QC-ok window-dates |
|---|---|---|---|---|---|
| 0 | `almond` | 75 | 33 / 33 / 32 / 32 / 32 | 162 | 1 970 |
| 1 | `shrubland` | 64, 152 | 119 / 113 / 119 / 119 / 117 | 587 | 6 852 |
| 2 | `water` | 83, 92, 111 | 39 / 39 / 39 / 39 / 39 | 195 | 2 308 |
| 3 | `evergreen_forest` | 142 | 74 / 82 / 82 / 82 / 82 | 402 | 4 504 |
| 4 | `mixed_forest` | 143 | 38 / 42 / 42 / 42 / 42 | 206 | 2 534 |
| 5 | `grassland_pasture` | 58, 59, 60, 176 | 77 / 75 / 74 / 74 / 76 | 376 | 4 851 |
| 6 | `barren` | 65, 131 | 29 / 30 / 30 / 29 / 30 | 148 | 1 476 |
| 7 | `developed_high` | 124 | 48 / 50 / 50 / 51 / 50 | 249 | 3 107 |
| 8 | `developed_low` | 122 | 134 / 133 / 133 / 134 / 122 | 656 | 7 995 |
| 9 | `developed_medium` | 123 | 131 / 125 / 130 / 133 / 132 | 651 | 7 954 |
| 10 | `developed_open` | 82, 121 | 116 / 122 / 122 / 117 / 122 | 599 | 7 258 |
| 11 | `grapes` | 69 | 39 / 39 / 38 / 39 / 39 | 194 | 2 445 |
| 12 | `other` | all other mapped codes | 60 / 58 / 60 / 60 / 60 | 298 | 3 904 |

Windows: number of windows where the class is present at window level.
QC-ok window-dates: number of `mode="date"` samples where it is present at
date level (`label_level="date"`). Both counts are checked against
`classes.csv` by the real-data tests.

### Multilabel definition

A class is **present** when it covers **at least 10 %** of the mapped CDL
pixels of the window:

- **per date** (`label_level="date"`, default of `mode="date"`): in the CDL of
  the acquisition year. The labels of a window can therefore change between
  dates (in 766 windows at least one class flips, mostly near the threshold);
- **per window** (`label_level="window"`, the only level of `mode="series"`):
  the proportions averaged over the *distinct* CDL years of the window's
  dates (each year counted once), then thresholded at 10 %.

A window has 1 to 5 labels (333, 876, 670, 147 and 8 windows with 1, 2, 3, 4
and 5 labels; 2.3 on average). `task="regression"` returns the proportions
themselves (13 values summing to 1); `task="fine"` returns the class of
largest proportion (1 182 to 3 167 window-dates per class).

```{dataset-figure} aviris_ca_ml/class_hist
```

```{dataset-figure} aviris_ca_ml/class_means
```

## Time dimension and quality control

Each window has up to 14 dates (median 14), irregularly spaced and
concentrated in 2013–2015, with gaps of several years; acquisition geometry, season, atmosphere and native
resolution change from one date to the next. A date is **QC-ok** unless one of
these heuristic flags is set (`metadata/dates.parquet`):

| Flag | Rule | Window-dates |
|---|---|---|
| no data | more than 20 % no-data pixels | 1 155 |
| snow | NDSI(560, 1 650 nm) > 0.4 and blue > 0.15 | 525 |
| very bright | blue > 0.3 and NIR > 0.4 | 876 |
| near zero | more than half of the base-mask bands with mean ≤ 0.0005 | 1 918 |

28 245 of the 28 476 window-dates are acquired and 24 211 pass the QC; only
these are used by default (`qc=True`). With `qc=False`, every acquired date
with at least `2p` valid pixels and an SPD estimate is kept (25 611 window-dates). The QC flags
concern the image only: a date failing the QC keeps its labels.

In `mode="series"`, the 14 dates of a window form a series of covariances:
`mask[t]` is `False` for a padded, failed-QC or too-sparse date, whose token
is the identity matrix (and whose mean is zero). The number of valid dates per
window ranges from 1 to 14 (median 12); `min_dates` drops windows with fewer
valid dates. The dates themselves are in `dataset.fetched.dates` (columns
`window_id`, `t`, `date`, `flight_id`); the series is irregular in time, and a
model should be given the dates.

## Covariance construction

For each QC-ok (window, date):

1. the 155 bands (one in `band_step`, default 1) of the 32 × 32 pixels;
2. reflectance **clipped to [−0.05, 1.5]** (`clip`): some QC-ok dates hold
   isolated values up to 10³–10⁴, mostly in the SWIR bands of a few 2014 and
   2018 flight lines (`outlier_frac` in `metadata/dates.parquet`);
3. pixels with a non-finite band dropped; the date needs `min_pixels` valid
   pixels (default and minimum `2p` = 310). With the QC at most 20 % of the
   pixels are missing: n = 821 to 1 024 (median 1 024), so n / p ≥ 5.3;
4. **Ledoit-Wolf** covariance (default `estimator`): the bands are strongly
   correlated and n / p ≈ 6.6, so the sample covariance is badly conditioned.
   Every one of the 24 211 estimates is SPD with a condition number below
   10¹².

| Estimator | p | n (median) | Samples | κ (5 / 50 / 95 %) |
|---|---|---|---|---|
| Ledoit-Wolf | 155 | 1 024 | 24 211 | 8.1e3 / 3.6e4 / 2.0e5 |

The covariances (24 211 × 155 × 155 float64, 4.6 GB, memory-mapped) are
cached once by {func}`~spdnet_datasets.core.fetch.run_pipeline`; reading the
7.1 GB of shards and estimating them took about 3 minutes on 3 threads.
`mode="date"` and `mode="series"` read the same cache. `return_raw` reads the
window from its shard on demand (the row group of 16 windows is kept in
memory, so a sequential pass is fast).

With `normalize=True`, the covariances are multiplied by the geometric scale
`s` of the training covariances, `s = exp(−mean(log det C) / p)` (8.8e4 for
`recommended`), the means and the raw pixels by `√s`.

```{dataset-figure} aviris_ca_ml/cov_samples
```

```{dataset-figure} aviris_ca_ml/eigen_spectrum
```

## Splits

The release groups the windows into 10.2 km × 10.2 km blocks (`block10km`) and
assigns whole blocks to 5 folds, balancing every class and the number of
windows (413 / 402 / 405 / 398 / 416 windows).

- `fold<k>`: test = fold `k`, validation = fold `k − 1` (mod 5), train = the
  three other folds;
- `recommended` = `train_val_test` = `fold4`: train = folds 0–2, validation =
  fold 3, test = fold 4, i.e. the `train_val_test` configuration of the
  Hugging Face release (14 331 / 4 872 / 5 008 window-dates; 1 218 / 397 / 416
  windows in `mode="series"`);
- `random`: sample-level, **leaky** (dates of one window and neighbouring
  windows on both sides) and not stratified with `task="multilabel"`.

`dataset.groups` is the 10 km block, the unit of the folds; `dataset.window_ids`
gives the window of each sample. `folds=` loads a subset of the folds; a
`fold<k>` split is then defined only when its training part is not empty.

```{dataset-figure} aviris_ca_ml/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| dates of one window on both sides | folds by window and by 10 km block | none (checked in the tests) |
| spatial autocorrelation between neighbouring windows | whole 10 km blocks per fold | 1.0 % of the windows touch a window of another fold at 600 m, 2.8 % within 1 km; median distance to the nearest window of another fold 4.8 km. Scores overestimate geographic generalisation |
| label noise | — | CDL is a 30 m classifier product; developed subtypes and natural vegetation are noisy; the 10 % threshold is sensitive near the limit |
| `random` split | warning, `split_leaky` | optimistic by design |

For a geographic-transfer test, the dataset card suggests holding out the zone
`Z2_SantaBarbara` or `Z4_SFBay_JasperRidge` (`meta["zone"]`) and training on
windows far from it (more than 25 km).

## Supported tasks

- `task="multilabel"` (default): float32 0/1 vector of the 13 classes
  (`torch.nn.BCEWithLogitsLoss`). `classes=[...]` keeps a subset of the label
  columns (in the dataset order); `balance="weights"` sets
  `dataset.class_weights` to the per-label positive weights
  `n_negative / n_positive` of the training part (the `pos_weight` of
  `BCEWithLogitsLoss`). `drop_classes`, `merge_classes`,
  `max_samples_per_class` and resampling do not apply.
- `task="regression"`: the 13 CDL proportions (float64).
- `task="fine"`: the dominant class, with every single-label option
  (`drop_classes`, `merge_classes`, `balance`...).

## Usage

```python
from spdnet_datasets.hyperspectral import AVIRISCAMultilabelDataset

R = "~/aviris-ca-multilabel"                    # local release, or omit to download
ds = AVIRISCAMultilabelDataset(R)               # 24 211 × 155 × 155, per-date labels (13,)
ds = AVIRISCAMultilabelDataset(R, label_level="window")       # window labels
ds = AVIRISCAMultilabelDataset(R, mode="series")              # 2 031 windows, 14 tokens + mask
ds = AVIRISCAMultilabelDataset(R, task="regression")          # CDL proportions
ds = AVIRISCAMultilabelDataset(R, band_step=2)                # p = 78
ds = AVIRISCAMultilabelDataset(R, split="fold2", subset="test")
(tokens, mask), y = AVIRISCAMultilabelDataset(R, mode="series")[0]
```

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/aviris_ca_ml.md
```
````
