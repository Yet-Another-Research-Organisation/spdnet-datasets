# AVIRIS

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral, airborne, time series
* - Task(s)
  - FINE: fine classification (12 classes) · coarse (7 groups) — MIX: regression on 13 proportions · fine on the dominant class (13 classes) · coarse (8 groups)
* - Covariance
  - 83×83 (166 clean bands, one in two), n = 1 024 × number of valid dates (≥ 1 024)
* - Outputs
  - cov · mean (83,) · raw (T_ok, 1024, 83) pixels of the valid dates
* - Splits
  - `fold5` (= `recommended`; FINE 949 / 322 / 311, MIX 1 229 / 390 / 412) · `block10km`
* - Module
  - {mod}`spdnet_datasets.hyperspectral.aviris` (`fetch_aviris`, `AVIRISDataset`, registry name `aviris`)
```

```{note}
LISTIC built both versions of this dataset on 13 November 2026. They are
**not published**, so the module lives on the `local` branch only. There is no
download URL, and the fetch cannot rebuild the dataset from the public
sources. It only checks the local copy against `MANIFEST_NAS.tsv`.
```

## Description and source

AVIRIS (Airborne Visible/InfraRed Imaging Spectrometer) Classic is the
imaging spectrometer of NASA (National Aeronautics and Space Administration)
and JPL (Jet Propulsion Laboratory), flown on ER-2 aircraft. It records 224 bands
from 366 to 2 496 nm, about 9.7 nm apart, with pixels of 14.8 to 17 m
depending on the flight line.

The dataset holds windows of 600 m × 600 m in California, in EPSG:5070
(EPSG: European Petroleum Survey Group code; 5070 = Albers equal-area
projection of the contiguous United States). Each window is resampled to 32 × 32 pixels of
18.75 m and followed at exactly 14 flight dates from 2008 to 2020. 70 % of
the flight pairs fall in 2013–2015, during the preparatory campaign of HyspIRI (Hyperspectral Infrared
Imager).
The labels come from the USDA NASS Cropland Data Layer (CDL), the crop and
land-cover map of the US Department of Agriculture's National Agricultural
Statistics Service.

- Images: NASA/JPL AVIRIS-Classic L2 calibrated reflectance products,
  distributed by the ORNL DAAC (Oak Ridge National Laboratory Distributed
  Active Archive Center), <https://daac.ornl.gov/AVIRIS/>. They are free to
  use with citation, still to be checked.
- Labels: USDA NASS CDL 2008–2021, public domain, still to be checked.
- Derived dataset: research use. No DOI.

Two versions are on one page because they share the windows' construction and
the module (`version=`):

| Version | Folder | Windows | Label |
|---|---|---|---|
| FINE | `HYPERSPECTRAL/AVIRIS-C_CA_TS14_FINE` | 1 590 (1 582 kept) | one of 12 classes; the dominant class covers at least 70 % of the window in every flight year and does not change between years |
| MIX | `HYPERSPECTRAL/AVIRIS-C_CA_TS14_MIX` | 2 034 (2 031 kept) | proportions of 13 classes (12 + `autre`); 1 494 mixed windows (dominant share 30–70 %) and 540 nearly pure ones |

The windows with no valid date are discarded and counted in `dataset.info`:
8 for FINE and 3 for MIX.

## Raw data

```text
windows/<window_id>_<class>.npz   (T=14, 224, 32, 32) float16 reflectance 0-1, NaN = no data
meta/bands.csv                    the 166 clean bands (column "clean")
meta/date_mask.npz                dates that passed the quality control (QC)
meta/labels.csv                   classes, proportions, fold5, block10km, coordinates
```

The QC thresholds are heuristic and were not validated:

- NDSI > 0.4 with blue > 0.15, where NDSI is the normalised difference snow
  index;
- blue > 0.3 and near infrared > 0.4;
- more than 50 % of the bands at 0.

`return_raw` gives the pixels of the QC-valid dates as `(T_ok, 1024, p)`,
clipped to [0, 1.5], with NaN kept. The figures below show the band mean of
four windows, with dates as rows and pixels as columns.

```{raw} html
<iframe src="../_static/figures/aviris_fine/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

The class identifiers are the French names of the CDL groups:

| Identifier | Class |
|---|---|
| `amandiers` | almond orchards |
| `arbustes` | shrubland |
| `eau` | water |
| `foret_coniferes` | evergreen (coniferous) forest |
| `foret_mixte` | mixed forest |
| `prairie_paturage` | grassland / pasture |
| `sol_nu` | barren |
| `urbain_dense` | developed, high intensity |
| `urbain_moyen` | developed, medium intensity |
| `urbain_faible` | developed, low intensity |
| `urbain_ouvert` | developed, open space |
| `vignes` | vineyards |
| `autre` | other types (MIX only) |

FINE has 57 to 171 windows per class (ratio at most 3:1), with at most 4
windows per 10 km block. With `task="fine"`, MIX is labelled by its dominant
class; `urbain_moyen` is then the largest class.

**FINE**

```{raw} html
<iframe src="../_static/figures/aviris_fine/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/aviris_fine/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

**MIX** (dominant class, `task="fine"`)

```{raw} html
<iframe src="../_static/figures/aviris_mix/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/aviris_mix/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

**Bands.** The 166 clean bands of `meta/bands.csv` are kept, one in
`band_step` (default 2), so p = 83.

**Observations: a "static" covariance.** The observations of a window are the
finite, non-zero pixels of **all its QC-valid dates pooled**, with reflectance
clipped to [0, 1.5]. This gives n = 1 024 × (number of valid dates), at least
1 024, which is above 2p = 166. A window with fewer than 2p valid pixels would
be discarded and counted; none is on the real data. The pooling mixes seasons
and years. The covariance therefore describes the spectral variability of the
window over time, not one acquisition.

**Normalisation.** The cache keeps unscaled covariances. At load time, the
geometric scale of `core.normalize` is fitted on the training part of the
chosen split. Neighbouring bands are strongly correlated, so the matrices are
badly conditioned. The median condition number of the SCM (sample covariance matrix) covariances is
about 2·10⁷ for both versions, and `estimator="ledoit_wolf"` reduces it.

FINE:

```{raw} html
<iframe src="../_static/figures/aviris_fine/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/aviris_fine/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

MIX:

```{raw} html
<iframe src="../_static/figures/aviris_mix/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/aviris_mix/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

The dataset comes with two spatial splits:

- `fold5` (= `recommended`) uses the atlas folds, assigned by 10 km blocks and
  balanced greedily by class (column `fold5`). The test set is fold `fold`
  (default 0) and the validation set fold `fold + 1`.
- `block10km` is a `StratifiedGroupKFold` on the 10 km blocks.

| Version | Split | train / val / test |
|---|---|---|
| FINE | `fold5`, fold 0 | 949 / 322 / 311 |
| FINE | `block10km` | 948 / 317 / 317 |
| MIX | `fold5`, fold 0 | 1 229 / 390 / 412 |
| MIX | `block10km` | 1 220 / 406 / 405 |

The fold assignment is drawn on the list of windows of the preprocessing,
which does not depend on the estimator. The SCM and Ledoit-Wolf runs therefore
share the same test groups.

```{raw} html
<iframe src="../_static/figures/aviris_fine/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/aviris_mix/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Neighbouring windows on both sides | 10 km blocks never cross parts | **no guard band**: 4.5 % (FINE) and 3.2 % (MIX) of the test windows lie less than 1 km from a training window (atlas; median distance 4.9 km) |
| Same flight line seen by train and test windows | none | acquisition conditions shared across blocks |
| Overlap between FINE and MIX | none (separate datasets) | 540 nearly pure MIX windows come from earlier datasets and may coincide with FINE windows; never train on one version and test on the other |
| Normalisation fitted on test data | geometric scale fitted on the training part only | none |

## Supported tasks

- **FINE** with `task="fine"`: 12 classes. `task="regression"` raises an
  error, because FINE has no continuous target.
- **MIX** with `task="regression"`: the target is the vector of 13
  proportions, shape (2 031, 13), each row summing to 1. With `task="fine"`,
  the target is the dominant class among 13, `autre` included.
- `task="coarse"` (`COARSE_MAP`) merges the classes into 7 groups: urbain,
  foret, arbustes, prairie, vergers_vignes (almonds and vineyards), eau and
  sol_nu. MIX adds `autre` as an eighth group.
- `drop_classes`, `merge_classes` and `balance` apply to the classification
  tasks.

## Usage

```python
from spdnet_datasets.hyperspectral.aviris import AVIRISDataset

fine = AVIRISDataset(version="fine", split="recommended")             # 12 classes, 83×83
mix = AVIRISDataset(version="mix", task="regression")                 # 13 proportions
dom = AVIRISDataset(version="mix", task="fine", subset="train")       # dominant class
```
