# CPAZMaL

```{list-table}
:stub-columns: 1

* - Domain
  - SAR (synthetic aperture radar) time series, cryosphere
* - Task(s)
  - fine classification (8 classes)
* - Covariance
  - 40×40 (amplitude of HH and HV on 20 dates; H horizontal, V vertical, emission then reception), n = valid pixels of the parcel (≥ 80; median ≈ 10 000)
* - Outputs
  - cov · mean (40,) · raw (n, 40) valid pixels
* - Splits
  - `sgkf5_fold0` … `sgkf5_fold4` (`recommended` = fold 0; descending orbit (DSC): 39 / 14 / 13 parcels)
* - Module
  - {mod}`spdnet_datasets.sar.cpazmal` (`fetch_cpazmal`, `CPAZMaLDataset`)
```

## Description and source

CPAZMaL (Cryosphere PAZ SAR data for MAchine Learning) is a set of time
series acquired by PAZ, the Spanish X-band SAR satellite, over the Mont-Blanc
massif. The images are dual-polarisation HH/HV: horizontal emission, with
horizontal or vertical reception. The series cover January 2020 to December
2022 in both orbits:

| Orbit | HH | HV |
|---|---|---|
| ascending (ASC) | 52 dates, 2020-01-10 to 2022-12-23 | 21 dates, until 2020-10-11 |
| descending (DSC) | 66 dates, 2020-01-08 to 2022-12-21 | 30 dates, until 2021-01-27 |

The dataset annotates 68 sites in 8 classes and adds two unlabelled study
areas. The stacks are linear intensities, geocoded on a common EPSG:4326 grid
(EPSG: European Petroleum Survey Group code; 4326 = WGS84 geographic
coordinates).

- Lin-Kwong-Chon C., Gallet M., *CPAZMAL: Cryosphere PAZ SAR Data for MAchine
  Learning*, Workshop SAR & Cryosphere, 19 September 2024, LISTIC
  ([slides](https://www.univ-smb.fr/listic/wp-content/uploads/sites/66/2024/10/04_christophe_linkwongchon_cpazmaldataset.pdf)).
- Context: Kaushik S. et al., *Analysis of the Temporal Evolution of Ice Aprons
  in the Mont-Blanc Massif Using X and C-Band SAR Images*, Frontiers in Remote
  Sensing, 2022, doi:10.3389/frsen.2022.930021.
- Data: `<root>/SAR/PAZ_S1_proc_v2/CPAZMaL.hdf5` (with `CPAZMaL.txt`). The
  same file is published on Hugging Face as
  [`musmb/CPAZMaL`](https://huggingface.co/datasets/musmb/CPAZMaL), Apache-2.0
  according to that page. The PAZ images fall under a Hisdesat/INTA licence,
  so check it before any redistribution.

## Raw data

```text
data/<SITE>/<ORBIT>/<POL>/   stack (H, W, T) of linear intensity, -999 = nodata
                     masks   -127 outside the polygon, else distortion code
```

The Hugging Face README documents the mask codes: 0 no distortion,
1 layover, 2 shadow, 3 both. A pixel is valid when it meets three
conditions:

- it lies inside the polygon (mask ≠ −127);
- its code is at most `max_mask_value` (default 3, all codes kept);
- I > 0 on all 40 channels, which excludes nodata and the interpolated negative
  values found on about 1 % of the border pixels.

The raw output is the (n, 40) matrix of these pixels. Its rows are pixels and
its columns the 20 HH amplitudes followed by the 20 HV amplitudes. The figure
shows four parcels, with at most 300 pixels each.

```{raw} html
<iframe src="../_static/figures/cpazmal/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

| Code | Class | Parcels (DSC) |
|---|---|---|
| ABL | ablation zone | 7 |
| ACC | accumulation zone | 9 |
| HAG | hanging glacier | 7 |
| ICA | ice apron | 9 |
| FOR | forest | 9 |
| LAC | lake | 9 |
| PLA | plain | 7 |
| ROC | rock | 9 |

The loader excludes four sites:

- the two unlabelled study areas (class `STUDY`: ALLGLA, ARGTOP);
- PLA006, which lacks HH or HV in both orbits;
- PLA005, which lacks HV in the descending orbit.

This leaves 66 parcels in DSC and 67 in ASC. `dataset.excluded` lists the
excluded sites with the reason.

```{raw} html
<iframe src="../_static/figures/cpazmal/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/cpazmal/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

**Sample unit: the parcel.** The parcels should come from the source
shapefile, but no shapefile is published: it is neither in the HDF5 file, nor on
Hugging Face, nor in the `CPAZMaL_dataset` GitHub repository, which cites
`DATASET/SHP/<class>/*.shp` without providing it. The polygons survive only
rasterised in the HDF5 masks (`mask != -127`). Each polygon is a single
connected component, which was checked on the 70 groups. A parcel is
therefore a site.

- `block=None` (default): one covariance per parcel, over all its valid
  pixels. A rectangular window around the polygon is never used.
- `block=b`: the parcel is cut by a b × b pixel grid. Every piece with at
  least `min_pixels` valid pixels is a sample, and its group is the parcel.
  With `block=16, min_pixels=120`, this gives 5 751 samples in DSC and 6 332
  in ASC.

**Channels.** p = 2 × `n_dates`. The loader uses the amplitude √I
(`scale="amplitude"`) of HH and then HV, on `n_dates` dates common to HH and
HV. There are 30 common dates in DSC and 21 in ASC. The default `n_dates=20`
dates are spread evenly over the common dates, identical for every parcel, and
stored in `dataset.dates`. The default `orbit` is `"DSC"`. HV stops in October
2020 (ASC) and January 2021 (DSC), so the covariance covers about one year.

```{raw} html
<iframe src="../_static/figures/cpazmal/dates.html" class="plotly-figure" loading="lazy"></iframe>
```

**Observations.** The observations are the valid pixels, and a sample needs
`min_pixels` of them: 80 by default, which is 2p, and 120 recommended. The
parcels have 317 to about 195 000 valid pixels, with a median of about
10 000, so n/p is always above 7.

**Normalisation.** The cache keeps unscaled covariances. At load time, the
geometric scale of `core.normalize` is fitted on the training part of `split`.
The median condition number of the SCM (sample covariance matrix) covariances is about 90 (DSC) and 150
(ASC) per parcel, and about 250 with 16 × 16 blocks.

```{raw} html
<iframe src="../_static/figures/cpazmal/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/cpazmal/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

The paper gives no split. The loader provides `sgkf5_fold{k}`,
`StratifiedGroupKFold(5)` folds grouped by parcel (site), with a grouped
validation part. `recommended` is fold 0:

| Configuration | Samples | train / val / test |
|---|---|---|
| DSC, one covariance per parcel | 66 | 39 / 14 / 13 |
| ASC, one covariance per parcel | 67 | 39 / 14 / 14 |
| DSC, `block=16, min_pixels=120` | 5 751 | 3 225 / 1 455 / 1 071 |
| ASC, `block=16, min_pixels=120` | 6 332 | 3 501 / 1 279 / 1 552 |

With one covariance per parcel, a test fold has one or two parcels per class,
so scores vary strongly between folds. Report the mean over the five folds.

```{raw} html
<iframe src="../_static/figures/cpazmal/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Overlapping windows of the same parcel on both sides (the historical loader cut rectangles) | samples are parcel pixels or blocks inside the polygon; group = parcel, checked by `check_split_groups` | none |
| Neighbouring sites of the same glacier on both sides | none (site positions are not used by the split) | possible spatial correlation between nearby sites |
| Dates | all parcels share the same dates; the dates do not separate train and test | the task is classification of the whole series, not forecasting |
| Normalisation fitted on test data | geometric scale fitted on the training part only | none |

## Supported tasks

- `task="fine"`: 8 classes.
- `drop_classes` / `merge_classes` take the class codes. For example,
  `merge_classes={"ABL": "ice", "ACC": "ice", "HAG": "ice", "ICA": "ice"}`
  merges the four ice classes (`{old_class: new_class}`).
- `balance` acts on the training part only.
- `orbit`, `n_dates`, `scale` (`"amplitude"`, `"intensity"` or `"log10"`), `max_mask_value`
  and `block` change the construction.

## Usage

```python
from spdnet_datasets.sar.cpazmal import CPAZMaLDataset

ds = CPAZMaLDataset(orbit="DSC", n_dates=20, split="recommended")
cov, y = ds[0]                       # (40, 40)
ds.dates                             # the 20 retained dates
blocks = CPAZMaLDataset(block=16, min_pixels=120, return_raw=True)
```
