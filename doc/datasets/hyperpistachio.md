# HyperPistachio

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), laboratory, food safety
* - Task(s)
  - binary classification at the EU limit of 8 µg/kg (default, `task="coarse"`) · 26 contamination levels (`task="fine"`) · regression of the concentration (`task="regression"`)
* - Covariance
  - `unit="pistachio"` (default): 462×462, one per pistachio, n = the pixels of the nut (median 25 333) · `unit="window"`: 77×77 (one band in 6), windows of 32×32 inside the nut (n = 1 024)
* - Outputs
  - cov · mean · raw (pixels × p of a pistachio, or 32 × 32 × p window)
* - Samples
  - 52 pistachios, or 2 541 windows (20 to 71 per pistachio, median 49); none discarded
* - Splits
  - `by_level` = `recommended` (5 folds grouped by level: 30 / 10 / 12 pistachios for fold 0) · `by_index` (cubes `_0001` → `_0002`) · `by_index_reverse`; with `task="fine"`, `recommended` = `by_index`
* - Module
  - `spdnet_datasets.hyperspectral.hyperpistachio` (`fetch_hyperpistachio`, `HyperPistachioDataset`, registry name `hyperpistachio`)
```

```{dataset-preview} hyperpistachio
```

## Description and source

HyperPistachio is the public subset of a dataset of pistachio nuts
contaminated with aflatoxin B1 (AFB1). It contains 52 hyperspectral cubes
taken by a Resonon Pika XC2 laboratory imager: 256 × 384 pixels, with 462
bands from 386.88 to 1 003.60 nm. The cubes are stored as ENVI (Environment
for Visualizing Images) files in BIL (band interleaved by line) order,
uint16, with reflectance × 10 000. Each cube shows **one** pistachio on a
dark background. There are 26 levels of controlled contamination, from 0 to
114.67 µg/kg, with 2 pistachios per level; the concentration of each level
comes from the description of the Zenodo record and is known per level (a
batch), not measured per nut.

- Reference: A. Sheikh-Akbari and H. Mehrabinejad, "HyperPistachio: A
  Hyperspectral Image Dataset of Aflatoxin B1 Contaminated Pistachio Nuts",
  Zenodo, 5 May 2026,
  doi:[10.5281/zenodo.20027441](https://doi.org/10.5281/zenodo.20027441).
  A. N. Ndu *et al.* (MECO 2026) use it for aflatoxin detection with spectral
  attention.
- **Licence conflict**: the Zenodo metadata say CC BY-ND 4.0. The
  description, however, restricts the subset to academic teaching and
  research, without redistribution or commercial use. Apply the stricter
  rule and do not redistribute the data.

## Getting the data

Direct download, no account. Without a local copy under
`<SPDNET_DATA_ROOT>/HYPERSPECTRAL/HyperPistachio`, `fetch_hyperpistachio()`
downloads `Dataset.zip` from the Zenodo record (2 859 334 911 bytes, md5
checked) into `<cache_dir>/HyperPistachio/downloads/` and extracts it there
(4.7 GB, folders `Level 01` … `Level 26`). A zip already downloaded:

```python
fetch_hyperpistachio(archive="~/Downloads/Dataset.zip")
```

## Covariance construction

1. **Mask of the pistachio**, the recipe of the dataset atlas: Otsu threshold
   on the mean reflectance over 450–900 nm, opening (2 iterations), largest
   connected component, holes filled.
2. **Band decimation**: one band in `decimate` is kept, the others are dropped
   (no averaging): p = ⌈462 / `decimate`⌉.
3. **Sample unit**:
   - `unit="pistachio"` (default, `decimate=1`): one covariance per cube, the
     observations are the pixels of the mask, p = 462 and n / p ≈ 55.
     `decimate=2` gives p = 231.
   - `unit="window"` (`decimate=6` by default, p = 77): windows of
     `window` × `window` pixels (32 by default) on a grid of step
     `window // 2`, kept only when they lie **entirely** in the mask; n = 1 024,
     n / p = 13. The windows of a pistachio are not independent samples.

Reflectances are DN / 10 000. Specular highlights exceed 1 (up to 14 % of
the values of one cube); they are kept.

| Unit | p | n (median) | Samples | κ median (SCM) |
|---|---|---|---|---|
| pistachio, `decimate=1` | 462 | 25 333 | 52 | 3.6e5 |
| pistachio, `decimate=2` | 231 | 25 333 | 52 | 1.7e5 |
| window 32×32, `decimate=6` | 77 | 1 024 | 2 541 | 8.9e3 |

## Splits

- `by_level` = `recommended`: `StratifiedGroupKFold(5)` grouped by level and
  stratified on the EU class; `fold` selects the test fold and the next one is
  the validation. The two pistachios of a level, and all their windows, stay
  on the same side.
- `by_index`: train = the `_0001` cubes, test = the `_0002` cubes, validation
  = 20 % of the training windows (empty with `unit="pistachio"`);
  `by_index_reverse` swaps them. It is the only split where every level is
  seen in training, hence the recommended split of `task="fine"`.

## Leakage risks and measures

- **Contamination level confounded with the session**: the two pistachios of
  a level share their shutter time, frame rate and acquisition minute, and
  every level above 8 µg/kg was acquired with a shutter of 14.74 ms
  (Spearman correlation between shutter and concentration 0.87). A classifier
  can learn the session. `meta` has the `shutter` and `framerate` columns:
  restricting the data to the 13 levels at 14.74 ms is one control.
- With `by_index`, train and test pistachios of a level come from the same
  session; a score on this split mixes contamination and session.
- **52 objects only**: a score on 12 test pistachios has a very wide
  confidence interval; windows multiply the samples, not the objects.

## Supported tasks

- `task="coarse"` (default): `above_limit` (levels 18–26, 18 pistachios) vs
  `below_limit` (levels 1–17, 34 pistachios). The limit of 8 µg/kg is the one
  of the European Union for AFB1 in pistachios sold to the final consumer
  (Regulation (EU) 2023/915).
- `task="fine"`: the 26 levels `L01` … `L26`, 2 pistachios each.
- `task="regression"`: the concentration in µg/kg (from 0 to 114.67, heavily
  skewed: a log transform of the target is advisable).

## Usage

```python
from spdnet_datasets.hyperspectral import HyperPistachioDataset

ds = HyperPistachioDataset()                                # 52 × 462 × 462, EU class
ds = HyperPistachioDataset(decimate=2)                      # p = 231
ds = HyperPistachioDataset(unit="window")                   # 2 541 × 77 × 77
ds = HyperPistachioDataset(unit="window", task="fine")      # 26 levels, by_index split
```

<!-- covariance-series: written by hand (not by write_sections.py) -->
## Covariance series

{class}`~spdnet_datasets.series.CovarianceSeries` cuts each pistachio into a series of
covariance tokens, the patches of a vision transformer (see {doc}`/guide/series` and
{doc}`series_overview`).

```{series-table} hyperpistachio
```

`inner_windows` (default): the windows of 32 × 32 pixels on a grid of step 16 that lie
entirely inside the nut, one band in 6 (`p = 77`, `n_t = 1 024`): the windows of
`unit="window"`, grouped by pistachio. The source is the cube cropped to the nut
(`HyperPistachioDataset.structured_raw`, crop aligned on multiples of 16 so that the grid is
the one of the frame). S depends on the size of the nut: 20 to 71 tokens (median 49, 2 541
in all), padded with masked tokens; it exceeds the S ≤ 32 rule of the other modes by design.
Measured: 100 % SPD tokens, κ median 8.9e3.

```{dataset-figure} hyperpistachio/series_inner_windows
```

```python
from spdnet_datasets.series import CovarianceSeries

series = CovarianceSeries("hyperpistachio")                         # (S, 77, 77), S = 20..71
series = CovarianceSeries("hyperpistachio", window=48, step=24, decimate=3)
```
<!-- /covariance-series -->

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/hyperpistachio.md
```
````
