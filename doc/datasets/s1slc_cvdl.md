# S1SLC_CVDL

```{list-table}
:stub-columns: 1

* - Domain
  - SAR (synthetic aperture radar), land cover; HH / HV polarisations (horizontal emission, horizontal or vertical reception)
* - Task(s)
  - fine classification (7 classes, named "1" to "7")
* - Covariance
  - none: raw or mean outputs only
* - Outputs
  - raw (100, 100, 4) complex parts or (100, 100, 2) amplitudes · mean (|HH|, |HV|)
* - Splits
  - `loco_Houston` (= `recommended`), `loco_Chicago`, `loco_Sao_Paulo` (leave one city out)
* - Module
  - {mod}`spdnet_datasets.sar.s1slc_cvdl` (`fetch_s1slc_cvdl`, `S1SLCCVDLDataset`)
```

```{dataset-preview} s1slc_cvdl
```

## Description and source

S1SLC_CVDL contains 100 × 100 pixel patches of Sentinel-1 (S1) SLC (single
look complex) images. The images are dual-polarisation HH/HV, that is,
horizontal emission with horizontal or vertical reception. The patches cover
three cities (Houston, Chicago, São Paulo), and each one carries one of seven
land-cover labels. The dataset was built for complex-valued deep learning
(CVDL).

- Asiyabi R. M. et al., S1SLC_CVDL, a Sentinel-1 SLC dataset for
  complex-valued deep learning, IEEE DataPort, 2023. The folder holds no paper,
  so this reference is still to be confirmed. The licence is not stated in the
  local copy.
- The data must be downloaded by hand. The local copy is a zarr v3 store
  converted from the original `.npy` files:
  `<root>/SAR/S1SLC_CVDL/complex_HHHV.zarr`. Reading it requires
  `pip install 'spdnet-datasets[sar]'`.

## Getting the data

Manual: [IEEE DataPort](https://ieee-dataport.org/open-access/s1slccvdl-complex-valued-annotated-single-look-complex-sentinel-1-sar-dataset-complex)
(doi:10.21227/nm4g-yd98, CC BY 4.0, free IEEE account, temporary download links).
The file is `S1SLC_CVDL.rar` (12.24 GB, no checksum published). Without a local
copy, `fetch_s1slc_cvdl()` opens the page and raises `ManualDownloadRequired`;
then pass the `.rar` (extracted with the `unrar`, `7z` or `bsdtar` command; Python
has no RAR decoder) or the extracted folder. The city `.npy` files are converted
into `<cache>/S1SLC_CVDL/downloads/complex_HHHV.zarr` (`npy_to_zarr`, port of the
local conversion script, checked on synthetic data only).

```python
fetch_s1slc_cvdl(archive="~/Downloads/S1SLC_CVDL.rar")
```

## Raw data

```text
<City>/data    (N, 100, 100, 4) float32 = [Re HH, Im HH, Re HV, Im HV]
<City>/labels  (N,) int32 in 1..7
```

| City | Patches |
|---|---|
| Chicago | 110 758 |
| São Paulo | 84 238 |
| Houston | 81 575 |
| total | 276 571 |

The values are uncalibrated, at int16 full scale. With `normalize=True`
(default), each city is divided by its RMS (root mean square) amplitude,
estimated on at most `n_stat_patches=2000` patches of that city in the
**training** part of the split. The held-out city gets the RMS of the pooled
training cities, so no test statistic is used. The statistics are cached in
`<cache_dir>/s1slc_cvdl/`.

The store is chunked by 100 patches, so random access is slow. The loader keeps
the last decompressed chunk in memory. Iterate over sorted indices, or sample
whole chunks, as the figures here do with `--contiguous 100`.

The figures show amplitudes (`features="amplitude"`), after the per-city RMS
normalisation.

```{dataset-figure} s1slc_cvdl/raw_samples
```

```{dataset-figure} s1slc_cvdl/amplitude_hist
```

## Classes

The folder does not document the names behind ids 1 to 7. The literature
lists agricultural fields, forest, dense urban, high-rise, low-density urban,
industrial and water, but the order is not verified. The classes are therefore
named `"1"` to `"7"`. They are strongly imbalanced, from 108 981 patches for
class 2 to 1 680 for class 4, and their proportions change from city to city.

```{dataset-figure} s1slc_cvdl/class_hist
```

```{dataset-figure} s1slc_cvdl/class_means
```

## Covariance construction

The loader builds no covariance. By design, S1SLC_CVDL gives raw or mean outputs only
({doc}`../guide/outputs`) (`OUTPUTS = ("raw", "mean")`), and exactly one of
`return_raw=True` and `return_mean=True` must be set. The mean is the mean
amplitude (|HH|, |HV|) of the patch, because the mean of the real and imaginary
parts is zero. A 2 × 2 polarimetric covariance [HH, HV] could be estimated on
the 10 000 complex pixels of a patch. The loader does not do it.

## Splits

`loco_<City>` is leave one city out. The held-out city is the test set. The
validation set is 10 % of the training cities' patches (`val_size`),
stratified by class. The group is the city. `recommended` is `loco_Houston`:
175 496 train, 19 500 validation, 81 575 test.

The dataset gives no patch positions, so no spatial split within a city is
possible. A random split would put neighbouring and overlapping patches on
both sides.

```{dataset-figure} s1slc_cvdl/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Neighbouring patches on both sides | whole city held out | none between train and test |
| Validation drawn from the training cities | stratified 10 % draw | validation is spatially correlated with train, hence optimistic |
| Normalisation statistics from the test city | RMS fitted on the training part only, pooled RMS for the held-out city | none |
| Class proportions differ between cities | none (domain shift by design) | test scores mix class shift and city shift |

## Supported tasks

- `task="fine"`: 7 classes.
- `drop_classes` / `merge_classes` take the names `"1"` to `"7"`.
- `balance` acts on the training part only.
- There is no coarse map and no regression target.

## Usage

```python
from spdnet_datasets.sar.s1slc_cvdl import S1SLCCVDLDataset

ds = S1SLCCVDLDataset(split="loco_Houston", features="complex", return_raw=True)
x, y = ds[0]                         # (100, 100, 4) float32: Re HH, Im HH, Re HV, Im HV
```

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/s1slc_cvdl.md
```
````
