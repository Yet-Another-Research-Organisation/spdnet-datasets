# AIR-PolSAR-Seg

```{list-table}
:stub-columns: 1

* - Domain
  - SAR (synthetic aperture radar), PolSAR (polarimetric SAR) terrain segmentation reduced to cell classification
* - Task(s)
  - fine classification (6 classes)
* - Covariance
  - none: raw or mean outputs only
* - Outputs
  - raw (32, 32, 4) log10(1 + amplitude) of HH, HV, VH, VV (H horizontal, V vertical; emission then reception) · mean (4,)
* - Splits
  - `macro1024_fold0` … `macro1024_fold4` (`recommended` = fold 0; 33 407 / 11 013 / 11 710)
* - Module
  - {mod}`spdnet_datasets.sar.air_polsar_seg` (`fetch_air_polsar_seg`, `AIRPolSARSegDataset`)
```

```{dataset-preview} air_polsar_seg
```

## Description and source

AIR-PolSAR-Seg contains 500 patches of 512 × 512 pixels. They are cut from
one Gaofen-3 (GF-3) C-band image of Hangzhou (Zhejiang, China), acquired on
29 April 2019 in its quad-polarisation stripmap mode (QPSI) at 8 m resolution. The scene is
9 082 × 9 805 pixels. Each patch holds the four amplitudes HH, HV, VH and VV
(H horizontal, V vertical; emission then reception) as uint16, with a
pixel-level ground truth in six classes. The official split is 350 training
and 150 test patches.

- Wang Z., Zeng X., Yan Z., Kang J., Sun X., *AIR-PolSAR-Seg: A Large-Scale
  Data Set for Terrain Segmentation in Complex-Scene PolSAR Images*, IEEE
  Journal of Selected Topics in Applied Earth Observations and Remote Sensing
  (JSTARS) 15, 3830–3841, 2022, doi:10.1109/JSTARS.2022.3170326. The PDF is in
  the dataset folder.
- Data: <https://github.com/AICyberTeam/AIR-PolSAR-Seg>. The licence is not
  stated; the dataset is described as "open to the community". See
  [Getting the data](#getting-the-data).

## Getting the data

Direct, no account: the Google Drive file `Raw_AIR-PolarSAR-Seg.zip` of the
authors (639 472 477 bytes, sha256 `d62e4391…` measured on 2026-11-15; its
2 502 files have the CRC-32 of the local copy). Without a local copy,
`fetch_air_polsar_seg()` downloads it into `<cache>/air_polsar_seg/downloads/`.
If Drive refuses (quota), the page is opened and `ManualDownloadRequired` gives
the Drive folder and the Baidu Pan link (code `ni88`).

```python
fetch_air_polsar_seg(archive="~/Downloads/Raw_AIR-PolarSAR-Seg.zip")
```

## Raw data

```text
{train_set,test_set}/AIR-PolarSAR-Seg-<k>_{HH,HV,VH,VV}.tiff   uint16, 512 × 512
{train_set,test_set}/AIR-PolarSAR-Seg-<k>_gt.png               RGB labels
```

4.9 % of the amplitudes are 0 and some saturate at 65 535. The raw output is
`log10(1 + A)` for each polarisation. The mean is the per-channel mean of that
raw cell.

```{dataset-figure} air_polsar_seg/raw_samples
```

```{dataset-figure} air_polsar_seg/amplitude_hist
```

## Classes

The ground-truth colours map to six classes:

| Class | RGB colour |
|---|---|
| housing | (255, 255, 0) |
| industrial | (0, 0, 255) |
| natural | (0, 255, 0) |
| land use | (255, 0, 0) |
| water | (0, 255, 255) |
| other | (255, 255, 255) |

Black (0, 0, 0) marks unannotated pixels. With 32 × 32 cells, housing has
23 210 cells and land use only 660.

```{dataset-figure} air_polsar_seg/class_hist
```

```{dataset-figure} air_polsar_seg/class_means
```

## Covariance construction

The loader builds no covariance. By design, AIR-PolSAR-Seg gives raw or mean
outputs only ({doc}`../guide/outputs`) (`OUTPUTS = ("raw", "mean")`), and exactly one of
`return_raw=True` and `return_mean=True` must be set. The dataset ships
amplitudes only, without phase. A PolSAR covariance or coherency matrix
therefore cannot be formed; only an amplitude covariance over the pixels of a
cell could be.

**Sample unit.** The patches overlap massively, so cutting each patch into
cells would duplicate pixels across splits. The loader first **rebuilds the
mosaic**:

1. It finds candidate displacements between patches by hashing 16-pixel row
   segments, probed on a regular grid. The procedure is deterministic.
2. It checks every candidate on its whole overlap and on the four
   polarisations.
3. It reads every patch back from the rebuilt mosaic to check the result.

The result has 1 333 overlapping pairs: the 1 241 pairs found by the atlas
plus 92 more exact overlaps of at least 5 250 pixels. These pairs form 11
connected components instead of 15, with no label conflict. Note that file
names `1` to `150` exist in both `train_set` and `test_set` for **different**
images.

The mosaic is then cut into non-overlapping `cell × cell` cells (default 32).
A cell is kept when three conditions hold:

- it is fully covered;
- at least `min_annotated=0.9` of its pixels are annotated;
- its majority class holds at least `purity=0.75` of the annotated pixels.

This gives 57 267 cells. The atlas reports 57 358, a difference of less than
2 % per class.

## Splits

The official per-image split is not used: 43 of the 150 test patches overlap a
training patch. `macro1024_fold{k}` are `StratifiedGroupKFold(5)` folds on
1 024 × 1 024 pixel macro-blocks of the mosaic, identified by
(component, row // 1024, col // 1024). This gives 123 macro-blocks (atlas:
121). The validation part is a grouped fold. A **guard band** removes the
training and validation cells that touch a test cell, within two cells.

For `recommended` (fold 0), the split is 33 407 train, 11 013 validation and
11 710 test cells. The guard band removes 1 137 cells. No cell is shared by
train and test, checked by md5.

```{dataset-figure} air_polsar_seg/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Overlapping patches across the official split | mosaic rebuilt from exact overlaps, official split not used | none |
| Same pixels in two cells | non-overlapping cells on the mosaic | none |
| Neighbouring cells on both sides | 1 024 px macro-blocks and a guard band of two cells | spatial correlation beyond the guard band (one scene, one date) |
| Same file name, different image | identification by content, not by name | none |

## Supported tasks

- `task="fine"`: 6 classes.
- `drop_classes` / `merge_classes` take class names.
- `balance` acts on the training part only.
- `cell`, `purity` and `min_annotated` change the sample definition.

## Usage

```python
from spdnet_datasets.sar.air_polsar_seg import AIRPolSARSegDataset

ds = AIRPolSARSegDataset(cell=32, return_raw=True)
x, y = ds[0]                         # (32, 32, 4) float32, log10(1 + A)
ds.positions.head()                  # patch positions in the rebuilt mosaic
```

<!-- covariance-series: generated section, edit write_sections.py -->
## Covariance series

{class}`~spdnet_datasets.series.CovarianceSeries` cuts each 512 x 512 cell into a series of
covariance tokens (see {doc}`/guide/series` and {doc}`series_overview`). The modes and
their defaults come from the registry; the first one is the default.

```{series-table} air_polsar_seg
```

The 8 × 8 grid of 64 × 64 windows needs 512 × 512 cells (loader `cell=512`)
instead of the 32 × 32 cells of the loader's default. With the same rules (cell fully covered,
90 % annotated, majority class at least 75 % of it), only 35 cells remain: the grid is the
user's choice, but its statistics rest on few samples. Variables: `log10(1 + A)` of the four
polarisations (the loader's raw output), then the absolute derivatives along columns and rows
of their mean, standardised on the training cells. The 64 tokens exceed the S ≤ 32 rule of
the other datasets.

```{dataset-figure} air_polsar_seg/series_spatial
```

Measured on the real data (BERTHA, November 2026, default parameters of each mode,
`describe()`): number of samples, tokens per series, observations per token among the
unmasked tokens, fraction of SPD (unmasked) tokens and condition number κ of the unmasked
tokens.

| Mode | Samples | S (max / median) | p | n_t (median, min) | n_t / p | SPD tokens | κ (5 / 50 / 95 %) |
|---|---|---|---|---|---|---|---|
| `spatial` | 35 | 64 / 64 | 6 | 4 096 (4 096) | 682.7 | 100.00 % | 6.4e+00 / 2.4e+01 / 6.9e+01 |

Every default can be overridden by name:

```python
from spdnet_datasets.series import CovarianceSeries, collate_series

series = CovarianceSeries("air_polsar_seg")
series = CovarianceSeries("air_polsar_seg", "spatial", window=(128, 128), step=(128, 128))  # 4 x 4 windows of 128 x 128
(tokens, mask), label = series[0]
```
<!-- /covariance-series -->

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/air_polsar_seg.md
```
````
