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

```{raw} html
<iframe src="../_static/figures/air_polsar_seg/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/air_polsar_seg/amplitude_hist.html" class="plotly-figure" loading="lazy"></iframe>
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

```{raw} html
<iframe src="../_static/figures/air_polsar_seg/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/air_polsar_seg/class_means.html" class="plotly-figure" loading="lazy"></iframe>
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

```{raw} html
<iframe src="../_static/figures/air_polsar_seg/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
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
