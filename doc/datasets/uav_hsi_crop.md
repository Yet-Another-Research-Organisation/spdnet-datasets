# UAV-HSI-Crop

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI) from an unmanned aerial vehicle (UAV), crop mapping
* - Task(s)
  - fine classification (28 crop codes, `code 1` to `code 28`)
* - Covariance
  - 100×100 (one band in two of 200), n = 200 to 225 non-zero pixels of a 15×15 window (n / 2p = 1.00 to 1.125)
* - Outputs
  - cov · mean · raw (15 × 15 × 100)
* - Samples
  - 12 923 windows (XJM 5 207, MJK_N 3 504, MJK_S 4 212); 303 discarded (< 2p non-zero pixels)
* - Splits
  - `spatial` = `recommended` (7 549 / 2 570 / 2 798) · `official` (7 116 / 773 / 1 912) · `scene_transfer` (MJK_N → MJK_S: 3 504 / 0 / 4 212) · `random` (sample-level 70 / 10 / 20, ignores the groups, see {doc}`/guide/splits`)
* - Module
  - `spdnet_datasets.hyperspectral.uav` (`fetch_uav`, `UAVDataset`, registry name `uav`)
```

```{dataset-preview} uav_hsi_crop
```

## Description and source

UAV-HSI-Crop is a crop-mapping dataset acquired by a UAV over three scenes
(XJM, MJK_N and MJK_S), during a single flight on 18 September 2019
according to the literature quoted in the atlas fiche. The data are reflectance from 400 to 1 000 nm, with
200 bands at 0.1 m. The published dataset is a semantic segmentation
benchmark: 433 patches of 96 × 96 × 200, with an official split drawn at
random by patch.

- Reference: B. Niu *et al.*, "HSI-TransUNet: a transformer based semantic
  segmentation model for crop mapping from UAV hyperspectral imagery",
  *Computers and Electronics in Agriculture* 201, 107297, 2022.
- Data: ScienceDB, doi:[10.57760/sciencedb.01898](https://doi.org/10.57760/sciencedb.01898),
  and <https://github.com/MrSuperNiu/UAV-HSI-Crop-Dataset>. Licence: see
  ScienceDB (not verified).
- The crop names behind the codes are not available locally: classes are
  `code k`.

## Getting the data

Manual. ScienceDB, doi:10.57760/sciencedb.01898, version V6, CC BY-SA 4.0:
<https://www.scidb.cn/detail?dataSetId=6de15e4ec9b74dacab12e29cb557f041> (page
rendered in JavaScript, no stable file URL). The record holds one file of
1 017 930 890 bytes (name not recorded, no checksum). Without a local copy,
`fetch_uav()` opens the page and raises `ManualDownloadRequired`; then pass the
file (zip or tar, size checked) or the extracted folder. The folder holding
`Train/Training/gt` and `Test/gt` is used as `source/`.

```python
fetch_uav(archive="~/Downloads/<file of the record>")
```

## Raw data

`source/{Train/Training, Train/Validation, Test}/{rs, gt}/<SCENE>_patch_<k>_<row>_<col>.npy`
holds `rs`, the 96 × 96 × 200 reflectance, and `gt`, the 96 × 96 crop codes
(0 = unlabelled). The (row, column) indices of the names form complete
grids: XJM 9 × 17, MJK_N 13 × 10 and MJK_S 15 × 10 patches. The official
split is 312 training, 34 validation and 87 test patches. The raw output is
the 15 × 15 × 100 window; the figure shows the band-mean image of four
windows.

```{dataset-figure} uav_hsi_crop/raw_samples
```

## Classes

The 28 codes are very imbalanced. The five largest are `code 1` (4 497
windows), `code 2` (2 502), `code 10` (1 752), `code 3` (1 320) and `code 19`
(887). Eleven codes have fewer than 30 windows: `code 7` has 2, and `code 6`
and `code 27` have 4 each. Each scene holds only part of the codes.

```{dataset-figure} uav_hsi_crop/class_hist
```

```{dataset-figure} uav_hsi_crop/class_means
```

## Covariance construction

1. **Scenes**: each scene is recomposed from its patches. The patch at
   (row, column) is placed at pixel (96 · row, 96 · column), which gives
   full grids of 864 × 1 632 (XJM), 1 248 × 960 (MJK_N) and 1 440 × 960
   (MJK_S) pixels.
2. **Windows**: 15 × 15 windows on the per-class best grid offset. A window
   belongs to a class when at least 80 % of its labelled pixels have that
   class (same rule as the original script and as GSOFF). Windows of
   different classes may overlap.
3. **Bands**: one in two (`band_step=2`), so p = 100. Each window has at most
   225 pixels, so n / 2p ≤ 1.125: this dataset sits at the limit of the
   n ≥ 2p rule.
4. **Observations**: the non-zero pixels of the window. 303 windows have
   fewer than 2p = 200 of them and are discarded: XJM 110, MJK_N 54 and
   MJK_S 139 (`info["scenes"]`).
5. **Estimator**: SCM (sample covariance matrix) by default. With n close
   to 2p, `estimator="ledoit_wolf"` is advisable.
6. **Normalisation**: covariances are scaled on the training part of `split`
   (geometric mean of the eigenvalues = 1). Means and raw windows are
   multiplied by √s.

```{dataset-figure} uav_hsi_crop/cov_samples
```

```{dataset-figure} uav_hsi_crop/eigen_spectrum
```

## Splits

- **`spatial` (recommended)**: the 480-pixel blocks (5 × 5 patches) of each
  scene are the groups of a `StratifiedGroupKFold(5)` (seed 0; 20 blocks
  hold windows). Test = fold `fold`, validation = fold `fold + 1`.
  Validation and test windows that overlap a training window, and test
  windows that overlap a validation window, are removed (6 windows). With
  `fold=0` this gives 7 549 / 2 570 / 2 798. Rare codes are missing from
  some parts: four codes are absent from train (`code 5`, `6`, `9`, `21`),
  seven from validation and eight from test.
- **`official`**: a window that lies inside a single patch takes the official
  split of that patch, which gives 7 116 / 773 / 1 912. The 3 122 windows that
  straddle two patches are left out. Patches were drawn at random, so
  neighbouring patches can fall on both sides of the split. Use this split to
  compare with the paper, not as a leakage-free benchmark.
- **`scene_transfer`**: train on MJK_N (3 504 windows), test on MJK_S
  (4 212). There is no validation part. This split is defined when both
  scenes are loaded.

```{dataset-figure} uav_hsi_crop/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| overlapping windows | buffer removal in `spatial`; checked by the real-data test | none (`spatial`) |
| neighbouring patches on both sides (random official split) | `spatial` recommended; `official` kept for comparison only | `official`: adjacent patches in train and test |
| non-official grid of the former prepared windows | scenes recomposed from `source/` by (row, column) | none |
| scenes in both train and test | `scene_transfer` | `spatial` mixes scenes, but blocks never span two scenes |

## Supported tasks

- `task="fine"`: 28 codes, all scenes. Use `scenes=("XJM",)` to evaluate one
  scene, or `split="scene_transfer"` to test transfer between scenes.
- There is no coarse map. `drop_classes`, `merge_classes`, `classes` and
  `max_classes` let you remove the rare codes. This is recommended, because
  four codes are absent from the spatial training set.
- `balance="undersample"`, `"oversample"` or `"weights"` acts on the training
  part only.

## Usage

```python
from spdnet_datasets.hyperspectral.uav import UAVDataset

ds = UAVDataset()                                         # 3 scenes, ds.splits: spatial, official, scene_transfer
xjm = UAVDataset(scenes=("XJM",), subset="train", estimator="ledoit_wolf")
transfer = UAVDataset(scenes=("MJK_N", "MJK_S"), split="scene_transfer")
```

## Migration notes

- The former prepared windows (`*_W15_T80`, built by `uav_preprocess.py`)
  took the patch index `k` as the row. They juxtaposed non-neighbouring
  patches: 23 to 41 % of the windows straddled two unrelated patches, and the
  official split was not respected. They are only read by the deprecated
  `real.uav.UAVLegacyDataset` (registry name `uav_legacy`).
- `UAVHSICropDataset` is an alias of `UAVDataset`.
- `split="train"` raises an error: use `subset="train"`. `subset=None` is the
  default.
- Use `estimator=`. `cov_method=...` is silently absorbed by `**kwargs` and
  gives the SCM.

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/uav_hsi_crop.md
```
````
