# FUSAR-Ship 1.0

```{list-table}
:stub-columns: 1

* - Domain
  - SAR (synthetic aperture radar), ship recognition
* - Task(s)
  - fine classification (10 categories with `class_level="basic"`; sub-categories with `"fine"`)
* - Covariance
  - none: raw or mean outputs only
* - Outputs
  - raw (224, 224, 1) log-amplitude · mean (1,)
* - Splits
  - `sgkf5_fold0` … `sgkf5_fold4` (`recommended` = fold 0; 3 118 / 1 040 / 1 040)
* - Module
  - {mod}`spdnet_datasets.sar.fusar_ship` (`fetch_fusar_ship`, `FUSARShipDataset`; registry names `fusar_ship` and `fusar`)
```

```{dataset-preview} fusar_ship
```

## Description and source

FUSAR-Ship 1.0 is a set of 8-bit amplitude chips cut from Gaofen-3 (GF-3)
C-band images in the DH and DV dual-polarisation modes (horizontal or
vertical emission, both receptions). Each chip is centred on a
ship. The ship is matched automatically with AIS (Automatic Identification
System) messages, which give its MMSI (Maritime Mobile Service Identity),
dimensions and type. The labels are a category / sub-category hierarchy taken
from the AIS type, with a very long tail. `meta.csv` gives the MMSI, the AIS
dimensions, the polarisation mode, the incidence angle and the pixel size
(about 1.12 m × 1.54–1.75 m). The dataset has no official split.

- Hou X. et al., *FUSAR-Ship: building a high-resolution SAR-AIS matchup
  dataset of Gaofen-3 for ship detection and recognition*, Science China
  Information Sciences 63, 140303, 2020, doi:10.1007/s11432-019-2772-5.
- Fudan University holds the copyright (`ReadMe.txt`). The reuse conditions are
  not stated in the folder. See
  [Getting the data](#getting-the-data).

## Getting the data

Direct, no account: the Google Drive file `FUSAR_Ship1.0.zip` linked by
<https://emwlab.fudan.edu.cn/resources/list.htm> (507 383 750 bytes, sha256
`0e985e3c…` measured on 2026-11-15; its 5 246 files have the CRC-32 of the local
copy). Without a local copy, `fetch_fusar_ship()` downloads it into
`<cache>/FUSAR_Ship1.0/downloads/`. If Drive refuses (quota),
`ManualDownloadRequired` gives the Drive and Baidu Pan (code `c6ea`) links.

```python
fetch_fusar_ship(archive="~/Downloads/FUSAR_Ship1.0.zip")
```

## Raw data

```text
<Category>/<SubCategory>/Ship_CxxSyyNzzzz.tiff   8-bit amplitude quick-looks (512 × 512, 256 × 256 or nearly square)
meta.csv                                          id, mmsi, ..., path
```

The loader applies three steps:

1. it takes `log1p` of the amplitude, because the amplitude is heavy-tailed;
2. it resizes the chip with antialiasing to `image_size × image_size`
   (default 224);
3. with `standardize=True` (default), it standardises the raw chip on its own
   (mean 0, standard deviation 1).

The mean output is the mean of the resized `log1p` chip, computed before
standardisation, which would make it zero. The figures show the standardised
raw chips and the distribution of their values.

```{dataset-figure} fusar_ship/raw_samples
```

```{dataset-figure} fusar_ship/amplitude_hist
```

## Classes

With `class_level="basic"`, the categories with fewer than
`min_samples_per_class=30` chips are dropped: DiveVessel, HighSpeedCraft,
PortTender, SAR and WingInGrnd. That leaves 10 categories and 5 198 chips:

| Cargo | Other | Fishing | Tanker | Unspecified | Passenger | Tug | Dredger | Reserved | LawEnforce |
|---|---|---|---|---|---|---|---|---|---|
| 2 114 | 1 644 | 789 | 248 | 142 | 64 | 64 | 61 | 37 | 35 |

Cargo and LawEnforce differ by a factor of 60, so use `balance="weights"` or a
balanced metric.

```{dataset-figure} fusar_ship/class_hist
```

```{dataset-figure} fusar_ship/class_means
```

## Covariance construction

The loader builds no covariance. By design, the SAR chip datasets of the package give raw or mean
outputs only ({doc}`../guide/outputs`) (`OUTPUTS = ("raw", "mean")`), and exactly one of
`return_raw=True` and `return_mean=True` must be set. This changes the API of
the historical `real/fusar.py` loader in two ways: raw chips are now
(H, W, 1) instead of (1, H, W), and setting one of the two flags is mandatory.

## Splits

One vessel can appear in several chips. The loader groups the chips by MMSI
with a union-find over `meta.csv`:

- a chip matched with several MMSI joins them into one group;
- rows with an empty MMSI are ignored;
- a chip without any MMSI is its own group.

The 5 198 chips form 3 871 groups. The splits `sgkf5_fold{k}` are
`StratifiedGroupKFold(5)` folds on these groups, with a grouped validation part
(`check_split_groups`). `recommended` is fold 0: 3 118 train, 1 040
validation, 1 040 test.

```{dataset-figure} fusar_ship/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| Same vessel in train and test | MMSI groups (union-find), split by group | chips with no MMSI cannot be linked |
| Same GF-3 scene on both sides (sea state, incidence angle, neighbouring ships) | none: the original scenes are not provided | possible scene-context leak |
| Rare classes | classes below 30 chips dropped; stratified group folds | Reserved and LawEnforce have only 7 or 8 test chips in fold 0: high variance |

## Supported tasks

- `class_level="basic"` (categories) or `"fine"` (`Category_SubCategory`),
  with the same `min_samples_per_class` filter.
- `exclude_classes` and `drop_classes` remove classes; `merge_classes` merges
  them.
- `balance="undersample"`, `"oversample"` or `"weights"` acts on the training
  part only.

## Usage

```python
from spdnet_datasets.sar.fusar_ship import FUSARShipDataset

ds = FUSARShipDataset(return_raw=True, image_size=224)
x, y = ds[0]                         # (224, 224, 1) float32
parts = ds.splits["recommended"]     # train / val / test indices
```
