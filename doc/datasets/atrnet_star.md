# ATRNet-STAR

```{list-table}
:stub-columns: 1

* - Domain
  - SAR (synthetic aperture radar), automatic target recognition (ATR)
* - Task(s)
  - fine classification (40 civilian vehicle classes)
* - Covariance
  - none: raw or mean outputs only
* - Outputs
  - raw (128, 128, 4) amplitudes, (128, 128, 8) complex parts, or (128, 128, 1) 8-bit amplitude · mean (per channel)
* - Splits
  - one protocol per instance (`split=`): `SOC_40classes`, `EOC_scene`, `EOC_depression`, `EOC_azimuth`, `EOC_band`, `EOC_pol`; inside it, `dataset.splits["official"]`
* - Module
  - {mod}`spdnet_datasets.sar.atrnet_star` (`fetch_atrnet_star`, `ATRNetSTARDataset`, `protocol_overlap`)
```

## Description and source

ATRNet-STAR holds 128 × 128 SAR chips of 40 civilian vehicles: cars, vans,
trucks and construction machines. An airborne radar acquired them in X and Ku
bands, with quad polarisation (HH, HV, VH, VV; H horizontal, V vertical,
emission then reception) and a resolution of 0.12–0.15 m. The acquisitions
cover several depression angles (15°, 30°, 45°, 60°), azimuths over 0–359°
and scenes. There are two representations:

- `Slant_Range`: complex chips, one `.mat` per chip with four complex
  channels;
- `Amplitude_8bit`: one 8-bit `.tif` per chip **and per polarisation**.

- Liu Y. et al., *ATRNet-STAR: A Large Dataset and Benchmark Towards Remote
  Sensing Object Recognition in the Wild*, arXiv:2501.13354
  (<https://arxiv.org/abs/2501.13354>). Version 4 of the paper is in the
  dataset folder.
- The licence is not stated in the local copy. The data must be downloaded by
  hand and placed under `<root>/SAR/ATRNet-STAR`. There is no checksum
  registry.

## Raw data

```text
{Slant_Range|Amplitude_8bit}/<protocol>/{train|test|test_<cond>}/<class>/
    <BAND>_<POL>_<DEP>_<AZ>_<id>.{mat|tif} (+ .xml)
```

| Representation | Raw output | Note |
|---|---|---|
| `Slant_Range`, `slant_features="amplitude"` (default) | (128, 128, 4) float32, \|S\| of HH, HV, VH, VV | |
| `Slant_Range`, `slant_features="complex"` | (128, 128, 8): hh.re, hh.im, hv.re, … | |
| `Amplitude_8bit` | (128, 128, 1) | the polarisations of one acquisition have different ids, so they are separate samples |

The mean output is the per-channel mean of the raw chip. The loader builds the
list of chips of a protocol from `MANIFEST_NAS.tsv`, without walking the
2.1 million files, and caches it as JSON in `<cache_dir>/atrnet_star/index/`.

```{raw} html
<iframe src="../_static/figures/atrnet_star/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

There are 40 vehicle models, with 266 to 530 training chips per class in
`SOC_40classes` (`Slant_Range`). The mean-profile figure shows the first 8
classes in the four polarisations.

```{raw} html
<iframe src="../_static/figures/atrnet_star/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/atrnet_star/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

The loader builds no covariance. The mission restricts the SAR chip datasets
to raw or mean outputs (`OUTPUTS = ("raw", "mean")`). Asking for a covariance
raises `ValueError`, and exactly one of `return_raw=True` and
`return_mean=True` must be set. With `slant_features="complex"`, the four
complex channels allow a 4 × 4 polarimetric covariance per pixel
neighbourhood, but the loader does not compute it.

## Splits

The protocols are those of Table 4 of the paper. The constructor argument
`split=` selects one of them:

| Protocol | Train → test | Slant_Range train / test |
|---|---|---|
| `SOC_40classes` | random 7:3 draw | 17 003 / 7 243 |
| `EOC_scene` | simple scene → urban, factory, woodland | 4 896 / 19 350 |
| `EOC_depression` | 15° → 30°, 45°, 60° | 5 557 / 18 689 |
| `EOC_azimuth` | 0–60° → other azimuths | 4 650 / 19 596 |
| `EOC_band` | X → Ku | 6 918 / 6 935 |
| `EOC_pol` | HH → HV, VH, VV (`Amplitude_8bit` only) | not available in Slant_Range (explicit error) |

Here SOC stands for standard operating conditions and EOC for extended
operating conditions. `Amplitude_8bit/SOC_40classes` has 68 091 training and
29 284 test chips.

Inside an instance, `dataset.splits` holds three kinds of split:

- `official`: the train / test of the protocol, with 10 % of train drawn as
  validation (`val_size`);
- `official_<cond>`: the same, restricted to one test condition;
- `recommended`.

The `test/` folder is the union of the `test_<cond>` folders, which was
checked. The condition of each test chip is in `meta["condition"]`. For
`SOC_40classes` in `Slant_Range`, the split is 15 302 train, 1 701 validation
and 7 243 test chips.

**Empty files.** `Amplitude_8bit/EOC_depression` has 0-byte chips: all of
`test_30`, `test_45` and `test_60`, and 6 987 chips in `train`. Each test chip
is read from its non-empty copy in `test/`, which holds 30 085 of them. Chips
without a non-empty copy are excluded and counted in `dataset.n_empty`:
44 770 test and 6 987 training chips.

The group of a chip is its own id, so every group has size 1 in the figure
below. The protocols are defined by acquisition conditions, not by groups.

```{raw} html
<iframe src="../_static/figures/atrnet_star/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

**Within a protocol**, train and test share no file name and no file content.
The md5 of every chip was measured on BERTHA for both representations, and
`assert_no_duplicates` holds.

**Across protocols**, chips are **reused**. Every protocol is a re-partition
of the same pool of chips. In `Slant_Range`, for example, 13 112 test chips of
`EOC_depression` are training chips of `SOC_40classes`. The figure counts,
with `protocol_overlap()`, the chips (same class and file name) shared by the
train and test parts of every pair of protocols. The diagonal blocks of each
protocol have zero train–test overlap.

```{raw} html
<iframe src="../_static/figures/atrnet_star/protocol_overlap.html" class="plotly-figure" style="height:580px" loading="lazy"></iframe>
```

| Risk | Measure | Residual |
|---|---|---|
| Same chip in train and test of one protocol | name and md5 checked (`assert_no_duplicates`) | none |
| Pre-training on one protocol, testing on another | not pruned, because pruning would break the official protocols; documented, and `protocol_overlap()` reports it | **never** pre-train on one protocol and test on another |
| Same physical vehicle in train and test (one vehicle per class) | none | SOC measures instance recognition under close conditions, not generalisation to new vehicles |
| Empty files in `Amplitude_8bit/EOC_depression` | read from the non-empty copy in `test/`, otherwise excluded and counted | 51 757 empty chips have a non-empty copy of the same name in `Amplitude_8bit/SOC_40classes` (md5 identical for 44 476 of the 44 522 that can be compared); they are not recovered |

## Supported tasks

- `task="fine"`: 40 classes.
- `drop_classes` / `merge_classes` take class names.
- `balance` acts on the training part only.
- The protocols give domain-shift tasks (scene, depression, azimuth, band,
  polarisation). Per-condition test sets are in `official_<cond>`.

## Usage

```python
from spdnet_datasets.sar.atrnet_star import ATRNetSTARDataset, protocol_overlap

ds = ATRNetSTARDataset(split="EOC_depression", representation="Slant_Range", return_raw=True)
x, y = ds[0]                         # (128, 128, 4) amplitudes
protocol_overlap(a="SOC_40classes", b="EOC_depression")
# {('train', 'train'): 3891, ('train', 'test'): 13112, ('test', 'train'): 1666, ('test', 'test'): 5577}
```
