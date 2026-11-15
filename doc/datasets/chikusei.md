# Chikusei

```{list-table}
:stub-columns: 1

* - Domain
  - hyperspectral imaging (HSI), airborne, land cover
* - Task(s)
  - fine classification (19 classes)
* - Covariance
  - 64×64 (one band in two of 128), n = 225 pixels of a 15×15 window (n / 2p = 1.76); 23×23 windows optional (n / 2p = 4.1)
* - Outputs
  - cov · mean · raw (15 × 15 × 64)
* - Samples
  - 4 199 windows (4 844 extracted, 645 in the buffer zone discarded)
* - Splits
  - `spatial` = `recommended` (320-pixel blocks with a buffer: 2 421 / 789 / 989)
* - Module
  - `spdnet_datasets.hyperspectral.chikusei` (`fetch_chikusei`, `ChikuseiDataset`, registry name `chikusei`)
```

## Description and source

Airborne scene over Chikusei (Ibaraki, Japan), acquired on 29 July 2014 by a
Headwall Hyperspec-VNIR-C imager. VNIR stands for visible and near infrared.
The scene is 2 517 × 2 335 pixels at 2.5 m, with 128 bands from 363 to
1 018 nm. Its ground truth (GT) has 19 land-cover classes and 77 592 labelled
pixels.

- Reference: N. Yokoya and A. Iwasaki, "Airborne hyperspectral data over
  Chikusei", Space Application Laboratory, University of Tokyo, technical
  report SAL-2016-05-27, May 2016. This citation comes from the `Readme.txt`
  of the dataset and was not re-verified.
- Source: <https://www.sal.t.u-tokyo.ac.jp/hyperdata/> (formerly
  park.itc.u-tokyo.ac.jp/sal/hyperdata). Licence: CC BY 3.0, with an acknowledgement of the Space Application
  Laboratory and a citation of the report.
- Local copy: `source/HyperspecVNIR_Chikusei_20140729.mat` (MATLAB v7.3 /
  HDF5, Hierarchical Data Format 5) with its GT, and `prepared_v2/chikusei_v2.h5`, the atlas
  re-extraction of 30 October 2026 that this loader reads.

```{note}
**Orientation.** h5py reads the MATLAB variable as (bands, columns, rows)
= (128, 2 335, 2 517), so the scene must be transposed with `(2, 1, 0)` to
match the frame of the GT (`read_scene_crop`). The former v1 windows (760
windows of 16 × 16) had been cut from the non-transposed array, and their
labels had been read in the other frame: 718 of the 760 windows were
mislabelled. They are no longer used. A real-data test checks that a v2 window
equals the transposed source scene.
```

## Getting the data

Direct, no account. Without a local copy, `fetch_chikusei()` downloads
`Hyperspec_Chikusei_MATLAB.zip` (1.44 GB) from
<https://www.sal.t.u-tokyo.ac.jp/hyperdata/> into
`<cache>/Chikusei/downloads/`. No checksum is published: the size of the zip and
the CRC-32 of its members are checked. The v2 windows are then rebuilt from the
scene with the block assignment of the atlas extraction (`prepare_v2`); on BERTHA
the rebuilt windows, labels, splits and blocks are identical to the local
`prepared_v2/chikusei_v2.h5`. Licence: Creative Commons (CC BY 3.0 on the
mirrors), with the acknowledgement and citation above.

```python
fetch_chikusei(archive="~/Downloads/Hyperspec_Chikusei_MATLAB.zip")  # already downloaded
```

## Raw data

The raw output is the 15 × 15 × 64 window (uint16 reflectance, integer values
from 0 to 15 133, scale undocumented). The figure shows the band-mean image of
four windows.

```{raw} html
<iframe src="../_static/figures/chikusei/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

Nineteen classes, with an imbalance of about 76 between the largest and
smallest:

| Class | Windows | Class | Windows |
|---|---|---|---|
| Forest | 836 | Rice field (first stage) | 79 |
| Rice field (grown) | 830 | Manmade (non-dark) | 77 |
| Manmade (dark) | 420 | Weeds in farmland | 72 |
| Grass | 355 | Manmade grass | 63 |
| Row crops | 354 | Asphalt | 45 |
| Bare soil (farmland) | 307 | Manmade (blue) | 30 |
| Natural plants | 212 | Bare soil (park) | 18 |
| Water | 177 | Manmade (red) | 13 |
| Bare soil (school) | 176 | Paved ground | 11 |
| Plastic house | 124 | | |

```{raw} html
<iframe src="../_static/figures/chikusei/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/chikusei/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

1. **Windows** (from `prepared_v2`): 15 × 15 windows, or 23 × 23 with
   `window=23`, centred on the labelled pixels of a grid of step 4. The label
   is the class of the central pixel. Windows of the same split therefore
   overlap, because the grid step of 4 is smaller than 15.
2. **Bands**: one in two (`band_step=2`), so p = 64 and n = 225 ≥ 2p = 128.
   With `band_step=1` (p = 128), a 15 × 15 window gives n = 225 < 256, and
   the loader raises `InsufficientSamplesError`.
3. **Observations**: the non-zero, finite pixels of the window. Windows with
   fewer than 2p of them are discarded and counted: none are.
4. **Estimator**: SCM (sample covariance matrix) by default, or
   `estimator="ledoit_wolf"`, `"tyler"` or `"student"`. The atlas measures a
   median condition number of about 10⁸ for the SCM at p = 128: regularise.
5. **Normalisation**: with `normalize=True` (the default), the covariances are
   multiplied by a scalar fitted on the training part of `split`, so that the
   geometric mean of their eigenvalues is 1. Means and raw windows are
   multiplied by √s.

The windows are not re-extracted from the source scene. The block assignment
of the split exists only in `prepared_v2`, and a re-extraction gives the same
windows.

```{raw} html
<iframe src="../_static/figures/chikusei/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/chikusei/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

The technical report gives no split. The loader uses the v2 spatial split of
the atlas:

- The scene is cut into an 8 × 8 grid of 320-pixel blocks, assigned to
  train, validation or test (target 60 / 20 / 20). The assignment is the best
  of 60 000 random draws (seed 0).
- A window is kept when its centre is at a Chebyshev distance ≥ w + 1 from
  every pixel of another split. This leaves at least 17 pixels between the
  footprints of two splits. The 645 windows in this buffer zone are discarded
  (`info["excluded_buffer"]`).
- Sizes: 2 421 train, 789 validation and 989 test windows, as in the atlas
  fiche. 36 blocks hold windows.
- Five classes are present in train only: Bare soil (park), Weeds in
  farmland, Rice field (first stage), Manmade grass and Paved ground. Seven
  classes are absent from validation and eight from test.

```{raw} html
<iframe src="../_static/figures/chikusei/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| overlapping windows (grid step 4 < 15) | spatial blocks with a buffer ≥ w + 1; checked by the real-data test (no window overlaps another split) | none |
| random split of windows | not offered | none |
| mislabelled v1 windows (transposed scene) | v2 windows only; orientation test against the source | none |
| classes clustered in one or a few fields | — | large val/test variance; 5 classes train-only |

## Supported tasks

- `task="fine"`: 19 classes. There is no coarse map yet (`task="coarse"`
  raises an error). Merge classes yourself with `merge_classes`, for example
  the four `Manmade (...)` classes, or the three `Bare soil (...)` classes.
- `drop_classes=[...]` is useful for the five train-only classes when
  evaluating on test.
- `balance="undersample"`, `"oversample"` or `"weights"` acts on the training
  part only.

## Usage

```python
from spdnet_datasets.hyperspectral.chikusei import ChikuseiDataset

ds = ChikuseiDataset()                                    # 4 199 windows, ds.splits["spatial"]
test = ChikuseiDataset(subset="test", window=23, estimator="ledoit_wolf")
```

## Migration notes

- The v1 windows (16 × 16, mislabelled) and the former `chikusei_preprocess`
  script are gone. Counts now follow the v2 fiche.
- `split="train"` raises an error: use `subset="train"`. `subset=None` (the
  default) keeps every sample.
- Use `estimator=`. `cov_method=...` is silently absorbed by `**kwargs` and
  gives the SCM.
