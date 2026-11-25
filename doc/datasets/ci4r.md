# CI4R

```{list-table}
:stub-columns: 1

* - Domain
  - micro-Doppler radar (human activities)
* - Task(s)
  - fine classification (11 activities)
* - Covariance
  - 77 GHz: 64×64, n = 1 475 time columns (n / 2p = 11.5) · xethru: 32×32, n = 1 176 (n / 2p = 18.4)
* - Outputs
  - cov · mean · raw (time columns × p)
* - Samples
  - 77 GHz: 645 · xethru: 633 (duplicates removed)
* - Splits
  - `fold0`…`fold3` (participant-grouped 4-fold; `recommended` = `fold<fold>`, default `fold0`), no validation part · `random` (sample-level 70 / 10 / 20, ignores the groups, see {doc}`/guide/splits`)
* - Module
  - `spdnet_datasets.radar_microdoppler.ci4r` (`fetch_ci4r`, `CI4RDataset`, registry name `ci4r`)
```

```{dataset-preview} ci4r
```

## Description and source

CI4R-MULTI3 is a dataset of the Laboratory of Computational Intelligence for
Radar (CI4R, S. Z. Gurbuz): eleven activities (walking
away from and towards the radar, bending, crawling, kneeling, limping, picking
up an object, side-stepping, scissor gait, sitting, walking on tiptoes)
recorded simultaneously by three co-located radars: a Texas Instruments
IWR1443 77 GHz FMCW (frequency-modulated continuous wave) radar, an Ancortek
24 GHz FMCW radar and a XeThru X4 UWB (ultra-wideband) radar.

- Source: <https://github.com/ci4r/CI4R-Activity-Recognition-datasets>
  (related publication: S. Z. Gurbuz, M. M. Rahman, E. Kurtoglu, T. Macks and
  F. Fioranelli, 2020, cross-frequency training; exact title not verified).
  Licence: not stated in the copy (ask the laboratory).
- Only the **77 GHz** and **xethru** modes are provided here; the 24 GHz mode
  is out of scope.
- Local copy: `<data root>/RADAR_DOPPLER/CI4R-MULTI3/11_class_activity_data/{77GHz,Xethru}/<class>/*.png`.

## Getting the data

Direct access, no account. Without a local copy, `fetch_ci4r(mode=...)` lists the
public Google Cloud Storage bucket `cross-frequency_dataset` and downloads the 653
PNG images of the requested sensor (77 GHz 150 MB, xethru 26 MB), each checked
against the MD5 of the listing. `fetch_ci4r(archive=...)` takes a folder, zip or
tar holding `11_class_activity_data`. No licence is stated.

## Raw data

Only micro-Doppler spectrograms rendered as jet-coloured PNG (Portable Network Graphics) images (no axes,
no physical scale) are distributed: 653 files in 77 GHz (804 Doppler rows ×
1 475 time columns) and 652 in xethru (626 × 1 176 useful pixels inside grey
margins). The dynamic range and the physical units are lost in the rendering.
The file name encodes the radar, the activity, the participant (characters 7–8,
inferred) and a Unix timestamp shared by the three radars.

Exact duplicates are removed: 7 files in 77 GHz and 19 in xethru, all named
`(copy)` or `(another copy)` and identical (same MD5, Message-Digest 5, hash) to an original file.

## Covariance construction

1. The jet colour map is inverted: each pixel takes the index (in [0, 1]) of
   the nearest of the 256 colours of the jet table; the grey margins of the
   xethru images are cropped.
2. Rows are flipped so that row 0 is the most negative Doppler bin; the
   central `band` fraction of the Doppler axis is kept and averaged into `p`
   contiguous bins, giving `X` of shape (T, p).
3. One symmetric positive definite (SPD) covariance over the T time columns
   (`n = T`).

| Mode | `band` | `p` | Default estimator | n | Samples |
|---|---|---|---|---|---|
| `"77GHz"` | 0.5 (central 50 %) | 64 | `"scm"` (sample covariance matrix) | 1 475 | 645 (646 unique, 1 rejected) |
| `"xethru"` | 0.2 (central 20 %) | 32 | `"ledoit_wolf"` | 1 176 | 633 |

- 77 GHz: one recording (Crawl, `04110014_1574698309_Raw_0.png`) has 18
  constant rows, hence a constant bin and a singular SCM: it is rejected and
  counted (`info["rejected_constant_channel"]`). p = 96 or 128 remain SPD but
  much worse conditioned (median condition number 10⁶ to 4·10⁶ against
  2.5·10⁵ for p = 64).
- xethru: most pixels sit on a constant background colour, so many bins never
  vary. The SCM is singular for 23 % of the recordings with the 20 % band and
  85 % with a 50 % band; hence the narrow band and the Ledoit-Wolf shrinkage.
  With `estimator="scm"`, the recordings with a constant bin are rejected, but
  others have varying yet collinear bins and the fetch raises
  `NotSPDError`: keep Ledoit-Wolf for xethru.

With `normalize=True` (default), the covariances are multiplied by the
scalar s fitted on the training part of the chosen split (geometric mean of the
eigenvalues = 1), the means and raw spectrograms by √s.

## Splits

There is no official split. The participant code is not documented; it is
inferred from the file name (characters 7–8), with code `00` merged into `14`
(same date), as in the data atlas. `splits["fold<k>"]` is a `GroupKFold(4)`
over participants; validation parts are empty (only 4 to 7 participant codes).
`"recommended"` is `fold<fold>` (`fold=0` by default).

| Mode | fold0 | fold1 | fold2 | fold3 |
|---|---|---|---|---|
| 77 GHz test participants | 29 (221) | 30 (212) | 10 (116) | 14 (96) |
| xethru test participants | 30 (211) | 29 (194) | 05, 11, 14 (117) | 10, 18 (111) |

In 77 GHz the 4 folds are exactly leave-one-subject-out (LOSO). In xethru there
are 7 codes, so two folds hold several participants. The codes 29 and 30 each
span two acquisition dates: they probably group two persons each, which keeps
the split at least as strict as a split by date.

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| same participant in train and test | participant-grouped folds (LOSO in 77 GHz) | participant inferred from the file name (undocumented) |
| exact duplicate files | `(copy)` files removed (MD5) | none |
| xethru codes 05, 11, 18 (February 2019 sessions, 50 files, few classes) | documented only | participant and class partly confounded |
| random split | not offered | the atlas measured 0.66 (LOSO) against 0.81 (random) accuracy with the same model |

## Supported tasks

- `task="fine"`: 11 activities. No coarse grouping is defined.
- `drop_classes`, `merge_classes`, `classes`; `balance="undersample" |
  "oversample"` resamples the training part of every split; class weights are
  computed on the training part (`dataset.class_weights`).
- `return_raw=True` gives the (T, p) spectrogram; `return_mean=True` its
  per-bin mean.

## 77 GHz mode

```{dataset-figure} ci4r_77ghz/raw_samples
```

```{dataset-figure} ci4r_77ghz/class_hist
```

```{dataset-figure} ci4r_77ghz/class_means
```

```{dataset-figure} ci4r_77ghz/cov_samples
```

```{dataset-figure} ci4r_77ghz/eigen_spectrum
```

```{dataset-figure} ci4r_77ghz/split_schema
```

The class histogram shows `fold0` (test = participant 29). The 645
recordings: Away 58, Bend 59, Crawl 60, Kneel 62, Limp 59, Pick 56, SStep 60,
Scissor 57, Sit 57, Toes 58, Towards 59. The raw figure shows the time axis
decimated to 400 columns.

## Xethru mode

```{dataset-figure} ci4r_xethru/raw_samples
```

```{dataset-figure} ci4r_xethru/class_hist
```

```{dataset-figure} ci4r_xethru/class_means
```

```{dataset-figure} ci4r_xethru/cov_samples
```

```{dataset-figure} ci4r_xethru/eigen_spectrum
```

```{dataset-figure} ci4r_xethru/split_schema
```

The class histogram shows `fold0` (test = participant 30). The 633
recordings: Away 60, Bend 59, Crawl 61, Kneel 49, Limp 59, Pick 58, SStep 60,
Scissor 57, Sit 52, Toes 58, Towards 60.

## Usage

```python
from spdnet_datasets import create_dataloaders
from spdnet_datasets.radar_microdoppler.ci4r import CI4RDataset

ds = CI4RDataset(mode="77GHz", fold=0)                 # every sample, splits fold0..fold3
train, val, test = create_dataloaders(ds, split="fold0", batch_size=16)
xe = CI4RDataset(mode="xethru", return_raw=True)       # Ledoit-Wolf 32x32 by default
```

## Migration notes

- `modality=` is a deprecated alias of `mode=`. The 24 GHz mode is no longer
  available.
- The precomputed `cov/` matrices (236×236; 75 % SPD in 77 GHz, 0 % in xethru
  according to the atlas audit) are no longer read.
- `scaling_factor` is replaced by `normalize=True`.
- `subset=None` is the default: the instance holds every sample and
  `dataset.splits` indexes it.

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/ci4r.md
```
````
