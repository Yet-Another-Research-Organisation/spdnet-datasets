# ESC-50

```{list-table}
:stub-columns: 1

* - Domain
  - audio (environmental sounds)
* - Task(s)
  - fine classification (50 classes) · coarse (5 major categories)
* - Covariance
  - 20×20 covariance of the MFCC (mel-frequency cepstral coefficients) (or 64×64 log-mel), n = number of non-silent frames (45 to 427, 427 for a clip without silence; n / 2p ≥ 1.1)
* - Outputs
  - cov · mean · raw (frames × p)
* - Samples
  - 1 979 clips (2 000 minus 4 mostly silent clips and 17 numerically singular covariances)
* - Splits
  - `official` = `recommended`: 5 official folds, test = fold `fold`, val = next fold, train = the 3 others (fold 1: 1 198 / 400 / 398)
* - Module
  - `spdnet_datasets.audio.esc50` (`fetch_esc50`, `ESC50Dataset`, registry name `esc50`)
```

```{dataset-preview} esc50
```

## Description and source

ESC-50 (Environmental Sound Classification, 50 classes) gathers 2 000 clips of
5 s at 44.1 kHz extracted from [Freesound](https://freesound.org) recordings.
The 50 classes are balanced (40 clips each) and grouped into 5 major
categories: animals, natural soundscapes and water sounds, human non-speech
sounds, interior/domestic sounds, and exterior/urban noises. The dataset comes
with 5 official folds of 400 clips.

- Reference: K. J. Piczak, "ESC: Dataset for Environmental Sound
  Classification", *Proc. ACM (Association for Computing Machinery) Multimedia*, 2015, pp. 1015–1018,
  doi:[10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390).
- Source: <https://github.com/karolpiczak/ESC-50>. Licence: CC BY-NC 3.0
  (Creative Commons Attribution-NonCommercial; Freesound clips).
- Local copy: a Hugging Face export in two parquet files under
  `<data root>/AUDIO/ESC-50` (columns `filename`, `fold`, `target`,
  `category`, `esc10`, `src_file`, `take`, `audio`). The ESC-10 subset (10
  classes, 400 clips) is flagged by the `esc10` column.

## Getting the data

Direct access, no account. Without a local copy, `fetch_esc50()` downloads the two
parquet shards (773 MB) of the Hugging Face mirror `ashraq/esc50` at a pinned
revision and checks their sha256. Already downloaded shards (folder, zip or tar)
can be given with `fetch_esc50(archive="~/Downloads/esc50/")`. Licence CC BY-NC 3.0.
The official GitHub archive (WAV + `meta/esc50.csv`) is not read.

## Raw data

Each clip is a mono 16-bit PCM (pulse-code modulation) WAV (Waveform Audio
File Format) file (220 500 samples at 44 100 Hz) stored as
bytes in the `audio` column and decoded to float in [−1, 1]. The figure shows
the preprocessed data the covariance is computed from: the 20 MFCC of each non-silent frame, as a (coefficient × frame)
image.

```{dataset-figure} esc50/raw_samples
```

## Classes

The 50 classes have 40 clips each before exclusion; 21 clips are excluded
(see *Covariance construction*), so 17 classes have 38 or 39 clips. The bar chart gives the counts per
class in the train, validation and test parts of the official split with test
fold 1. The second figure shows the mean MFCC profile of the first 8 classes.

```{dataset-figure} esc50/class_hist
```

```{dataset-figure} esc50/class_means
```

## Covariance construction

One symmetric positive definite (SPD) covariance per clip, over its frames
(`audio/_features.py`, same chain as the dataset atlas):

1. Power spectrum on Hann frames of `n_fft = 2 ** round(log2(0.046 · sr))`
   samples (2 048 at 44.1 kHz, about 46 ms) with a hop of `n_fft / 4` (512).
2. 64 triangular mel filters from 50 Hz to the Nyquist frequency, in dB, with
   a floor at `max − 80 dB`.
3. `features="mfcc"` (default): orthonormal DCT-II (discrete cosine transform)
   of the 64 log-mel bands, first `n_mfcc = 20` coefficients, so p = 20.
   `features="logmel"` keeps the 64 log-mel bands (p = 64).
4. Frames with more than 50 % exact zeros (digital silence, padding) are
   dropped.
5. One covariance per clip over the remaining frames (n = number of frames,
   427 for a full clip) with the chosen estimator (`estimator="scm"` by
   default: sample covariance matrix; `"ledoit_wolf"`, `"tyler"`, `"student"`
   are available).

A clip with fewer than 2p non-silent frames, or whose estimate is not SPD, is
excluded and listed in `dataset.excluded`; nothing is regularised silently.
With MFCC, 21 clips are excluded (measured on 2026-11-15):

- 4 clips have too few non-silent frames (30 to 34 for 2p = 40):
  `1-100032-A-0` (dog), `1-223162-A-25` (footsteps), `3-155556-A-31`
  (mouse_click) and `5-103422-A-2` (pig);
- 17 clips give a numerically singular covariance (condition number above
  `1e12`, from `1.2e12` to `9.8e16`): near-constant MFCC over the clip.

The atlas counts 216 clips that are mostly silent; the non-silent frames of the
others are enough.

With `normalize=True` (default) the covariances are multiplied by a scalar
`s` fitted on the training part so that the geometric mean of their eigenvalues
is 1; means and raw frames are multiplied by √s.

```{dataset-figure} esc50/cov_samples
```

```{dataset-figure} esc50/eigen_spectrum
```

## Splits

The official 5-fold protocol of the paper is used (`split="official"`, alias
`"recommended"`): test = fold `fold` (1 to 5), validation = the next fold
(cyclically), train = the three other folds. With `fold=1`: 1 198 train
(folds 3, 4, 5), 400 validation (fold 2) and 398 test (fold 1, which holds two
of the excluded clips). The literature trains on 4 folds and tests on the
fifth (1 600 / 400, the protocol of the atlas benchmark): merge `train` and
`val` to reproduce it.

Groups (`dataset.groups`) are the source Freesound recording (`src_file`):
1 506 recordings among the 1 979 kept clips. The second panel of the figure
shows the number of clips per recording. Its title says that groups never
cross splits: for ESC-50 this holds except for the 4 recordings that span two
official folds (see the next section).

```{dataset-figure} esc50/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| several takes (A to H) of one Freesound recording | official folds, which keep the takes of a recording together; `groups = src_file` | 4 recordings (of 1 524) span two folds: known leakage, kept to stay on the official protocol, checked by the test suite |
| random split of clips | not offered: only the official folds | none |
| silent clips | frames > 50 % zeros dropped; clips with n < 2p excluded and listed | none |

## Supported tasks

- `task="fine"`: 50 classes.
- `task="coarse"`: the 5 major categories (`animals`, `natural_soundscapes`,
  `human_non_speech`, `domestic`, `urban`), derived from `target // 10`.
- `drop_classes=[...]`, `merge_classes={...}`, `classes=[...]` and
  `max_classes` restrict or regroup classes; `balance="undersample"`,
  `"oversample"` or `"weights"` acts on the training part only.

## Usage

```python
from spdnet_datasets import create_dataloaders
from spdnet_datasets.audio.esc50 import ESC50Dataset

ds = ESC50Dataset(split="official", subset=None, fold=1)   # every clip, splits indexed
train, val, test = create_dataloaders(ds, split="official", batch_size=32)

ds_test = ESC50Dataset(subset="test", fold=1, features="logmel")  # 64x64 log-mel covariances
```

## Migration notes

- `scaling_factor` is replaced by `normalize=True` (geometric scale fitted on
  the training part of the chosen split).
- The former `series_audio.esc50_series` (branch `timeseries`) builds a series
  of Ledoit-Wolf covariances per 1 s window; this loader gives one covariance
  per clip.

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/esc50.md
```
````
