# BirdSet XCM

```{list-table}
:stub-columns: 1

* - Domain
  - audio (bird species, bioacoustics)
* - Task(s)
  - fine classification (95 species, single label)
* - Covariance
  - 20×20 covariance of the MFCC (mel-frequency cepstral coefficients) (or 64×64 log-mel), n = number of non-silent frames of the recording (42 to 199 756, median 1 982)
* - Outputs
  - cov · mean · raw (frames × p)
* - Samples
  - 5 183 recordings of the 95 species with at least 10 recordings in the copy
* - Splits
  - `group_kfold` = `recommended`: 5-fold `StratifiedGroupKFold` by recordist (fold 1: 3 120 / 1 032 / 1 031)
* - Module
  - `spdnet_datasets.audio.birdset_xcm` (`fetch_birdset_xcm`, `BirdSetXCMDataset`, registry name `birdset_xcm`)
```

```{warning}
**Partial copy.** Only 21 parquet shards out of at least 341 were downloaded
(`xcm_00000`, `xcm_00017`, …, `xcm_00340`, one in 17), i.e. 5 371 recordings.
The shards are sorted by species, so only 136 of the 411 species are present,
and the official BirdSet test sets (soundscapes) are absent. Results on this
copy are not comparable with the BirdSet benchmark. `BirdSetXCMDataset` emits a
`UserWarning`.
```

## Description and source

BirdSet is a benchmark for audio classification in avian bioacoustics. XCM is
its "medium" training set: focal [Xeno-Canto](https://xeno-canto.org)
recordings of the species that occur in the 8 soundscape test sets (PER, NES,
UHH, HSN, NBP, POW, SSW, SNE).

- Reference: L. Rauch et al., "BirdSet: A Large-Scale Dataset for Audio
  Classification in Avian Bioacoustics", [arXiv:2403.10380](https://arxiv.org/abs/2403.10380),
  2024 (ICLR 2025, International Conference on Learning Representations).
- Source: <https://huggingface.co/datasets/DBD-research-group/BirdSet>
  (configuration XCM).
- Licence: Creative Commons per recording (column `license`, mostly
  CC BY-NC-SA 4.0, Attribution-NonCommercial-ShareAlike, where it is filled in).
- Local copy: `<data root>/AUDIO/BirdSet_XCM/xcm_*.parquet`.

## Raw data

Each row holds an Ogg Vorbis recording at 32 kHz (mono or stereo, averaged to
mono) of variable length (0.6 to 3 196 s, median 32 s), and metadata:
`ebird_code` (species), `ebird_code_multilabel` (one species per row in this
copy, so the task is single-label), `detected_events` (start and end of the
detected vocalisations, in s), `recordist`, `quality`, … The figure shows the
20 MFCC of the non-silent frames of four
recordings (time axis decimated for display).

```{raw} html
<iframe src="../_static/figures/birdset_xcm/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

Species with at least `min_recordings = 10` recordings in the copy are kept:
95 species and 5 185 recordings (186 recordings of rarer species are left
out). The classes are imbalanced: 10 to 256 recordings per species. The class
names are eBird codes.

```{raw} html
<iframe src="../_static/figures/birdset_xcm/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/birdset_xcm/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

Same chain as {doc}`esc50`: Hann frames of 2 048 samples at 32 kHz (64 ms:
0.046·sr rounded to the nearest power of two) with a hop of 512, 64 mel bands from 50 Hz in
dB with a floor at `max − 80 dB`, first 20 coefficients of the orthonormal
DCT-II (discrete cosine transform), so p = 20 (`features="logmel"`: p = 64).
Frames more than half made of exact zeros are dropped; one symmetric positive
definite (SPD) covariance per recording over its frames, with the chosen
estimator (`"scm"`, the sample covariance matrix, by default).

- Default: the **whole recording** (this is what the figures show).
- `crop_s=5.0`: only an excerpt of 5 s centred on the first detected event, as
  in the atlas benchmark (start of the recording if no event is detected).

Exclusions (listed in `dataset.excluded`): 2 recordings of the 5 185, so 5 183
are kept. `XC539245` (easwar1) has 35 non-silent frames for 2p = 40, and the
sample covariance matrix (SCM) of `XC592530` (blutit) is not SPD (smallest eigenvalue −5·10⁻¹³).

```{raw} html
<iframe src="../_static/figures/birdset_xcm/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/birdset_xcm/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

The official BirdSet protocol (train on XCM, test on the soundscapes) cannot
be used: the test sets are absent. The proposed split groups the recordings by
recordist, since one recordist often records the same individual or the same
site: `split="group_kfold"` (alias `"recommended"`) is a 5-fold
`StratifiedGroupKFold` on the species grouped by `recordist` (seed 0); test =
fold `fold`, validation = the next fold, train = the three others. Empty
recordist names form one group `"NA"`. The 5 183 recordings come from 950
recordists. With `fold=1`: 3 120 / 1 032 / 1 031.

```{raw} html
<iframe src="../_static/figures/birdset_xcm/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| same recordist (individual, site, equipment) in train and test | `StratifiedGroupKFold` by recordist; no recordist shared between train, val and test (tested on all 5 folds) | none for recordists |
| 10 pairs of identical 5 s excerpts under two Xeno-Canto identifiers (atlas) | kept | documented: they can fall in different folds if the two recordists differ |
| domain shift to soundscapes | not measurable (test sets absent) | documented |

## Supported tasks

- `task="fine"`: 95 species. No coarse grouping is defined.
- `min_recordings` changes the species kept; `drop_classes`, `merge_classes`,
  `classes`, `max_classes`; `balance="undersample" | "oversample" | "weights"`
  (training part only) is useful given the 1:26 imbalance.

## Usage

```python
from spdnet_datasets.audio.birdset_xcm import BirdSetXCMDataset

train = BirdSetXCMDataset(subset="train", fold=1, balance="weights")
test = BirdSetXCMDataset(subset="test", fold=1)
crop = BirdSetXCMDataset(subset=None, crop_s=5.0)   # 5 s excerpts, as the atlas benchmark
```

## Migration notes

- `scaling_factor` is replaced by `normalize=True`.
- The former covariance series (`series_audio.birdset_series`, branch
  `timeseries`) used the 5 s crop; the default is now the whole recording.
