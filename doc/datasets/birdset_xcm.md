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

```{dataset-preview} birdset_xcm
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

## Getting the data

Direct access, no account: Hugging Face `DBD-research-group/BirdSet`, branch `data`
(pinned commit), folder `XCM/`: `XCM_metadata.parquet` (16.7 MB) and 182 tar.gz
shards of Vorbis files (95.9 GB in total). The local `xcm_*.parquet` copy is a
partial re-export with no published equivalent, so the published shards are read
directly (labels and recordists from the metadata, audio from the tar members).
Every file is checked against its registered sha256.

```python
fetch_birdset_xcm(data_dir="absent", shards=4)      # 4 shards evenly spread (~2 GB)
fetch_birdset_xcm(data_dir="absent", shards=[1, 50])
fetch_birdset_xcm(archive="~/hf/BirdSet/XCM")      # metadata + shards downloaded by hand
```

The shards are sorted by species: a few shards hold few species (a warning gives
the number of shards used). Licence CC BY-NC 4.0, per-recording Creative Commons
licences.

## Raw data

Each row holds an Ogg Vorbis recording at 32 kHz (mono or stereo, averaged to
mono) of variable length (0.6 to 3 196 s, median 32 s), and metadata:
`ebird_code` (species), `ebird_code_multilabel` (one species per row in this
copy, so the task is single-label), `detected_events` (start and end of the
detected vocalisations, in s), `recordist`, `quality`, … The figure shows the
20 MFCC of the non-silent frames of four
recordings (time axis decimated for display).

```{dataset-figure} birdset_xcm/raw_samples
```

## Classes

Species with at least `min_recordings = 10` recordings in the copy are kept:
95 species and 5 185 recordings (186 recordings of rarer species are left
out). The classes are imbalanced: 10 to 256 recordings per species. The class
names are eBird codes.

```{dataset-figure} birdset_xcm/class_hist
```

```{dataset-figure} birdset_xcm/class_means
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

```{dataset-figure} birdset_xcm/cov_samples
```

```{dataset-figure} birdset_xcm/eigen_spectrum
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

```{dataset-figure} birdset_xcm/split_schema
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

<!-- covariance-series: generated section, edit write_sections.py -->
## Covariance series

{class}`~spdnet_datasets.series.CovarianceSeries` cuts each recording into a series of
covariance tokens (see {doc}`/guide/series` and {doc}`series_overview`). The modes and
their defaults come from the registry; the first one is the default.

```{series-table} birdset_xcm
```

The modes read the 5 s excerpt centred on the first detected event (loader
`crop_s=5`), as the data atlas and `real.series_audio.birdset_series`: whole recordings last up
to several minutes (6 443 windows of 1 s for the longest). The log-mel front end keeps 5 141
recordings, the MFCC one 5 184 (exclusions of the loader's preprocessing: fewer than 2p
non-silent frames).

```{dataset-figure} birdset_xcm/series_temporal
```

```{dataset-figure} birdset_xcm/series_spectral
```

```{dataset-figure} birdset_xcm/series_spectro_temporal
```

Measured on the real data (BERTHA, 2026-11-19, default parameters of each mode,
`describe()`): number of samples, tokens per series, observations per token among the
unmasked tokens, fraction of SPD (unmasked) tokens and condition number κ of the unmasked
tokens.

| Mode | Samples | S (max / median) | p | n_t (median, min) | n_t / p | SPD tokens | κ (5 / 50 / 95 %) |
|---|---|---|---|---|---|---|---|
| `temporal` | 5 184 | 9 / 9 | 20 | 62 (42) | 3.1 | 97.82 % | 2.4e+02 / 5.6e+03 / 5.9e+05 |
| `spectral` | 5 141 | 7 / 7 | 16 | 309 (129) | 19.3 | 90.22 % | 1.1e+01 / 2.6e+02 / 2.2e+03 |
| `spectro_temporal` | 5 141 | 16 / 16 | 16 | 123 (51) | 7.7 | 83.37 % | 1.3e+01 / 2.1e+02 / 2.7e+03 |

Every default can be overridden by name:

```python
from spdnet_datasets.series import CovarianceSeries, collate_series

series = CovarianceSeries("birdset_xcm")
series = CovarianceSeries("birdset_xcm", "temporal", window=124, step=62)  # 2 s windows
(tokens, mask), label = series[0]
```
<!-- /covariance-series -->

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/birdset_xcm.md
```
````
