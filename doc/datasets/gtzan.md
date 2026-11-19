# GTZAN

```{list-table}
:stub-columns: 1

* - Domain
  - audio (music genres)
* - Task(s)
  - fine classification (10 genres)
* - Covariance
  - 20×20 covariance of the MFCC (mel-frequency cepstral coefficients) (or 64×64 log-mel), n = number of non-silent frames (2 397 to 2 632, median 2 582; n / 2p ≥ 59)
* - Outputs
  - cov · mean · raw (frames × p)
* - Samples
  - 985 excerpts (999 in the copy, minus 14 exact duplicates)
* - Splits
  - `stratified` = `recommended`: 5 stratified folds (fold 1: 591 / 197 / 197). No official split.
* - Module
  - `spdnet_datasets.audio.gtzan` (`fetch_gtzan`, `GTZANDataset`, registry name `gtzan`)
```

```{warning}
GTZAN has no artist metadata. The stratified folds put excerpts of one artist
(and repeated songs) in train and test, so scores are optimistic.
`GTZANDataset` emits a `UserWarning` saying so.
```

```{dataset-preview} gtzan
```

## Description and source

The GTZAN genre collection holds 10 genres × 100 excerpts of 30 s (blues,
classical, country, disco, hiphop, jazz, metal, pop, reggae, rock), taken from
CDs (compact discs), radio and other recordings.

- Reference: G. Tzanetakis and P. Cook, "Musical genre classification of audio
  signals", *IEEE (Institute of Electrical and Electronics Engineers) Transactions on Speech and Audio Processing* 10(5), 293–302,
  2002, doi:[10.1109/TSA.2002.800560](https://doi.org/10.1109/TSA.2002.800560).
- Known defects: B. L. Sturm, "The GTZAN dataset: its contents, its faults,
  their effects on evaluation, and its future use", 2013,
  [arXiv:1306.1461](https://arxiv.org/abs/1306.1461): repetitions,
  mislabelings and distortions.
- Licence: not clarified (commercial music excerpts); research use.
- Local copy: Hugging Face export, parquet files under
  `<data root>/AUDIO/GTZAN` (columns `file`, `audio`, `genre`).

## Getting the data

Direct access, no account. Without a local copy, `fetch_gtzan()` downloads the three
parquet shards (1.3 GB) of the Hugging Face dataset `sanchit-gandhi/gtzan` at a
pinned revision and checks their sha256; `fetch_gtzan(archive=...)` takes shards
already downloaded (folder, zip or tar). The original server no longer answers.
No explicit licence (research use only).

## Raw data

The copy holds 999 excerpts: `jazz.00054` is missing (a known corrupted
file). Each excerpt is a mono 16-bit WAV (Waveform Audio File Format) file of 660 000 to 675 808 samples. The
Hugging Face metadata announce 32 kHz; the real rate, read from every WAV
header, is 22 050 Hz, and the features use the real rate. The figure shows the
20 MFCC of the non-silent frames of four
excerpts (time axis decimated for display).

```{dataset-figure} gtzan/raw_samples
```

## Classes

After removing duplicates: blues 100, classical 100, country 100, disco 99,
hiphop 98, jazz 99, metal 93, pop 98, reggae 99, rock 99.

```{dataset-figure} gtzan/class_hist
```

```{dataset-figure} gtzan/class_means
```

## Covariance construction

Same chain as {doc}`esc50`: Hann frames of 1 024 samples (about 46 ms at
22.05 kHz) with a hop of 256, 64 mel bands from 50 Hz to Nyquist in dB with a
floor at `max − 80 dB`, then the first 20 coefficients of the orthonormal
DCT-II (discrete cosine transform), so p = 20 (`features="logmel"`: p = 64).
Frames more than half made of exact zeros are dropped. One symmetric positive
definite (SPD) covariance per excerpt over its frames (n ≈ 2 580), with the
chosen estimator (`"scm"`, the sample covariance matrix, by default). No
excerpt is excluded by the n ≥ 2p check or the SPD check.

**Duplicates.** The 14 pairs of identical signals measured by the atlas are
removed (the second file of each pair is dropped): `pop.00060`, `pop.00071`,
`metal.00094`, `metal.00063`, `disco.00099`, `metal.00064`, `metal.00061`,
`reggae.00082`, `rock.00016`, `metal.00065`, `metal.00066`, `hiphop.00045`,
`metal.00062`, `hiphop.00078`. One pair crosses classes: `metal.00058` =
`rock.00016` (the metal copy is kept). The list is `gtzan.DUPLICATES`.

With `normalize=True` (default), covariances are multiplied by the scalar
fitted on the training part (geometric mean of the eigenvalues = 1).

```{dataset-figure} gtzan/cov_samples
```

```{dataset-figure} gtzan/eigen_spectrum
```

## Splits

GTZAN has no official split. The literature recommends the "fault-filtered"
artist split of Kereliuk, Sturm and Larsen (2015, 443 / 197 / 290 excerpts),
but the copy carries no artist information, so it cannot be rebuilt.
`split="stratified"` (alias `"recommended"`): 5 folds stratified by genre
(`StratifiedKFold`, seed 0); test = fold `fold`, validation = the next fold,
train = the three others. With `fold=1`: 591 / 197 / 197.

Groups are the excerpts themselves (one group per excerpt): the right panel of
the figure is therefore trivial, and the folds do not protect against artist
or song leakage.

```{dataset-figure} gtzan/split_schema
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| exact duplicates (14 pairs, one across classes) | second copy removed | none for exact copies |
| same artist / repeated songs in train and test | none possible (no artist metadata); `UserWarning` | **high**: scores optimistic |
| near-duplicates and distortions (Sturm 2013) | not detected | documented only |

## Supported tasks

- `task="fine"`: 10 genres. No coarse grouping is defined.
- `drop_classes`, `merge_classes`, `classes`, `max_classes` and
  `balance="undersample" | "oversample" | "weights"` (training part only).

## Usage

```python
from spdnet_datasets.audio.gtzan import GTZANDataset

train = GTZANDataset(split="stratified", subset="train", fold=1)   # warns about artists
test = GTZANDataset(split="stratified", subset="test", fold=1)
```

## Migration notes

- `scaling_factor` is replaced by `normalize=True`.
- Duplicates are now removed (985 excerpts instead of 999).

<!-- covariance-series: generated section, edit write_sections.py -->
## Covariance series

{class}`~spdnet_datasets.series.CovarianceSeries` cuts each clip into a series of
covariance tokens (see {doc}`/guide/series` and {doc}`series_overview`). The modes and
their defaults come from the registry; the first one is the default.

```{series-table} gtzan
```

The spectral modes and `temporal_mel64` need the 64 log-mel bands (loader built with
`features="logmel"`), with which the loader keeps 885 of the 985 MFCC clips. `temporal_mel64`
(R23) has `n_t = 258 ≥ 2p`: no derogation.

```{dataset-figure} gtzan/series_temporal
```

```{dataset-figure} gtzan/series_spectral
```

```{dataset-figure} gtzan/series_spectro_temporal
```

```{dataset-figure} gtzan/series_temporal_mel64
```

Measured on the real data (BERTHA, 2026-11-19, default parameters of each mode,
`describe()`): number of samples, tokens per series, observations per token among the
unmasked tokens, fraction of SPD (unmasked) tokens and condition number κ of the unmasked
tokens.

| Mode | Samples | S (max / median) | p | n_t (median, min) | n_t / p | SPD tokens | κ (5 / 50 / 95 %) |
|---|---|---|---|---|---|---|---|
| `temporal` | 985 | 20 / 20 | 20 | 258 (258) | 12.9 | 99.92 % | 1.3e+02 / 5.2e+02 / 5.9e+03 |
| `spectral` | 885 | 7 / 7 | 16 | 2 582 (2 397) | 161.4 | 100.00 % | 5.0e+01 / 2.2e+02 / 1.4e+03 |
| `spectro_temporal` | 885 | 16 / 16 | 16 | 1 032 (958) | 64.5 | 99.27 % | 5.0e+01 / 2.3e+02 / 2.0e+03 |
| `temporal_mel64` | 885 | 20 / 20 | 64 | 258 (258) | 4.0 | 100.00 % | 2.8e+02 / 1.2e+03 / 4.0e+03 |

Every default can be overridden by name:

```python
from spdnet_datasets.series import CovarianceSeries, collate_series

series = CovarianceSeries("gtzan")
series = CovarianceSeries("gtzan", "temporal", window=129, step=129)  # 1.5 s windows, no overlap
(tokens, mask), label = series[0]
```
<!-- /covariance-series -->

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/gtzan.md
```
````
