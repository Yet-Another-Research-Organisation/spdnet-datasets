# Covariance series

A **covariance series** replaces the single covariance of a sample by a sequence of
`S` covariances, the **tokens**. Each token is the `p × p` covariance of `n_t`
observations taken from one part of the sample: a time window, a frequency sub-band,
a spatial window, a body part. A model then reads the sample as an `S × p × p`
sequence of symmetric positive definite (SPD) matrices.

{class}`~spdnet_datasets.series.CovarianceSeries` builds these series on top of the
loaders of the package. The wrapped loader still does the fetch, the class
selection, the labels, the groups and the named splits, so a series has exactly the
samples and the leakage-free splits of its loader.

```python
from spdnet_datasets.series import CovarianceSeries, collate_series

series = CovarianceSeries("esc50")                 # default mode: 1 s windows of 20 MFCC
(tokens, mask), label = series[0]                  # (9, 20, 20) float64, (9,) bool
series.describe()                                  # S, p, n_t, n_t / p, SPD fraction, kappa
```

## Two families of tokens

The modes belong to one of two families. The registry records the family of every
mode (`SeriesMode.family`), and the tables never mix them.

- **Time series of covariances** (`family="temporal"`). The tokens follow the time
  axis: windows of audio frames, of radar spectrogram columns, of skeleton frames,
  dates of a satellite time series. Token `k + 1` comes after token `k`.
- **Vision tokens** (`family="vision"`). The tokens are windows of **one**
  measurement, as the patches of a vision transformer: spatial and/or spectral
  windows of a hyperspectral cube, sub-bands of a whole recording, joints of a
  skeleton. Their order is a convention (row-major, spatial-major), not time.

## Token types

Each diagram below is drawn from the registry for a typical sample. On the left is
the sample with its axes and the token windows, coloured in series order (for a
product mode, one hue per outer window and lighter steps for the inner windows);
token 1 is filled. In the middle is one token: `n_t` observations of `p` variables
and their covariance. On the right is the resulting `S × p × p` series. Every
dataset page shows the diagrams of its own modes.

### Temporal windows

Windows along the time axis; the variables are the channels of each frame
(MFCC, Doppler bins, possibly averaged into fewer channels with `n_channels`).

```{dataset-figure} esc50/series_temporal
```

### Spectral windows

Windows of consecutive bands or frequency channels; the observations are every
pixel (or every frame) of the sample.

```{dataset-figure} gsoff/series_spectral
```

### Spatial windows

A grid of sub-windows of an image; the variables are the bands, possibly pooled into
fewer channels.

```{dataset-figure} gsoff/series_spatial
```

### Spatio-spectral windows

The Cartesian product of spatial windows and band windows, ordered spatial-major:
every band window of the first spatial window, then those of the second, and so on.

```{dataset-figure} hyperleaf/series_spatio_spectral
```

### Spectro-temporal windows

For a spectrogram: time windows × frequency sub-bands, ordered time-major.

```{dataset-figure} ci4r/series_spectro_temporal
```

### Groups of variables × time

Body parts (groups of joints) in each time window, ordered window-major.

```{dataset-figure} ntu120/series_part_temporal
```

### Dates and periods

For a satellite image time series: one token per date or per period; the
observations are the pixels of the parcel.

```{dataset-figure} pastis_r/series_s2_dates
```

## How windows are placed

The tokenizers ({mod}`spdnet_datasets.series.tokenizers`) are pure functions of an
array. Along one axis of length `n`:

- `window` and `step`: a window every `step` positions (default `step = window // 2`,
  50 % overlap). When the regular windows leave the end of the axis uncovered, a last
  window flush with the end is added (`cover=True`). `cover=False` keeps the regular
  windows only, as the former builders of `real/series*.py` did, so that their
  series can be reproduced exactly (CI4R, Dop-NET, the R23 large-p modes).
- `n_windows`: windows spread evenly over the axis. Without `window`, the window is
  `2n / (n_windows + 1)`, so the windows overlap by 50 % and cover the whole axis.
- A `window` or `step` given as a float at most 1 is a fraction of the axis: `0.5`
  cuts an object (a grain, a fruit) into halves whatever its size.
- An axis shorter than one window gives one token over the whole axis (a short audio
  clip, a short gesture).

## Default parameters

The defaults follow rules, not a benchmark:

1. `n_t ≥ 2p` for every token, and `n_t ≥ 4p` when the data allow it;
2. the tokens cover the whole sample, with 0 or 50 % overlap;
3. between 4 and 32 tokens;
4. `p` from the data atlas or from the users' choices when they exist (CI4R, Dop-NET,
   NTU, PASTIS-R and the large-p modes reuse the parameters of `real/series*.py`);
5. a degenerate token becomes a masked identity token (see below).

The table of every mode with its defaults is on {doc}`/datasets/series_overview`.
Every default can be changed: pass the parameter by name.

```python
# HyperLeaf, spatio-spectral (the default): 6 windows of 48 x 58 pixels x 4 windows of 48 bands
series = CovarianceSeries("hyperleaf")
# 4 windows of 51 bands that cover the 204 bands exactly, and 8 spatial windows
series = CovarianceSeries("hyperleaf", "spatio_spectral", band_window=51,
                          n_windows=(1, 8))
# another loader setting: pass loader_kwargs (here the training fold of the loader)
series = CovarianceSeries("gsoff", "spatial", window=(10, 10), step=(5, 5),
                          loader_kwargs={"fold": 2})
```

An unknown parameter raises `ValueError` and lists the parameters of the mode.

## The `n_t ≥ 2p` rule

A sample covariance of `p` variables needs at least `2p` observations to be a stable
SPD estimate (see {doc}`principles`). Before building anything,
`CovarianceSeries` tokenizes a few samples spread over the dataset and raises
`ValueError` when the median size of their smallest token is below `2p`, with the
values that would fix it.

`allow_small_n=True` accepts such tokens and estimates them with Ledoit-Wolf
shrinkage. It is a documented derogation: the large-p modes of the R23 study
(`temporal_mel64` of ESC-50, `temporal_w64` of MVDoppler, `window_p45` and
`window_p75` of NTU) have it by default, and their tables say so.

## Masked tokens

A token is **masked** when it has fewer than `2p` valid observations (image
background, a short window), when its observations are not finite (a PASTIS-R
period without acquisition) or when its estimate is not SPD (constant channels,
condition number above `1e12`). A masked token is the identity matrix and its
`mask` entry is `False`. Nothing is regularised silently: `describe()` reports the
fraction of SPD tokens, and the dataset pages give it for the real data.

The series of a dataset can have different lengths (audio clips, Dop-NET gestures).
`series[i]` returns the tokens of sample `i` only; {func}`~spdnet_datasets.series.collate_series`
pads a batch with masked identity tokens:

```python
from torch.utils.data import DataLoader

loader = DataLoader(series, batch_size=32, collate_fn=collate_series)
(tokens, mask), labels = next(iter(loader))     # (32, S_max, p, p), (32, S_max)
```

## Normalisation and standardisation

As for one covariance per sample (see {doc}`outputs`), a geometric scale `s` is
fitted on the **training tokens** of the chosen split (masked tokens left out) so
that the geometric mean of their eigenvalues is 1. The tokens are multiplied by `s`
and the token means (`return_mean=True`) by `√s`. The cache stores unscaled tokens:
changing the split does not recompute them.

A few modes also standardise each variable before tokenizing, with means and standard
deviations fitted on the training samples of the split (`SeriesMode.standardize`):
the PASTIS-R fusion of reflectances and decibels, the Pol-InSAR log-powers, and the
NTU coordinates of the `part`, `jointcov` and `axis` modes.

## Splits and leakage

`series.splits` and `series.groups` are those of the loader, so
{func}`~spdnet_datasets.base.create_dataloaders` works unchanged:

```python
from spdnet_datasets import create_dataloaders

series = CovarianceSeries("ci4r", split="recommended")
train, val, test = create_dataloaders(series, split="recommended", batch_size=16)
test_only = CovarianceSeries("ci4r", subset="test")   # one part of the split
```

Built by name, the loader estimates its own (unused) covariance with Ledoit-Wolf
(`estimator="ledoit_wolf"`, changed through `loader_kwargs`): a sample is never lost because
its single `p × p` covariance is singular, since every token is checked on its own. The
exclusions of the loader's preprocessing are inherited: silent or empty recordings, files
without data, and sequences shorter than `2p` frames or pixels for the loader's own `p`
(the dataset pages count them).

The loader must expose every sample (`subset=None`); pass `subset` to
`CovarianceSeries` instead. A loader instance can also be wrapped:
`CovarianceSeries(ESC50Dataset(features="logmel"), "spectral")`. The mode then
checks that the instance has the settings it needs (here `features="logmel"`).

## Cache

The tokens are computed once and stored, unscaled, under
`<SPDNET_CACHE_DIR>/series/<name>/covariance/<hash>/` through
{func}`~spdnet_datasets.core.fetch.run_pipeline`. The hash covers the mode, its
parameters, the estimator, the standardisation statistics and the identity of the
loader samples (its cached stage, the selected samples and their labels), so a
different class selection or loader setting gives a new cache entry.

## Where the samples come from

A mode reads each sample with its axes through the `structured_raw(i)` hook of the
loader (`T × F` spectrogram, `H × W × B` window or image, `T × J × 3` skeleton). The
loaders whose `raw` output had lost the layout got a hook that reads it again
without changing their outputs: HyperLeaf (whole image instead of the leaf pixels),
NTU (skeleton), Rices_90 and DeepHS-Fruit (object box), Pol-InSAR-Island (cell
grid), PASTIS-R (`s2_dates`, `s1s2_periods`) and Dop-NET (`doppler_bands`: the 128
central Doppler bins of `real.series.dopnet_series`). HDM05 has no trajectories:
the local and ETH copies hold covariances only, and its hook raises a clear error.

The SAR loaders without covariance get real variables, never a complex covariance:
log-amplitudes of the channels and the absolute derivatives of their mean (S1SLC_CVDL,
ATRNet-STAR, AIR-PolSAR-Seg), the region-covariance features of Tuzel et al. on the
complex MSTAR chips, and every dual-polarisation date of a CPAZMaL block
(`CPAZMaLDataset.date_stack`, one orbit or both). The adapters have hooks too:
`MOABBCovariances.structured_raw` (epoch `T × C`, with `sfreq` and `passband`) and
`NilearnConnectomes.structured_raw` (scan `T × regions`). Built from objects, they are
passed as instances; a mode parameter listed in `from_loader` (the sampling rate of the
filter bank) is then read from the instance.

### Samples the loader drops

A loader drops a sample whose own `p × p` covariance cannot be estimated: fewer than `2p`
non-silent frames (audio), frames with a body (NTU), frames (Dop-NET) or pixels (Rices_90
grains). That rule is global, while a token has its own `n_t`: a clip of 100 log-mel frames
has no `64 × 64` covariance, but its tokens of 16 bands do. Built by name, these loaders
store their short samples (`min_frames=1`, `min_pixels=1`), and `CovarianceSeries` wraps
the copy their `series_samples()` hook returns: every stored sample, with task, class
selection and splits rebuilt by the loader's rules on that list (groups kept, so no
leakage). Silent or empty samples stay excluded and listed (`loader.excluded`).
`recover=False` keeps the loader's samples only; `describe()["n_loader"]` gives their
number.

The former builders of `spdnet_datasets.real.series*` stay available for the
datasets outside the loaders (finance, UEA, PTB-XL, AVIRIS, TreeSat, BreizhCrops,
TenGeoP, FMCW, UrbanSound8K); their docstrings point to this API for the others.
