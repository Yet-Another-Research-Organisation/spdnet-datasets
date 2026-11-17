# UrbanSound8K

```{warning}
⚠️ **Outside the maintained selection; only available as a covariance-series
function on the `timeseries` branch, not updated.** There is no
`UrbanSound8K` dataset class, no `fetch_*` function and no entry in the
registry of the main package.
```

```{dataset-preview} urbansound8k
```

## Description and source

UrbanSound8K holds 8 732 labelled slices (at most 4 s) of urban sounds from
[Freesound](https://freesound.org), in 10 classes, with predefined folds
(column `fold` of the metadata file).

- Reference: J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy
  for Urban Sound Research", *Proc. ACM (Association for Computing Machinery) Multimedia*, 2014.
- Expected layout: `<data root>/AUDIO/UrbanSound8K/metadata/UrbanSound8K.csv`
  and `audio/fold<k>/<slice>.wav`.

## What exists

On the `timeseries` branch, `spdnet_datasets.real.series_audio.urbansound8k_series`
returns a series of covariances per slice: each slice is resampled to
22.05 kHz, cut into 1 s windows (half-window step, at most `t_max = 6`
windows), and one Ledoit-Wolf covariance of the 20 MFCC (mel-frequency
cepstral coefficients) is estimated per window (n = 86 frames, n / p = 4.3).
Groups are the source recordings (`fsID`), never split by the official folds.
A slice shorter than one window gives a single token over its frames; windows
with fewer than 2p frames become identity tokens (masked).

This code is kept as it is: it does not use the `core` interface of the other
pages (no `split`, `normalize`, `task` or `keep` options, no checks against a
data atlas sheet). For a maintained environmental sound dataset, use
{doc}`esc50`.

## How to cite

````{admonition} If you use this dataset in a publication, cite
:class: tip

```{include} _cite/urbansound8k.md
```
````
