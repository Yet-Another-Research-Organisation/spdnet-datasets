# Covariance series: overview

Every covariance-series mode of {class}`~spdnet_datasets.series.CovarianceSeries`, read
from the registry {data}`~spdnet_datasets.series.SERIES_MODES` when the documentation
is built, so this page always matches the code. `S`, `p` and `n_t` are those of a
typical sample (column *Sample*, every observation valid); `n_t` is the smallest token.
The first mode of a dataset is its default. How the series are built, and how to
change any parameter: {doc}`/guide/series`.

The two families are kept apart (see {doc}`/guide/series`): **time series of
covariances**, whose tokens follow time, and **vision tokens**, windows of one
measurement.

## Time series of covariances

```{series-table}
:family: temporal
```

## Vision tokens

```{series-table}
:family: vision
```

## Measured on the real data

The statistics of every mode on the real data (fraction of SPD tokens, condition
numbers, `n_t / p`, samples kept) are on the dataset pages, section *Covariance series*.
MOABB and nilearn are adapters: their series wrap an instance (`MOABBCovariances`,
`NilearnConnectomes`), see {doc}`moabb` and {doc}`nilearn`. FUSAR-Ship has no series, by
decision. HDM05 has modes but no data to build them from: its copies hold covariances only.

Some loaders drop, while preprocessing, the samples too short for their own `p × p`
covariance (fewer than `2p` frames or pixels). A token has its own `n_t`, so the series keep
them: built by name, these loaders store their short samples and
`CovarianceSeries` wraps the copy returned by their `series_samples()` hook (audio, NTU
RGB+D 120, Dop-NET, Rices_90), with splits rebuilt by the loader's rules on the longer list
(groups kept). Silent or empty samples stay excluded and counted.
