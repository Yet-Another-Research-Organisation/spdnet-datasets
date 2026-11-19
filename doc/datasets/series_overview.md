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

The statistics of every mode on the full datasets (fraction of SPD tokens, condition
numbers, `n_t / p`) are on the dataset pages, section *Covariance series*. Datasets
without a mode yet (CPAZMaL, S1SLC_CVDL, ATRNet-STAR, AIR-PolSAR-Seg, FUSAR-Ship,
MSTAR, MOABB, nilearn) wait for a choice of tokens. HDM05 has modes but no data to
build them from: its copies hold covariances only.
