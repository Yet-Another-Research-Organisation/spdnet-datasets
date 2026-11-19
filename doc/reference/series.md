# Covariance series

See {doc}`/guide/series` for the concepts and {doc}`/datasets/series_overview` for the
modes of every dataset.

## Dataset

```{eval-rst}
.. automodule:: spdnet_datasets.series.dataset
   :members: CovarianceSeries, collate_series
```

## Modes

```{eval-rst}
.. automodule:: spdnet_datasets.series.modes
   :members: SeriesMode, get_mode, FAMILIES

.. data:: SERIES_MODES

   ``{dataset: {mode: SeriesMode}}``; the first mode of a dataset is its default.
```

## Tokenizers

```{eval-rst}
.. automodule:: spdnet_datasets.series.tokenizers
   :members:
```
