# User guide

How every loader behaves, whatever the dataset. Read {doc}`principles` first;
the other pages can be read in any order.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`light-bulb` Principles
:link: principles
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

One sample, one covariance: `n ≥ 2p`, SPD checks, normalisation fitted on the
training split, grouped splits.
:::

:::{grid-item-card} {octicon}`stack` Outputs
:link: outputs
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Covariance, channel mean and raw data: what `dataset[i]` returns and how it
is normalised.
:::

:::{grid-item-card} {octicon}`tag` Tasks
:link: tasks
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Fine and coarse classification, regression, class selection, merging and
balancing.
:::

:::{grid-item-card} {octicon}`download` Fetching and caching
:link: fetch
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

`fetch_<name>()`, the staged cache, checksums, credentials and manual
downloads.
:::

:::{grid-item-card} {octicon}`shield-check` Splits and data leakage
:link: splits
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Groups, official and recommended splits, and how leakage is measured.
:::

:::{grid-item-card} {octicon}`gear` Loading a dataset
:link: loading
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

The configuration-dictionary entry point used by the Hydra configs of
`spdnet-training`.
:::

:::{grid-item-card} {octicon}`stack` Covariance series
:link: series
:link-type: doc
:class-card: sd-card-hover
:class-item: tile tile-here

Sequences of SPD tokens per sample: time windows or vision tokens, defaults,
masking, normalisation and how to override the parameters.
:::
::::

```{seealso}
The dataset-specific pages (catalogue, covariance estimation, synthetic data,
adding a dataset, MOABB and nilearn adapters) are in {doc}`/datasets/index`.
```

```{toctree}
:hidden:

principles
outputs
tasks
fetch
splits
loading
series
```
