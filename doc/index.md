# spdnet-datasets

**spdnet-datasets** loads the datasets used in SPDNet research as batches of
symmetric positive definite (SPD) matrices, mostly covariance matrices, ready for
[yetanotherspdnet](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet),
and generates synthetic SPD datasets for controlled experiments.

```text
spdnet-datasets ──▶ yetanotherspdnet (models) ──▶ spdnet-training (training loop, Hydra configs)
   loaders, splits,       SPDnet, RResNet,             spdnet-train / spdnet-optimize
   synthetic data         batch normalization
```

```python
from spdnet_datasets import PastisRDataset, create_dataloaders

ds = PastisRDataset(scenario="C", estimator="ledoit_wolf", return_mean=True)
train, val, test = create_dataloaders(ds, split="official", batch_size=64)
(cov, mean), y = next(iter(train))      # cov: (64, 16, 16) SPD, mean: (64, 16)
```

::::{grid} 1 2 3 3
:gutter: 3
:class-container: landing-cards

:::{grid-item-card} Getting started
:link: getting_started/installation
:link-type: doc

Install the package, point it to the data and load a first dataset
(installation, quickstart).
:::

:::{grid-item-card} User guide
:link: guide/principles
:link-type: doc

What a sample is, outputs and tasks, the fetch cache, leakage-free splits,
covariance estimators, adding a dataset.
:::

:::{grid-item-card} Datasets
:link: datasets/index
:link-type: doc

One page per dataset (audio, radar, SITS, hyperspectral, SAR, skeleton) with
interactive figures, splits and leakage risks.
:::

:::{grid-item-card} Examples
:link: auto_examples/index
:link-type: doc

A gallery of scripts: tutorials that run without any data, and one example
per dataset family. Download them as `.py` or `.ipynb`.
:::

:::{grid-item-card} API reference
:link: reference/index
:link-type: doc

Dataset classes, the `core` contract (estimators, outputs, tasks, splits,
fetch), the manager and the synthetic generators.
:::

:::{grid-item-card} External datasets
:link: guide/moabb
:link-type: doc

MOABB (EEG): covariances from the MOABB datasets, paradigms and evaluations.
:::

::::

```{toctree}
:hidden:
:caption: Getting started

getting_started/installation
getting_started/quickstart
```

```{toctree}
:hidden:
:caption: User guide

guide/principles
guide/outputs
guide/tasks
guide/fetch
guide/splits
guide/loading
guide/datasets
guide/adding
guide/synthetic
guide/covariance
guide/moabb
```

```{toctree}
:hidden:
:caption: Datasets

datasets/index
```

```{toctree}
:hidden:
:caption: Examples

auto_examples/index
```

```{toctree}
:hidden:
:caption: API

reference/index
```
