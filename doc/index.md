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

A dataset is a PyTorch dataset whose samples are covariances, optionally with
their channel means and the raw data they come from ({doc}`guide/outputs`):

```python
from spdnet_datasets import PastisRDataset, create_dataloaders

ds = PastisRDataset(scenario="C", estimator="ledoit_wolf", return_mean=True)
train, val, test = create_dataloaders(ds, split="official", batch_size=64)
(cov, mean), y = next(iter(train))      # cov: (64, 16, 16) SPD, mean: (64, 16)
```

The configuration-dictionary entry point used by the Hydra configs of
`spdnet-training` is kept ({doc}`guide/loading`):

```python
from spdnet_datasets import DatasetManager

train, val, test, n_classes = DatasetManager.create_dataloaders({
    "name": "pastis_r", "scenario": "A", "split_strategy": "official",
    "batch_size": 64,
})
```

## Install

```bash
pip install "spdnet-datasets @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"
# development
git clone https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
cd spdnet-datasets && pip install -e ".[dev,test,docs]"
```

The datasets themselves are not distributed with the package: they are read
from `$SPDNET_DATA_ROOT/<DOMAIN>/<Name>` (default `/DATA/DATASET`) or fetched
into `$SPDNET_CACHE_DIR` ({doc}`guide/fetch`).

## Contents

| Page | For |
|---|---|
| {doc}`guide/principles` | what a sample is, the `n ≥ 2p` rule, estimators, package layout |
| {doc}`guide/outputs` | covariances, means and raw data; normalisation |
| {doc}`guide/tasks` | fine / coarse classification, regression, dropping and merging classes, balancing |
| {doc}`guide/fetch` | `fetch_*` functions, cache stages, checksums, the `keep` option |
| {doc}`guide/splits` | leakage-free splits and the checks behind them |
| {doc}`datasets/index` | one page per dataset, with interactive figures |
| {doc}`guide/loading` | the configuration keys, how splits are made, what a batch contains |
| {doc}`guide/datasets` | the catalogue of loaders, their options and expected directory layouts |
| {doc}`guide/adding` | registering a new dataset |
| {doc}`guide/synthetic` | synthetic SPD datasets with controlled spectra |
| {doc}`guide/covariance` | covariance estimators and precomputation scripts |
| {doc}`guide/nilearn` | fMRI connectomes from nilearn datasets (ABIDE, ADHD-200, development) |
| {doc}`reference/index` | API reference |

```{toctree}
:hidden:
:caption: Guide

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
guide/nilearn
```

```{toctree}
:hidden:
:caption: Datasets

datasets/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```
