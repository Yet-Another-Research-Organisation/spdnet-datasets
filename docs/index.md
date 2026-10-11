# spdnet-datasets

**spdnet-datasets** loads the datasets used in SPDNet research as batches of
SPD matrices (covariances) ready for
[yetanotherspdnet](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet),
and generates synthetic SPD datasets for controlled experiments.

```text
spdnet-datasets ──▶ yetanotherspdnet (models) ──▶ spdnet-training (training loop, Hydra configs)
   loaders, splits,       SPDnet, RResNet,             spdnet-train / spdnet-optimize
   synthetic data         batch normalization
```

Every loader is reached through one entry point, a configuration dictionary
that also comes out of the Hydra configs of `spdnet-training`:

```python
from spdnet_datasets import DatasetManager

train, val, test, n_classes = DatasetManager.create_dataloaders({
    "name": "hdm05",              # registered dataset name
    "path": "/DATA", "subdir": "HDM05",
    "scaling_factor": 190.0,      # dataset-specific option
    "batch_size": 16, "val_ratio": 0.15, "test_ratio": 0.15,
    "stratified": True, "seed": 0,
})
x, y = next(iter(train))          # x: (16, 93, 93) SPD matrices, y: (16,) labels
```

## Install

```bash
pip install "spdnet-datasets @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"
# development
git clone https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
cd spdnet-datasets && pip install -e ".[dev,test,docs]"
```

The datasets themselves are not distributed with the package: point `path`
at the directory holding them (for instance `/DATA`); each loader documents
the layout it expects ({doc}`guide/datasets`).

## Contents

| Page | For |
|---|---|
| {doc}`guide/loading` | the configuration keys, how splits are made, what a batch contains |
| {doc}`guide/datasets` | the catalogue of loaders, their options and expected directory layouts |
| {doc}`guide/adding` | registering a new dataset |
| {doc}`guide/synthetic` | synthetic SPD datasets with controlled spectra |
| {doc}`guide/covariance` | covariance estimators and precomputation scripts |
| {doc}`reference/index` | API reference |

```{toctree}
:hidden:
:caption: Guide

guide/loading
guide/datasets
guide/adding
guide/synthetic
guide/covariance
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```
