# Quickstart

A dataset is a PyTorch dataset whose samples are covariances, optionally with
their channel means and the raw data they come from ({doc}`/guide/outputs`):

```python
from spdnet_datasets import PastisRDataset, create_dataloaders

ds = PastisRDataset(scenario="C", estimator="ledoit_wolf", return_mean=True)
train, val, test = create_dataloaders(ds, split="official", batch_size=64)
(cov, mean), y = next(iter(train))      # cov: (64, 16, 16) SPD, mean: (64, 16)
```

`split="official"` selects a leakage-free split provided by the dataset
({doc}`/guide/splits`); every dataset page lists its splits
({doc}`/datasets/index`).

The configuration-dictionary entry point used by the Hydra configs of
`spdnet-training` is kept ({doc}`/guide/loading`):

```python
from spdnet_datasets import DatasetManager

train, val, test, n_classes = DatasetManager.create_dataloaders({
    "name": "pastis_r", "scenario": "A", "split_strategy": "official",
    "batch_size": 64,
})
```

## Next steps

- The {ref}`tutorials` run without any data: covariance estimation, outputs
  and tasks, leakage-free splits, the fetch cache.
- The {ref}`dataset-examples` load one dataset per family.
- The {doc}`/guide/principles` page explains what a sample is and why
  `n ≥ 2p`.
