# spdnet-datasets

Dataset loaders, covariance estimators and synthetic data generators for SPDNet research.

Documentation: build it with `pip install -e ".[docs]" && make -C doc html` (published on GitHub Pages at each release).

## Overview

Loaders that turn heterogeneous measurements into symmetric positive definite
(SPD) covariance matrices for SPDNet-style networks, with the raw data and
channel means available for other models (CNN, Transformer, ViT). Every dataset
comes with a standalone `fetch_<name>()` (download or local check, staged
cache), leakage-free splits and a documentation page.

| Domain | Datasets |
|---|---|
| Audio | ESC-50, GTZAN, BirdSet_XCM |
| Skeleton | HDM05, NTU RGB+D 120 |
| Micro-Doppler radar | CI4R (77 GHz, xethru), MVDoppler, DopNet |
| Satellite image time series (SITS) | PASTIS-R (scenarios A–E) |
| Hyperspectral (HSI) | GSOFF, Chikusei, UAV-HSI-Crop, Wheat disease, Placenta, Rices_90, HyperLeaf2024, DeepHS-Fruit |
| Synthetic aperture radar (SAR) | Pol-InSAR-Island, CPAZMaL, S1SLC_CVDL, AIR-PolSAR-Seg, ATRNet-STAR, FUSAR-Ship 1.0, MSTAR |

Branches: `timeseries` (series of covariances for Transformers), `local`
(datasets not published yet: S1 Iceland SAR, AVIRIS), `moabb` (adapter for
MOABB EEG datasets).

## Installation

```bash
pip install "spdnet-datasets @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets"
pip install "spdnet-datasets[robust,audio,sar] @ git+..."   # Tyler/Student estimators, audio, SAR readers
```

Development: `pip install -e ".[dev,test,docs]"`.

## Quick start

```python
from spdnet_datasets import PastisRDataset, create_dataloaders

ds = PastisRDataset(scenario="C", estimator="ledoit_wolf", return_mean=True)
train, val, test = create_dataloaders(ds, split="official", batch_size=64)
(cov, mean), y = next(iter(train))      # cov: (64, 16, 16) SPD, mean: (64, 16)
```

- Datasets are read from `$SPDNET_DATA_ROOT/<DOMAIN>/<Name>` (default
  `/DATA/DATASET`, never written to); caches go to `$SPDNET_CACHE_DIR`
  (default `~/.cache/spdnet_datasets`).
- `estimator` is `"scm"`, `"ledoit_wolf"`, `"tyler"` or `"student"`; every
  covariance needs `n >= 2p` observations and is checked to be SPD.
- `task="fine" | "coarse" | "regression"`, `drop_classes`, `merge_classes` and
  `balance` select the task; named splits (`dataset.splits`) never separate a
  subject, patch, block or site.

The configuration-dictionary entry point is kept:
`DatasetManager.create_dataloaders({"name": "pastis_r", "scenario": "A", "split_strategy": "official", ...})`.

## Package structure

```
src/spdnet_datasets/
├── core/                 config, estimators, normalize, outputs, tasks, splits, fetch
├── audio/ skeleton/ radar_microdoppler/ sits/ hyperspectral/ sar/
│                         one module per dataset: fetch_<name>() + <Name>Dataset
├── base.py, manager.py   BaseDataset, create_dataloaders, DatasetManager
├── real/                 deprecated aliases of the former loaders
├── synthetic/            synthetic SPD datasets (extra "synthetic")
└── estimator/            former estimator classes
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{spdnet_datasets,
  title = {spdnet-datasets: Dataset loaders and generators for SPDNet research},
  author = {Gallet, Matthieu and Mian, Ammar},
  year = {2026},
  url = {https://github.com/Yet-Another-Research-Organisation/spdnet-datasets}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Support

For issues and questions, please use the GitHub issue tracker:
https://github.com/Yet-Another-Research-Organisation/spdnet-datasets/issues
