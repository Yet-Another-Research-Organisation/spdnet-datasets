# spdnet-datasets

A public shared infrastructure package for SPDNet research, providing dataset loaders, covariance estimators, and synthetic data generators.

## Overview

This package extracts core dataset and data generation functionality from the monolithic `spdnet-benchmarks` codebase, making it available as a reusable library for the SPDNet research community.

## Features

### Real Dataset Loaders

Pre-built loaders for hyperspectral and motion capture datasets with SPD matrices:

- **Rices90**: Vietnamese rice varieties (90 classes, 256×256 covariance matrices)
- **HyperLeaf**: Hyperspectral barley leaf imagery (4 cultivars, 3 fertilizer levels)
- **HDM05**: Human motion capture dataset
- **UAV-HSI-Crop**: UAV hyperspectral crop classification
- **GSOFF**: Tree species classification from airborne hyperspectral
- **Chikusei**: Land cover classification (19 classes, 128 spectral bands)
- **DeepHS-Fruit**: Fruit classification and ripeness assessment
- **Placenta**: Hyperspectral tissue classification
- **Kaggle Wheat**: Wheat disease classification

### Synthetic Data Generators

Tools for generating synthetic SPD matrix datasets for controlled experiments:

- **ScaleMatrixGeneratorDiagonal**: Diagonal eigenvalue control with random rotation
- **ScaleMatrixGeneratorBlock**: Block-structured SPD matrices
- **WishartGenerator**: Wishart/Inverse-Wishart distributions with random SPD scales

### Covariance Estimation

Utilities for computing covariance matrices from hyperspectral images:

- Sample Covariance Matrix (SCM)
- Ledoit-Wolf shrinkage estimator
- PyTorch-compatible estimators for GPU acceleration

## Installation

```bash
pip install spdnet-datasets
```

Or for development:

```bash
git clone https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
cd spdnet-datasets
pip install -e ".[dev,test]"
```

## Quick Start

### Loading a Real Dataset

```python
from spdnet_datasets import DatasetManager

# Configure dataset
config = {
    'name': 'rices90',
    'path': '/path/to/data',
    'max_classes': 10,
    'seed': 42
}

# Create dataloaders
train_loader, val_loader, test_loader, num_classes = \
    DatasetManager.create_dataloaders(config)
```

### Generating Synthetic Data

```python
from spdnet_datasets.synthetic import ScaleMatrixGeneratorDiagonal

# Create generator
generator = ScaleMatrixGeneratorDiagonal(matrix_size=16, n_classes=3, seed=42)

# Generate data
data, labels, info = generator.generate_data(
    n_samples_per_class=120,
    max_value=100.0,
    conditioning=100.0,
    mode='geomspace'
)
```

### Computing Covariance Matrices

```python
from spdnet_datasets.estimator import EstimateCovariance
import numpy as np

# Initialize estimator
estimator = EstimateCovariance(method='scm', remove_mean=True)

# Compute covariance from hyperspectral image
image = np.random.rand(100, 100, 128)  # H x W x C
cov_matrix = estimator.from_image(image)  # C x C
```

## Package Structure

```
spdnet-datasets/
├── src/spdnet_datasets/
│   ├── base.py              # Base dataset class
│   ├── manager.py           # Dataset manager
│   ├── real/                # Real dataset loaders
│   ├── synthetic/           # Synthetic data generators
│   ├── estimator/           # Covariance estimators
│   └── utils/               # Preprocessing utilities
└── tests/                   # Test suite
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
