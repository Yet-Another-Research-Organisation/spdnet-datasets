"""
SPDNet Datasets - Dataset loaders, covariance estimators, and synthetic data generators.

This package provides:
- Real dataset loaders for SPD matrix classification tasks
- Synthetic data generators for simulation experiments
- Covariance estimation utilities
"""

from .base import BaseDataset, create_dataloaders
from .estimator import EstimateCovariance, EstimateCovarianceTorch

# Hyperspectral datasets (registered on import)
from .hyperspectral import (
    AVIRISDataset,
    ChikuseiDataset,
    DeepHSFruitDataset,
    GSOFFDataset,
    GSOffDataset,
    HyperLeafDataset,
    KaggleWheatDataset,
    PlacentaDataset,
    Rices90Dataset,
    UAVDataset,
)
from .manager import DatasetManager

# Other real datasets (registered on import)
from .real import (
    CI4RDataset,
    DopNetDataset,
    HDM05Dataset,
    MVDopplerDataset,
    NTU120Dataset,
)

__version__ = "0.1.0"

__all__ = [
    # Core infrastructure
    'BaseDataset',
    'DatasetManager',
    'create_dataloaders',

    # Estimators
    'EstimateCovariance',
    'EstimateCovarianceTorch',

    # Real datasets
    'Rices90Dataset',
    'HyperLeafDataset',
    'HDM05Dataset',
    'UAVDataset',
    'GSOffDataset',
    'GSOFFDataset',
    'AVIRISDataset',
    'ChikuseiDataset',
    'DeepHSFruitDataset',
    'PlacentaDataset',
    'KaggleWheatDataset',
    'NTU120Dataset',
    'CI4RDataset',
    'DopNetDataset',
    'MVDopplerDataset',
]
