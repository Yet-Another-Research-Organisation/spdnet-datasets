"""
SPDNet Datasets - Dataset loaders, covariance estimators, and synthetic data generators.

This package provides:
- Real dataset loaders for SPD matrix classification tasks
- Synthetic data generators for simulation experiments
- Covariance estimation utilities
"""

from .base import BaseDataset, create_dataloaders
from .manager import DatasetManager
from .estimator import EstimateCovariance, EstimateCovarianceTorch

# Import real datasets to register them
from .real import (
    Rices90Dataset,
    HyperLeafDataset,
    HDM05Dataset,
    UAVDataset,
    GSOffDataset,
    ChikuseiDataset,
    DeepHSFruitDataset,
    PlacentaDataset,
    KaggleWheatDataset,
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
    'ChikuseiDataset',
    'DeepHSFruitDataset',
    'PlacentaDataset',
    'KaggleWheatDataset',
]
