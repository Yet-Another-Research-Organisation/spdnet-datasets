"""
SPDNet Datasets - Dataset loaders, covariance estimators, and synthetic data generators.

This package provides:
- Real dataset loaders for SPD matrix classification tasks
- Synthetic data generators for simulation experiments
- Covariance estimation utilities
"""

from .base import BaseDataset, create_dataloaders
from .estimator import EstimateCovariance, EstimateCovarianceTorch
from .manager import DatasetManager

# Import real datasets to register them
from .radar_microdoppler import CI4RDataset, DopNetDataset, MVDopplerDataset
from .real import (
    ChikuseiDataset,
    DeepHSFruitDataset,
    GSOffDataset,
    HDM05Dataset,
    HyperLeafDataset,
    KaggleWheatDataset,
    NTU120Dataset,
    PlacentaDataset,
    Rices90Dataset,
    UAVDataset,
)
from .sits import PastisRDataset, fetch_pastis_r

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
    'NTU120Dataset',
    'CI4RDataset',
    'DopNetDataset',
    'MVDopplerDataset',

    # SITS
    'PastisRDataset',
    'fetch_pastis_r',
]
