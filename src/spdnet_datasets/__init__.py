"""
SPDNet Datasets - Dataset loaders, covariance estimators, and synthetic data generators.

This package provides:
- Real dataset loaders for SPD matrix classification tasks
- Synthetic data generators for simulation experiments
- Covariance estimation utilities
"""

# Domain modules: importing them registers their classes in DatasetManager
from .audio import BirdSetXCMDataset, ESC50Dataset, GTZANDataset
from .base import BaseDataset, create_dataloaders
from .estimator import EstimateCovariance, EstimateCovarianceTorch
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
from .radar_microdoppler import CI4RDataset, DopNetDataset, MVDopplerDataset

# Deprecated loader kept for its precomputed cov/ layout (superseded by NTURGBD120Dataset)
from .real import NTU120Dataset
from .sar import (
    AIRPolSARSegDataset,
    ATRNetSTARDataset,
    CPAZMaLDataset,
    FUSARShipDataset,
    MSTARDataset,
    PolInSARIslandDataset,
    S1SLCCVDLDataset,
)
from .sar.s1_iceland import S1IcelandDataset, fetch_s1_iceland
from .sits import PastisRDataset, fetch_pastis_r
from .skeleton import HDM05Dataset, NTURGBD120Dataset

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
    'ESC50Dataset',
    'GTZANDataset',
    'BirdSetXCMDataset',
    'NTURGBD120Dataset',

    # SITS
    'PastisRDataset',
    'fetch_pastis_r',

    # SAR datasets
    'AIRPolSARSegDataset',
    'ATRNetSTARDataset',
    'CPAZMaLDataset',
    'FUSARShipDataset',
    'MSTARDataset',
    'PolInSARIslandDataset',
    'S1SLCCVDLDataset',
    'S1IcelandDataset',
    'fetch_s1_iceland',
]
