"""
Real dataset implementations for SPD matrix classification.
"""

# Hyperspectral loaders moved to spdnet_datasets.hyperspectral (the real.<name>
# modules are deprecated aliases); imported from there to avoid their warning.
from ..hyperspectral import (
    ChikuseiDataset,
    DeepHSFruitDataset,
    GSOffDataset,
    HyperLeafDataset,
    KaggleWheatDataset,
    PlacentaDataset,
    Rices90Dataset,
    UAVDataset,
)
from .ci4r import CI4RDataset
from .dopnet import DopNetDataset
from .fusar import FUSARShipDataset
from .hdm05 import HDM05Dataset
from .mvdoppler import MVDopplerDataset
from .ntu120 import NTU120Dataset

__all__ = [
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
    'FUSARShipDataset',
]
