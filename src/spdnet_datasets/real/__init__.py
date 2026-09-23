"""
Real dataset implementations for SPD matrix classification.
"""

from .chikusei import ChikuseiDataset
from .ci4r import CI4RDataset
from .deephsfruit import DeepHSFruitDataset
from .dopnet import DopNetDataset
from .gsoff import GSOffDataset
from .hdm05 import HDM05Dataset
from .hyperleaf import HyperLeafDataset
from .kaggle_wheat import KaggleWheatDataset
from .mvdoppler import MVDopplerDataset
from .ntu120 import NTU120Dataset
from .placenta import PlacentaDataset
from .rices90 import Rices90Dataset
from .uav import UAVDataset

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
]
