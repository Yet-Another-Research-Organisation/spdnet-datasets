"""
Real dataset implementations for SPD matrix classification.
"""

from .rices90 import Rices90Dataset
from .hyperleaf import HyperLeafDataset
from .hdm05 import HDM05Dataset
from .uav import UAVDataset
from .gsoff import GSOffDataset
from .chikusei import ChikuseiDataset
from .deephsfruit import DeepHSFruitDataset
from .placenta import PlacentaDataset
from .kaggle_wheat import KaggleWheatDataset

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
]
