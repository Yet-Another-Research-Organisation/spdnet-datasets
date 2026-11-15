"""Hyperspectral datasets: one module per dataset, ``fetch_<name>()`` + ``<Name>Dataset``."""

from .aviris import AVIRISDataset, fetch_aviris
from .chikusei import ChikuseiDataset, fetch_chikusei
from .gsoff import GSOFFDataset, GSOffDataset, fetch_gsoff
from .kaggle_wheat import KaggleWheatDataset, fetch_kaggle_wheat
from .placenta import PlacentaDataset, fetch_placenta
from .rices90 import Rices90Dataset, fetch_rices90
from .uav import UAVDataset, UAVHSICropDataset, fetch_uav

__all__ = [
    "AVIRISDataset", "ChikuseiDataset", "GSOFFDataset", "GSOffDataset", "KaggleWheatDataset", "PlacentaDataset", "Rices90Dataset", "UAVDataset", "UAVHSICropDataset",
    "fetch_aviris", "fetch_chikusei", "fetch_gsoff", "fetch_kaggle_wheat", "fetch_placenta", "fetch_rices90", "fetch_uav",
]
