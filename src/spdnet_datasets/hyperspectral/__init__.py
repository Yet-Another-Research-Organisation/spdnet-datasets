"""Hyperspectral datasets: one module per dataset, ``fetch_<name>()`` + ``<Name>Dataset``."""

from .aviris import AVIRISDataset, fetch_aviris
from .chikusei import ChikuseiDataset, fetch_chikusei
from .gsoff import GSOFFDataset, GSOffDataset, fetch_gsoff
from .kaggle_wheat import KaggleWheatDataset, fetch_kaggle_wheat
from .uav import UAVDataset, UAVHSICropDataset, fetch_uav

__all__ = [
    "AVIRISDataset", "ChikuseiDataset", "GSOFFDataset", "GSOffDataset", "KaggleWheatDataset", "UAVDataset", "UAVHSICropDataset",
    "fetch_aviris", "fetch_chikusei", "fetch_gsoff", "fetch_kaggle_wheat", "fetch_uav",
]
