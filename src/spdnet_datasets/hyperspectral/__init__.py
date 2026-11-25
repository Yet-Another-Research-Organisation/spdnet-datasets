"""Hyperspectral datasets: one module per dataset, ``fetch_<name>()`` + ``<Name>Dataset``."""

from .aviris_ca_ml import AVIRISCAMultilabelDataset, fetch_aviris_ca_ml
from .chikusei import ChikuseiDataset, fetch_chikusei
from .deephsfruit import DeepHSFruitDataset, fetch_deephsfruit
from .gsoff import GSOFFDataset, GSOffDataset, fetch_gsoff
from .hyperleaf import HyperLeafDataset, fetch_hyperleaf
from .hyperpistachio import HyperPistachioDataset, fetch_hyperpistachio
from .kaggle_wheat import KaggleWheatDataset, fetch_kaggle_wheat
from .placenta import PlacentaDataset, fetch_placenta
from .rices90 import Rices90Dataset, fetch_rices90
from .uav import UAVDataset, UAVHSICropDataset, fetch_uav

__all__ = [
    "AVIRISCAMultilabelDataset", "ChikuseiDataset", "DeepHSFruitDataset", "GSOFFDataset", "GSOffDataset",
    "HyperLeafDataset", "HyperPistachioDataset", "KaggleWheatDataset", "PlacentaDataset",
    "Rices90Dataset", "UAVDataset", "UAVHSICropDataset",
    "fetch_aviris_ca_ml", "fetch_chikusei", "fetch_deephsfruit", "fetch_gsoff", "fetch_hyperleaf",
    "fetch_hyperpistachio", "fetch_kaggle_wheat", "fetch_placenta", "fetch_rices90", "fetch_uav",
]
