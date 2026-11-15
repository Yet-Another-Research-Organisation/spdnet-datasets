"""Hyperspectral datasets: one module per dataset, ``fetch_<name>()`` + ``<Name>Dataset``."""

from .aviris import AVIRISDataset, fetch_aviris
from .chikusei import ChikuseiDataset, fetch_chikusei
from .gsoff import GSOFFDataset, GSOffDataset, fetch_gsoff

__all__ = [
    "AVIRISDataset", "ChikuseiDataset", "GSOFFDataset", "GSOffDataset",
    "fetch_aviris", "fetch_chikusei", "fetch_gsoff",
]
