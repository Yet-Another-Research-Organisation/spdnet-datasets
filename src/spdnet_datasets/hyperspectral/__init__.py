"""Hyperspectral datasets: one module per dataset, ``fetch_<name>()`` + ``<Name>Dataset``."""

from .chikusei import ChikuseiDataset, fetch_chikusei
from .gsoff import GSOFFDataset, GSOffDataset, fetch_gsoff

__all__ = ["ChikuseiDataset", "fetch_chikusei", "GSOFFDataset", "GSOffDataset", "fetch_gsoff"]
