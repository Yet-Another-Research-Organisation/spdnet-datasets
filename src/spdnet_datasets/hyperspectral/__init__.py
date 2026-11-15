"""Hyperspectral datasets: one module per dataset, ``fetch_<name>()`` + ``<Name>Dataset``."""

from .gsoff import GSOFFDataset, GSOffDataset, fetch_gsoff

__all__ = ["GSOFFDataset", "GSOffDataset", "fetch_gsoff"]
