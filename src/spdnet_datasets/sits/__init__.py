"""Satellite image time series (SITS) datasets."""

from .pastis_r import PastisRDataset, fetch_pastis_r

__all__ = ["PastisRDataset", "fetch_pastis_r"]
