"""Micro-Doppler radar datasets: CI4R (77 GHz, xethru), MVDoppler and DopNet.

Each module provides ``fetch_<dataset>()`` (staged cache, unscaled
covariances) and a ``<Dataset>Dataset`` registered in
:class:`~spdnet_datasets.manager.DatasetManager`.
"""

from .ci4r import CI4RDataset, fetch_ci4r
from .dopnet import DopNetDataset, fetch_dopnet
from .mvdoppler import MVDopplerDataset, fetch_mvdoppler

__all__ = ["CI4RDataset", "DopNetDataset", "MVDopplerDataset", "fetch_ci4r", "fetch_dopnet",
           "fetch_mvdoppler"]
