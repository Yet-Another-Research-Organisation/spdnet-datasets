"""Locations of the datasets and of the preprocessing cache.

No absolute path is hard-coded in the dataset modules: they all resolve their
directories through these functions, which read environment variables.

- ``SPDNET_DATA_ROOT``: root of the read-only dataset copies, laid out as
  ``<root>/<DOMAIN>/<Name>`` (default ``/DATA/DATASET``, the LISTIC convention).
- ``SPDNET_CACHE_DIR``: writable directory for downloads, preprocessed tensors
  and covariances (default ``~/.cache/spdnet_datasets``).
"""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = "/DATA/DATASET"
DEFAULT_CACHE_DIR = "~/.cache/spdnet_datasets"


def get_data_root() -> Path:
    """Return the dataset root (``$SPDNET_DATA_ROOT`` or ``/DATA/DATASET``)."""
    return Path(os.environ.get("SPDNET_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()


def get_cache_dir() -> Path:
    """Return the cache directory (``$SPDNET_CACHE_DIR`` or ``~/.cache/spdnet_datasets``)."""
    return Path(os.environ.get("SPDNET_CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser()


def dataset_path(domain: str, name: str, data_dir=None) -> Path:
    """Directory of one dataset: ``data_dir`` if given, else ``<root>/<domain>/<name>``."""
    if data_dir is not None:
        return Path(data_dir).expanduser()
    return get_data_root() / domain / name


def existing_dir(root, cache_dir=None) -> Path:
    """``root`` if it exists, else the cache directory (created).

    :class:`~spdnet_datasets.base.BaseDataset` needs an existing folder; without
    a local copy, the dataset classes give it the cache, where their fetch
    downloads (or finds) the sources, then set ``data_dir`` to ``bunch.root``.
    """
    root = Path(root)
    if root.exists():
        return root
    cache = Path(cache_dir).expanduser() if cache_dir else get_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    return cache
