"""Shared fixtures for spdnet-datasets tests.

Real-dataset tests use the actual data under DATA_ROOT
(``$SPDNET_DATA_ROOT``, default ``/DATA/DATASET``, laid out as
``<DOMAIN>/<Name>``) and are skipped automatically when a dataset's
directory isn't present, rather than mocking the on-disk layout.
"""

from pathlib import Path

import pytest

from spdnet_datasets.core.config import get_data_root

DATA_ROOT = get_data_root()


def dataset_dir(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"dataset not available at {path} (set SPDNET_DATA_ROOT)")
