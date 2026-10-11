"""Shared fixtures for spdnet-datasets tests.

Real-dataset tests use the actual data under DATA_ROOT (see CLAUDE.md's
`/DATA` convention) and are skipped automatically when a dataset's
directory isn't present, rather than mocking the on-disk layout.
"""

import os
from pathlib import Path

import pytest

DATA_ROOT = Path(os.environ.get("SPDNET_DATA_ROOT", "/DATA"))


def dataset_dir(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"dataset not available at {path} (set SPDNET_DATA_ROOT)")
