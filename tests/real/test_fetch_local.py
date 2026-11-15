"""Local copies of the datasets against their ``MANIFEST_NAS.tsv`` and the download registries."""

import importlib
from pathlib import Path

import pytest

from spdnet_datasets.core import dataset_path
from spdnet_datasets.core.fetch import sha256sum, verify_local
from tests.unit.test_fetch_registry import MODULES

# Copies without MANIFEST_NAS.tsv on BERTHA (2026-11-15): verify_local cannot check them.
NO_MANIFEST = {"HYPERSPECTRAL/UAV-HSI-Crop", "SAR/PAZ_S1_proc_v2", "SAR/AIR-PolSAR-Seg"}


def _root(name):
    mod = importlib.import_module(f"spdnet_datasets.{name}")
    root = dataset_path(mod.DOMAIN, mod.NAME)
    if not root.is_dir():
        pytest.skip(f"dataset not available at {root}")
    return mod, root


@pytest.mark.real_data
@pytest.mark.parametrize("name", MODULES)
def test_local_copy_matches_manifest(name):
    mod, root = _root(name)
    key = f"{mod.DOMAIN}/{mod.NAME}"
    if key in NO_MANIFEST:
        assert not (root / "MANIFEST_NAS.tsv").exists(), "manifest added: drop it from NO_MANIFEST"
        pytest.skip(f"{key} has no MANIFEST_NAS.tsv")
    report = verify_local(root, verify="size")
    assert report.n_checked > 0
    assert report.missing == [] and report.wrong_size == []


@pytest.mark.real_data
@pytest.mark.slow
@pytest.mark.parametrize("name", ["audio.esc50", "audio.gtzan", "sar.cpazmal"])
def test_registry_hashes_equal_local_copy(name):
    """The registered sha256 (Hugging Face LFS oid) are those of the local files."""
    mod, root = _root(name)
    for url, sha in mod.FILES.items():
        local = {"audio.esc50": "esc50_{}.parquet", "audio.gtzan": "gtzan_{}.parquet"}
        if name in local:  # shard k of the Hugging Face export = local file k
            k = int(Path(url).name.split("-")[1])
            path = root / local[name].format(k)
        else:
            path = root / mod.FILENAME
        assert sha256sum(path) == sha, path
