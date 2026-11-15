"""Preparations run after a download, checked against the local copies (read only).

The sources are read from ``SPDNET_DATA_ROOT``; the outputs go to ``tmp_path``.
"""

import collections
import json

import h5py
import numpy as np
import pytest

from spdnet_datasets.hyperspectral import chikusei as CHK
from spdnet_datasets.utils import placenta_preprocess as PP
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = [pytest.mark.real_data, pytest.mark.slow]


def test_chikusei_prepare_v2_reproduces_the_local_windows(tmp_path):
    """About 50 s and 1.5 GB of memory."""
    root = dataset_dir("HYPERSPECTRAL", "Chikusei")
    skip_if_missing(root / CHK.PREPARED)
    summary = json.loads((root / "prepared_v2/summary_v2.json").read_text())
    assert tuple(summary["block_assignment"]) == CHK.BLOCK_SPLIT
    assert (summary["block_size"], summary["n_blocks_cols"]) == (CHK.BLOCK, CHK.BLOCK_COLS)
    out = CHK.prepare_v2(root / "source", tmp_path / "chikusei_v2.h5", sizes=(15,))
    with h5py.File(out) as a, h5py.File(root / CHK.PREPARED) as b:
        for key in ("centers", "labels", "split", "block", "purity", "n_valid"):
            assert np.array_equal(a[f"w15/{key}"][:], b[f"w15/{key}"][:]), key
        idx = np.random.default_rng(0).choice(len(a["w15/labels"]), 100, replace=False)
        for i in sorted(idx):
            assert np.array_equal(a["w15/windows"][i], b["w15/windows"][i])


def test_placenta_build_windows_reproduces_the_local_imagettes(tmp_path):
    root = dataset_dir("HYPERSPECTRAL", "Hyperspectral_Placenta_Dataset")
    folder = next(iter(PP.DYE_FOLDERS))
    skip_if_missing(root / folder)
    ref_meta = json.loads((root / "imagettes/placenta_windows_metadata.json").read_text())
    ref = np.load(root / "imagettes/placenta_windows_data.npy", mmap_mode="r")
    PP.build_windows(root, tmp_path, folders={folder: PP.DYE_FOLDERS[folder]})
    meta = json.loads((tmp_path / "placenta_windows_metadata.json").read_text())
    data = np.load(tmp_path / "placenta_windows_data.npy", mmap_mode="r")
    n = len(meta)
    assert n > 20_000 and meta == ref_meta[:n]
    for i in range(0, n, 997):
        assert np.array_equal(data[i], ref[i])
    counts = collections.Counter(r["sample_id"] for r in ref_meta)
    assert sum(counts.values()) == 77_203 and len(counts) == 81
