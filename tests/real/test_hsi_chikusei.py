"""Chikusei v2 on the real data (skipped when the dataset folder is absent)."""

import collections

import h5py
import numpy as np
import pytest

from spdnet_datasets.core import verify_local
from spdnet_datasets.hyperspectral.chikusei import (
    PREPARED,
    SOURCE,
    fetch_chikusei,
    read_scene_crop,
)
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "Chikusei")


def test_manifest_is_checked():
    skip_if_missing(ROOT)
    report = verify_local(ROOT)
    assert report.n_checked > 0 and not report.missing and not report.wrong_size


def test_prepared_windows_are_in_the_source_frame():
    """A window of prepared_v2 equals the transposed source scene (not the v1 frame)."""
    skip_if_missing(ROOT / SOURCE)
    with h5py.File(ROOT / PREPARED) as f:
        r, c = f["w15/centers"][100]
        win = f["w15/windows"][100]
    crop = read_scene_crop(ROOT / SOURCE, r - 7, r + 8, c - 7, c + 8)
    assert crop.shape == win.shape
    np.testing.assert_allclose(crop, win, atol=1)


def test_counts_match_the_atlas_fiche():
    """Fiche (15 x 15): train 2421, val 789, test 989, buffer 645."""
    skip_if_missing(ROOT)
    b = fetch_chikusei()
    sp = b.splits["recommended"]
    assert [len(sp[k]) for k in ("train", "val", "test")] == [2421, 789, 989]
    assert b.info["excluded_buffer"] == 645 and b.info["excluded_n_lt_2p"] == 0
    assert b.covariances.shape[1:] == (64, 64)
    train = collections.Counter(b.labels[sp["train"]])
    assert train["Forest"] == 539 and train["Manmade (dark)"] == 340
