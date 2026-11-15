"""GSOFF on the real scene (skipped when the dataset folder is absent)."""

import collections

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_window_overlap, verify_local
from spdnet_datasets.hyperspectral._common import purity_grid_windows, window_boxes
from spdnet_datasets.hyperspectral.gsoff import GROUND_TRUTH, fetch_gsoff
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "GSOFF")


def test_manifest_is_checked():
    skip_if_missing(ROOT)
    report = verify_local(ROOT)
    assert report.n_checked > 1000 and not report.missing and not report.wrong_size


def test_16px_extraction_reproduces_the_prepared_windows():
    """Atlas fiche: 1106 windows of 16 x 16, counts per code in gaofeng_metadata.json."""
    skip_if_missing(ROOT)
    tifffile = pytest.importorskip("tifffile")
    gt = tifffile.imread(ROOT / GROUND_TRUTH)
    assert gt.shape == (572, 906)
    pos, lab, _ = purity_grid_windows(gt, 16, 0.8)
    fiche = {1: 309, 2: 37, 3: 40, 4: 79, 5: 204, 6: 82, 7: 27, 8: 63, 9: 71, 10: 108,
             11: 82, 12: 4}
    assert dict(collections.Counter(lab.tolist())) == fiche
    ref = np.load(ROOT / "gaofeng_windows_positions.npy")
    assert set(map(tuple, ref.tolist())) == set(map(tuple, pos.tolist()))


def test_15px_windows_and_spatial_split():
    skip_if_missing(ROOT)
    b = fetch_gsoff()
    assert b.covariances.shape == (1249, 63, 63)
    assert b.info["excluded_n_lt_2p"] == 0 and len(set(b.labels)) == 12
    sp = b.splits["recommended"]
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (739, 250, 255)
    boxes = window_boxes(b.positions, 15)
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_window_overlap(boxes[sp[a]], boxes[sp[c]])
