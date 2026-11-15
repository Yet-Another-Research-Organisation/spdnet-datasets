"""UAV-HSI-Crop rebuilt from source/ (skipped when absent)."""

import pytest

from spdnet_datasets.core import assert_no_window_overlap
from spdnet_datasets.hyperspectral._common import window_boxes
from spdnet_datasets.hyperspectral.uav import fetch_uav, list_patches
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "UAV-HSI-Crop")


def test_source_grids_are_complete():
    """Fiche: 433 patches, grids XJM 9 x 17, MJK_N 13 x 10, MJK_S 15 x 10."""
    skip_if_missing(ROOT / "source")
    p = list_patches(ROOT / "source")
    assert len(p) == 433
    grids = {s: (g.row.max() + 1, g.col.max() + 1, len(g)) for s, g in p.groupby("scene")}
    assert grids == {"XJM": (9, 17, 153), "MJK_N": (13, 10, 130), "MJK_S": (15, 10, 150)}
    assert dict(p.part.value_counts()) == {"train": 312, "test": 87, "val": 34}


def test_windows_and_spatial_split():
    skip_if_missing(ROOT / "source")
    b = fetch_uav()
    assert b.covariances.shape[1:] == (100, 100)
    assert {s: v["shape"] for s, v in b.info["scenes"].items()} == {
        "XJM": [864, 1632], "MJK_N": [1248, 960], "MJK_S": [1440, 960]}
    sp = b.splits["recommended"]
    boxes = window_boxes(b.positions, 15, scene=b.scene)
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_window_overlap(boxes[sp[a]], boxes[sp[c]])
