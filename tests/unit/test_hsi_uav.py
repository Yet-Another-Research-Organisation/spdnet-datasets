"""UAV-HSI-Crop on a tiny synthetic source tree (official patch layout)."""

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_window_overlap
from spdnet_datasets.hyperspectral._common import window_boxes
from spdnet_datasets.hyperspectral.uav import UAVDataset, fetch_uav, list_patches


@pytest.fixture
def uav_dir(tmp_path):
    root = tmp_path / "UAV-HSI-Crop"
    rng = np.random.default_rng(0)
    # MJK_N: 2 x 2 patches; MJK_S: 1 x 2 patches; k is NOT the row index
    layout = {"MJK_N": [(7, 0, 0, "Train/Training"), (3, 0, 1, "Test"),
                        (9, 1, 0, "Train/Validation"), (1, 1, 1, "Train/Training")],
              "MJK_S": [(5, 0, 0, "Train/Training"), (2, 0, 1, "Test")]}
    for scene, patches in layout.items():
        for k, r, c, sub in patches:
            for kind in ("rs", "gt"):
                (root / "source" / sub / kind).mkdir(parents=True, exist_ok=True)
            rs = (rng.random((96, 96, 200)) + 0.1).astype(np.float32)
            gt = np.full((96, 96), 1 + (r + c) % 2, np.float32)
            name = f"{scene}_patch_{k}_{r}_{c}.npy"
            np.save(root / "source" / sub / "rs" / name, rs)
            np.save(root / "source" / sub / "gt" / name, gt)
    return root


def test_patches_are_placed_by_row_and_column(uav_dir):
    p = list_patches(uav_dir / "source")
    assert len(p) == 6 and set(p.part) == {"train", "val", "test"}
    assert p.set_index(["scene", "row", "col"]).loc[("MJK_N", 0, 1), "k"] == 3


def test_windows_splits_and_raw(uav_dir, tmp_path):
    b = fetch_uav(uav_dir, cache_dir=tmp_path / "c", scenes=("MJK_N", "MJK_S"), block=48)
    assert b.covariances.shape[1:] == (100, 100)
    assert set(b.labels) == {"code 1", "code 2"}
    sp = b.splits["recommended"]
    boxes = window_boxes(b.positions, 15, scene=b.scene)
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_window_overlap(boxes[sp[a]], boxes[sp[c]])
    off = b.splits["official"]
    assert (b.meta["official"].to_numpy()[off["test"]] == "test").all()
    assert len(b.splits["scene_transfer"]["test"]) == (b.meta["scene"] == "MJK_S").sum()
    ds = UAVDataset(uav_dir, cache_dir=tmp_path / "c", scenes=("MJK_N", "MJK_S"), block=48,
                    verbose=False, return_raw=True)
    (cov, raw), _ = ds[0]
    r, c = int(ds.meta.row[0]), int(ds.meta.col[0])
    assert raw.shape == (15, 15, 100) and cov.shape == (100, 100)
    # first MJK_N window comes from the (row, col) patch, not from patch k
    row0 = ds.meta.iloc[0]
    assert row0.scene == "MJK_N" and (r // 96, c // 96) in {(0, 0), (0, 1), (1, 0), (1, 1)}
