"""AIR-PolSAR-Seg: mosaic rebuilt from overlapping fake patches, macro-block folds."""

import numpy as np
import pytest
import tifffile
from PIL import Image

from spdnet_datasets.sar.air_polsar_seg import COLOURS, AIRPolSARSegDataset

S = 64
# (folder, k, row, col) in a 200 x 214 scene; component 1 = A, B, C; component 2 = D
PATCHES = [("train_set", 1, 0, 0), ("train_set", 2, 0, 40), ("test_set", 1, 40, 0), ("test_set", 2, 136, 150)]


@pytest.fixture
def air_dir(tmp_path):
    rng = np.random.default_rng(0)
    scene = rng.integers(1, 60000, (4, 200, 214), dtype=np.uint16)
    colours = list(COLOURS)
    lab = np.zeros((200, 214, 3), np.uint8)
    for r in range(0, 200, 16):
        for c in range(0, 214, 16):
            lab[r:r + 16, c:c + 16] = colours[(r // 16 + c // 16) % 3]
    root = tmp_path / "AIR"
    for folder, k, r, c in PATCHES:
        (root / folder).mkdir(parents=True, exist_ok=True)
        for p, pol in enumerate(("HH", "HV", "VH", "VV")):
            tifffile.imwrite(root / folder / f"AIR-PolarSAR-Seg-{k}_{pol}.tiff", scene[p, r:r + S, c:c + S])
        Image.fromarray(lab[r:r + S, c:c + S]).save(root / folder / f"AIR-PolarSAR-Seg-{k}_gt.png")
    return root, scene


def test_mosaic_is_rebuilt_and_cells_are_unique(air_dir, tmp_path):
    root, scene = air_dir
    ds = AIRPolSARSegDataset(root, cache_dir=tmp_path / "c", cell=8, macro=32, return_raw=True, verbose=False)
    pos = ds.positions.set_index("name")
    assert pos.loc["train/AIR-PolarSAR-Seg-1", "component"] == pos.loc["test/AIR-PolarSAR-Seg-1", "component"]
    assert pos.loc["test/AIR-PolarSAR-Seg-2", "component"] != pos.loc["train/AIR-PolarSAR-Seg-1", "component"]
    rel = pos.loc["train/AIR-PolarSAR-Seg-2", ["py", "px"]] - pos.loc["train/AIR-PolarSAR-Seg-1", ["py", "px"]]
    assert tuple(rel) == (0, 40)
    assert len(ds.overlaps) == 3  # A-B, A-C, B-C
    assert ds.label_conflicts == 0
    # union of the first component: 104 x 104 minus the empty corner, plus D (64 x 64): 8 px cells
    n_cells = (13 * 13 - 5 * 5) + 8 * 8  # rows and cols 64..104 are not covered
    assert len(ds) == n_cells
    keys = set(zip(ds.meta["component"], ds.meta["row"], ds.meta["col"], strict=True))
    assert len(keys) == len(ds)  # no duplicated cell
    x, _ = ds[0]
    assert x.shape == (8, 8, 4) and x.dtype.is_floating_point
    with pytest.raises(ValueError, match="exactly one"):
        AIRPolSARSegDataset(root, cache_dir=tmp_path / "c", cell=8, macro=32, verbose=False)


def test_macro_block_folds_with_guard(air_dir, tmp_path):
    root, _ = air_dir
    ds = AIRPolSARSegDataset(root, cache_dir=tmp_path / "c", cell=8, macro=32, return_mean=True, verbose=False)
    assert ds[0][0].shape == (4,)
    xy = ds.meta[["row", "col"]].to_numpy() + ds.meta[["component"]].to_numpy() * 10_000
    for parts in ds.splits.values():
        assert not set(ds.groups[parts["train"]]) & set(ds.groups[parts["test"]])
        if len(parts["train"]) and len(parts["test"]):
            d = np.abs(xy[parts["train"]][:, None] - xy[parts["test"]][None]).max(-1)
            assert d.min() >= 16  # no train cell touches a test cell
