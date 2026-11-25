"""AIR-PolSAR-Seg on the real data: mosaic and cells vs the atlas fiche."""

import hashlib

import numpy as np
import pytest

from spdnet_datasets.core.config import get_data_root
from spdnet_datasets.core.splits import assert_no_duplicates
from spdnet_datasets.sar.air_polsar_seg import AIRPolSARSegDataset

ROOT = get_data_root() / "SAR" / "AIR-PolSAR-Seg"
pytestmark = [pytest.mark.real_data, pytest.mark.slow,
              pytest.mark.skipif(not ROOT.exists(), reason="AIR-PolSAR-Seg not available")]
# fiche air_polsar_seg: benchmark_cellules_32x32
CELLS = {"housing": 23634, "industrial": 11598, "natural": 16125, "land use": 677, "water": 4116, "other": 1208}


def test_mosaic_cells_and_groups():
    """The atlas found 1241 overlapping pairs and 15 components; the grid probing here
    finds the same 1241 pairs plus 92 exact overlaps (>= 5250 px), merging them into 11
    consistent components, so the cell grid and counts differ slightly (< 2 % per class)."""
    ds = AIRPolSARSegDataset(return_mean=True, verbose=False)
    assert len(ds.overlaps) == 1333
    assert ds.positions["component"].nunique() == 11
    counts = ds.meta["label"].value_counts().to_dict()
    assert sum(counts.values()) == 57267
    for k, v in CELLS.items():
        assert abs(counts[k] - v) <= 0.02 * v
    assert ds.meta["block"].nunique() == 123
    for name, parts in ds.splits.items():
        if name == "random":  # sample-level split, leaky by design
            continue
        assert not set(ds.groups[parts["train"]]) & set(ds.groups[parts["test"]])
    assert ds.label_conflicts == 0
    # leakage: no cell content shared between train and test
    cells = ds._cells
    h = np.array([hashlib.md5(np.asarray(cells[i]).tobytes()).hexdigest() for i in ds._keep])
    rec = ds.splits["recommended"]
    assert_no_duplicates(h[np.r_[rec["train"], rec["val"]]], h[rec["test"]])
