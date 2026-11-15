"""AIR-PolSAR-Seg on the real data: mosaic and cells vs the atlas fiche."""

import pytest

from spdnet_datasets.core.config import get_data_root
from spdnet_datasets.sar.air_polsar_seg import AIRPolSARSegDataset

ROOT = get_data_root() / "SAR" / "AIR-PolSAR-Seg"
pytestmark = [pytest.mark.real_data, pytest.mark.slow,
              pytest.mark.skipif(not ROOT.exists(), reason="AIR-PolSAR-Seg not available")]
# fiche air_polsar_seg: benchmark_cellules_32x32
CELLS = {"housing": 23634, "industrial": 11598, "natural": 16125, "land use": 677, "water": 4116, "other": 1208}


def test_mosaic_cells_and_groups():
    """The atlas found 1241 overlapping pairs and 15 components; the denser probing here
    finds the same 1241 pairs plus 89 exact overlaps (>= 5625 px), merging them into 12
    consistent components, so the cell grid and counts differ slightly (< 0.5 %)."""
    ds = AIRPolSARSegDataset(return_mean=True, verbose=False)
    assert len(ds.overlaps) == 1330
    assert ds.positions["component"].nunique() == 12
    counts = ds.meta["label"].value_counts().to_dict()
    assert sum(counts.values()) == 57269
    for k, v in CELLS.items():
        assert abs(counts[k] - v) <= 0.02 * v
    assert ds.meta["block"].nunique() == 119
    for parts in ds.splits.values():
        assert not set(ds.groups[parts["train"]]) & set(ds.groups[parts["test"]])
