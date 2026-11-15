"""FUSAR-Ship on the real data: counts and vessel groups vs the atlas fiche."""

import pytest

from spdnet_datasets.core.config import get_data_root
from spdnet_datasets.sar.fusar_ship import FUSARShipDataset

ROOT = get_data_root() / "SAR" / "FUSAR_Ship1.0"
pytestmark = [pytest.mark.real_data, pytest.mark.skipif(not ROOT.exists(), reason="FUSAR-Ship not available")]


def test_counts_groups_and_folds():
    ds = FUSARShipDataset(return_mean=True, verbose=False)
    assert len(ds) == 5198 and len(ds.classes) == 10  # fiche: categories >= 30 chips
    counts = ds.meta["label"].value_counts()
    assert counts["Cargo"] == 2114 and counts["LawEnforce"] == 35
    for parts in ds.splits.values():
        assert not set(ds.groups[parts["train"]]) & set(ds.groups[parts["test"]])
        assert 900 < len(parts["test"]) < 1200
