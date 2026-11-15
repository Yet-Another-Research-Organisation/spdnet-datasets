"""Pol-InSAR-Island on the real data: cell counts vs the atlas fiche."""

import numpy as np
import pytest

from spdnet_datasets.core.config import get_data_root
from spdnet_datasets.core.estimators import check_spd
from spdnet_datasets.sar.pol_insar_island import CLASSES, PolInSARIslandDataset

ROOT = get_data_root() / "SAR" / "Pol-InSAR-Island"
pytestmark = [pytest.mark.real_data, pytest.mark.skipif(not ROOT.exists(), reason="Pol-InSAR-Island not available")]
# fiche pol_insar_island: class_distribution.benchmark_cellules_union (same class order as CLASSES)
UNION = {"tidal flat": 397, "water": 908, "coastal shrub": 151, "dense high vegetation": 62,
         "white dune": 337, "peat bog": 53, "grey dune": 960, "couch grass": 388,
         "upper saltmarsh": 433, "lower saltmarsh": 279, "sand": 773, "settlement": 292}


def test_union_cells_logpower():
    ds = PolInSARIslandDataset(verbose=False)
    assert len(ds) == 5033 and len(np.unique(ds.groups)) == 52
    counts = ds.meta["label"].value_counts().to_dict()
    assert counts == UNION and set(counts) == set(CLASSES)
    assert ds.covariances.shape[1:] == (12, 12)
    check_spd(ds.covariances)
    assert ds.meta["n_pixels"].min() >= 512
    for name, parts in ds.splits.items():
        if name.startswith("spatial"):
            assert not set(ds.groups[parts["train"]]) & set(ds.groups[parts["test"]])
    t = ds.splits["recommended"]
    assert 800 < len(t["test"]) < 1300


def test_official_cells():
    with pytest.warns(UserWarning, match="optimistic"):
        ds = PolInSARIslandDataset(split="official", verbose=False)
    parts = ds.splits["official"]
    assert (len(parts["train"]), len(parts["test"])) == (3315, 3403)


def test_t6_representation():
    ds = PolInSARIslandDataset(representation="t6", verbose=False)
    assert ds.covariances.shape == (5033, 24, 24)
    check_spd(ds.covariances)
