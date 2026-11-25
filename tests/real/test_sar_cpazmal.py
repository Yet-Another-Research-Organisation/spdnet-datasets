"""CPAZMaL on the real HDF5 file: parcels per class vs the atlas fiche."""

import pytest

from spdnet_datasets.core.config import get_data_root
from spdnet_datasets.core.estimators import check_spd
from spdnet_datasets.sar.cpazmal import CPAZMaLDataset

ROOT = get_data_root() / "SAR" / "PAZ_S1_proc_v2"
# sites per class in the fiche (paz_s1_proc_v2, class_distribution.sites)
FICHE_SITES = {"ABL": 7, "ACC": 9, "FOR": 9, "HAG": 7, "ICA": 9, "LAC": 9, "PLA": 9, "ROC": 9}

pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not (ROOT / "CPAZMaL.hdf5").exists(), reason="CPAZMaL not available")]


@pytest.mark.parametrize("orbit, missing", [("DSC", {"PLA005", "PLA006"}), ("ASC", {"PLA006"})])
def test_parcels_per_class(orbit, missing):
    ds = CPAZMaLDataset(orbit=orbit, verbose=False)
    sites = ds.meta.groupby("class")["site"].nunique().to_dict()
    expected = dict(FICHE_SITES, PLA=FICHE_SITES["PLA"] - len(missing))
    assert sites == expected
    assert missing <= set(ds.excluded) and {"ALLGLA", "ARGTOP"} <= set(ds.excluded)
    assert ds.covariances.shape == (sum(expected.values()), 40, 40)
    check_spd(ds.covariances)
    assert ds.meta["n_pixels"].min() >= 80
    for name, parts in ds.splits.items():
        if name == "random":  # sample-level split, leaky by design
            continue
        assert not set(ds.groups[parts["train"]]) & set(ds.groups[parts["test"]])
    assert len(set(ds.groups[ds.splits["recommended"]["test"]])) >= 10
