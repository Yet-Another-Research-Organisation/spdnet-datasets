"""S1SLC_CVDL on the real zarr store: counts per city and class vs the atlas fiche."""

import pytest

from spdnet_datasets.core.config import get_data_root

pytest.importorskip("zarr")
from spdnet_datasets.sar.s1slc_cvdl import S1SLCCVDLDataset  # noqa: E402

ROOT = get_data_root() / "SAR" / "S1SLC_CVDL"
pytestmark = [pytest.mark.real_data, pytest.mark.skipif(not ROOT.exists(), reason="S1SLC_CVDL not available")]
CHICAGO = {"1": 42359, "2": 33089, "3": 1247, "4": 191, "5": 11348, "6": 22515, "7": 9}


def test_counts_and_loco():
    ds = S1SLCCVDLDataset(return_raw=True, n_stat_patches=300, verbose=False)
    assert ds.meta["city"].value_counts().to_dict() == {"Chicago": 110758, "Sao Paulo": 84238, "Houston": 81575}
    assert ds.meta[ds.meta["city"] == "Chicago"]["label"].value_counts().to_dict() == CHICAGO
    test = ds.splits["recommended"]["test"]
    assert set(ds.meta["city"][test]) == {"Houston"}
    x, y = ds[int(test[0])]
    assert x.shape == (100, 100, 4) and 0 <= y < 7
    assert set(ds.rms) == {"Houston", "Chicago", "Sao Paulo"}
