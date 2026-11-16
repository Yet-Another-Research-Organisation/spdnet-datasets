"""MSTAR SOC on the real data: counts per class vs the atlas fiche."""

import pytest

from spdnet_datasets.core.config import get_data_root
from spdnet_datasets.sar.mstar import MSTARDataset

ROOT = get_data_root() / "SAR" / "MSTAR_SOC"
pytestmark = [pytest.mark.real_data, pytest.mark.skipif(not ROOT.exists(), reason="MSTAR not available")]
TRAIN = {"2S1": 299, "BMP2": 233, "BRDM2": 298, "BTR60": 256, "BTR70": 233, "D7": 299, "T62": 299,
         "T72": 232, "ZIL131": 299, "ZSU234": 299}
TEST = {"2S1": 274, "BMP2": 195, "BRDM2": 274, "BTR60": 195, "BTR70": 196, "D7": 274, "T62": 273,
        "T72": 196, "ZIL131": 274, "ZSU234": 274}


@pytest.mark.parametrize("source", ["jpg", "complex"])
def test_counts(source):
    ds = MSTARDataset(source=source, return_raw=True, verbose=False)
    m = ds.meta
    assert m[m["part"] == "train"]["label"].value_counts().to_dict() == TRAIN
    assert m[m["part"] == "test"]["label"].value_counts().to_dict() == TEST
    x, _ = ds[0]
    assert x.shape == (128, 128, 1 if source == "jpg" else 2)
