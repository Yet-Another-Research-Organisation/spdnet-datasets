"""ATRNet-STAR on the real data: counts vs the atlas fiche, no train/test reuse."""

import numpy as np
import pytest

from spdnet_datasets.core.config import get_data_root
from spdnet_datasets.core.fetch import md5sum
from spdnet_datasets.core.splits import assert_no_duplicates
from spdnet_datasets.sar.atrnet_star import ATRNetSTARDataset, protocol_overlap

ROOT = get_data_root() / "SAR" / "ATRNet-STAR"
pytestmark = [pytest.mark.real_data, pytest.mark.skipif(not ROOT.exists(), reason="ATRNet-STAR not available")]

# fiche atrnet_star, n_samples (Slant_Range)
SLANT = {"SOC_40classes": (17003, 7243), "EOC_band": (6918, 6935), "EOC_depression": (5557, 18689),
         "EOC_azimuth": (4650, 19596), "EOC_scene": (4896, 19350)}


def _make(**kw):
    return ATRNetSTARDataset(return_raw=True, verbose=False, **kw)


@pytest.mark.parametrize("protocol", sorted(SLANT))
def test_slant_range_counts(protocol):
    ds = _make(split=protocol)
    parts = ds.splits["official"]
    assert (len(parts["train"]) + len(parts["val"]), len(parts["test"])) == SLANT[protocol]
    assert ds.n_empty == {}
    names = ds.meta["path"].map(lambda p: p.rsplit("/", 1)[-1]).to_numpy()
    assert not set(names[parts["train"]]) & set(names[parts["test"]])
    assert protocol_overlap(None, protocol, protocol)[("train", "test")] == 0


def test_soc40_class_counts_and_chip():
    ds = _make(split="SOC_40classes")
    assert len(ds.classes) == 40
    test = ds.meta.iloc[ds.splits["official"]["test"]]
    assert test["label"].value_counts()["Buick_GL8"] == 192 and test["label"].value_counts()["Hongqi_h5"] == 160
    x, _ = ds[0]
    assert x.shape == (128, 128, 4)


def test_amplitude_counts_and_empty_files():
    ds = _make(split="SOC_40classes", representation="Amplitude_8bit")
    parts = ds.splits["official"]
    assert (len(parts["train"]) + len(parts["val"]), len(parts["test"])) == (68091, 29284)
    ds = _make(split="EOC_depression", representation="Amplitude_8bit")
    assert ds.n_empty == {"train": 6987, "test": 74855 - 30085}
    parts = ds.splits["official"]
    assert (len(parts["train"]) + len(parts["val"]), len(parts["test"])) == (21424 - 6987, 30085)
    assert set(ds.meta["condition"].iloc[parts["test"]]) == {"30", "45", "60"}


def test_no_content_shared_between_train_and_test_eoc_band_amplitude():
    ds = _make(split="EOC_band", representation="Amplitude_8bit")
    h = np.array([md5sum(ds.data_dir / p) for p in ds.meta["path"]])
    parts = ds.splits["official"]
    assert_no_duplicates(h[np.r_[parts["train"], parts["val"]]], h[parts["test"]])


def test_cross_protocol_reuse_is_measured():
    """EOC test chips are SOC training chips: the protocols must not be chained."""
    ov = protocol_overlap(None, "SOC_40classes", "EOC_depression")
    assert ov[("train", "test")] == 13112
