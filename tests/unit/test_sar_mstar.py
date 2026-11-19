"""MSTAR SOC loader on fake jpg and Phoenix files."""

import numpy as np
import pytest
from PIL import Image

from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.sar.mstar import MSTARDataset, center_crop, read_phoenix

SIZES = {"2S1": (20, 20), "BMP2": (16, 16), "D7": (23, 22)}


def _phoenix(path, z, dep):
    nr, nc = z.shape
    head = (f"[PhoenixHeaderVer01.04]\nPhoenixHeaderLength= 00512\nNumberOfColumns= {nc}\n"
            f"NumberOfRows= {nr}\nTargetSerNum= b01\nMeasuredDepression= {dep}\n[EndofPhoenixHeader]\n")
    data = np.concatenate([np.abs(z).ravel(), np.angle(z).ravel()]).astype(">f4").tobytes()
    path.write_bytes(head.encode().ljust(512, b" ") + data)


@pytest.fixture
def mstar_dir(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "MSTAR"
    k = 0
    for part, n, dep in (("train", 6, 17.1), ("test", 4, 15.0)):
        for cls, (h, w) in SIZES.items():
            (root / "soc" / part / cls).mkdir(parents=True)
            (root / "soc_complex" / part / cls).mkdir(parents=True)
            for _ in range(n):
                k += 1
                Image.fromarray(rng.integers(0, 256, (h, w), dtype=np.uint8)).save(
                    root / "soc" / part / cls / f"HB{k:05d}.jpg")
                z = rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))
                _phoenix(root / "soc_complex" / part / cls / f"HB{k:05d}.0{k % 20:02d}", z, dep)
    return root


def test_phoenix_reader(tmp_path):
    z = np.arange(12).reshape(3, 4) * np.exp(1j * 0.5)
    _phoenix(tmp_path / "x.000", z, 17.0)
    out, info = read_phoenix(tmp_path / "x.000")
    np.testing.assert_allclose(out, z, rtol=1e-6, atol=1e-6)
    assert info["serial"] == "b01"


def test_raw_mean_only_and_crop(mstar_dir):
    with pytest.raises(ValueError, match="exactly one"):
        MSTARDataset(mstar_dir, verbose=False)
    ds = MSTARDataset(mstar_dir, crop=16, return_raw=True, verbose=False)
    assert len(ds) == 30 and ds.classes == sorted(SIZES)
    assert all(ds[i][0].shape == (16, 16, 1) for i in range(len(ds)))
    ds = MSTARDataset(mstar_dir, source="complex", crop=12, return_raw=True, verbose=False)
    assert ds[0][0].shape == (12, 12, 2)
    ds = MSTARDataset(mstar_dir, source="amplitude", crop=12, return_mean=True, verbose=False)
    assert ds[0][0].shape == (1,)
    ds = MSTARDataset(mstar_dir, crop=17, return_raw=True, verbose=False)
    with pytest.raises(ValueError, match="crop"):
        ds[int((ds.meta["label"] == "BMP2").idxmax())]  # 16 x 16 chips
    assert center_crop(np.zeros((23, 22)), 16).shape == (16, 16)


def test_official_split_by_depression(mstar_dir):
    ds = MSTARDataset(mstar_dir, crop=16, return_raw=True, verbose=False)
    s = ds.splits["official"]
    assert set(ds.meta["part"][s["test"]]) == {"test"} and len(s["test"]) == 12
    assert set(ds.meta["part"][np.r_[s["train"], s["val"]]]) == {"train"}
    assert_no_leakage(ds.meta["chip_id"][s["train"]], ds.meta["chip_id"][s["test"]])


def test_no_groups_and_stratified_validation(mstar_dir):
    """The file stem is not a physical unit and the vehicle is the class: no groups."""
    ds = MSTARDataset(mstar_dir, crop=16, return_raw=True, verbose=False, val_size=0.5)
    assert ds.groups is None
    s = ds.splits["official"]
    # stratified draw in the 17 deg chips: 3 of the 6 training chips of each class
    assert ds.meta["label"][s["val"]].value_counts().to_dict() == dict.fromkeys(ds.classes, 3)
