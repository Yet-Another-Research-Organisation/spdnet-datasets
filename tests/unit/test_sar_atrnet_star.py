"""ATRNet-STAR loader on a tiny fake folder tree."""

import numpy as np
import pytest
import tifffile
from scipy.io import savemat

from spdnet_datasets.core.fetch import md5sum
from spdnet_datasets.core.splits import assert_no_duplicates, assert_no_leakage
from spdnet_datasets.sar.atrnet_star import (
    ATRNetSTARDataset,
    fetch_atrnet_star,
    parse_name,
    protocol_overlap,
)

CLASSES = ("Buick_GL8", "Chery_qq3")


def _chip(path, rep, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rep == "Slant_Range":
        z = lambda: (rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))  # noqa: E731
        savemat(path, {f"data_{p}": z() for p in ("hh", "hv", "vh", "vv")})
    else:
        tifffile.imwrite(path, rng.integers(0, 256, (8, 8), dtype=np.uint8))


@pytest.fixture
def atr_dir(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "ATR"
    ids = iter(range(10_000, 99_999))
    pool = {c: [f"X_quad_15_{az}_{next(ids)}" for az in range(0, 360, 30)] for c in CLASSES}
    for c, names in pool.items():
        for i, n in enumerate(names):  # SOC: 8 train, 4 test
            part = "train" if i < 8 else "test"
            _chip(root / f"Slant_Range/SOC_40classes/{part}/{c}/{n}.mat", "Slant_Range", rng)
        for i, n in enumerate(names):  # EOC_depression reuses the same chips
            dep = ("15", "30", "45")[i % 3]
            name = n.replace("_15_", f"_{dep}_")
            if dep == "15":
                _chip(root / f"Slant_Range/EOC_depression/train/{c}/{name}.mat", "Slant_Range", rng)
            else:
                for folder in ("test", f"test_{dep}"):
                    _chip(root / f"Slant_Range/EOC_depression/{folder}/{c}/{name}.mat", "Slant_Range", rng)
        # amplitude EOC_depression: empty test_<cond>, non-empty copies in test/, one empty train chip
        for i in range(6):
            name = f"X_HH_{(15, 30, 45)[i % 3]}_{i * 10}_{next(ids)}.tif"
            dep = name.split("_")[2]
            if dep == "15":
                _chip(root / f"Amplitude_8bit/EOC_depression/train/{c}/{name}", "Amplitude_8bit", rng)
            else:
                _chip(root / f"Amplitude_8bit/EOC_depression/test/{c}/{name}", "Amplitude_8bit", rng)
                empty = root / f"Amplitude_8bit/EOC_depression/test_{dep}/{c}/{name}"
                empty.parent.mkdir(parents=True, exist_ok=True)
                empty.touch()
        bad = root / f"Amplitude_8bit/EOC_depression/train/{c}/X_HH_15_5_{next(ids)}.tif"
        bad.touch()
    return root


def _make(root, tmp_path, **kw):
    kw.setdefault("return_raw", True)
    return ATRNetSTARDataset(root, cache_dir=tmp_path / "cache", verbose=False, **kw)


def test_raw_or_mean_only(atr_dir, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        _make(atr_dir, tmp_path, return_raw=False)
    with pytest.raises(ValueError, match="exactly one"):
        _make(atr_dir, tmp_path, return_raw=True, return_mean=True)
    ds = _make(atr_dir, tmp_path)
    x, y = ds[0]
    assert x.shape == (8, 8, 4) and x.dtype.is_floating_point and (x >= 0).all()
    ds = _make(atr_dir, tmp_path, slant_features="complex")
    assert ds[0][0].shape == (8, 8, 8)
    ds = _make(atr_dir, tmp_path, return_raw=False, return_mean=True)
    assert ds[0][0].shape == (4,)


def test_official_split_has_no_shared_name_or_content(atr_dir, tmp_path):
    ds = _make(atr_dir, tmp_path)
    assert len(ds) == 24 and ds.classes == list(CLASSES)
    parts = ds.splits["official"]
    assert len(parts["test"]) == 8 and len(parts["train"]) + len(parts["val"]) == 16
    names = ds.meta["path"].map(lambda p: p.split("/")[-1]).to_numpy()
    assert_no_leakage(names[parts["train"]], names[parts["test"]])
    hashes = np.array([md5sum(atr_dir / p) for p in ds.meta["path"]])
    assert_no_duplicates(hashes[parts["train"]], hashes[parts["test"]])
    assert protocol_overlap(atr_dir, "SOC_40classes", "SOC_40classes",
                            cache_dir=tmp_path / "cache")[("train", "test")] == 0


def test_cross_protocol_reuse_is_reported(atr_dir, tmp_path):
    ov = protocol_overlap(atr_dir, "SOC_40classes", "EOC_depression", cache_dir=tmp_path / "c")
    # the 15 deg chips of EOC_depression are SOC chips (same names): reuse is counted
    assert ov == {("train", "train"): 6, ("train", "test"): 0, ("test", "train"): 2, ("test", "test"): 0}
    src = next((atr_dir / "Slant_Range/EOC_depression/test_30").rglob("*.mat"))
    dst = atr_dir / "Slant_Range/SOC_40classes/train" / src.parent.name / src.name
    dst.write_bytes(src.read_bytes())
    ov = protocol_overlap(atr_dir, "SOC_40classes", "EOC_depression", cache_dir=tmp_path / "c2")
    assert ov[("train", "test")] == 1


def test_eoc_conditions_and_recommended(atr_dir, tmp_path):
    ds = _make(atr_dir, tmp_path, split="EOC_depression")
    assert set(ds.meta["condition"]) == {"train", "30", "45"}
    assert {"official_30", "official_45"} <= set(ds.splits)
    rec = ds.splits["recommended"]
    assert set(ds.meta["depression"].to_numpy()[rec["val"]]) == {30}
    assert set(ds.meta["depression"].to_numpy()[rec["test"]]) == {45}


def test_empty_amplitude_files_are_excluded_and_counted(atr_dir, tmp_path):
    ds = _make(atr_dir, tmp_path, split="EOC_depression", representation="Amplitude_8bit")
    assert ds.n_empty == {"train": 2}
    assert len(ds) == 12  # 4 test chips per class read from test/, 2 train chips per class
    assert all("/test/" in p for p in ds.meta["path"][ds.meta["part"] == "test"])
    assert ds[0][0].shape == (8, 8, 1)
    with pytest.raises(ValueError, match="EOC_pol"):
        fetch_atrnet_star(atr_dir, split="EOC_pol", cache_dir=tmp_path)


def test_parse_name():
    assert parse_name("KU_HH_30_110_225733.tif") == {
        "band": "KU", "pol": "HH", "depression": 30, "azimuth": 110, "chip_id": "225733"}
