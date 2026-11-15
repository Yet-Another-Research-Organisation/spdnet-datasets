"""S1SLC_CVDL loader on a tiny zarr store."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from spdnet_datasets.sar.s1slc_cvdl import S1SLCCVDLDataset  # noqa: E402

SCALE = {"Houston": 10.0, "Chicago": 100.0, "Sao Paulo": 1000.0}


@pytest.fixture
def slc_dir(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "S1SLC"
    g = zarr.open_group(str(root / "complex_HHHV.zarr"), mode="w")
    for city, s in SCALE.items():
        n = 30
        d = g.create_array(f"{city}/data", shape=(n, 6, 6, 4), chunks=(10, 6, 6, 4), dtype="float32")
        d[:] = (rng.normal(size=(n, 6, 6, 4)) * s).astype(np.float32)
        lab = g.create_array(f"{city}/labels", shape=(n,), chunks=(10,), dtype="int32")
        lab[:] = 1 + np.arange(n) % 3
    return root


def _make(root, tmp_path, **kw):
    kw.setdefault("return_raw", True)
    return S1SLCCVDLDataset(root, cache_dir=tmp_path / "cache", verbose=False, **kw)


def test_raw_mean_only(slc_dir, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        _make(slc_dir, tmp_path, return_raw=False)
    ds = _make(slc_dir, tmp_path)
    assert len(ds) == 90 and ds.classes == ["1", "2", "3"]
    assert ds[0][0].shape == (6, 6, 4)
    assert _make(slc_dir, tmp_path, features="amplitude")[0][0].shape == (6, 6, 2)
    assert _make(slc_dir, tmp_path, return_raw=False, return_mean=True)[0][0].shape == (2,)


def test_leave_one_city_out_and_train_only_normalisation(slc_dir, tmp_path):
    ds = _make(slc_dir, tmp_path, split="loco_Sao_Paulo")
    for name in ("loco_Houston", "loco_Chicago", "loco_Sao_Paulo"):
        parts = ds.splits[name]
        test_city = set(ds.groups[parts["test"]])
        assert len(test_city) == 1 and not test_city & set(ds.groups[np.r_[parts["train"], parts["val"]]])
    # train cities are normalised by their own RMS, the held-out city by the pooled train RMS
    rms = np.sqrt(2) * np.array([SCALE["Houston"], SCALE["Chicago"]])  # |z| of std-s Re and Im
    np.testing.assert_allclose([ds.rms["Houston"], ds.rms["Chicago"]], rms, rtol=0.1)
    assert ds.rms["Sao Paulo"] < 1000 / 2
    i = int(np.flatnonzero(ds.meta["city"] == "Chicago")[0])
    assert np.sqrt(np.mean(ds[i][0].numpy() ** 2)) == pytest.approx(1, rel=0.5)
