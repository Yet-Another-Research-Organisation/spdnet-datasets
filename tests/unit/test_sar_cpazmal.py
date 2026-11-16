"""CPAZMaL loader on a tiny synthetic HDF5 file."""

import h5py
import numpy as np
import pytest

from spdnet_datasets.core.estimators import NotSPDError, check_spd
from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.sar import cpazmal
from spdnet_datasets.sar.cpazmal import CPAZMaLDataset, fetch_cpazmal, select_dates

HH_DATES = [f"2020{m:02d}{d:02d}" for m in range(1, 13) for d in (5, 20)]  # 24 dates
HV_DATES = HH_DATES[:22]  # 22 common dual-pol dates


def _write_site(h, name, cls, rng, size=(20, 20), poly=None, orbits=("ASC", "DSC"), pols=("HH", "HV")):
    g = h.create_group(f"data/{name}")
    g.attrs["class"] = cls
    H, W = size
    inside = np.zeros((H, W), bool)
    if poly is None:
        inside[2:-2, 3:-3] = True
    else:
        inside[poly] = True
    for orbit in orbits:
        for pol in pols:
            dates = HH_DATES if pol == "HH" else HV_DATES
            img = rng.gamma(2.0, 0.05, size=(H, W, len(dates))).astype(np.float32)
            mask = np.where(inside[..., None], 0, -127).astype(np.int8).repeat(len(dates), -1)
            img[~inside] = -999.0
            mask[0 + 2, 3, 0] = 2  # one shadow pixel
            s = g.create_group(f"{orbit}/{pol}")
            s["images"], s["masks"] = img, mask
            s["timestamps"] = np.array(dates, dtype="S8")


@pytest.fixture
def cpaz_dir(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "PAZ"
    root.mkdir()
    with h5py.File(root / "CPAZMaL.hdf5", "w") as h:
        for c in ("ABL", "FOR", "LAC"):
            for k in range(5):
                _write_site(h, f"{c}{k:03d}", c, rng)
        # a too small parcel (fewer than 80 valid pixels) and two excluded groups
        small = (slice(2, 10), slice(2, 10))  # 64 px
        _write_site(h, "ROC000", "ROC", rng, poly=small)
        _write_site(h, "ALLGLA", "STUDY", rng)
        _write_site(h, "PLA006", "PLA", rng, orbits=("ASC",), pols=("HH",))
    return root


def _make(root, tmp_path, **kw):
    kw.setdefault("n_dates", 20)
    return CPAZMaLDataset(root, cache_dir=tmp_path / "cache", verbose=False, **kw)


def test_parcel_covariances_are_spd_with_enough_pixels(cpaz_dir, tmp_path):
    ds = _make(cpaz_dir, tmp_path)
    assert ds.covariances.shape[1:] == (40, 40)
    check_spd(ds.covariances)
    assert (ds.meta["n_pixels"] >= 80).all() and (ds.meta["n_pixels"] >= 2 * 40).all()
    assert len(ds) == 15  # one per parcel
    assert ds.classes == ["ABL", "FOR", "LAC"]
    assert {"ROC000", "ALLGLA", "PLA006"} <= set(ds.excluded)
    assert len(ds.dates) == 20 and set(ds.dates) <= set(HV_DATES)
    x, y = ds[0]
    assert x.shape == (40, 40) and x.dtype.is_floating_point


def test_only_polygon_pixels_are_used(cpaz_dir, tmp_path):
    ds = _make(cpaz_dir, tmp_path, return_raw=True, return_mean=True)
    (cov, mean, raw), _ = ds[0]
    assert raw.shape == (16 * 14, 40)  # polygon pixels only (the shadow code 2 is kept)
    assert np.isfinite(raw.numpy()).all() and (raw.numpy() > 0).all()
    np.testing.assert_allclose(mean.numpy(), raw.numpy().astype(float).mean(0), rtol=1e-5)
    ds = _make(cpaz_dir, tmp_path, return_raw=True, max_mask_value=1)
    (cov, raw), _ = ds[0]
    assert raw.shape == (16 * 14 - 1, 40)  # the shadow pixel is dropped


def test_min_pixels_threshold(cpaz_dir, tmp_path):
    with pytest.raises(ValueError, match="2p"):
        _make(cpaz_dir, tmp_path, min_pixels=79)
    ds = _make(cpaz_dir, tmp_path, n_dates=10, min_pixels=60)  # p = 20
    assert "ROC000" not in ds.excluded and ds.covariances.shape[-1] == 20


def test_blocks_keep_parcel_groups_and_threshold(cpaz_dir, tmp_path):
    ds = _make(cpaz_dir, tmp_path, n_dates=10, block=10, min_pixels=40)
    assert (ds.meta["n_pixels"] >= 40).all()
    assert len(ds) > 15
    for parts in ds.splits.values():
        assert_no_leakage(ds.groups[parts["train"]], ds.groups[parts["test"]])
        assert_no_leakage(ds.groups[parts["val"]], ds.groups[parts["test"]])


def test_splits_and_normalisation(cpaz_dir, tmp_path):
    ds = _make(cpaz_dir, tmp_path)
    assert "recommended" in ds.splits and "sgkf5_fold4" in ds.splits
    train = ds.splits["recommended"]["train"]
    covs = np.stack([ds[i][0].numpy() for i in train])
    assert abs(np.mean(np.log(np.linalg.eigvalsh(covs)))) < 1e-8


def test_normalisation_scales_mean_and_raw_by_sqrt_s(cpaz_dir, tmp_path):
    ds = _make(cpaz_dir, tmp_path, return_mean=True, return_raw=True)
    (cov, mean, raw), _ = ds[0]
    (cov0, mean0, raw0), _ = _make(cpaz_dir, tmp_path, normalize=False, return_mean=True, return_raw=True)[0]
    s = ds.scale
    assert s != 1.0
    np.testing.assert_allclose(cov.numpy(), s * cov0.numpy(), rtol=1e-12)
    np.testing.assert_allclose(mean.numpy(), np.sqrt(s) * mean0.numpy(), rtol=1e-12)
    np.testing.assert_allclose(raw.numpy(), np.sqrt(s) * raw0.numpy(), rtol=1e-6)


def test_non_spd_sample_is_rejected_and_counted(cpaz_dir, tmp_path, monkeypatch):
    real = cpazmal.estimate_covariance

    def estimate(x, estimator):
        if estimate.calls == 0:
            estimate.calls += 1
            raise NotSPDError("forced")
        return real(x, estimator)

    estimate.calls = 0
    monkeypatch.setattr(cpazmal, "estimate_covariance", estimate)
    ds = _make(cpaz_dir, tmp_path, return_mean=True, return_raw=True, normalize=False)
    assert len(ds) == 14 and len(ds.rejected) == 1
    assert ds.rejected[0]["proc_index"] == 0 and "NotSPDError" in ds.rejected[0]["reason"]
    assert ds.rejected[0]["site"] not in set(ds.meta["site"])
    (cov, mean, raw), _ = ds[0]  # raw pixels still aligned with the kept sample
    np.testing.assert_allclose(mean.numpy(), raw.numpy().astype(float).mean(0), rtol=1e-5)


def test_select_dates_and_task_options(cpaz_dir, tmp_path):
    assert select_dates(HV_DATES, 20)[0] == HV_DATES[0] and select_dates(HV_DATES, 20)[-1] == HV_DATES[-1]
    with pytest.raises(ValueError):
        select_dates(HV_DATES, 23)
    ds = _make(cpaz_dir, tmp_path, drop_classes=("LAC",))
    assert ds.classes == ["ABL", "FOR"] and len(ds) == 10
    ds = _make(cpaz_dir, tmp_path, merge_classes={"FOR": "LAC"})
    assert ds.classes == ["ABL", "LAC"]


def test_fetch_replays_from_cache(cpaz_dir, tmp_path):
    b1 = fetch_cpazmal(cpaz_dir, cache_dir=tmp_path / "c", n_dates=20)
    b2 = fetch_cpazmal(cpaz_dir, cache_dir=tmp_path / "c", n_dates=20)
    np.testing.assert_array_equal(b1.covariances, b2.covariances)
    b3 = fetch_cpazmal(cpaz_dir, cache_dir=tmp_path / "c", n_dates=20, keep="keep_covariance")
    assert b3.processed_dir is None
    with pytest.raises(ValueError, match="return_raw"):
        CPAZMaLDataset(cpaz_dir, cache_dir=tmp_path / "c", keep="keep_covariance",
                       return_raw=True, verbose=False)
