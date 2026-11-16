"""Pol-InSAR-Island loader on tiny synthetic ENVI rasters."""

import numpy as np
import pytest

from spdnet_datasets.core.estimators import NotSPDError, check_spd
from spdnet_datasets.sar import pol_insar_island
from spdnet_datasets.sar.pol_insar_island import (
    PolInSARIslandDataset,
    fetch_pol_insar_island,
    realify,
    t6_matrix,
)

H, W, CELL = 48, 64, 8  # 6 x 8 cells per flight path


def _hdr(path, lines, samples, east=None):
    text = f"ENVI\nsamples = {samples}\nlines   = {lines}\nbands   = 1\n"
    if east is not None:
        text += f"map info = {{UTM, 1, 1, {east}, 5000, 1, 1, 32, North,WGS-84}}\n"
    path.with_name(path.name + ".hdr").write_text(text)


@pytest.fixture
def pol_dir(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "POL"
    for fp, east in (("FP1", 1000), ("FP2", 1032)):  # 32 m overlap
        lab = np.zeros((H, W), np.uint8)
        for r in range(H // CELL):
            for c in range(W // CELL):
                lab[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = 1 + (r + c) % 3
        lab[:CELL, :CELL] = 0  # one unlabelled cell
        train = np.where((np.arange(W) // 16)[None, :] % 2 == 0, lab, 0).astype(np.uint8)
        test = (lab - train).astype(np.uint8)
        (root / f"label/{fp}").mkdir(parents=True)
        for name, arr in (("label_train", train), ("label_test", test)):
            p = root / f"label/{fp}/{name}.bin"
            arr.tofile(p)
            _hdr(p, H, W, east)
        for band in ("L", "S"):
            d = root / f"data/{fp}/{band}/T6"
            d.mkdir(parents=True)
            k = (rng.normal(size=(H, W, 6)) + 1j * rng.normal(size=(H, W, 6))) * (1 + lab[..., None])
            T = k[..., :, None] * k[..., None, :].conj()  # rank one per pixel
            for i in range(6):
                p = d / f"T{i + 1}{i + 1}.bin"
                T[..., i, i].real.astype("<f4").tofile(p)
                _hdr(p, H, W)
                for j in range(i + 1, 6):
                    for part, v in (("real", T[..., i, j].real), ("imag", T[..., i, j].imag)):
                        p = d / f"T{i + 1}{j + 1}_{part}.bin"
                        v.astype("<f4").tofile(p)
                        _hdr(p, H, W)
    return root


def _make(root, tmp_path, **kw):
    kw.setdefault("cell", CELL)
    kw.setdefault("block", 16)
    return PolInSARIslandDataset(root, cache_dir=tmp_path / "cache", verbose=False, **kw)


def test_logpower_covariances(pol_dir, tmp_path):
    ds = _make(pol_dir, tmp_path, return_raw=True, return_mean=True)
    assert len(ds) == 2 * (6 * 8 - 1)
    assert ds.covariances.shape[1:] == (12, 12)
    check_spd(ds.covariances)
    assert (ds.meta["n_pixels"] >= 2 * 12).all()
    (cov, mean, raw), y = ds[0]
    assert raw.shape == (CELL, CELL, 12) and mean.shape == (12,)
    assert ds.classes == ["coastal shrub", "tidal flat", "water"]


def test_normalisation_scales_mean_and_raw_by_sqrt_s(pol_dir, tmp_path):
    ds = _make(pol_dir, tmp_path, return_mean=True, return_raw=True)
    (cov, mean, raw), _ = ds[0]
    (cov0, mean0, raw0), _ = _make(pol_dir, tmp_path, normalize=False, return_mean=True, return_raw=True)[0]
    s = ds.scale
    assert s != 1.0
    np.testing.assert_allclose(cov.numpy(), s * cov0.numpy(), rtol=1e-12)
    np.testing.assert_allclose(mean.numpy(), np.sqrt(s) * mean0.numpy(), rtol=1e-12)
    np.testing.assert_allclose(raw.numpy(), np.sqrt(s) * raw0.numpy(), rtol=1e-6)


@pytest.mark.parametrize("representation", ["logpower", "t6"])
def test_non_spd_cell_is_rejected_and_counted(pol_dir, tmp_path, monkeypatch, representation):
    name = "estimate_covariance" if representation == "logpower" else "check_spd"
    real = getattr(pol_insar_island, name)

    def fail_first(C, *a, **kw):
        if fail_first.calls == 0:
            fail_first.calls += 1
            raise NotSPDError("forced")
        return real(C, *a, **kw)

    fail_first.calls = 0
    monkeypatch.setattr(pol_insar_island, name, fail_first)
    ds = _make(pol_dir, tmp_path, representation=representation, return_mean=True,
               return_raw=representation == "logpower", normalize=False)
    assert len(ds) == 2 * (6 * 8 - 1) - 1 and len(ds.rejected) == 1
    assert ds.rejected["proc_index"].tolist() == [0] and ds.meta["proc_index"].iat[0] == 1
    assert len(ds.means) == len(ds.covariances) == len(ds)
    if representation == "logpower":
        (cov, mean, raw), _ = ds[0]
        X = raw.numpy().reshape(-1, 12)  # raw still aligned with the kept cell
        np.testing.assert_allclose(mean.numpy(), X.astype(float).mean(0), rtol=1e-5)


def test_t6_means_are_scaled_like_the_covariances(pol_dir, tmp_path):
    ds = _make(pol_dir, tmp_path, representation="t6", return_mean=True)
    (cov, mean), _ = ds[0]
    (cov0, mean0), _ = _make(pol_dir, tmp_path, representation="t6", return_mean=True,
                             normalize=False)[0]
    np.testing.assert_allclose(cov.numpy(), ds.scale * cov0.numpy(), rtol=1e-12)
    np.testing.assert_allclose(mean.numpy(), ds.scale * mean0.numpy(), rtol=1e-12)


def test_t6_representation_is_realified_block_diagonal(pol_dir, tmp_path):
    ds = _make(pol_dir, tmp_path, representation="t6", normalize=False)
    C = ds.covariances
    assert C.shape[1:] == (24, 24)
    check_spd(C)
    assert np.allclose(C[:, :12, 12:], 0)  # no L-S cross terms
    T = t6_matrix(ds.means[:, :36])
    np.testing.assert_allclose(realify(T), C[:, :12, :12])
    # eigenvalues of the real form are those of T, doubled
    ev_c = np.linalg.eigvalsh(T[0])
    ev_r = np.linalg.eigvalsh(C[0, :12, :12])
    np.testing.assert_allclose(np.sort(np.r_[ev_c, ev_c]), ev_r, rtol=1e-8, atol=1e-10)
    with pytest.raises(ValueError, match="scm"):
        fetch_pol_insar_island(pol_dir, cache_dir=tmp_path, representation="t6", estimator="ledoit_wolf")
    with pytest.raises(ValueError, match="logpower"):
        _make(pol_dir, tmp_path, representation="t6", return_raw=True)


def test_spatial_split_has_guard_band_and_no_shared_block(pol_dir, tmp_path):
    ds = _make(pol_dir, tmp_path)
    xy = ds.meta[["easting", "northing"]].to_numpy()
    for name, parts in ds.splits.items():
        if not name.startswith("spatial"):
            continue
        assert not set(ds.groups[parts["train"]]) & set(ds.groups[parts["test"]])
        if len(parts["train"]) and len(parts["test"]):
            d = np.abs(xy[parts["train"]][:, None] - xy[parts["test"]][None]).max(-1)
            assert d.min() >= 2 * CELL
    fp = ds.splits["fp1_to_fp2"]
    assert set(ds.meta["path"][fp["train"]]) == {"FP1"} and set(ds.meta["path"][fp["test"]]) == {"FP2"}
    assert ds.meta["easting"][fp["train"]].max() + CELL / 2 <= 1032
    assert ds.meta["easting"][fp["test"]].min() - CELL / 2 >= 1064


def test_official_checkerboard_warns(pol_dir, tmp_path):
    with pytest.warns(UserWarning, match="optimistic"):
        ds = _make(pol_dir, tmp_path, split="official")
    parts = ds.splits["official"]
    assert set(ds.meta["part"][parts["train"]]) == {"train"} and set(ds.meta["part"][parts["test"]]) == {"test"}
    tr = np.flatnonzero(ds.meta["part"] == "train")
    covs = np.stack([ds[i][0].numpy() for i in tr])
    assert abs(np.mean(np.log(np.linalg.eigvalsh(covs)))) < 1e-8
