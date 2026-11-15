"""S1 Iceland SAR loader on a tiny fake zarr copy (64 x 64 patches, 32 px sub-patches)."""

from datetime import date, timedelta

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from spdnet_datasets.core.estimators import InsufficientSamplesError  # noqa: E402
from spdnet_datasets.core.splits import (  # noqa: E402
    assert_no_leakage,
    assert_no_window_overlap,
)
from spdnet_datasets.manager import DatasetManager  # noqa: E402
from spdnet_datasets.sar import s1_iceland as S  # noqa: E402

H = 64


def _dates(start, n):
    d0 = date(*start)
    return [(d0 + timedelta(days=6 * i)).strftime("%Y%m%d") for i in range(n)]


DATES_2019 = _dates((2019, 6, 18), 15)  # as in the real train/val groups
DATES_2025 = _dates((2025, 6, 16), 16)  # as in the real test group


def _patch(rng, kind, T):
    """Labels, complex SLC and amplitudes of one patch; four 32 px sub-patches."""
    y = np.zeros((H, H), np.uint8)
    y[:32, 32:] = 4
    y[32:, :32] = 2
    A = rng.gamma(4.0, 300.0, size=(T, 2, H, H)).astype(np.uint16) + 1
    Z = rng.standard_normal((T, 4, H, H)).astype(np.float32)
    if kind == "rejects":
        y[:32, :32] = 255  # unlabelled -> rejected
        Z[2, :, :8, 32:] = 0  # out of swath at one date: 25 % invalid -> rejected
        y[32:, 32:][:, :16] = 1  # 50 / 50 -> purity < 0.6, rejected
        y[32:, 32:][:, 16:] = 5
    if kind == "zero_vh":
        A[3, 1, 32:, 32:] = 0  # constant channel -> covariance not SPD
    if T == 16:
        Z[15] = 0  # swath gap on the dropped 2025 date: does not invalidate pixels
    return y, Z, A


def _make_fake(root, n_patches=(3, 2, 2)):
    rng = np.random.default_rng(0)
    seg = zarr.open_group(str(root / S.SEG), mode="w", zarr_format=2)
    amp = zarr.open_group(str(root / S.AMP), mode="w", zarr_format=2)
    amp.attrs.update(amp_scale=6000.0, channels=["vv_amp", "vh_amp"])
    col0 = {"train": 0, "val": 3000, "test": 6000}  # three distinct 10 km columns
    for sp, n in zip(S.SPLITS, n_patches, strict=True):
        dates = DATES_2025 if sp == "test" else DATES_2019
        kinds = ["rejects", "zero_vh"] + ["plain"] * (n - 2)
        ys, Zs, As = zip(*(_patch(rng, k, len(dates)) for k in kinds), strict=True)
        g, ga = seg.create_group(sp), amp.create_group(sp)
        g.attrs.update(dates=dates, n_dates=len(dates), year=dates[0][:4])
        g.create_array("y", data=np.stack(ys))
        g.create_array("X", data=np.stack(Zs))
        g.create_array("r0", data=np.arange(n, dtype=np.int32) * 100)  # overlapping patches
        g.create_array("c0", data=np.full(n, col0[sp], np.int32))
        ga.create_array("X_amp", data=np.stack(As))
    return root


@pytest.fixture
def fake(tmp_path):
    return _make_fake(tmp_path / "ICELAND_SAR"), tmp_path / "cache"


def test_select_dates_drops_the_last_2025_date():
    keep = S.select_dates(DATES_2025, DATES_2019)
    assert len(keep) == 15 and [DATES_2025[i] for i in set(range(16)) - set(keep)] == ["20250914"]
    # 16 summer dates of 2019 (change product, 16 June - 14 Sept.): the last one goes too
    assert S.select_dates(_dates((2019, 6, 16), 16), DATES_2019).tolist() == list(range(15))
    with pytest.raises(ValueError):
        S.select_dates(DATES_2019[:10], DATES_2019)


def test_fetch_shapes_selection_and_exclusions(fake):
    root, cache = fake
    b = S.fetch_s1_iceland(root, cache_dir=cache)
    # per split: 1 sub-patch kept on the "rejects" patch, 3 SPD ones on the zero-VH patch
    # (1 excluded), 4 per plain patch
    assert b.n_excluded == {"train": 1, "val": 1, "test": 1}
    assert {sp: len(v) for sp, v in b.splits["official"].items()} == {"train": 8, "val": 4, "test": 4}
    assert b.covariances.shape == (16, 30, 30) and b.means.shape == (16, 30)
    assert b.dropped_dates == {"train": [], "val": [], "test": ["20250914"]}
    assert all(len(d) == 15 for d in b.dates.values())
    np.testing.assert_allclose(b.covariances, np.swapaxes(b.covariances, 1, 2))
    assert np.linalg.eigvalsh(b.covariances).min() > 0
    assert set(b.labels.tolist()) <= {0, 2, 4}


def test_channels_are_vv_then_vh_dates(fake):
    root, cache = fake
    ds = S.S1IcelandDataset(root, cache_dir=cache, return_mean=True, return_raw=True,
                            verbose=False)
    (cov, mean, raw), y = ds[0]
    assert cov.shape == (30, 30) and mean.shape == (30,) and tuple(raw.shape) == (32, 32, 30)
    m = ds.meta.iloc[0]
    A = zarr.open_group(str(root / S.AMP), mode="r")[m.split]["X_amp"][int(m.patch)]
    r, c = int(m.row) - int(m.patch) * 100, int(m.col)
    keep = S.select_dates(DATES_2019, DATES_2019)
    expected = np.concatenate([A[keep, 0, r:r + 32, c:c + 32], A[keep, 1, r:r + 32, c:c + 32]])
    np.testing.assert_allclose(raw.numpy(), np.moveaxis(expected, 0, -1) / 6000.0, rtol=1e-6)
    np.testing.assert_allclose(mean.numpy(), raw.numpy().reshape(-1, 30).mean(0), rtol=1e-5)


def test_sample_size_rule(fake):
    root, cache = fake
    b = S.fetch_s1_iceland(root, cache_dir=cache)
    raw, valid, info = S._load_raw(b.processed_path)
    assert valid.reshape(len(valid), -1).sum(1).min() >= 2 * info["p"]
    with pytest.raises(InsufficientSamplesError):
        S.fetch_s1_iceland(root, cache_dir=cache, block=4)  # 16 px < 2p = 60


def test_official_split_has_no_spatial_or_temporal_leakage(fake):
    root, cache = fake
    b = S.fetch_s1_iceland(root, cache_dir=cache)
    parts = b.splits["official"]
    meta = b.meta
    boxes = np.stack([meta.row, meta.col, meta.row + 32, meta.col + 32], 1)
    for a, c in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert_no_leakage(b.groups[parts[a]], b.groups[parts[c]])
        assert_no_window_overlap(boxes[parts[a]], boxes[parts[c]])
    years = {sp: set(meta.year[parts[sp]]) for sp in parts}
    assert years == {"train": {2019}, "val": {2019}, "test": {2025}}


def test_normalisation_fitted_on_train(fake):
    root, cache = fake
    ds = S.S1IcelandDataset(root, cache_dir=cache, verbose=False)
    tr = ds.splits["official"]["train"]
    eig = np.linalg.eigvalsh(ds.covariances[tr])
    assert abs(np.mean(np.log(eig))) < 1e-10
    raw = S.S1IcelandDataset(root, cache_dir=cache, normalize=False, verbose=False)
    np.testing.assert_allclose(ds.covariances, raw.covariances * ds.scale)


def test_tasks_subset_and_registry(fake):
    root, cache = fake
    ds = S.S1IcelandDataset(root, cache_dir=cache, subset="test", task="coarse", verbose=False)
    assert len(ds) == 4 and set(ds.classes) <= {"vegetation", "water"}
    assert list(ds.splits) == ["official"] and len(ds.splits["official"]["test"]) == 4
    ds = S.S1IcelandDataset(root, cache_dir=cache, drop_classes=("water",), verbose=False)
    assert "water" not in ds.classes and len(ds) == sum(len(v) for v in ds.splits["official"].values())
    assert DatasetManager.DATASETS["s1_iceland"] is S.S1IcelandDataset


def test_cache_replay_and_keep_covariance(fake, monkeypatch):
    root, cache = fake
    S.fetch_s1_iceland(root, cache_dir=cache)

    def boom(*a, **k):
        raise AssertionError("recomputed")

    monkeypatch.setattr(S, "_process", boom)
    monkeypatch.setattr(S, "_covariances", boom)
    b = S.fetch_s1_iceland(root, cache_dir=cache, keep="keep_covariance")
    assert b.processed_path is None
    with pytest.raises(ValueError, match="return_raw"):
        S.S1IcelandDataset(root, cache_dir=cache, keep="keep_covariance", return_raw=True,
                           verbose=False)


def test_manifest_mismatch_is_reported(fake):
    root, cache = fake
    (root / "MANIFEST_NAS.tsv").write_text(f"path\tsize\tmtime\n{S.SEG}/.zgroup\t1\tx\n")
    with pytest.raises(OSError, match="MANIFEST"):
        S.fetch_s1_iceland(root, cache_dir=cache)
