"""PASTIS-R loader on a tiny synthetic copy of the dataset layout."""

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from spdnet_datasets import DatasetManager
from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.sits import pastis_r as pr
from spdnet_datasets.sits.pastis_r import (
    SCENARIOS,
    SPLITS,
    PastisRDataset,
    fetch_pastis_r,
    homogeneity_mask,
    pair_s1,
    select_dates,
)

H = W = 16
REF = "2019-07-16"


def _dates(start, step, n):
    d0 = datetime.date.fromisoformat(start)
    return {str(i): int((d0 + datetime.timedelta(days=step * i)).strftime("%Y%m%d"))
            for i in range(n)}


def _write_patch(root, pid, rng, n_s2, n_s1, cloudy=None, bad_pixel=False):
    ids = np.zeros((H, W), np.int32)
    target = np.zeros((3, H, W), np.uint8)
    flat_ids, flat_t = ids.reshape(-1), target[0].reshape(-1)
    # parcel 1: 80 px, majority Meadow (1) with 5 px of class 2
    flat_ids[:80], flat_t[:80] = pid * 10 + 1, 1
    flat_t[:5] = 2
    flat_ids[80:120], flat_t[80:120] = pid * 10 + 2, 2      # 40 px, Soft winter wheat
    flat_ids[120:140], flat_t[120:140] = pid * 10 + 3, 3    # 20 px: too small
    flat_ids[140:210], flat_t[140:210] = pid * 10 + 4, 19   # 70 px void: dropped
    flat_t[210:] = 0                                        # background, id 0
    ann = root / "ANNOTATIONS"
    np.save(ann / f"ParcelIDs_{pid}.npy", ids)
    np.save(ann / f"TARGET_{pid}.npy", target)
    s2 = rng.normal(1000, 200, size=(n_s2, 10, H, W)).astype(np.int16)
    if cloudy is not None:
        s2[cloudy, 0] = 5000
    if bad_pixel:
        s2[:, 2, 0, 10] = -600   # B04 + B08 <= 0 at pixel 10 (parcel 1)
        s2[:, 6, 0, 10] = 100
    np.save(root / "DATA_S2" / f"S2_{pid}.npy", s2)
    for s in ("S1A", "S1D"):
        vv = rng.normal(-10, 2, size=(n_s1, H, W))
        vh = rng.normal(-17, 2, size=(n_s1, H, W))
        np.save(root / f"DATA_{s}" / f"{s}_{pid}.npy",
                np.stack([vv, vh, vv - vh], 1).astype(np.float16))


@pytest.fixture
def pastis_dir(tmp_path):
    root = tmp_path / "PASTIS-R"
    for d in ("ANNOTATIONS", "DATA_S2", "DATA_S1A", "DATA_S1D"):
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    n_s2, n_s1 = 10, 9
    feats = []
    for k, pid in enumerate([10, 11, 12, 13, 14]):  # folds 1..5, two tiles
        tile = "tA" if k % 2 == 0 else "tB"
        # tile A: S2 date 2019-07-16 is cloudy in patch 10
        _write_patch(root, pid, rng, n_s2, n_s1, cloudy=3 if pid == 10 else None,
                     bad_pixel=pid == 11)
        feats.append({"type": "Feature", "properties": {
            "Fold": k + 1, "ID_PATCH": pid, "TILE": tile,
            "dates-S2": _dates("2019-07-01", 5, n_s2),
            "dates-S1A": _dates("2019-07-02" if tile == "tA" else "2019-07-03", 6, n_s1),
            "dates-S1D": json.dumps(_dates("2019-06-30", 6, n_s1)).replace('"', "'"),
        }})
    _write_patch(root, 99, rng, n_s2, n_s1)  # images without metadata: ignored
    (root / "metadata.geojson").write_text(json.dumps({"features": feats}))
    return tmp_path  # dataset dir, files in PASTIS-R/


def _fetch(pastis_dir, tmp_path, **kw):
    kw.setdefault("ref_date", REF)
    return fetch_pastis_r(pastis_dir, tmp_path / "cache", **kw)


# ------------------------------------------------------------------- date logic
def test_pair_s1_uses_distinct_acquisitions():
    # both S2 dates are nearest to S1 day 3: the pairing must use two images
    js, gaps = pair_s1([0, 5], [3, 9, 15], max_gap_days=4)
    assert js == (0, 1) and tuple(gaps) == (3, 4)
    assert pair_s1([0, 5], [3, 9, 15], max_gap_days=3) is None
    js, gaps = pair_s1([10], [3, 9, 15], max_gap_days=3)
    assert js == (1,) and tuple(gaps) == (-1,)


def test_select_dates_prefers_small_gap_within_window():
    s2 = np.arange(0, 50, 5)
    s1 = np.array([1, 7, 13, 19, 26, 31, 37])  # S2 day 25 -> gap 1, day 20 -> gap 1
    sel, _ = select_dates(s2, s1, s1, ref_day=20, date_window=6, max_gap_days=3)
    assert sel["s2"] == (4,) and sel["max_gap"] == 1   # day 20 (gap 1, offset 0)
    sel, reason = select_dates(s2, s1, s1, ref_day=20, date_window=6, max_gap_days=0)
    assert sel is None and reason == "gap"
    sel, reason = select_dates(s2, s1, s1, ref_day=500, date_window=6)
    assert sel is None and reason == "no_window"


def test_select_dates_skips_cloudy_and_long_windows():
    s2 = np.array([0, 5, 10, 40, 45, 50])
    cloud = np.array([0, 0, 0.9, 0, 0, 0])
    sel, _ = select_dates(s2, ref_day=10, date_window=10, cloud=cloud, max_cloud_fraction=0.1)
    assert sel["s2"] == (1,) and sel["cloud"] == 0
    sel, reason = select_dates(s2, ref_day=10, date_window=0, cloud=cloud,
                               max_cloud_fraction=0.1)
    assert sel is None and reason == "cloud"
    sel, _ = select_dates(s2, n_dates=3, ref_day=30, date_window=30, max_span_days=20)
    assert sel["s2"] in ((0, 1, 2), (3, 4, 5))
    sel, reason = select_dates(s2, n_dates=3, date_index=1, max_span_days=20)
    assert sel is None and reason == "span"


def test_homogeneity_mask():
    mask, median = homogeneity_mask([1, 2, 3], 1)
    assert median == 2 and mask.all()
    mask, median = homogeneity_mask([1, 2, 3], 1, weights=[5, 1, 1])  # median over covariances
    assert median == 1 and mask.tolist() == [True, True, False]
    assert homogeneity_mask([3], None)[0].all()


# ------------------------------------------------------------------ extraction
@pytest.mark.parametrize("scenario,n_parcels", [("A", 10), ("B", 10), ("C", 10),
                                                ("D", 5), ("E", 5)])
def test_scenario_dimension_and_pixel_threshold(pastis_dir, tmp_path, scenario, n_parcels):
    b = _fetch(pastis_dir, tmp_path, scenario=scenario)
    p = SCENARIOS[scenario]["p"]
    assert b.covariances.shape == (n_parcels, p, p) and len(b.channels) == p
    assert (b.meta["n_pixels"] >= SCENARIOS[scenario]["min_pixels"]).all()
    assert np.all(np.linalg.eigvalsh(b.covariances)[:, 0] > 0)
    # all parcels of one patch share the same dates
    assert (b.meta.groupby("patch")["s2_dates"].nunique() == 1).all()
    assert b.meta["s2_dates"].str.split(";").str.len().eq(SCENARIOS[scenario]["n_dates"]).all()


def test_parcels_labels_and_exclusions(pastis_dir, tmp_path):
    b = _fetch(pastis_dir, tmp_path, scenario="A")
    assert sorted(b.meta["parcel"] % 10) == [1] * 5 + [2] * 5  # 20 px and void parcels dropped
    names = dict(zip(b.meta["parcel"] % 10, b.meta["class_name"], strict=True))
    assert names == {1: "Meadow", 2: "Soft winter wheat"}  # majority label of parcel 1
    assert set(b.labels) == {0, 1}
    assert b.report["n_patches_without_metadata"] == 1
    assert 99 not in set(b.groups)
    assert sum(v for k, v in b.report["rejected"]["parcel"].items()
               if k.startswith("few_pixels")) == 5


def test_cloudy_date_is_avoided(pastis_dir, tmp_path):
    b = _fetch(pastis_dir, tmp_path, scenario="A")
    dates = b.meta.groupby("patch")["s2_dates"].first()
    assert dates[10] != "2019-07-16" and dates[12] == "2019-07-16"


def test_ndvi_ndwi_channels(pastis_dir, tmp_path):
    files = pastis_dir / "PASTIS-R"
    sel = {"s2": (3,)}
    cube = pr.patch_channels(files, 12, sel, "B")
    s2 = np.load(files / "DATA_S2" / "S2_12.npy")[3].astype(np.float64)
    ndvi = (s2[6] - s2[2]) / (s2[6] + s2[2])
    ndwi = (s2[1] - s2[6]) / (s2[1] + s2[6])
    np.testing.assert_allclose(cube[10], ndvi, rtol=1e-5)
    np.testing.assert_allclose(cube[11], ndwi, rtol=1e-5)
    assert pr.channel_names("B")[10:] == ["t0:NDVI", "t0:NDWI"]
    # invalid index pixel (B04 + B08 <= 0) is dropped from the parcel
    b = _fetch(pastis_dir, tmp_path, scenario="B")
    row = b.meta[(b.meta.patch == 11) & (b.meta.parcel == 111)].iloc[0]
    assert row.n_pixels == 79 and row.n_pixels_parcel == 80


def test_s1_channels_are_not_collinear():
    rng = np.random.default_rng(1)
    vv, vh = rng.normal(-10, 2, (64,)), rng.normal(-17, 2, (64,))
    frame = np.stack([vv, vh, vv - vh]).astype(np.float16)[:, :, None]
    ch = pr.s1_channels(frame)
    np.testing.assert_allclose(ch[2], 10 ** ((ch[0] - ch[1]) / 10), rtol=1e-5)
    X = ch.reshape(3, -1).T
    X = (X - X.mean(0)) / X.std(0)
    assert np.linalg.cond(np.cov(X.T)) < 1e3
    lin = pr.s1_channels(frame, "linear")
    np.testing.assert_allclose(lin[2], lin[0] / lin[1], rtol=1e-4)


def test_s1_dates_and_gaps_in_meta(pastis_dir, tmp_path):
    b = _fetch(pastis_dir, tmp_path, scenario="E")
    row = b.meta.iloc[0]
    assert len(set(row.s1a_dates.split(";"))) == 2   # distinct S1 images
    gaps = [abs(int(g)) for g in row.gap_s1a.split(";") + row.gap_s1d.split(";")]
    assert max(gaps) <= 3 and row.max_abs_gap == max(gaps)
    # tile tB has no S2 date with both S1 gaps <= 1 day: its patches are rejected
    strict = _fetch(pastis_dir, tmp_path, scenario="C", max_gap_days=1)
    assert strict.report["rejected"]["patch"] == {"gap|tB|fold2": 1, "gap|tB|fold4": 1}
    assert (strict.meta["max_abs_gap"] <= 1).all() and set(strict.meta.tile) == {"tA"}


def test_gap_tables_written(pastis_dir, tmp_path):
    b = _fetch(pastis_dir, tmp_path, scenario="C")
    table = (b.processed_path / "date_gaps.csv").read_text()
    assert "gap_s1a" in table and (b.processed_path / "gap_histogram.csv").exists()
    assert b.report["gap_quantiles_all_s2_dates"]["S1A"]["p100"] <= 3


def test_too_few_pixels_are_counted_not_fatal(pastis_dir, tmp_path):
    b = _fetch(pastis_dir, tmp_path, scenario="D", min_pixels=30)  # 2p = 60
    assert b.report["covariance"]["failures"]["InsufficientSamplesError"] == 5
    assert len(b.covariances) == 5


# ------------------------------------------------------- normalisation, splits
def test_standardization_uses_training_folds_only(pastis_dir, tmp_path):
    b = _fetch(pastis_dir, tmp_path, scenario="A")
    std = np.load(b.covariance_path / "standardization.npz")
    offsets = np.load(b.processed_path / "offsets.npy")
    pix = np.fromfile(b.processed_path / "pixels.f32", np.float32).reshape(-1, 10)
    proc_meta = pd.read_csv(b.processed_path / "parcels.csv")
    rows = [pix[offsets[i]:offsets[i + 1]] for i in range(len(proc_meta))
            if proc_meta.fold[i] in (1, 2, 3)]
    np.testing.assert_allclose(std["mean"], np.concatenate(rows).mean(0), rtol=1e-6)


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_splits_have_no_patch_leakage(pastis_dir, tmp_path, split):
    ds = PastisRDataset(pastis_dir, cache_dir=tmp_path / "c", ref_date=REF, split=split,
                        verbose=False)
    parts = ds.splits[split]
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(ds.groups[parts[a]], ds.groups[parts[c]])
    folds = ds.meta["fold"].to_numpy()
    assert set(folds[parts["test"]]) == set(SPLITS[split]["test"])
    assert sum(len(v) for v in parts.values()) == len(ds)
    eig = np.linalg.eigvalsh(ds.covariances[parts["train"]])
    assert abs(np.mean(np.log(eig))) < 1e-10   # geometric mean 1 on train


def test_cv5_equals_official():
    assert SPLITS["cv5"] == SPLITS["official"]


# ------------------------------------------------------------------ dataset API
def test_output_tuples(pastis_dir, tmp_path):
    ds = PastisRDataset(pastis_dir, cache_dir=tmp_path / "c", ref_date=REF, scenario="B",
                        verbose=False)
    x, y = ds[0]
    assert x.shape == (12, 12) and x.dtype.is_floating_point and isinstance(y, int)
    ds = PastisRDataset(pastis_dir, cache_dir=tmp_path / "c", ref_date=REF, scenario="B",
                        return_mean=True, return_raw=True, verbose=False)
    (cov, mean, raw), _ = ds[0]
    assert cov.shape == (12, 12) and mean.shape == (12,)
    assert raw.ndim == 2 and raw.shape[1] == 12 and raw.shape[0] == ds.meta.n_pixels[0]
    np.testing.assert_allclose(raw.numpy().mean(0), mean.numpy(), atol=1e-5)


def test_registered_and_tasks(pastis_dir, tmp_path):
    assert DatasetManager.DATASETS["pastis_r"] is PastisRDataset
    ds = PastisRDataset(pastis_dir, cache_dir=tmp_path / "c", ref_date=REF, task="coarse",
                        verbose=False)
    assert ds.classes == ["Grassland and fodder", "Winter cereals"]
    ds = PastisRDataset(pastis_dir, cache_dir=tmp_path / "c", ref_date=REF,
                        drop_classes=("Meadow",), verbose=False)
    assert ds.classes == ["Soft winter wheat"] and len(ds) == 5
    ds = PastisRDataset(pastis_dir, cache_dir=tmp_path / "c", ref_date=REF, subset="test",
                        verbose=False)
    assert set(ds.meta.fold) == {5} and len(ds) == 2


def test_cache_is_replayed_and_keep_covariance(pastis_dir, tmp_path, monkeypatch):
    _fetch(pastis_dir, tmp_path, scenario="A")
    monkeypatch.setattr(pr, "_process", lambda *a: (_ for _ in ()).throw(RuntimeError))
    b = _fetch(pastis_dir, tmp_path, scenario="A", estimator="ledoit_wolf")  # new cov stage
    assert b.covariances.shape == (10, 10, 10)
    b = _fetch(pastis_dir, tmp_path, scenario="A", keep="keep_covariance")
    assert b.processed_path is None and len(b.covariances) == 10
    monkeypatch.undo()
    with pytest.raises(ValueError, match="return_raw"):
        PastisRDataset(pastis_dir, cache_dir=tmp_path / "cache", ref_date=REF,
                       keep="keep_covariance", return_raw=True, verbose=False)


def test_bad_arguments(pastis_dir, tmp_path):
    with pytest.raises(ValueError, match="scenario"):
        _fetch(pastis_dir, tmp_path, scenario="F")
    with pytest.raises(ValueError, match="split"):
        _fetch(pastis_dir, tmp_path, split="random")
