"""AVIRIS-C California multilabel on tiny synthetic Parquet shards with the release layout.

20 windows (4 per fold, 2 shards per fold, row groups of one row), 3 dates of 8 kept
bands (of 12) over 6 x 6 pixels. Window 0 has a padded date, window 1 a date failing
QC, window 2 a date with too many no-data pixels, window 3 a few outliers (clipped).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from sklearn.covariance import LedoitWolf  # noqa: E402

from spdnet_datasets.base import create_dataloaders  # noqa: E402
from spdnet_datasets.core import assert_no_leakage, pos_weights  # noqa: E402
from spdnet_datasets.hyperspectral import aviris_ca_ml as A  # noqa: E402
from spdnet_datasets.manager import DatasetManager  # noqa: E402

T, B_ALL, B, H = 3, 12, 8, 6
K = len(A.CLASSES)


def _nested(values, dims):
    """Nested ``list`` array (as written by ``datasets``) of rows of shape ``dims``."""
    arr = pa.array(values.ravel())
    for d in reversed(dims):
        n = len(arr) // d
        arr = pa.ListArray.from_arrays(pa.array(np.arange(n + 1, dtype=np.int32) * d), arr)
    return arr


# CDL proportions per date by window type (w % 5), see the scenario tests:
# 0 pure water, constant (change 0); 1 pure almond 0.80 then 0.95 (change 0.15);
# 2 grapes 0.6 / 0.2 / 0.6 against other (change 0.4); 3 mixed, change 0.07;
# 4 random (Dirichlet, change > 0.1)
def _props(rng, w):
    kind, P = w % 5, np.zeros((T, K), np.float32)
    ci = {c: i for i, c in enumerate(A.CLASSES)}
    if kind == 0:
        P[:, ci["water"]], P[:, ci["barren"]] = 0.9, 0.1
    elif kind == 1:
        P[:, ci["almond"]] = [0.80, 0.95, 0.95]
        P[:, ci["other"]] = 1 - P[:, ci["almond"]]
    elif kind == 2:
        P[:, ci["grapes"]] = [0.6, 0.2, 0.6]
        P[:, ci["other"]] = 1 - P[:, ci["grapes"]]
    elif kind == 3:
        P[:, ci["shrubland"]] = [0.5, 0.43, 0.5]
        P[:, ci["grassland_pasture"]] = 1 - P[:, ci["shrubland"]]
    else:
        P = rng.dirichlet(np.ones(K) * 0.3, size=T).astype(np.float32)
    return P


def _window(rng, w, fold):
    dates = ["2013-04-10", "2014-05-07", "2015-06-11"]
    acquired = np.ones(T, bool)
    qc = np.ones(T, bool)
    cube = (0.05 + 0.3 * rng.random((T, B, H, H))).astype(np.float16)
    if w == 0:
        acquired[2], qc[2] = False, False
        cube[2] = np.nan
        dates[2] = ""
    if w == 1:
        qc[1] = False
    if w == 2:
        cube[0, :, :, :4] = np.nan  # 12 valid pixels < 2p = 16
    if w == 3:
        cube[0, 3, 0, :3] = 5000.0  # SWIR-like outliers, clipped to 1.5
    props = _props(rng, w)
    labels = (props >= 0.1).astype(np.int8)
    labels[~acquired] = -1
    props[~acquired] = np.nan
    pw = np.nanmean(props, 0).astype(np.float32)
    return {
        "window_id": 1000 + w, "fold": fold, "zone": "none", "block10km": 10 * fold + w % 2,
        "lat": 36.0 + w, "lon": -120.0, "x_min": 600.0 * w, "y_max": 0.0, "cube": cube,
        "dates": dates, "acquired": acquired.tolist(), "qc_ok": qc.tolist(),
        "flight_ids": [f"f{w}t{k}" if acquired[k] else "" for k in range(T)],
        "nan_frac": [float(np.isnan(cube[k]).any(0).mean()) for k in range(T)],
        "n_ok_dates": int(qc.sum()), "labels_date": labels, "props_date": props,
        "labels_window": (pw >= 0.1).astype(np.int8).tolist(), "props_window": pw.tolist(),
    }


def _table(rows):
    cols = {}
    for c in rows[0]:
        if c == "cube":
            cols[c] = _nested(np.stack([r[c] for r in rows]), (T, B, H, H))
        elif c in ("labels_date", "props_date"):
            cols[c] = _nested(np.stack([r[c] for r in rows]), (T, K))
        else:
            cols[c] = pa.array([r[c] for r in rows])
    return pa.table(cols)


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    (root / "data").mkdir(parents=True)
    rng = np.random.default_rng(0)
    kept = np.r_[1:5, 6:10]
    pd.DataFrame({"band": np.arange(B_ALL), "wavelength_nm": 400 + 150 * np.arange(B_ALL),
                  "fwhm_nm": 10.0, "kept": np.isin(np.arange(B_ALL), kept).astype(int),
                  "release_index": [list(kept).index(b) if b in kept else -1
                                    for b in range(B_ALL)]}).to_csv(root / "bands.csv",
                                                                     index=False)
    w, cdl = 0, []
    for fold in range(5):
        rows = []
        for _ in range(4):
            rows.append(_window(rng, w, fold))
            w += 1
        for r in rows:  # exact proportions per (window, year) of the acquired dates
            for k, d in enumerate(r["dates"]):
                if d:
                    cdl.append({"window_id": r["window_id"], "year": int(d[:4]),
                                **{f"prop_{c}": float(v) for c, v in
                                   zip(A.CLASSES, r["props_date"][k], strict=True)}})
        for i, part in enumerate((rows[:2], rows[2:])):
            pq.write_table(_table(part), root / "data" / f"fold_{fold}-0000{i}-of-00002.parquet",
                           row_group_size=1)
    (root / "metadata").mkdir()
    pd.DataFrame(cdl).to_parquet(root / A.CDL_BY_YEAR)
    return root


KW = {"verify": None, "verbose": False}


def _ds(release, tmp_path, **kw):
    return A.AVIRISCAMultilabelDataset(release, cache_dir=tmp_path / "cache", **{**KW, **kw})


def test_fetch_counts_and_ledoit_wolf(release, tmp_path):
    b = A.fetch_aviris_ca_ml(release, cache_dir=tmp_path / "cache", verify=None)
    assert b.covariances.shape[1:] == (B, B) and len(b.windows) == 20
    # 60 window-dates, 1 padded, 1 failing QC -> 58 eligible; 1 with 12 valid pixels
    assert b.info["n_acquired"] == 59 and b.info["n_eligible"] == 58
    assert b.info["excluded"]["n_lt_min_pixels"] == 1 and b.ok.sum() == 57
    assert list(b.wavelengths) == [400 + 150 * k for k in (1, 2, 3, 4, 6, 7, 8, 9)]
    # the covariance of window 3, date 0 is Ledoit-Wolf on the clipped pixels
    row = b.dates[(b.dates.window_id == 1003) & (b.dates.t == 0)].iloc[0]
    f = pq.ParquetFile(release / "data" / "fold_0-00001-of-00002.parquet")
    cube = A.decode_cubes(f.read_row_group(1, columns=["cube"]).column("cube"), T, B)[0]
    X = np.clip(cube[0].astype(np.float64).reshape(B, -1).T, *A.CLIP)
    assert X.max() == 1.5
    np.testing.assert_allclose(b.covariances[int(row.cov_index)],
                               LedoitWolf().fit(X).covariance_, rtol=1e-10)
    np.testing.assert_allclose(b.logdet[int(row.cov_index)],
                               np.linalg.slogdet(b.covariances[int(row.cov_index)])[1])


def test_date_mode_multilabel_and_folds(release, tmp_path):
    ds = _ds(release, tmp_path)
    assert len(ds) == 57 and ds.num_classes == K and ds.estimator == "ledoit_wolf"
    x, y = ds[0]
    assert x.dtype == torch.float64 and x.shape == (B, B)
    assert y.dtype == torch.float32 and y.shape == (K,) and set(y.tolist()) <= {0.0, 1.0}
    # per-date labels: those of the acquisition year
    m = ds.meta.iloc[0]
    D = ds.fetched.dates
    ref = D[(D.window_id == m.window_id) & (D.t == m.t)][[f"label_{c}" for c in A.CLASSES]]
    np.testing.assert_array_equal(y.numpy(), ref.to_numpy()[0])
    sp = ds.splits["recommended"]
    folds = ds.meta["fold"].to_numpy()
    assert set(folds[sp["train"]]) == {1, 2, 3} and set(folds[sp["val"]]) == {0}
    assert set(folds[sp["test"]]) == {4}
    for k in range(5):
        s = ds.splits[f"fold{k}"]
        assert set(folds[s["test"]]) == {k} and set(folds[s["val"]]) == {(k + 1) % 5}
        assert_no_leakage(ds.groups[s["train"]], ds.groups[s["test"]])
        assert_no_leakage(ds.groups[s["train"]], ds.groups[s["val"]])
    # normalisation: geometric mean of the training eigenvalues is 1
    train = np.stack([ds[i][0].numpy() for i in sp["train"]])
    assert abs(np.mean(np.log(np.linalg.eigvalsh(train)))) < 1e-10


def test_window_labels_regression_and_fine(release, tmp_path):
    win = _ds(release, tmp_path, label_level="window", normalize=False)
    W = win.fetched.windows.set_index("window_id")
    m = win.meta.iloc[5]
    np.testing.assert_array_equal(
        win[5][1].numpy(), W.loc[m.window_id, [f"label_{c}" for c in A.CLASSES]].to_numpy())
    reg = _ds(release, tmp_path, task="regression")
    assert reg[0][1].dtype == torch.float64 and abs(float(reg[0][1].sum()) - 1) < 1e-5
    fine = _ds(release, tmp_path, task="fine")
    _, c = fine[0]
    assert isinstance(c, int)
    assert fine.classes[c] == A.CLASSES[int(np.argmax(reg[0][1].numpy()))]


def test_series_mode(release, tmp_path):
    ds = _ds(release, tmp_path, mode="series", return_mean=True, return_raw=True)
    assert len(ds) == 20
    i = int(np.flatnonzero(ds.meta.window_id == 1000)[0])
    (tokens, mask, mean, raw), y = ds[i]
    assert tokens.shape == (T, B, B) and mask.tolist() == [True, True, False]
    np.testing.assert_array_equal(tokens[2].numpy(), np.eye(B))
    assert mean.shape == (T, B) and float(mean[2].abs().sum()) == 0
    assert raw.shape == (T, H, H, B) and torch.isnan(raw[2]).all()
    assert y.shape == (K,)
    j = int(np.flatnonzero(ds.meta.window_id == 1001)[0])
    assert ds[j][0][1].tolist() == [True, False, True]  # QC failure masked
    two = _ds(release, tmp_path, mode="series", min_dates=3)
    assert len(two) == 17
    with pytest.raises(ValueError, match="window labels"):
        _ds(release, tmp_path, mode="series", label_level="date")


def test_raw_and_mean_follow_the_scale(release, tmp_path):
    ds = _ds(release, tmp_path, return_mean=True, return_raw=True)
    (cov, mean, raw), _ = ds[3]
    X = raw.numpy().reshape(-1, B).astype(np.float64)
    X = X[np.isfinite(X).all(1)]
    np.testing.assert_allclose(cov.numpy(), LedoitWolf().fit(X).covariance_, rtol=1e-3)
    np.testing.assert_allclose(mean.numpy(), X.mean(0), rtol=1e-4)
    assert ds.scale != 1.0


def test_classes_balance_and_task_errors(release, tmp_path):
    ds = _ds(release, tmp_path, classes=["water", "almond"], balance="weights")
    assert ds.classes == ["almond", "water"] and ds[0][1].shape == (2,)
    tr = ds.splits["recommended"]["train"]
    np.testing.assert_allclose(ds.class_weights, pos_weights(ds.labels[tr]))
    with pytest.raises(ValueError, match="balance"):
        _ds(release, tmp_path, balance="undersample")
    with pytest.raises(ValueError, match="multilabel"):
        _ds(release, tmp_path, drop_classes=("water",))
    with pytest.raises(ValueError, match="mode"):
        _ds(release, tmp_path, mode="window")


def test_random_split_warns_and_subset(release, tmp_path):
    with pytest.warns(UserWarning, match="random"):
        ds = _ds(release, tmp_path, split="random")
    assert ds.split_leaky and not ds.random_split_stratified
    test = _ds(release, tmp_path, subset="test")
    assert set(test.meta["fold"]) == {4} and test.splits == {}


def test_folds_restrict_the_shards(release, tmp_path):
    # fold1: test 1, train 0 (val = fold 2, not loaded); fold0 would have no training part
    ds = _ds(release, tmp_path, folds=[0, 1], split="fold1")
    assert set(ds.meta["fold"]) == {0, 1} and set(ds.splits) == {"fold1", "random"}
    assert ds.fetched.shards == A.shard_paths([0, 1])
    with pytest.raises(ValueError, match="Unknown split"):
        _ds(release, tmp_path, folds=[0, 1])


def test_local_copy_is_checked_against_the_pinned_sizes(release, tmp_path):
    with pytest.raises(OSError, match="expected"):
        A.locate_release(release)
    assert A.locate_release(tmp_path / "nowhere", archive=release, verify=None) == release


def test_download_with_token_into_the_cache(release, tmp_path, monkeypatch):
    import hashlib

    from spdnet_datasets.core import access

    files = {}
    for rel in ["bands.csv", A.CDL_BY_YEAR] + A.shard_paths():
        data = (release / rel).read_bytes()
        files[rel] = (len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(A, "HF_FILES", files)
    calls = []

    def fake_hf_download(repo, filename, dest_dir, *, revision, gated, checksum, size,
                         interactive=None, **kw):
        assert repo == A.HF_REPO and revision == A.HF_REVISION and gated
        calls.append(filename)
        out = dest_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((release / filename).read_bytes())
        assert (size, checksum) == files[filename]
        return out

    monkeypatch.setattr(access, "hf_download", fake_hf_download)
    cache = tmp_path / "cache"
    ds = A.AVIRISCAMultilabelDataset(tmp_path / "absent", cache_dir=cache, verbose=False)
    assert len(ds) == 57 and len(calls) == 12
    assert ds.data_dir == cache / "aviris_ca_ml" / "downloads" / A.HF_REVISION
    A.AVIRISCAMultilabelDataset(tmp_path / "absent", cache_dir=cache, verbose=False)
    assert len(calls) == 12  # nothing downloaded again


def test_registry_and_dataloaders(release, tmp_path):
    ds = DatasetManager.create_dataset({"name": "aviris_ca_ml", "data_dir": str(release),
                                        "cache_dir": str(tmp_path / "cache"), **KW})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tr, _, te = create_dataloaders(ds, batch_size=4, num_workers=0, split="recommended",
                                       verbose=False)
    xb, yb = next(iter(tr))
    assert xb.shape == (4, B, B) and yb.shape == (4, K) and yb.dtype == torch.float32


def test_pos_weights():
    Y = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(pos_weights(Y), [1 / 3, 1.0, 0.0])
    with pytest.raises(ValueError):
        pos_weights(np.zeros(3))


# --------------------------------------------------------------- learning scenarios
KIND = {1000 + w: w % 5 for w in range(20)}


def _kinds(ds):
    return {KIND[int(w)] for w in ds.meta["window_id"]}


def test_window_change_statistics(release, tmp_path):
    ds = _ds(release, tmp_path)
    st = ds.window_stats
    assert st.loc[1000, "change"] == 0 and st.loc[1000, "n_years"] == 2  # padded date
    assert st.loc[1001, "change"] == pytest.approx(0.15, abs=1e-6)  # QC-failed 2014 unused
    assert st.loc[1002, "change"] == pytest.approx(0.4, abs=1e-6)  # sparse 2013 unused
    assert st.loc[1003, "change"] == pytest.approx(0.07, abs=1e-6)
    assert (st.loc[[1004, 1009, 1014, 1019], "change"] > 0.10).all()
    assert st.loc[1000, "pure_class"] == "water" and st.loc[1006, "pure_class"] == "almond"
    assert st.loc[1003, "pure_class"] == ""
    assert "change" in ds.meta and np.isfinite(ds.meta["change"]).all()


@pytest.mark.parametrize("mode", ["date", "series"])
def test_pure_scenario(release, tmp_path, mode):
    ds = _ds(release, tmp_path, mode=mode, scenario="pure", min_class_windows=3)
    assert ds.task == "fine" and ds.classes == ["almond", "water"]
    assert _kinds(ds) == {0, 1} and len(np.unique(ds.window_ids)) == 8
    assert len(ds) == (22 if mode == "date" else 8)
    for i in range(len(ds)):
        want = "water" if KIND[int(ds.meta.window_id[i])] == 0 else "almond"
        assert ds.classes[ds[i][1]] == want
    # strict: almond windows change by 0.15 -> dropped; too few windows -> class dropped
    strict = _ds(release, tmp_path, mode=mode, scenario="pure", min_class_windows=3,
                 pure_max_change=0.05)
    assert strict.classes == ["water"] and _kinds(strict) == {0}
    with pytest.raises(ValueError, match="keeps no sample"):
        _ds(release, tmp_path, mode=mode, scenario="pure", min_class_windows=5)
    hi = _ds(release, tmp_path, mode=mode, scenario="pure", min_class_windows=3, purity=0.85)
    assert hi.classes == ["water"]  # almond is 0.80 in its first year


def test_stable_scenario(release, tmp_path):
    reg = _ds(release, tmp_path, scenario="stable")
    assert reg.task == "regression" and _kinds(reg) == {0} and len(reg) == 11
    W = reg.fetched.windows.set_index("window_id")
    m = reg.meta.iloc[0]
    np.testing.assert_allclose(reg[0][1].numpy(),
                               W.loc[m.window_id, [f"prop_{c}" for c in A.CLASSES]].to_numpy(
                                   np.float64))
    ml = _ds(release, tmp_path, scenario="stable", task="multilabel", mode="series")
    assert len(ml) == 4 and ml[0][1].shape == (K,)
    wider = _ds(release, tmp_path, scenario="stable", stable_max_change=0.08,
                changing_min_change=0.1)
    assert _kinds(wider) == {0, 3}
    # splits and normalisation on the kept samples
    folds = reg.meta["fold"].to_numpy()
    sp = reg.splits["recommended"]
    assert set(folds[sp["test"]]) <= {4} and len(np.concatenate(list(sp.values()))) == len(reg)
    train = np.stack([reg[i][0].numpy() for i in sp["train"]])
    assert abs(np.mean(np.log(np.linalg.eigvalsh(train)))) < 1e-10


def test_spacetime_scenario(release, tmp_path):
    d = _ds(release, tmp_path, scenario="spacetime")
    assert len(d) == 57 and d.task == "regression" and d.label_level == "date"
    D = d.fetched.dates
    m = d.meta.iloc[4]
    ref = D[(D.window_id == m.window_id) & (D.t == m.t)][[f"prop_{c}" for c in A.CLASSES]]
    np.testing.assert_allclose(d[4][1].numpy(), ref.to_numpy(np.float64)[0])
    s = _ds(release, tmp_path, scenario="spacetime", mode="series", task="multilabel")
    assert len(s) == 20 and s.label_level == "window"


@pytest.mark.parametrize("mode", ["date", "series"])
def test_stability_scenario(release, tmp_path, mode):
    ds = _ds(release, tmp_path, mode=mode, scenario="stability")
    assert ds.classes == ["stable", "changing"]  # stable = 0, changing = 1
    assert _kinds(ds) == {0, 1, 2, 4}  # 0.07 (kind 3) is in the excluded band
    for i in range(len(ds)):
        assert ds[i][1] == (0 if KIND[int(ds.meta.window_id[i])] == 0 else 1)
    reg = _ds(release, tmp_path, mode=mode, scenario="stability", task="regression")
    assert len(reg) == (57 if mode == "date" else 20)
    y = reg[0][1]
    assert y.ndim == 0 and float(y) == pytest.approx(reg.meta["change"][0])


def test_scenario_errors(release, tmp_path):
    with pytest.raises(ValueError, match="scenario must be"):
        _ds(release, tmp_path, scenario="change_type")
    with pytest.raises(ValueError, match="not available"):
        _ds(release, tmp_path, scenario="pure", task="regression")
    with pytest.raises(ValueError, match="not available"):
        _ds(release, tmp_path, scenario="stable", task="fine")
    with pytest.raises(ValueError, match="not available"):
        _ds(release, tmp_path, scenario="stability", task="multilabel")
    with pytest.raises(ValueError, match="label_level"):
        _ds(release, tmp_path, scenario="stable", label_level="date")
    with pytest.raises(ValueError, match="stable_max_change"):
        _ds(release, tmp_path, scenario="stability", stable_max_change=0.2)
    # no scenario: unchanged behaviour (multilabel per date, every sample)
    ds = _ds(release, tmp_path)
    assert ds.scenario is None and ds.task == "multilabel" and len(ds) == 57
