"""ESC-50, GTZAN and BirdSet XCM loaders on tiny synthetic parquet copies."""

import io
import json
import warnings

import numpy as np
import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
sf = pytest.importorskip("soundfile")

from spdnet_datasets.audio import (  # noqa: E402
    BirdSetXCMDataset,
    ESC50Dataset,
    GTZANDataset,
    fetch_birdset_xcm,
    fetch_esc50,
    fetch_gtzan,
)
from spdnet_datasets.audio._features import frame_features  # noqa: E402
from spdnet_datasets.core import assert_no_leakage  # noqa: E402
from spdnet_datasets.manager import DatasetManager  # noqa: E402

SR = 8000  # n_fft = 512, hop = 128: a 1 s clip has 59 frames >= 2p = 40


def _wav(seed, silent=False, seconds=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(int(SR * seconds)) if silent else rng.standard_normal(int(SR * seconds))
    if not silent:  # class-independent colouring so that the MFCC are not degenerate
        x = np.convolve(x, rng.random(8), mode="same") * 0.1
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), SR, format="WAV")
    return buf.getvalue()


def _audio(blob, name):
    return {"bytes": blob, "path": name}


@pytest.fixture
def esc50_dir(tmp_path):
    """5 folds x 4 classes x 2 clips; clip 0 of fold 1 is silent (excluded)."""
    root = tmp_path / "ESC-50"
    root.mkdir()
    rows = []
    for fold in range(1, 6):
        for target in (0, 1, 10, 41):  # categories in 3 major groups
            for take in "AB":
                src = fold * 100 + target  # one source recording per (fold, class)
                name = f"{fold}-{src}-{take}-{target}.wav"
                silent = fold == 1 and target == 0 and take == "A"
                rows.append({"filename": name, "fold": fold, "target": target,
                             "category": f"cat{target:02d}", "esc10": target == 0,
                             "src_file": src, "take": take,
                             "audio": _audio(_wav(len(rows), silent), name)})
    half = len(rows) // 2
    pq.write_table(pa.Table.from_pylist(rows[:half]), root / "esc50_0.parquet")
    pq.write_table(pa.Table.from_pylist(rows[half:]), root / "esc50_1.parquet")
    return root


def _esc(esc50_dir, tmp_path, **kw):
    return ESC50Dataset(esc50_dir, cache_dir=tmp_path / "cache", verbose=False, **kw)


def test_frame_features_drop_silence():
    x = np.random.default_rng(0).standard_normal(SR)
    x[: SR // 2] = 0
    full = frame_features(np.random.default_rng(0).standard_normal(SR), SR)
    part = frame_features(x, SR)
    assert full.shape == (59, 20) and part.shape[1] == 20 and len(part) < 35
    assert frame_features(x, SR, "logmel", n_mels=32).shape[1] == 32


def test_esc50_fetch_shapes_and_exclusion(esc50_dir, tmp_path):
    b = fetch_esc50(esc50_dir, cache_dir=tmp_path / "cache")
    assert b.covariances.shape == (39, 20, 20) and b.means.shape == (39, 20)
    assert (b.meta["n_obs"] >= 40).all()
    assert len(b.excluded) == 1 and b.excluded.reason.iloc[0].startswith("n < 2p")
    eig = np.linalg.eigvalsh(b.covariances)
    assert (eig > 0).all()
    assert set(b.splits) == {"official", "recommended"}


def test_esc50_outputs_and_normalisation(esc50_dir, tmp_path):
    ds = _esc(esc50_dir, tmp_path, subset="train")
    x, y = ds[0]
    assert x.shape == (20, 20) and x.dtype == torch.float64 and isinstance(y, int)
    train = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    log_eig = np.log(np.linalg.eigvalsh(train))
    assert abs(log_eig.mean()) < 1e-8  # geometric mean of the train eigenvalues = 1

    ds = _esc(esc50_dir, tmp_path, return_mean=True, return_raw=True)
    (cov, mean, raw), _ = ds[0]
    assert mean.shape == (20,) and raw.dtype == torch.float32 and raw.shape[1] == 20
    assert raw.shape[0] >= 40
    raw64 = raw.double().numpy()
    np.testing.assert_allclose(np.cov(raw64.T, bias=True), cov.numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(raw64.mean(0), mean.numpy(), rtol=1e-4, atol=1e-6)


def test_esc50_manager_official_split_by_default(esc50_dir, tmp_path):
    train, val, test, n = DatasetManager.create_dataloaders({
        "name": "esc50", "path": str(esc50_dir), "cache_dir": tmp_path / "cache",
        "split_strategy": "official", "batch_size": 4, "num_workers": 0,
        "persistent_workers": False, "pin_memory": False, "verbose": False})
    assert n == 4 and len(train.dataset) + len(val.dataset) + len(test.dataset) == 39


def test_esc50_official_folds_without_leakage(esc50_dir, tmp_path):
    ds = _esc(esc50_dir, tmp_path, subset=None, fold=3)
    parts = ds.splits["official"]
    folds = ds.meta["fold"].to_numpy()
    assert set(folds[parts["test"]]) == {3} and set(folds[parts["val"]]) == {4}
    assert set(folds[parts["train"]]) == {1, 2, 5}
    assert len(parts["train"]) + len(parts["val"]) + len(parts["test"]) == len(ds)
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        assert_no_leakage(ds.groups[parts[a]], ds.groups[parts[b]])
    assert len(_esc(esc50_dir, tmp_path, subset="test", fold=3)) == len(parts["test"])


def test_subset_attributes_are_aligned(esc50_dir, tmp_path):
    from spdnet_datasets.base import create_dataloaders

    ds = _esc(esc50_dir, tmp_path, subset="train", fold=2)
    assert len(ds.groups) == len(ds.labels) == len(ds.meta) == len(ds) == 23
    assert set(ds.meta["fold"]) == {1, 4, 5}
    assert [ds[i][1] for i in range(len(ds))] == ds.labels.tolist()
    assert ds.splits == {} and set(ds.full_splits) == {"official", "recommended"}
    with pytest.raises(ValueError, match="Unknown split"):
        create_dataloaders(ds, num_workers=0, split="recommended", verbose=False)


def test_esc50_tasks(esc50_dir, tmp_path):
    coarse = _esc(esc50_dir, tmp_path, task="coarse", subset=None)
    assert coarse.classes == ["animals", "natural_soundscapes", "urban"]
    dropped = _esc(esc50_dir, tmp_path, drop_classes=["cat41"], subset=None)
    assert "cat41" not in dropped.classes and len(dropped) == 29
    merged = _esc(esc50_dir, tmp_path, merge_classes={"cat01": "cat00"}, subset=None)
    assert merged.classes == ["cat00", "cat10", "cat41"]
    limited = _esc(esc50_dir, tmp_path, classes=["cat01", "cat10"], max_samples_per_class=3,
                   subset=None, normalize=False)
    assert limited.classes == ["cat01", "cat10"] and len(limited) == 6
    assert sorted({limited[i][1] for i in range(len(limited))}) == [0, 1]


def test_esc50_balance_and_errors(esc50_dir, tmp_path):
    under = _esc(esc50_dir, tmp_path, subset="train", fold=2, balance="undersample")
    counts = np.bincount([under[i][1] for i in range(len(under))])
    assert counts.tolist() == [5, 5, 5, 5]  # cat00 lost its silent clip (fold 1)
    weights = _esc(esc50_dir, tmp_path, balance="weights")
    assert weights.class_weights is not None and len(weights.class_weights) == 4
    with pytest.raises(ValueError, match="keep"):
        _esc(esc50_dir, tmp_path, return_raw=True, keep="keep_covariance")
    with pytest.raises(ValueError, match="Unknown split"):
        _esc(esc50_dir, tmp_path, split="nope")


def test_esc50_ledoit_wolf_and_registry(esc50_dir, tmp_path):
    ds = _esc(esc50_dir, tmp_path, estimator="ledoit_wolf")
    assert ds[0][0].shape == (20, 20)
    assert DatasetManager.DATASETS["esc50"] is ESC50Dataset


@pytest.fixture
def gtzan_dir(tmp_path):
    """10 genres x 6 excerpts, including two names of the duplicate list."""
    root = tmp_path / "GTZAN"
    root.mkdir()
    rows = []
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop",
              "reggae", "rock"]
    for g, genre in enumerate(genres):
        for k in [54, 60, 61, 62, 63, 64] if genre == "pop" else range(6):
            name = f"{genre}.{k:05d}.wav"
            rows.append({"file": f"/x/genres/{genre}/{name}", "audio": _audio(_wav(len(rows)), name),
                         "genre": g})
    pq.write_table(pa.Table.from_pylist(rows), root / "gtzan_0.parquet")
    return root


def test_gtzan_duplicates_and_warning(gtzan_dir, tmp_path):
    b = fetch_gtzan(gtzan_dir, cache_dir=tmp_path / "cache")
    assert "pop.00060.wav" not in set(b.meta["id"]) and "pop.00054.wav" in set(b.meta["id"])
    assert len(b.labels) == 59
    with pytest.warns(UserWarning, match="artist"):
        ds = GTZANDataset(gtzan_dir, cache_dir=tmp_path / "cache", subset=None, verbose=False)
    parts = ds.splits["recommended"]
    assert sorted(np.concatenate(list(parts.values())).tolist()) == list(range(len(ds)))
    assert_no_leakage(ds.groups[parts["train"]], ds.groups[parts["test"]])


@pytest.fixture
def birdset_dir(tmp_path):
    """3 species x 12 recordings over 6 recordists + 1 rare species (3 recordings)."""
    root = tmp_path / "BirdSet_XCM"
    root.mkdir()
    rows = []
    for code, n in ((0, 12), (1, 12), (2, 12), (3, 3)):
        for k in range(n):
            rows.append({"audio": _audio(_wav(len(rows), seconds=1.5), "x.ogg"),
                         "filepath": f"/x/XC{code}{k:03d}.ogg", "ebird_code": code,
                         "recordist": "" if k == 0 else f"rec{(code + k) % 6}", "length": 1,
                         "detected_events": [[0.2, 0.6]]})
    table = pa.Table.from_pylist(rows)
    names = {"info": {"features": {"ebird_code": {"names": ["sp_a", "sp_b", "sp_c", "sp_d"]}}}}
    table = table.replace_schema_metadata({b"huggingface": json.dumps(names).encode()})
    pq.write_table(table, root / "xcm_00000.parquet")
    return root


def test_birdset_species_filter_groups_and_crop(birdset_dir, tmp_path):
    b = fetch_birdset_xcm(birdset_dir, cache_dir=tmp_path / "cache")
    assert sorted(set(b.labels)) == ["sp_a", "sp_b", "sp_c"] and len(b.labels) == 36
    assert "NA" in set(b.groups)
    cropped = fetch_birdset_xcm(birdset_dir, cache_dir=tmp_path / "cache", crop_s=1.0)
    assert cropped.meta["n_obs"].max() < b.meta["n_obs"].min()
    with pytest.warns(UserWarning, match="partial"):
        ds = BirdSetXCMDataset(birdset_dir, cache_dir=tmp_path / "cache", subset=None,
                               verbose=False)
    parts = ds.splits["group_kfold"]
    for a, c in (("train", "test"), ("train", "val"), ("val", "test")):
        assert_no_leakage(ds.groups[parts[a]], ds.groups[parts[c]])


def test_limit_is_part_of_the_cache_key(esc50_dir, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small = fetch_esc50(esc50_dir, cache_dir=tmp_path / "cache", limit=10)
    full = fetch_esc50(esc50_dir, cache_dir=tmp_path / "cache")
    assert len(small.labels) + len(small.excluded) == 10 and len(full.labels) == 39
