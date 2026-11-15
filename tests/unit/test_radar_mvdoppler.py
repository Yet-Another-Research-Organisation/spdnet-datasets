"""MVDoppler on a tiny synthetic copy (.h5 snapshots and design_table.csv under tmp_path)."""

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from spdnet_datasets import DatasetManager
from spdnet_datasets.core import assert_no_leakage, check_spd
from spdnet_datasets.radar_microdoppler.mvdoppler import (
    CLASSES,
    MVDopplerDataset,
    fetch_mvdoppler,
    snapshot_features,
)

FOLD_SUBJECTS = {0: [4, 6], 1: [2, 8], 2: [1, 7], 3: [0, 3]}


def make_mvdoppler(root):
    rng = np.random.default_rng(0)
    (root / "Data").mkdir(parents=True)
    (root / "Labels_and_metadata").mkdir()
    rows = []
    for fold, subjects in FOLD_SUBJECTS.items():
        for s in subjects:
            for c, pattern in enumerate(CLASSES):
                for rep in range(2):
                    name = f"{pattern}_{len(rows):05d}"
                    x = np.exp(rng.standard_normal((128, 256, 2)) * (0.5 + 0.1 * c))
                    with h5py.File(root / "Data" / f"{name}.h5", "w") as f:
                        f["radar_dat"] = x
                    rows.append({"ID": len(rows), "exp_fname": f"{s}_{c}_{rep}.npz", "fname": name,
                                 "pattern": pattern, "subject": s, "fold": fold,
                                 "val_set": "TRUE" if rep == 1 and s == subjects[0] else "FALSE"})
    (root / "Data" / f"{rows[0]['fname']}.h5").write_bytes(b"\0" * 1024)  # zero-filled copy
    with h5py.File(root / "Data" / "normal_99999.h5", "w") as f:  # not in the design table
        f["radar_dat"] = np.ones((128, 256, 2))
    pd.DataFrame(rows).to_csv(root / "Labels_and_metadata" / "design_table.csv")
    return root, len(rows)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_mvdoppler(tmp_path_factory.mktemp("mv") / "MVDoppler")


def test_two_views_are_pooled_and_stacked():
    x = np.ones((128, 256, 2))
    x[:, :, 1] = np.e
    x[:4, :, 0] = np.e ** 2
    X = snapshot_features(x)
    assert X.shape == (256, 64)
    assert np.allclose(X[:, 0], 2) and np.allclose(X[:, 1:32], 0) and np.allclose(X[:, 32:], 1)


def test_fetch_counts_and_official_folds(data, tmp_path):
    root, n_rows = data
    b = fetch_mvdoppler(root, tmp_path, verify=None)
    assert b.info["unreadable"] == 1 and b.info["h5_not_in_design_table"] == 1
    assert b.covariances.shape == (n_rows - 1, 64, 64)
    check_spd(b.covariances)
    assert (b.meta.n_obs == 256).all()
    for k in range(4):
        s = b.splits[f"fold{k}"]
        assert_no_leakage(b.groups[s["train"]], b.groups[s["test"]])
        assert_no_leakage(b.meta.exp_fname[s["val"]], b.meta.exp_fname[s["train"]])
        assert set(b.meta.fold[s["test"]]) == {k} and b.meta.val_set[s["val"]].all()
        assert len(s["train"]) + len(s["val"]) + len(s["test"]) == len(b.labels)
    assert b.splits["official"] is b.splits["fold0"]


def test_limit(data, tmp_path):
    b = fetch_mvdoppler(data[0], tmp_path, verify=None, limit=1)
    assert len(b.labels) == 4 * len(CLASSES)


def test_dataset(data, tmp_path):
    root, _ = data
    kw = {"cache_dir": tmp_path, "verify": None, "verbose": False}
    ds = MVDopplerDataset(root, fold=2, **kw)
    assert DatasetManager.DATASETS["mvdoppler"] is MVDopplerDataset
    x, y = ds[0]
    assert x.shape == (64, 64) and x.dtype == torch.float64 and 0 <= y < 4
    (cov, mean, raw), _ = MVDopplerDataset(root, return_mean=True, return_raw=True, **kw)[0]
    assert mean.shape == (64,) and raw.shape == (256, 64)
    train = ds.splits["official"]["train"]
    eig = np.linalg.eigvalsh(np.stack([ds[i][0].numpy() for i in train]))
    assert np.exp(np.mean(np.log(eig))) == pytest.approx(1.0, rel=1e-8)
    assert set(ds.meta.fold[ds.splits["official"]["test"]]) == {2}
    assert MVDopplerDataset(root, classes=["texting"], **kw).classes == ["texting"]
    assert len(MVDopplerDataset(root, max_classes=3, **kw).classes) == 3
    assert np.bincount(MVDopplerDataset(root, max_samples_per_class=5, **kw).labels).tolist() == [5] * 4
    lw = MVDopplerDataset(root, estimator="ledoit_wolf", **kw)
    assert lw._bunch.info["estimator"] == "ledoit_wolf"
