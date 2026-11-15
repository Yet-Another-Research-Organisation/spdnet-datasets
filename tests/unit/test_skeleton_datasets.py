"""HDM05 and NTU RGB+D 120 loaders on tiny synthetic copies."""

import warnings

import numpy as np
import pytest
import torch

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.manager import DatasetManager
from spdnet_datasets.skeleton import (
    HDM05Dataset,
    NTURGBD120Dataset,
    fetch_hdm05,
    fetch_ntu120,
)
from spdnet_datasets.skeleton.hdm05 import c31
from spdnet_datasets.skeleton.ntu120 import TRAIN_SUBJECTS, ntu_index

pytestmark = pytest.mark.filterwarnings("ignore:HDM05 has no actor:UserWarning")


def _spd(rng, p):
    A = rng.standard_normal((p, 2 * p))
    return A @ A.T / (2 * p)


@pytest.fixture
def hdm05_dir(tmp_path):
    """4 classes x 6 sequences of 9 x 9 covariances (3 joints x 3 coordinates).

    Class 2 has two sequences of 15 frames < 2p = 18 (excluded for c93, kept for c31).
    """
    root = tmp_path / "HDM05"
    root.mkdir()
    rng = np.random.default_rng(0)
    seq = 0
    for cls in range(4):
        for k in range(6):
            frames = 15 if cls == 2 and k < 2 else 40
            np.save(root / f"{seq}_{frames}_{cls}.npy", _spd(rng, 9))
            seq += 1
    return root


def test_hdm05_fetch_c93_excludes_short_sequences(hdm05_dir, tmp_path):
    b = fetch_hdm05(hdm05_dir, cache_dir=tmp_path / "c")
    assert b.covariances.shape == (22, 9, 9)
    assert len(b.excluded) == 2 and b.excluded.reason.str.startswith("n < 2p").all()
    assert (b.meta["n_obs"] >= 18).all()
    b31 = fetch_hdm05(hdm05_dir, cache_dir=tmp_path / "c", features="c31")
    assert b31.covariances.shape == (24, 3, 3) and len(b31.excluded) == 0


def test_c31_sums_coordinate_blocks():
    C = np.arange(81, dtype=float).reshape(9, 9)
    np.testing.assert_array_equal(c31(C), C[:3, :3] + C[3:6, 3:6] + C[6:, 6:])


def test_hdm05_dataset_limits_and_outputs(hdm05_dir, tmp_path):
    kw = {"cache_dir": tmp_path / "c", "verbose": False}
    with pytest.warns(UserWarning, match="actor"):
        ds = HDM05Dataset(hdm05_dir, subset=None, **kw)
    assert ds.classes == ["000", "001", "002", "003"] and len(ds) == 22
    x, y = ds[0]
    assert x.shape == (9, 9) and x.dtype == torch.float64 and 0 <= y < 4
    assert len(HDM05Dataset(hdm05_dir, subset=None, classes=["001", "003"], **kw)) == 12
    two = HDM05Dataset(hdm05_dir, subset=None, max_classes=2, **kw)
    assert len(two.classes) == 2
    with pytest.warns(UserWarning, match="No split"):  # 3 per class < 5 folds
        capped = HDM05Dataset(hdm05_dir, subset=None, max_samples_per_class=3,
                              normalize=False, **kw)
    assert np.bincount(capped.labels).tolist() == [3, 3, 3, 3]
    with pytest.raises(ValueError, match="not available"):
        HDM05Dataset(hdm05_dir, return_mean=True, **kw)
    with pytest.raises(ValueError, match="scm"):
        HDM05Dataset(hdm05_dir, estimator="ledoit_wolf", **kw)
    assert DatasetManager.DATASETS["hdm05"] is HDM05Dataset


def test_hdm05_split_partitions_samples(hdm05_dir, tmp_path):
    ds = HDM05Dataset(hdm05_dir, subset=None, cache_dir=tmp_path / "c", verbose=False,
                      features="c31")
    parts = ds.splits["recommended"]
    idx = np.concatenate([parts[k] for k in ("train", "val", "test")])
    assert sorted(idx.tolist()) == list(range(len(ds)))
    assert_no_leakage(ds.groups[parts["train"]], ds.groups[parts["test"]])
    train = HDM05Dataset(hdm05_dir, cache_dir=tmp_path / "c", verbose=False, features="c31")
    log_eig = np.log(np.linalg.eigvalsh(np.stack([train[i][0].numpy()
                                                  for i in range(len(train))])))
    assert abs(log_eig.mean()) < 1e-8


def test_legacy_hdm05_shim(hdm05_dir, tmp_path):
    from spdnet_datasets.real.hdm05 import HDM05Dataset as Legacy

    with pytest.warns(DeprecationWarning):
        ds = Legacy(data_dir=hdm05_dir, cache_dir=tmp_path / "c", scaling_factor=2.0,
                    verbose=False)
    assert len(ds) == 22 and ds.scale == 2.0


def _record(rng, frames, bodies=1, lost=0):
    skel = rng.standard_normal((frames, 25, 3)) * 0.1 + np.linspace(0, 1, 25)[None, :, None]
    skel[:lost] = 0  # body lost on the first frames
    return {"file_name": "x", "nbodys": [bodies] * frames, "njoints": 25, "nframes": frames,
            **({"skel_body0": skel} if bodies else {})}


@pytest.fixture
def ntu_dir(tmp_path):
    """Actions A001 (daily), A050 (mutual), A103 (medical); setups 1-4; subjects 1-6."""
    root = tmp_path / "NTU_RGBD_120"
    rng = np.random.default_rng(0)
    for action in (1, 50, 103):
        d = root / "npy_skeletons_by_class" / f"A{action:03d}"
        d.mkdir(parents=True)
        for setup in (1, 2, 3, 4):
            for subject in (1, 2, 3, 4, 5, 6):
                name = f"S{setup:03d}C001P{subject:03d}R001A{action:03d}"
                if setup == 1 and subject == 3:
                    rec = _record(rng, 40, bodies=0)  # file without a body
                elif setup == 1 and subject == 6:
                    rec = _record(rng, 40, lost=15)  # 25 frames with a body < 30
                else:
                    rec = _record(rng, 40 + subject)
                np.save(d / f"{name}.npy", rec, allow_pickle=True)
    return root


def test_ntu_index_parses_names(ntu_dir):
    index = ntu_index(ntu_dir)
    assert len(index) == 72 and set(index["label"]) == {"A001", "A050", "A103"}
    row = index[index["id"] == "S002C001P005R001A050"].iloc[0]
    assert (row.setup, row.subject, row.camera, row.replication) == (2, 5, 1, 1)


def test_ntu_fetch_exclusions_and_features(ntu_dir, tmp_path):
    b = fetch_ntu120(ntu_dir, cache_dir=tmp_path / "c")
    assert b.covariances.shape == (66, 15, 15)
    reasons = b.excluded.reason.tolist()
    assert reasons.count("no body") == 3 and sum(r.startswith("n < 2p") for r in reasons) == 3
    assert (b.meta["n_obs"] >= 30).all()
    with pytest.raises(ValueError, match="features"):
        fetch_ntu120(ntu_dir, cache_dir=tmp_path / "c", features="bones")


def test_ntu_joints_need_150_frames(ntu_dir, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = fetch_ntu120(ntu_dir, cache_dir=tmp_path / "c", features="joints")
    assert len(b.labels) == 0  # every sequence is shorter than 2p = 150 frames
    assert len(b.excluded) == 72


def test_ntu_official_splits(ntu_dir, tmp_path):
    ds = NTURGBD120Dataset(ntu_dir, cache_dir=tmp_path / "c", subset=None, verbose=False)
    subject, setup = ds.meta["subject"].to_numpy(), ds.meta["setup"].to_numpy()
    xsub = ds.splits["xsub"]
    assert set(subject[xsub["test"]]) == {3, 6}  # subjects 3 and 6 are not training subjects
    assert set(subject[np.concatenate([xsub["train"], xsub["val"]])]) <= set(TRAIN_SUBJECTS)
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        assert_no_leakage(subject[xsub[a]], subject[xsub[b]])
    xsetup = ds.splits["xsetup"]
    assert set(setup[xsetup["test"]]) == {1, 3} and set(setup[xsetup["train"]]) <= {2, 4}
    assert_no_leakage(setup[xsetup["train"]], setup[xsetup["test"]])
    assert ds.splits["official"] is xsub


def test_ntu_dataset_outputs_and_coarse_task(ntu_dir, tmp_path):
    kw = {"cache_dir": tmp_path / "c", "verbose": False}
    ds = NTURGBD120Dataset(ntu_dir, return_mean=True, return_raw=True, **kw)
    (cov, mean, raw), y = ds[0]
    assert cov.shape == (15, 15) and mean.shape == (15,) and raw.shape[1] == 15
    np.testing.assert_allclose(np.cov(raw.double().numpy().T, bias=True), cov.numpy(),
                               rtol=1e-4, atol=1e-7)
    coarse = NTURGBD120Dataset(ntu_dir, task="coarse", subset=None, **kw)
    assert coarse.classes == ["daily", "medical", "mutual"]
    dropped = NTURGBD120Dataset(ntu_dir, drop_classes=["A050"], subset=None, **kw)
    assert dropped.classes == ["A001", "A103"]
    assert DatasetManager.DATASETS["ntu120"] is NTURGBD120Dataset
