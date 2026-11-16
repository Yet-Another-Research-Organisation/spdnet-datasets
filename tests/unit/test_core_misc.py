"""core.config, core.normalize, core.outputs, core.tasks, core.splits."""

from pathlib import Path

import numpy as np
import pytest

from spdnet_datasets.core import (
    apply_scale,
    apply_task,
    assert_no_duplicates,
    assert_no_leakage,
    assert_no_window_overlap,
    balance_indices,
    class_weights,
    dataset_path,
    geometric_scale,
    get_cache_dir,
    get_data_root,
    group_split,
    pack_outputs,
    spatial_block_split,
)

# --- config -----------------------------------------------------------------


def test_config_defaults_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SPDNET_DATA_ROOT", raising=False)
    monkeypatch.delenv("SPDNET_CACHE_DIR", raising=False)
    assert get_data_root() == Path("/DATA/DATASET")
    assert get_cache_dir() == Path("~/.cache/spdnet_datasets").expanduser()
    monkeypatch.setenv("SPDNET_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("SPDNET_CACHE_DIR", str(tmp_path / "cache"))
    assert dataset_path("SAR", "MSTAR_SOC") == tmp_path / "root" / "SAR" / "MSTAR_SOC"
    assert get_cache_dir() == tmp_path / "cache"
    assert dataset_path("SAR", "X", data_dir=tmp_path) == tmp_path


# --- normalize --------------------------------------------------------------


def test_geometric_scale_brings_geometric_mean_eigenvalue_to_one():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 5, 20))
    covs = np.einsum("bij,bkj->bik", A, A) / 20 * 37.0
    s = geometric_scale(covs)
    eig = np.linalg.eigvalsh(apply_scale(covs, s))
    assert np.exp(np.mean(np.log(eig))) == pytest.approx(1.0, abs=1e-10)


def test_geometric_scale_rejects_non_spd():
    with pytest.raises(ValueError):
        geometric_scale(np.diag([1.0, 0.0])[None])


# --- outputs ----------------------------------------------------------------


def test_pack_outputs_all_combinations():
    c, m, r = "cov", "mean", "raw"
    assert pack_outputs(c, m, r) == c
    assert pack_outputs(c, m, r, return_mean=True) == (c, m)
    assert pack_outputs(c, m, r, return_raw=True) == (c, r)
    assert pack_outputs(c, m, r, return_mean=True, return_raw=True) == (c, m, r)


def test_pack_outputs_raw_mean_only_datasets():
    out = ("raw", "mean")
    assert pack_outputs(None, "m", "r", return_raw=True, outputs=out) == "r"
    assert pack_outputs(None, "m", "r", return_mean=True, outputs=out) == "m"
    with pytest.raises(ValueError, match="exactly one"):
        pack_outputs(None, "m", "r", outputs=out)
    with pytest.raises(ValueError, match="exactly one"):
        pack_outputs(None, "m", "r", return_raw=True, return_mean=True, outputs=out)


def test_pack_outputs_rejects_unavailable_output():
    with pytest.raises(ValueError, match="return_raw"):
        pack_outputs("c", "m", None, return_raw=True, outputs=("cov", "mean"))


# --- tasks ------------------------------------------------------------------


def test_fine_task_reindexes_classes():
    y, keep, names = apply_task(["b", "a", "c", "a"])
    assert names == ["a", "b", "c"] and y.tolist() == [1, 0, 2, 0] and keep.all()


def test_drop_then_merge_classes():
    y, keep, names = apply_task([0, 1, 2, 3, 3], drop_classes=[0], merge_classes={3: 2})
    assert keep.tolist() == [False, True, True, True, True]
    assert names == [1, 2] and y.tolist() == [0, 1, 1, 1]


def test_coarse_task_requires_complete_map():
    y, _, names = apply_task(["wheat", "barley", "rapeseed"], task="coarse",
                             coarse_map={"wheat": "cereal", "barley": "cereal",
                                         "rapeseed": "oilseed"})
    assert names == ["cereal", "oilseed"] and y.tolist() == [0, 0, 1]
    with pytest.raises(ValueError, match="coarse_map"):
        apply_task(["x"], task="coarse")
    with pytest.raises(ValueError, match="no entry"):
        apply_task(["x", "y"], task="coarse", coarse_map={"x": 0})


def test_regression_keeps_targets():
    t = np.array([0.1, 0.7])
    y, keep, names = apply_task(t, task="regression")
    assert names is None and np.array_equal(y, t) and keep.all()
    with pytest.raises(ValueError):
        apply_task(t, task="regression", drop_classes=[0])


def test_balancing():
    y = np.array([0] * 6 + [1] * 2)
    under = balance_indices(y, "undersample")
    over = balance_indices(y, "oversample")
    assert np.bincount(y[under]).tolist() == [2, 2]
    assert np.bincount(y[over]).tolist() == [6, 6]
    assert balance_indices(y).tolist() == list(range(8))
    w = class_weights(y)
    assert w[1] / w[0] == pytest.approx(3.0)


def test_class_weights_keep_classes_absent_from_train():
    w = class_weights(np.array([0, 0, 2]), n_classes=4)
    assert len(w) == 4 and w[1] == w[3] == 0 and w[2] / w[0] == pytest.approx(2.0)


# --- splits -----------------------------------------------------------------


def test_group_split_never_shares_a_group():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 10)
    y = rng.integers(0, 3, size=40)[groups]
    parts = group_split(y, groups, val_size=0.1, test_size=0.2, seed=0)
    assert sum(len(v) for v in parts.values()) == len(y)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert_no_leakage(groups[parts[a]], groups[parts[b]])
    assert 0.1 < len(parts["test"]) / len(y) < 0.35


def test_spatial_block_split_buffer_removes_overlapping_windows():
    rr, cc = np.meshgrid(np.arange(0, 200, 8), np.arange(0, 200, 8), indexing="ij")
    coords = np.c_[rr.ravel(), cc.ravel()]
    parts = spatial_block_split(coords, block=50, window=16, n_splits=4, fold=0)
    w = 16
    boxes = np.c_[coords, coords + w]
    assert_no_window_overlap(boxes[parts["train"]], boxes[parts["test"]])
    assert_no_leakage(parts["groups"][parts["train"]], parts["groups"][parts["test"]])
    assert len(parts["test"]) > 0


def test_leakage_assertions_fire():
    with pytest.raises(AssertionError, match="shared"):
        assert_no_leakage([1, 2], [2, 3])
    with pytest.raises(AssertionError, match="overlaps"):
        assert_no_window_overlap([[0, 0, 10, 10]], [[5, 5, 15, 15]])
    assert_no_window_overlap([[0, 0, 10, 10]], [[10, 0, 20, 10]])  # touching is fine
    with pytest.raises(AssertionError, match="duplicate"):
        assert_no_duplicates(["h1"], ["h1"])
