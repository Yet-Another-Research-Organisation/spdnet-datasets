"""Cross-subject folds ``subject_fold_<k>`` (:func:`subject_fold_splits`)."""

import numpy as np
import pytest

from spdnet_datasets.core.normalize import geometric_scale
from spdnet_datasets.moabb import subject_fold_splits


def _trials(n_subjects=7, n_sessions=2, n_per_session=6, blocks=False):
    """Labels, subjects and sessions of each trial, in recording order.

    Two classes, alternating in time, or stored class by class with ``blocks``.
    """
    subjects = np.repeat(np.arange(1, n_subjects + 1), n_sessions * n_per_session)
    sessions = np.tile(np.repeat([f"{k}train" for k in range(n_sessions)], n_per_session),
                       n_subjects)
    half = n_per_session // 2
    one = np.repeat([0, 1], [half, n_per_session - half]) if blocks else \
        np.arange(n_per_session) % 2
    return np.tile(one, n_subjects * n_sessions), subjects, sessions


@pytest.mark.parametrize("n_folds", [3, 4, 7])
def test_each_subject_is_tested_once(n_folds):
    labels, subjects, sessions = _trials()
    folds = subject_fold_splits(labels, subjects, sessions, n_folds, seed=1)
    tested = [np.unique(subjects[f["test"]]) for f in folds.values()]
    assert len(folds) == n_folds
    assert sorted(np.concatenate(tested)) == list(range(1, 8))
    assert {len(t) for t in tested} <= {7 // n_folds, 7 // n_folds + 1}  # unequal allowed


@pytest.mark.parametrize("val_groups", [1, 2])
def test_train_val_and_test_subjects_are_disjoint_and_cover_all_trials(val_groups):
    labels, subjects, sessions = _trials()
    for f in subject_fold_splits(labels, subjects, sessions, 7, val_groups).values():
        parts = [f[k] for k in ("train", "val", "test", "calib_val", "calib")]
        merged = np.concatenate(parts)
        assert len(merged) == len(subjects) == len(np.unique(merged))
        train = set(subjects[f["train"]])
        val = set(subjects[np.concatenate([f["val"], f["calib_val"]])])
        test = set(subjects[np.concatenate([f["test"], f["calib"]])])
        assert len(val) == val_groups and len(test) == 1 and len(train) == 6 - val_groups
        assert not (train & val or train & test or val & test)


def test_validation_subjects_are_the_next_groups():
    labels, subjects, sessions = _trials()
    folds = list(subject_fold_splits(labels, subjects, sessions, 7, val_groups=2).values())
    for k, f in enumerate(folds):
        nxt = {subjects[folds[(k + i) % 7]["test"][0]] for i in (1, 2)}
        assert set(subjects[f["val"]]) == nxt


def test_calibration_is_the_start_of_each_session():
    labels, subjects, sessions = _trials(n_per_session=7)
    f = subject_fold_splits(labels, subjects, sessions, 3)["subject_fold_0"]
    for pool, rest in ((f["calib"], f["test"]), (f["calib_val"], f["val"])):
        for s in np.unique(subjects[rest]):
            for session in ("0train", "1train"):
                p = pool[(subjects[pool] == s) & (sessions[pool] == session)]
                r = rest[(subjects[rest] == s) & (sessions[rest] == session)]
                assert len(p) == 3 and len(r) == 4  # int(0.5 * 7), rounded down
                assert list(p) == sorted(p) and p.max() < r.min()


def test_folds_depend_on_the_seed_only():
    data = _trials()
    a = subject_fold_splits(*data, 3, seed=0)["subject_fold_0"]["test"]
    b = subject_fold_splits(*data, 3, seed=0)["subject_fold_0"]["test"]
    others = [subject_fold_splits(*data, 3, seed=s)["subject_fold_0"]["test"] for s in (1, 2)]
    assert np.array_equal(a, b) and not all(np.array_equal(a, c) for c in others)


@pytest.mark.parametrize(("n_folds", "val_groups"), [(2, 1), (8, 1), (3, 2)])
def test_invalid_number_of_folds_raises(n_folds, val_groups):
    with pytest.raises(ValueError, match="<= n_folds <= 7"):
        subject_fold_splits(*_trials(), n_folds, val_groups)


@pytest.mark.parametrize("calib_fraction", [0, 1, 1.5, -0.2])
def test_invalid_calib_fraction_raises(calib_fraction):
    with pytest.raises(ValueError, match="calib_fraction"):
        subject_fold_splits(*_trials(), 3, calib_fraction=calib_fraction)


def test_sessions_stored_class_by_class_raise():
    with pytest.raises(ValueError, match="class by class"):
        subject_fold_splits(*_trials(blocks=True), 3)


def test_dataset_builds_the_folds_with_a_fixed_partition(tmp_path):
    pytest.importorskip("moabb")
    from tests.unit.test_moabb_dataset import FakeDataset, _make

    dataset = FakeDataset(n_subjects=4)
    a = _make(tmp_path / "cache", dataset, subject_folds=4, val_groups=2,
              split="subject_fold_1", seed=0)
    b = _make(tmp_path / "cache", dataset, subject_folds=4, val_groups=2,
              split="subject_fold_1", seed=1)
    fold = a.splits["subject_fold_1"]
    assert {f"subject_fold_{k}" for k in range(4)} <= set(a.splits)
    assert np.array_equal(fold["test"], b.splits["subject_fold_1"]["test"])  # run seed
    assert a.scale == pytest.approx(geometric_scale(a.covs[fold["train"]]))
    assert len(set(a.subjects[fold["val"]])) == 2
    assert not set(a.subjects[fold["train"]]) & set(a.subjects[fold["test"]])
