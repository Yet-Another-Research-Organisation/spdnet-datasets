"""MOABB adapter: ``MOABBCovariances`` on a fake dataset and paradigm (no network).

The splits come from MOABB's splitters, so these tests need ``moabb`` installed.
"""

import numpy as np
import pandas as pd
import pytest

from spdnet_datasets.core.estimators import InsufficientSamplesError, check_spd
from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.moabb import MOABBCovariances, epochs_to_covariances

pytest.importorskip("moabb")

N_CH, N_T = 4, 40  # n_times = 10 * n_channels


class FakeDataset:
    """Minimal stand-in for a MOABB dataset."""

    code = "Fake-2026"

    def __init__(self, n_subjects=3, n_sessions=2, n_per_class=6, n_times=N_T):
        self.subject_list = list(range(1, n_subjects + 1))
        self.n_sessions, self.n_per_class, self.n_times = n_sessions, n_per_class, n_times


class FakeParadigm:
    """Returns ``(X, y, metadata)`` like ``moabb.paradigms.*.get_data``."""

    def __init__(self, fmin=8, fmax=32):
        self.fmin, self.fmax = fmin, fmax
        self.events = ["left_hand", "right_hand"]
        self._calls = 0

    def get_data(self, dataset, subjects=None):
        self._calls += 1
        rng = np.random.default_rng(0)
        X, y, rows = [], [], []
        for s in subjects:
            for k in range(dataset.n_sessions):
                for label in self.events:
                    for _ in range(dataset.n_per_class):
                        scale = 1.0 if label == "left_hand" else 3.0
                        X.append(rng.standard_normal((N_CH, dataset.n_times)) * scale)
                        y.append(label)
                        rows.append({"subject": s, "session": f"{k}train", "run": "0"})
        return np.stack(X), np.array(y), pd.DataFrame(rows)


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


def _make(cache, dataset=None, paradigm=None, **kw):
    return MOABBCovariances(dataset or FakeDataset(), paradigm or FakeParadigm(),
                            cache_dir=cache, **kw)


# ---------------------------------------------------------------- dataset
def test_dataset_shapes_labels_groups(cache):
    ds = _make(cache)
    assert len(ds) == 3 * 2 * 2 * 6
    x, y = ds[0]
    assert tuple(x.shape) == (N_CH, N_CH) and str(x.dtype) == "torch.float64"
    assert y in (0, 1) and ds.classes == ["left_hand", "right_hand"]
    check_spd(ds.covariances)
    assert set(ds.groups) == {f"{s}-{k}train" for s in (1, 2, 3) for k in (0, 1)}
    assert ds.metadata.columns.tolist()[:3] == ["subject", "session", "run"]


@pytest.mark.parametrize("return_mean,return_raw", [(False, False), (True, False),
                                                    (False, True), (True, True)])
def test_output_tuples(cache, return_mean, return_raw):
    x, _ = _make(cache, return_mean=return_mean, return_raw=return_raw)[1]
    parts = x if isinstance(x, tuple) else (x,)
    assert len(parts) == 1 + return_mean + return_raw
    assert tuple(parts[0].shape) == (N_CH, N_CH)
    if return_mean:
        assert tuple(parts[1].shape) == (N_CH,)
    if return_raw:
        raw = parts[-1]
        assert tuple(raw.shape) == (N_T, N_CH) and str(raw.dtype) == "torch.float32"


def test_raw_epoch_matches_covariance(cache):
    ds = _make(cache, return_raw=True)
    cov, raw = ds[3][0]
    np.testing.assert_allclose(epochs_to_covariances(raw.numpy().T[None].astype(float))[0],
                               cov.numpy(), rtol=1e-5)


def test_too_short_epochs_raise(cache):
    # every trial fails: nothing left to exclude, the error is raised
    with pytest.raises(InsufficientSamplesError):
        _make(cache, dataset=FakeDataset(n_times=2 * N_CH - 1))


def test_cache_is_replayed_without_get_data(cache):
    paradigm = FakeParadigm()
    _make(cache, paradigm=paradigm)
    _make(cache, paradigm=paradigm)
    assert paradigm._calls == 1
    _make(cache, paradigm=paradigm, force=True)
    assert paradigm._calls == 2
    _make(cache, paradigm=FakeParadigm(fmax=30))  # other paradigm settings: new entry
    assert len(list((cache / "moabb_Fake-2026" / "covariance").iterdir())) == 2


def test_cache_key_covers_dataset_and_paradigm_options(cache):
    _make(cache, dataset=FakeDataset(n_per_class=6))
    ds = _make(cache, dataset=FakeDataset(n_per_class=5))  # same code, other options
    assert len(ds) == 3 * 2 * 2 * 5
    _make(cache, paradigm=FakeParadigm(fmax=np.float64(30)))  # numpy scalar option
    paradigm = FakeParadigm()
    paradigm.filters = [object()]  # non-plain option: keyed by its repr, without address
    _make(cache, paradigm=paradigm)
    paradigm = FakeParadigm()
    paradigm.filters = [object()]
    _make(cache, paradigm=paradigm)
    assert paradigm._calls == 0  # same repr: cache replayed
    assert len(list((cache / "moabb_Fake-2026" / "processed").iterdir())) == 4


def test_estimator_change_reuses_the_epochs(cache):
    paradigm = FakeParadigm()
    scm = _make(cache, paradigm=paradigm)
    lw = _make(cache, paradigm=paradigm, estimator="ledoit_wolf")
    assert paradigm._calls == 1
    assert scm.paths.processed_dir == lw.paths.processed_dir
    assert scm.paths.covariance_dir != lw.paths.covariance_dir


class ConstantTrialParadigm(FakeParadigm):
    """Trial 5 is constant: its covariance is singular (not SPD)."""

    def get_data(self, dataset, subjects=None):
        X, y, metadata = super().get_data(dataset, subjects)
        X[5] = 1.0
        metadata["trial"] = np.arange(len(y))
        return X, y, metadata


def test_failing_trials_are_excluded_and_counted(cache):
    paradigm = ConstantTrialParadigm()
    ds = _make(cache, paradigm=paradigm, return_raw=True)
    X, y, metadata = paradigm.get_data(FakeDataset(), [1, 2, 3])
    assert ds.n_excluded == 1 and len(ds) == len(y) - 1
    assert 5 not in ds.trial_index and ds.metadata["trial"].tolist() == ds.trial_index.tolist()
    assert [ds.classes[k] for k in ds.labels] == y[ds.trial_index].tolist()
    check_spd(ds.covariances)
    np.testing.assert_allclose(ds[5][0][1].numpy().T, X[6], rtol=1e-6)


def test_keep_covariance_drops_epochs_and_forbids_raw(cache):
    ds = _make(cache, keep="keep_covariance")
    assert ds.paths.processed_dir is None and ds.paths.raw_dir is None
    with pytest.raises(ValueError, match="return_raw"):
        _make(cache, keep="keep_covariance", return_raw=True)
    with pytest.raises(ValueError, match="deleted"):
        _make(cache, return_raw=True)  # the cached covariances have no epochs left


# ---------------------------------------------------------------- splits
def test_splits_follow_moabb_protocols(cache):
    ds = _make(cache)
    names = set(ds.splits)
    assert {n for n in names if n.startswith("cross_subject")} == {
        f"cross_subject_{k}" for k in range(3)}
    assert sum(n.startswith("cross_session") for n in names) == 3 * 2
    assert sum(n.startswith("within_session") for n in names) == 3 * 2 * 5
    assert ds.splits["recommended"] is ds.splits["cross_subject_0"]
    for name, part in ds.splits.items():
        tr, te = part["train"], part["test"]
        assert len(np.intersect1d(tr, te)) == 0 and len(te) > 0
        if name.startswith("cross_subject"):
            assert_no_leakage(ds.subjects[tr], ds.subjects[te], what="subject")
        elif name.startswith("cross_session"):
            assert len(set(ds.subjects[te])) == 1
            assert_no_leakage(ds.groups[tr], ds.groups[te], what="session")
        elif name.startswith("within_session"):
            assert len(set(ds.groups[np.r_[tr, te]])) == 1


def test_single_subject_has_no_cross_subject_split(cache):
    ds = _make(cache, dataset=FakeDataset(n_subjects=1))
    assert not any(n.startswith("cross_subject") for n in ds.splits)
    assert ds.splits["recommended"] is ds.splits["cross_session_0"]


def test_normalization_is_fitted_on_the_split_train(cache):
    assert _make(cache).scale == 1.0
    ds = _make(cache, split="cross_subject_1")
    train = ds.splits["cross_subject_1"]["train"]
    assert abs(np.mean(np.log(np.linalg.eigvalsh(ds.covariances[train])))) < 1e-10
    assert _make(cache, split="cross_subject_1", normalize=False).scale == 1.0
    with pytest.raises(ValueError, match="Unknown split"):
        _make(cache, split="nope")


def test_create_dataloaders_uses_moabb_split(cache):
    from spdnet_datasets.base import create_dataloaders

    ds = _make(cache)
    tr, va, te = create_dataloaders(ds, batch_size=8, num_workers=0, split="cross_subject_0",
                                    verbose=False)
    assert len(tr.dataset) + len(te.dataset) == len(ds) and len(va.dataset) == 0
    x, y = next(iter(tr))
    assert tuple(x.shape) == (8, N_CH, N_CH)


def test_create_dataloaders_without_split(cache):
    from spdnet_datasets.base import create_dataloaders

    ds = _make(cache)
    assert len(ds.samples) == len(ds) and ds.samples[0]["class_name"] in ds.classes
    with pytest.warns(UserWarning, match="Sample-level split"):
        tr, va, te = create_dataloaders(ds, num_workers=0)
    assert len(tr.dataset) + len(va.dataset) + len(te.dataset) == len(ds)
