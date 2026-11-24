"""``MOABBDataset``: the MOABB adapter built from names through ``DatasetManager``."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from spdnet_datasets import DatasetManager
from spdnet_datasets.moabb import MOABBCovariances, MOABBDataset, stratified_split

moabb = pytest.importorskip("moabb")

from .test_moabb_dataset import N_CH, FakeDataset, FakeParadigm, as_epochs  # noqa: E402


@pytest.fixture
def fakes(monkeypatch):
    """Expose the fakes under moabb.datasets.Fake and moabb.paradigms.Fake."""
    import moabb.datasets
    import moabb.paradigms

    monkeypatch.setattr(moabb.datasets, "Fake", FakeDataset, raising=False)
    monkeypatch.setattr(moabb.paradigms, "Fake", FakeParadigm, raising=False)


def _config(tmp_path, **kw):
    return {"name": "moabb", "dataset": "Fake", "paradigm": "Fake",
            "dataset_params": {"n_subjects": 3}, "paradigm_params": {"fmin": 8},
            "cache_dir": str(tmp_path), "split_strategy": "stratified",
            "batch_size": 8, "num_workers": 0, "persistent_workers": False,
            "verbose": False, **kw}


def test_registered():
    assert DatasetManager.DATASETS["moabb"] is MOABBDataset


def test_dataloaders_from_config(fakes, tmp_path):
    train, val, test, n_classes = DatasetManager.create_dataloaders(
        _config(tmp_path, input_dim=N_CH))
    n = 3 * 2 * 2 * 6
    assert n_classes == 2
    assert len(train.dataset) + len(val.dataset) + len(test.dataset) == n
    assert len(test.dataset) == round(0.15 * n)
    x, y = next(iter(train))
    assert tuple(x.shape[1:]) == (N_CH, N_CH) and str(x.dtype) == "torch.float64"
    ds = train.dataset.dataset
    assert ds.split == "stratified" and ds.scale != 1.0  # fitted on the training part


def test_stratified_split_matches_riemannian_fl_eeg():
    y = np.repeat([0, 1, 2, 3], 50)
    # create_fl_clients of riemannian-fl-eeg (val_size=0.1, test_size=0.15, seed 42)
    rest, test = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42,
                                  stratify=y)
    train, val = train_test_split(rest, test_size=0.1 / 0.85, random_state=42,
                                  stratify=y[rest])
    split = stratified_split(y, 0.1, 0.15, seed=42)
    for part, ref in (("train", train), ("val", val), ("test", test)):
        np.testing.assert_array_equal(split[part], np.sort(ref))


def test_subjects_and_checks(fakes, tmp_path):
    ds = MOABBDataset("Fake", "Fake", dataset_params={"n_subjects": 4},
                      exclude_subjects=[2], cache_dir=tmp_path)
    assert sorted(set(ds.subjects)) == [1, 3, 4]
    with pytest.raises(ValueError, match="input_dim"):
        MOABBDataset("Fake", "Fake", input_dim=N_CH + 1, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="no 'Nope'"):
        MOABBDataset("Nope", "Fake", cache_dir=tmp_path)


class MixedChannels(FakeParadigm):
    """Subject 2 has other channel names, subject 3 has sessions of 4 and 3 channels.

    Like MOABB, the runs are joined with ``mne.concatenate_epochs``.
    """

    def get_data(self, dataset, subjects=None, return_epochs=False):
        import mne

        self._calls += 1
        rng = np.random.default_rng(0)
        epochs, y, meta = [], [], []
        for s in subjects:
            for k in range(2):
                names = [f"C{i}" for i in range(N_CH)]
                if s == 2:
                    names[-1] = "Other"
                if s == 3 and k == 1:
                    names = names[:-1]
                epochs.append(as_epochs(rng.standard_normal((12, len(names), 40)), names))
                y += self.events * 6
                meta += [{"subject": s, "session": f"{k}train", "run": "0"}] * 12
        epochs = mne.concatenate_epochs(epochs, verbose="error")
        return epochs, np.array(y), pd.DataFrame(meta)


@pytest.mark.parametrize("keep", [None, "keep_covariance"])
def test_subjects_with_other_channels_are_excluded(tmp_path, keep):
    data = FakeDataset(n_subjects=4)
    with pytest.warns(UserWarning, match=r"subjects \[2, 3\] excluded"):
        ds = MOABBCovariances(data, MixedChannels(), cache_dir=tmp_path, keep=keep)
    assert ds.excluded_subjects == [2, 3]
    assert sorted(set(ds.subjects)) == [1, 4]
    assert ds.channels == [f"C{i}" for i in range(N_CH)]
    with pytest.warns(UserWarning, match="excluded"):  # also on a cached load
        again = MOABBCovariances(data, MixedChannels(), cache_dir=tmp_path, keep=keep)
    assert again.excluded_subjects == [2, 3]


class OtherOrder(FakeParadigm):
    """Subject 2 has the same channels in another order; subjects 3 and 4 have one less."""

    def get_data(self, dataset, subjects=None, return_epochs=False):
        X, y, meta = self._data(dataset, subjects)
        names = [f"C{i}" for i in range(N_CH)]
        if subjects == [2]:
            names = names[::-1]
        if subjects in ([3], [4]):
            X, names = X[:, :-1], names[:-1]
        return as_epochs(X, names), y, meta


def test_channel_order_counts(tmp_path):
    with pytest.warns(UserWarning):
        ds = MOABBCovariances(FakeDataset(n_subjects=4), OtherOrder(), cache_dir=tmp_path)
    # two subjects with N_CH - 1 channels, one with N_CH, one with N_CH in another order
    assert ds.channels == [f"C{i}" for i in range(N_CH - 1)]
    assert ds.excluded_subjects == [1, 2]


class OneLess(FakeParadigm):
    """Subjects 3 and 4 have one channel less than subjects 1 and 2."""

    def get_data(self, dataset, subjects=None, return_epochs=False):
        X, y, meta = self._data(dataset, subjects)
        names = [f"C{i}" for i in range(N_CH)]
        if subjects in ([3], [4]):
            X, names = X[:, :-1], names[:-1]
        return as_epochs(X, names), y, meta


def test_channel_ties_go_to_the_longer_list(tmp_path):
    with pytest.warns(UserWarning):
        ds = MOABBCovariances(FakeDataset(n_subjects=4), OneLess(), cache_dir=tmp_path)
    assert ds.channels == [f"C{i}" for i in range(N_CH)]  # 2 subjects against 2
    assert ds.excluded_subjects == [3, 4]


def test_no_subject_left_raises(fakes, tmp_path):
    with pytest.raises(ValueError, match="no subject left"):
        MOABBCovariances(FakeDataset(n_subjects=1), _OnlySubject3(), cache_dir=tmp_path,
                         subjects=[3])
    with pytest.raises(ValueError, match="no subject left"):
        MOABBDataset("Fake", "Fake", exclude_subjects=[1, 2, 3], cache_dir=tmp_path)


class _OnlySubject3(MixedChannels):
    """Every subject has sessions with different channels (as subject 3)."""

    def get_data(self, dataset, subjects=None, return_epochs=False):
        return super().get_data(dataset, [3], return_epochs)


class TwoBands(FakeParadigm):
    """A filter bank: twice as many epochs as labels, as MOABB returns them."""

    def get_data(self, dataset, subjects=None, return_epochs=False):
        X, y, meta = self._data(dataset, subjects)
        return as_epochs(np.concatenate([X, X])), y, meta


def test_epochs_and_labels_must_match(tmp_path):
    with pytest.raises(ValueError, match="several filters"):
        MOABBCovariances(FakeDataset(), TwoBands(), cache_dir=tmp_path)


def test_data_dir_is_the_cache_dir(fakes, tmp_path):
    ds = MOABBDataset("Fake", "Fake", data_dir=tmp_path)
    assert str(ds.paths.covariance_dir).startswith(str(tmp_path))


def test_same_data_as_the_moabb_arrays(tmp_path):
    """Epochs path (unit factor applied) equals the arrays MOABB returns."""
    from moabb.datasets.fake import FakeDataset as MoabbFake
    from moabb.paradigms import MotorImagery

    data = MoabbFake(n_subjects=2, n_sessions=2, n_runs=1, sfreq=128, duration=30,
                     event_list=("left_hand", "right_hand"), seed=0)
    data.unit_factor = 1e3
    paradigm = MotorImagery(n_classes=2, events=["left_hand", "right_hand"], fmin=8, fmax=32)
    ds = MOABBCovariances(data, paradigm, cache_dir=tmp_path, keep=None, return_raw=True)
    X, y, meta = paradigm.get_data(dataset=data, subjects=data.subject_list)
    np.testing.assert_allclose(np.asarray(ds.epochs), X.astype(np.float32)[ds.trial_index],
                               rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(ds.classes)[ds.labels], y[ds.trial_index])
    np.testing.assert_array_equal(ds.subjects, meta["subject"].to_numpy()[ds.trial_index])
