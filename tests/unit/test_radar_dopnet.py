"""DopNet on a tiny synthetic copy (.mat files under tmp_path)."""

import numpy as np
import pytest
import torch
from scipy.io import savemat

from spdnet_datasets import DatasetManager
from spdnet_datasets.core import assert_no_duplicates, assert_no_leakage, check_spd
from spdnet_datasets.radar_microdoppler.dopnet import (
    CELL_CLASSES,
    CLASSES,
    DopNetDataset,
    energy_band,
    fetch_dopnet,
)

F, P = 64, 8  # Doppler bins, channels (min_frames = 2p = 16)
PERSONS = ["A", "B", "C"]
LENGTHS = [20, 30, 12, 40]  # one gesture of 12 frames per cell is rejected (< 16)


def _signal(rng, t, cell):
    profile = np.exp(-0.5 * ((np.arange(F) - F // 2) / (2 + cell)) ** 2)[:, None]
    noise = rng.standard_normal((F, t)) + 1j * rng.standard_normal((F, t))
    return (profile + 0.01) * noise


def make_dopnet(root):
    rng = np.random.default_rng(0)
    (root / "Training Data").mkdir(parents=True)
    (root / "Test Data").mkdir()
    train = {}
    for person in PERSONS:
        cells = np.empty((1, 4), dtype=object)
        for c in range(4):
            sig = np.empty((len(LENGTHS), 1), dtype=object)
            for i, t in enumerate(LENGTHS):
                sig[i, 0] = _signal(rng, t, c)
            cells[0, c] = sig
            train[(person, c)] = sig
        extended = np.empty((len(LENGTHS) + 1, 1), dtype=object)
        extended[:-1] = cells[0, 1]
        extended[-1, 0] = cells[0, 1][0, 0]  # intra-train copy
        cells[0, 1] = extended
        savemat(root / "Training Data" / f"Data_Per_PersonData_Training_Person_{person}.mat",
                {"Data_Training": {"name": person, "Doppler_Signals": cells}})
    rows = []
    for person in PERSONS:
        for c in range(4):
            rows.append([_signal(rng, 25, c), f"{CELL_CLASSES[c]} 1 Person {person}"])
    rows.append([train[("A", 3)][0, 0], "Click 2 Person A"])  # test copy of a training gesture
    test = np.empty((len(rows), 2), dtype=object)
    for i, (s, label) in enumerate(rows):
        test[i, 0], test[i, 1] = s, label
    savemat(root / "Test Data" / "Data_For_Test_Random.mat", {"Data_rand": test})
    return root


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_dopnet(tmp_path_factory.mktemp("dop") / "Dop-NET")


@pytest.fixture(scope="module")
def bunch(data, tmp_path_factory):
    return fetch_dopnet(data, tmp_path_factory.mktemp("cache"), p=P, verify=None)


def test_energy_band():
    profile = np.zeros(100)
    profile[45:55] = 1.0
    assert energy_band(profile, 0.95) == (45, 55)
    assert energy_band(profile, 0.5) == (47, 53)
    assert energy_band(profile, 0.5, min_width=20) == (40, 60)


def test_counts_duplicates_and_rejections(bunch):
    info = bunch.info
    n_train = len(PERSONS) * (4 * len(LENGTHS) + 1)
    assert info["n_gestures"] == {"train": n_train, "test": len(PERSONS) * 4 + 1}
    assert info["duplicates_removed"] == {"train": len(PERSONS), "test": 1}
    assert info["test_in_train"] == 1
    assert info["rejected_short"] == {"train": len(PERSONS) * 4, "test": 0}
    assert bunch.covariances.shape == (len(PERSONS) * 4 * 3 + len(PERSONS) * 4, P, P)
    assert (bunch.meta.n_obs >= 2 * P).all()
    check_spd(bunch.covariances)
    assert bunch.meta.md5.is_unique


def test_band_holds_the_requested_energy(bunch):
    a, b = bunch.info["band"]
    assert bunch.info["band_energy"] >= 0.95 and b - a >= P
    assert (a + b) // 2 == F // 2  # centred on zero Doppler


def test_cell_order_is_wave_pinch_swipe_click(data, bunch):
    train = bunch.meta[bunch.meta.source == "train"]
    first_cell_of_a = train[(train.person == "A") & (train["index"] == 0)]
    assert set(first_cell_of_a.class_name) == set(CLASSES)
    assert CELL_CLASSES == ["Wave", "Pinch", "Swipe", "Click"]


def test_splits_have_no_leakage(bunch):
    off = bunch.splits["official"]
    assert_no_duplicates(bunch.meta.md5[off["train"]], bunch.meta.md5[off["test"]])
    for k in range(len(PERSONS)):
        s = bunch.splits[f"person{k}"]
        assert_no_leakage(bunch.groups[s["train"]], bunch.groups[s["test"]])
        assert set(bunch.groups[s["test"]]) == {PERSONS[k]}


def test_min_frames_below_2p_stores_short_gestures_for_the_series(data, bunch, tmp_path):
    with pytest.raises(ValueError, match=">= 1"):
        fetch_dopnet(data, tmp_path, p=P, min_frames=0, verify=None)
    short = fetch_dopnet(data, tmp_path, p=P, min_frames=1, verify=None)
    # same covariances and energy band: the short gestures have no p x p covariance
    assert short.info["band"] == bunch.info["band"]
    assert len(short.meta) == len(bunch.meta)
    assert short.info["rejected_n_lt_2p"] == len(PERSONS) * 4
    ds = DopNetDataset(data, p=P, min_frames=1, cache_dir=tmp_path, verify=None, verbose=False)
    full = ds.series_samples()
    assert len(full) == len(ds) + len(PERSONS) * 4
    assert (full.meta.n_frames < 2 * P).sum() == len(PERSONS) * 4
    i = int(np.flatnonzero(full.meta.n_frames.to_numpy() < 2 * P)[0])
    assert full.structured_raw(i).shape == (full.meta.n_frames.iloc[i], P)
    assert full.doppler_bands(i).shape[0] == full.meta.n_frames.iloc[i]
    for k in range(len(PERSONS)):
        s = full.splits[f"person{k}"]
        assert_no_leakage(full.groups[s["train"]], full.groups[s["test"]])
        assert set(full.groups[s["test"]]) == {PERSONS[k]}


def test_dataset(data, bunch, tmp_path_factory):
    cache = bunch.covariance_path.parents[2]
    kw = {"p": P, "cache_dir": cache, "verify": None, "verbose": False}
    ds = DopNetDataset(data, **kw)
    assert DatasetManager.DATASETS["dopnet"] is DopNetDataset
    x, y = ds[0]
    assert x.shape == (P, P) and x.dtype == torch.float64
    (cov, raw), _ = DopNetDataset(data, return_raw=True, **kw)[0]
    assert raw.shape == (ds.meta.n_obs[0], P)
    legacy_test = DopNetDataset(data, split="test", **kw)
    assert len(legacy_test) == len(ds.splits["official"]["test"])
    assert set(DopNetDataset(data, classes=["Wave"], **kw).labels) == {0}
    assert len(DopNetDataset(data, max_classes=2, **kw).classes) == 2
    assert np.bincount(DopNetDataset(data, max_samples_per_class=2, **kw).labels).tolist() == [2] * 4
    lopo = DopNetDataset(data, split="recommended", **kw)
    train = lopo.splits["recommended"]["train"]
    eig = np.linalg.eigvalsh(np.stack([lopo[i][0].numpy() for i in train]))
    assert np.exp(np.mean(np.log(eig))) == pytest.approx(1.0, rel=1e-8)
    merged = DopNetDataset(data, merge_classes={"Pinch": "Click"}, **kw)
    assert merged.classes == ["Click", "Swipe", "Wave"]
