"""Dop-NET on the real copy: counts and cell order compared with the atlas sheet ``dop_net``."""

import collections

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_duplicates, assert_no_leakage, check_spd
from spdnet_datasets.radar_microdoppler.dopnet import (
    CLASSES,
    fetch_dopnet,
    iter_gestures,
    signal_md5,
)
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("RADAR_DOPPLER", "Dop-NET")

# atlas sheet dop_net: class_distribution of the training files per person
PER_PERSON = {"A": (56, 98, 64, 105), "B": (112, 116, 72, 105), "C": (85, 132, 80, 137),
              "D": (70, 112, 71, 93), "E": (56, 98, 91, 144), "F": (87, 140, 101, 208)}
CELLS = ("Wave", "Pinch", "Swipe", "Click")


@pytest.fixture(autouse=True)
def _data():
    skip_if_missing(ROOT / "Training Data")


@pytest.fixture(scope="module")
def gestures():
    skip_if_missing(ROOT / "Training Data")  # module scope: runs before the autouse skip
    return [(name, person, source, signal_md5(s), s.shape) for s, name, person, source, _
            in iter_gestures(ROOT)]


def test_cell_counts_per_person(gestures):
    counts = collections.Counter((p, n) for n, p, src, _, _ in gestures if src == "train")
    for person, expected in PER_PERSON.items():
        assert tuple(counts[(person, c)] for c in CELLS) == expected


def test_cell_order_matches_the_labelled_test_copies(gestures):
    """Every test gesture identical to a training one carries the same class name."""
    train = {md5: n for n, _, src, md5, _ in gestures if src == "train"}
    pairs = collections.Counter((n, train[md5]) for n, _, src, md5, _ in gestures
                                if src == "test" and md5 in train)
    assert sum(pairs.values()) == 111
    assert all(a == b for a, b in pairs)
    assert {a for a, _ in pairs} == set(CLASSES)
    assert {s[0] for *_, s in gestures} == {800}


def test_fetch_counts_and_splits():
    b = fetch_dopnet()
    info = b.info
    assert info["n_gestures"] == {"train": 2433, "test": 643}
    assert info["test_in_train"] == 111
    assert info["n_gestures"]["train"] - info["duplicates_removed"]["train"] == 2211
    kept = 2433 + 643 - sum(info["duplicates_removed"].values()) - sum(info["rejected_short"].values())
    assert len(b.labels) == kept and b.covariances.shape[1:] == (48, 48)
    assert b.meta.n_obs.min() >= 96 and info["band_energy"] >= 0.95
    check_spd(b.covariances)
    off = b.splits["official"]
    assert_no_duplicates(b.meta.md5[off["train"]], b.meta.md5[off["test"]])
    for k in range(6):
        s = b.splits[f"person{k}"]
        assert_no_leakage(b.groups[s["train"]], b.groups[s["test"]])
    assert np.all(np.isfinite(b.means))
