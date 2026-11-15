"""MVDoppler on the real copy: counts compared with the atlas sheet ``mvdoppler``."""

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_leakage, check_spd
from spdnet_datasets.radar_microdoppler.mvdoppler import (
    CLASSES,
    fetch_mvdoppler,
    index_mvdoppler,
    official_folds,
)
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("RADAR_DOPPLER", "MVDoppler")

# atlas sheet mvdoppler: n_samples per fold and subjects per fold
FOLDS = {0: 11403, 1: 13741, 2: 12688, 3: 17004}
SUBJECTS = {0: {4, 6, 11}, 1: {2, 8, 10}, 2: {1, 7, 9}, 3: {0, 3, 5, 12}}
CLASS_TOTAL = {"normal": 14570, "phone_call": 13152, "pockets": 13266, "texting": 13848}


@pytest.fixture(scope="module")
def index():
    skip_if_missing(ROOT / "Labels_and_metadata" / "design_table.csv")
    return index_mvdoppler(ROOT)


def test_counts_folds_and_subjects(index):
    meta, info = index
    assert info["design_rows"] == sum(FOLDS.values())
    assert info["h5_not_in_design_table"] == 24
    assert info["unreadable"] <= 2  # zero-filled copies (phone_call_02515, texting_12478)
    assert len(meta) == info["design_rows"] - info["unreadable"]
    for k, n in FOLDS.items():
        rows = meta[meta.fold == k]
        assert n - info["unreadable"] <= len(rows) <= n
        assert set(rows.subject) == SUBJECTS[k]
    counts = meta.pattern.value_counts().to_dict()
    assert sum(CLASS_TOTAL.values()) - sum(counts.values()) == info["unreadable"]
    assert all(0 <= CLASS_TOTAL[c] - counts[c] <= info["unreadable"] for c in CLASSES)


def test_official_folds_are_subject_disjoint(index):
    meta, _ = index
    for k, s in official_folds(meta).items():
        assert_no_leakage(meta.subject.to_numpy()[s["train"]], meta.subject.to_numpy()[s["test"]])
        assert_no_leakage(meta.exp_fname.to_numpy()[s["train"]], meta.exp_fname.to_numpy()[s["val"]])
        assert len(s["train"]) + len(s["val"]) + len(s["test"]) == len(meta), k


def test_small_fetch(tmp_path):
    skip_if_missing(ROOT / "Data")
    b = fetch_mvdoppler(cache_dir=tmp_path, limit=5, verify=None)
    assert b.covariances.shape == (4 * 4 * 5, 64, 64)
    check_spd(b.covariances)
    assert np.all(b.meta.n_obs == 256)
