"""CI4R-MULTI3 on the real copy: counts compared with the atlas sheet ``ci4r_multi3``."""

import collections

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_leakage, check_spd
from spdnet_datasets.radar_microdoppler.ci4r import CI4RDataset, fetch_ci4r, index_ci4r
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("RADAR_DOPPLER", "CI4R-MULTI3")

# atlas sheet ci4r_multi3: n_samples (total / dedoublonne) and participants of 77 GHz
FILES = {"77GHz": (653, 646), "xethru": (652, 633)}
PARTICIPANTS_77 = {"10": 116, "14": 88 + 9, "29": 221, "30": 212}


@pytest.fixture(autouse=True)
def _data():
    skip_if_missing(ROOT / "11_class_activity_data")


@pytest.mark.parametrize("mode", sorted(FILES))
def test_index_counts_and_duplicates(mode):
    meta = index_ci4r(ROOT, mode)
    total, unique = FILES[mode]
    assert len(meta) == total and (~meta.duplicate).sum() == unique
    named_copy = meta.path.str.contains("copy")
    assert (meta.duplicate == named_copy).all()  # every "(copy)" file is an md5 duplicate


def test_77ghz_participants():
    meta = index_ci4r(ROOT, "77GHz")
    assert dict(collections.Counter(meta.participant[~meta.duplicate])) == PARTICIPANTS_77


@pytest.mark.parametrize("mode", sorted(FILES))
def test_fetch_and_folds(mode):
    b = fetch_ci4r(mode=mode)
    unique = FILES[mode][1]
    assert len(b.labels) == unique - b.info["rejected_constant_channel"]
    assert b.info["rejected_constant_channel"] <= 1
    assert (b.meta.n_obs >= 2 * b.covariances.shape[-1]).all()
    check_spd(b.covariances)
    for k in range(4):
        s = b.splits[f"fold{k}"]
        assert_no_leakage(b.groups[s["train"]], b.groups[s["test"]])
    if mode == "77GHz":  # leave-one-participant-out
        assert sorted(len(set(b.groups[b.splits[f"fold{k}"]["test"]])) for k in range(4)) == [1] * 4


def test_dataset_sample():
    ds = CI4RDataset(verbose=False, return_raw=True)
    (cov, raw), y = ds[0]
    assert cov.shape == (64, 64) and raw.shape == (1475, 64)
    assert np.all(np.linalg.eigvalsh(cov.numpy()) > 0) and 0 <= y < 11
