"""DeepHS-Fruit v2 covariances from the raw cubes on the real data (skipped when absent)."""

import collections

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.hyperspectral.deephsfruit import fetch_deephsfruit
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data
ROOT = dataset_dir("HYPERSPECTRAL", "DeepHS-Fruit")


def test_ripeness_counts_official_split_and_leak():
    skip_if_missing(ROOT)
    r = fetch_deephsfruit(verify=None, classification_mode="ripeness")
    assert len(r.labels) == 596 and r.covariances.shape[1:] == (224, 224)
    assert set(collections.Counter(r.labels)) == {"unripe", "perfect", "overripe"}
    off = r.splits["official"]
    assert [len(off[k]) for k in ("train", "val", "test")] == [384, 72, 140]
    train = set(r.groups[off["train"]])
    assert sum(g in train for g in r.groups[off["test"]]) == 46  # documented leak
    sp = r.splits["recommended"]
    assert_no_leakage(r.groups[sp["train"]], r.groups[sp["test"]])
    assert np.median(r.n_obs) > 2 * 224
