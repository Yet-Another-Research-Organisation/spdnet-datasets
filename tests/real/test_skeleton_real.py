"""HDM05 and NTU RGB+D 120 against the local copies and their atlas sheets."""

import json
from collections import Counter

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.skeleton import hdm05, ntu120
from tests.conftest import DATA_ROOT, dataset_dir, skip_if_missing

pytestmark = [pytest.mark.real_data,
              pytest.mark.filterwarnings("ignore:HDM05 has no actor:UserWarning")]
CATALOG = DATA_ROOT.parent / "_atlas_v2" / "site" / "catalog.json"


def _sheet(dataset_id):
    skip_if_missing(CATALOG)
    return {d["id"]: d for d in json.loads(CATALOG.read_text())["datasets"]}[dataset_id]


def _root(name):
    root = dataset_dir("SKELETON", name)
    skip_if_missing(root)
    return root


# -- HDM05 ----------------------------------------------------------------------
def test_hdm05_counts_match_sheet():
    sheet, index = _sheet("hdm05"), hdm05._index(_root("HDM05"))
    assert len(index) == sheet["n_samples"]["total"] == 2086
    expected = {f"{int(k.split()[1]):03d}": v
                for k, v in sheet["class_distribution"]["total"].items()}
    assert dict(Counter(index["label"])) == expected
    assert index["n_obs"].min() == 94 and (index["n_obs"] < 186).sum() == 604


def test_hdm05_c93_and_c31(tmp_path):
    b = hdm05.fetch_hdm05(cache_dir=tmp_path)
    assert b.covariances.shape == (1482, 93, 93) and len(b.excluded) == 604
    assert (b.meta["n_obs"] >= 186).all()
    b31 = hdm05.fetch_hdm05(cache_dir=tmp_path, features="c31")
    assert b31.covariances.shape == (2086, 31, 31) and len(b31.excluded) == 0
    parts = b31.splits["recommended"]
    assert sorted(np.concatenate(list(parts.values())).tolist()) == list(range(2086))
    assert_no_leakage(b31.groups[parts["train"]], b31.groups[parts["test"]])


def test_hdm05_dataset_honours_class_limits(tmp_path):
    ds = hdm05.HDM05Dataset(cache_dir=tmp_path, features="c31", max_classes=10,
                            max_samples_per_class=10, subset=None, verbose=False)
    assert len(ds.classes) == 10 and np.bincount(ds.labels).max() <= 10


# -- NTU RGB+D 120 ----------------------------------------------------------------
@pytest.fixture(scope="module")
def ntu_index():
    return ntu120.ntu_index(_root("NTU_RGBD_120"))


def test_ntu_counts_and_official_splits_match_sheet(ntu_index):
    sheet = _sheet("ntu_rgbd_120")
    n = sheet["n_samples"]
    assert len(ntu_index) == n["total"] == 114480
    assert dict(Counter(ntu_index["label"])) == sheet["class_distribution"]["total"]
    in_xsub = ntu_index["subject"].isin(ntu120.TRAIN_SUBJECTS)
    assert (in_xsub.sum(), (~in_xsub).sum()) == (n["X-Sub train"], n["X-Sub test"])
    in_xsetup = ntu_index["setup"] % 2 == 0
    assert (in_xsetup.sum(), (~in_xsetup).sum()) == (n["X-Setup train"], n["X-Setup test"])


def test_ntu_split_function_has_no_subject_leakage(ntu_index):
    labels = ntu_index["label"].to_numpy()
    splits = ntu120.ntu120_splits(ntu_index, labels, ntu_index["group"].to_numpy())
    subject, setup = ntu_index["subject"].to_numpy(), ntu_index["setup"].to_numpy()
    xsub, xsetup = splits["xsub"], splits["xsetup"]
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        assert_no_leakage(subject[xsub[a]], subject[xsub[b]], what="subject")
        assert_no_leakage(setup[xsetup[a]], setup[xsetup[b]], what="setup")
    assert len(xsub["train"]) + len(xsub["val"]) == 63360


def test_ntu_subset_covariances(tmp_path):
    b = ntu120.fetch_ntu120(cache_dir=tmp_path, limit=300, verify=None)
    assert b.covariances.shape[1:] == (15, 15)
    assert (b.meta["n_obs"] >= 30).all()
    assert len(b.labels) + len(b.excluded) == 300


@pytest.mark.slow
def test_ntu_full_excludes_the_356_files_without_body():
    b = ntu120.fetch_ntu120(verify=None)
    reasons = b.excluded.reason
    assert (reasons == "no body").sum() == 356
    assert len(b.labels) + len(b.excluded) == 114480
