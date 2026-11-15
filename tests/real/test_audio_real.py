"""ESC-50, GTZAN and BirdSet XCM against the local copies and their atlas sheets.

The counts come from the atlas (``catalog.json`` next to the data root). Tests
that decode every recording are marked ``slow``; with a warm cache
(``SPDNET_CACHE_DIR``) they only read the cached covariances.
"""

import json
from collections import Counter

import numpy as np
import pytest

from spdnet_datasets.core import assert_no_leakage
from tests.conftest import DATA_ROOT, dataset_dir, skip_if_missing

pytest.importorskip("pyarrow")
pytest.importorskip("soundfile")

from spdnet_datasets.audio import birdset_xcm, esc50, gtzan  # noqa: E402

pytestmark = [pytest.mark.real_data,
              pytest.mark.filterwarnings("ignore:(GTZAN|BirdSet_XCM):UserWarning")]
CATALOG = DATA_ROOT.parent / "_atlas_v2" / "site" / "catalog.json"


def _sheet(dataset_id):
    skip_if_missing(CATALOG)
    return {d["id"]: d for d in json.loads(CATALOG.read_text())["datasets"]}[dataset_id]


def _root(name):
    root = dataset_dir("AUDIO", name)
    skip_if_missing(root)
    return root


def _check_covariances(b, p=20):
    assert b.covariances.shape[1:] == (p, p)
    assert (b.meta["n_obs"] >= 2 * p).all()
    assert (np.linalg.eigvalsh(b.covariances)[:, 0] > 0).all()


# -- ESC-50 ---------------------------------------------------------------------
def test_esc50_counts_match_sheet():
    sheet, index = _sheet("esc50"), esc50._index(_root("ESC-50"))
    assert len(index) == sheet["n_samples"]["total"] == 2000
    assert dict(Counter(index["category"])) == sheet["class_distribution"]["total"]
    for k in range(1, 6):
        assert (index["fold"] == k).sum() == sheet["n_samples"][f"pli {k}"]


def test_esc50_official_folds_leakage_is_the_known_one():
    index = esc50._index(_root("ESC-50"))
    folds_per_source = index.groupby("src_file")["fold"].nunique()
    assert (folds_per_source > 1).sum() == 4  # atlas: 4 of 1 524 recordings span two folds
    parts = esc50.esc50_splits(index, None, None, fold=1)["official"]
    shared = set(index["src_file"].iloc[parts["train"]]) & set(
        index["src_file"].iloc[parts["test"]])
    assert shared <= set(folds_per_source.index[folds_per_source > 1])


def test_esc50_subset_covariances(tmp_path):
    b = esc50.fetch_esc50(cache_dir=tmp_path, limit=60)
    _check_covariances(b)
    assert b.meta["n_obs"].max() == 427


@pytest.mark.slow
def test_esc50_full():
    b = esc50.fetch_esc50()
    _check_covariances(b)
    assert len(b.labels) + len(b.excluded) == 2000
    assert b.excluded.reason.str.startswith(("n < 2p", "not SPD")).all()
    parts = b.splits["official"]
    assert len(parts["test"]) == 400 - (b.excluded.id.str.startswith("1-")).sum()


# -- GTZAN ----------------------------------------------------------------------
def test_gtzan_counts_and_duplicates_match_sheet():
    sheet, index = _sheet("gtzan"), gtzan._index(_root("GTZAN"))
    assert len(index) == sheet["n_samples"]["total"] == 999
    assert dict(Counter(g for _, g in index)) == sheet["class_distribution"]["total"]
    names = {n for n, _ in index}
    assert all(a in names and b in names for a, b in gtzan.DUPLICATES)
    assert all(f"{a} = {b}" in sheet["known_issues"][1] for a, b in gtzan.DUPLICATES)
    assert len(names - {b for _, b in gtzan.DUPLICATES}) == 985


def test_gtzan_subset_covariances(tmp_path):
    b = gtzan.fetch_gtzan(cache_dir=tmp_path, limit=30)
    _check_covariances(b)
    assert (b.meta["sr"] == 22050).all()  # not the 32 kHz of the Hugging Face metadata


@pytest.mark.slow
def test_gtzan_full():
    b = gtzan.fetch_gtzan()
    _check_covariances(b)
    assert len(b.labels) + len(b.excluded) == 985
    parts = b.splits["recommended"]
    assert sum(len(v) for v in parts.values()) == len(b.labels)


# -- BirdSet XCM ----------------------------------------------------------------
def test_birdset_counts_match_sheet():
    sheet = _sheet("birdset_xcm")
    index, kept = birdset_xcm._index(_root("BirdSet_XCM"))
    assert len(index) == sheet["n_samples"]["total"] == 5371
    assert kept.sum() == sheet["n_samples"]["benchmark (espèces >= 10 enr.)"] == 5185
    assert dict(Counter(index["label"])) == sheet["class_distribution"]["total"]
    assert index["label"][kept].nunique() == 95


def test_birdset_recordist_folds_do_not_leak():
    index, kept = birdset_xcm._index(_root("BirdSet_XCM"))
    meta = index[kept].reset_index(drop=True)
    for fold in range(1, 6):
        parts = birdset_xcm.birdset_xcm_splits(meta, meta["label"].to_numpy(),
                                               meta["group"].to_numpy(), fold=fold)
        g = meta["group"].to_numpy()
        for a, c in (("train", "test"), ("train", "val"), ("val", "test")):
            assert_no_leakage(g[parts["recommended"][a]], g[parts["recommended"][c]])


def test_birdset_subset_covariances(tmp_path):
    b = birdset_xcm.fetch_birdset_xcm(cache_dir=tmp_path, limit=20)
    _check_covariances(b)
