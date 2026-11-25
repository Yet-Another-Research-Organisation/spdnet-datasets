"""AVIRIS-C California multilabel on the real release: counts vs the dataset card.

The release folder is ``$SPDNET_AVIRIS_CA_ML_DIR`` if set, else
``<SPDNET_DATA_ROOT>/HYPERSPECTRAL/AVIRIS-CA-Multilabel``. The first run reads the
7.1 GB of shards and estimates about 24 000 Ledoit-Wolf covariances (4.6 GB of cache).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from spdnet_datasets.core import assert_no_leakage
from spdnet_datasets.core.config import dataset_path
from spdnet_datasets.core.estimators import check_spd
from spdnet_datasets.hyperspectral import aviris_ca_ml as A

ROOT = Path(os.environ.get("SPDNET_AVIRIS_CA_ML_DIR") or dataset_path(A.DOMAIN, A.NAME)).expanduser()
pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not (ROOT / "bands.csv").exists(),
                                 reason="AVIRIS-C CA multilabel release not available")]
FOLD_WINDOWS = [413, 402, 405, 398, 416]  # dataset card


@pytest.fixture(scope="module")
def ds():
    pytest.importorskip("pyarrow")
    return A.AVIRISCAMultilabelDataset(ROOT, verbose=False)


def test_counts_match_the_card(ds):
    info = ds.fetched.info
    assert info["n_windows"] == 2034 and info["n_acquired"] == 28245
    assert info["n_qc_ok"] == 24211 == info["n_eligible"]
    assert len(ds.bands) == 155 and ds.wavelengths[0] == pytest.approx(433.65)
    assert len(ds) == int(ds.fetched.ok.sum()) >= 24000
    W = ds.fetched.windows
    assert np.bincount(W["fold"], minlength=5).tolist() == FOLD_WINDOWS
    classes = pd.read_csv(ROOT / "classes.csv")
    assert W[[f"label_{c}" for c in A.CLASSES]].sum().tolist() == classes["windows_total"].tolist()
    D = ds.fetched.dates  # the qc_ok dates
    assert D[[f"label_{c}" for c in A.CLASSES]].sum().tolist() == \
        classes["window_dates_qc_ok"].tolist()


def test_samples_splits_and_scale(ds):
    x, y = ds[0]
    assert x.shape == (155, 155) and y.shape == (13,)
    idx = np.random.default_rng(0).choice(len(ds), 50, replace=False)
    check_spd(np.stack([ds[int(i)][0].numpy() for i in idx]))
    assert ds.fetched.n_obs[ds.fetched.ok].min() >= 2 * 155
    rec = ds.splits["recommended"]
    folds = ds.meta["fold"].to_numpy()
    assert set(folds[rec["train"]]) == {1, 2, 3} and set(folds[rec["val"]]) == {0}
    assert set(folds[rec["test"]]) == {4}
    for k in range(5):
        s = ds.splits[f"fold{k}"]
        assert_no_leakage(ds.groups[s["train"]], ds.groups[s["test"]])
        assert_no_leakage(ds.window_ids[s["train"]], ds.window_ids[s["test"]])
    assert np.isfinite(ds.scale) and ds.scale > 0


def test_series_mode():
    s = A.AVIRISCAMultilabelDataset(ROOT, mode="series", verbose=False)
    assert len(s) <= 2034
    (tokens, mask), y = s[0]
    assert tokens.shape == (14, 155, 155) and mask.dtype == torch.bool
    assert int(mask.sum()) == int((s._series_index[0] >= 0).sum())
    w = s.fetched.windows.set_index("window_id").loc[int(s.meta.window_id[0])]
    np.testing.assert_array_equal(y.numpy(), w[[f"label_{c}" for c in A.CLASSES]].to_numpy())


# learning scenarios (qc=True): windows, window-dates, number of classes (None: regression)
SCENARIOS = {
    "pure": ({"scenario": "pure"}, 401, 4679, 12),
    "pure_strict": ({"scenario": "pure", "pure_max_change": 0.05}, 217, 2463, 9),
    "stable": ({"scenario": "stable"}, 569, 6713, None),
    "spacetime": ({"scenario": "spacetime"}, 2031, 24211, None),
    "stability": ({"scenario": "stability"}, 1538, 18384, 2),
    "stability_reg": ({"scenario": "stability", "task": "regression"}, 2031, 24211, None),
}


def test_change_distribution(ds):
    ch = ds.window_stats["change"].to_numpy()
    assert len(ch) == 2031 and ch.max() == pytest.approx(0.2)  # cap of the window selection
    assert int((ch <= 0.05).sum()) == 569 and int((ch > 0.10).sum()) == 969
    assert np.median(ch) == pytest.approx(0.0975, abs=1e-3)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_counts(name):
    kw, n_windows, n_dates, n_classes = SCENARIOS[name]
    d = A.AVIRISCAMultilabelDataset(ROOT, verbose=False, **kw)
    s = A.AVIRISCAMultilabelDataset(ROOT, mode="series", verbose=False, **kw)
    assert len(d) == n_dates and len(s) == n_windows == len(np.unique(d.window_ids))
    if n_classes is not None:
        assert len(d.classes) == len(s.classes) == n_classes
    folds = d.meta["fold"].to_numpy()
    rec = d.splits["recommended"]
    assert set(folds[rec["train"]]) <= {1, 2, 3} and set(folds[rec["test"]]) == {4}
    assert_no_leakage(d.groups[rec["train"]], d.groups[rec["test"]])
    if name == "stability":
        counts = np.bincount(s.labels)
        assert d.classes == ["stable", "changing"] and counts.tolist() == [569, 969]
