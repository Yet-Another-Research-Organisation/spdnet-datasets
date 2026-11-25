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
