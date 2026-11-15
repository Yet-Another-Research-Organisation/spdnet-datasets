"""MOABB adapter on real data: BNCI2014_001, subject 1 (downloaded by MOABB, ~80 MB).

Run with ``pytest -m "slow and network"``. MOABB stores the recordings under
``MNE_DATA``; the covariances go to ``tmp_path``.
"""

import numpy as np
import pytest

from spdnet_datasets.core.estimators import check_spd
from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.moabb import Covariances, MOABBCovariances

pytestmark = [pytest.mark.slow, pytest.mark.network]

moabb = pytest.importorskip("moabb")


@pytest.fixture(scope="module")
def bnci():
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    return BNCI2014_001(), LeftRightImagery()


def test_moabb_covariances_bnci2014_001(bnci, tmp_path):
    dataset, paradigm = bnci
    ds = MOABBCovariances(dataset, paradigm, subjects=[1], estimator="ledoit_wolf",
                          return_raw=True, split="cross_session_0", cache_dir=tmp_path)
    assert len(ds) == 288 and ds.classes == ["left_hand", "right_hand"]
    (cov, raw), y = ds[0]
    assert tuple(cov.shape) == (22, 22) and tuple(raw.shape) == (1001, 22)
    check_spd(ds.covariances)
    assert set(ds.sessions) == {"0train", "1test"}
    part = ds.splits["cross_session_0"]
    assert_no_leakage(ds.sessions[part["train"]], ds.sessions[part["test"]], what="session")
    assert sum(n.startswith("within_session") for n in ds.splits) == 2 * 5


def test_moabb_evaluations_accept_the_transformer(bnci, tmp_path):
    from moabb.evaluations import CrossSessionEvaluation, WithinSessionEvaluation
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    dataset, paradigm = bnci
    dataset.subject_list = [1]
    pipelines = {"LW+TS+LR": make_pipeline(Covariances("ledoit_wolf"), TangentSpace(),
                                           LogisticRegression(max_iter=1000))}
    for Evaluation in (WithinSessionEvaluation, CrossSessionEvaluation):
        evaluation = Evaluation(paradigm=paradigm, datasets=[dataset], overwrite=True,
                                hdf5_path=str(tmp_path), random_state=0)
        results = evaluation.process(pipelines)
        print(Evaluation.__name__, results[["session", "score"]].to_dict("records"))
        assert len(results) == 2 and np.all(results["score"] > 0.6)
