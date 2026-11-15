"""Loaders of the R17 bilan (BreizhCrops, MVDoppler, UrbanSound8K, PASTIS-R S2 + S1): shapes, SPD tokens, official splits, and the same
samples with and without series (``pooled``). Skipped when the data is not on this machine."""

import os

import numpy as np
import pytest

from spdnet_datasets.core import InsufficientSamplesError
from spdnet_datasets.core.config import dataset_path, get_data_root
from spdnet_datasets.real.series_audio import _series_job
from spdnet_datasets.real.series_breizhcrops import breizhcrops_series
from spdnet_datasets.real.series_mvdoppler import mvdoppler_series
from spdnet_datasets.real.series_pastis_fusion import pastis_fusion_series

AUDIO_ROOT = str(get_data_root() / "AUDIO")
BREIZHCROPS = str(dataset_path("SITS", "BreizhCrops") / "L2A_h5")
MVDOPPLER = str(dataset_path("RADAR_DOPPLER", "MVDoppler"))
PASTIS = str(dataset_path("SITS", "PASTIS-R") / "PASTIS-R")
pytestmark = pytest.mark.real_data


def _check(series, pooled, T, p):
    (data, labels, _, extra), (flat, flat_labels, _, flat_extra) = series, pooled
    assert data.shape[1:] == (T, p, p) and flat.shape[1:] == (1, p, p)
    np.testing.assert_array_equal(labels, flat_labels)
    np.testing.assert_array_equal(extra["split"], flat_extra["split"])
    assert set(np.unique(extra["split"])) == {0, 1, 2}
    valid = (data != np.eye(p)).any((-1, -2))
    assert valid.any(1).all() and np.all(np.linalg.eigvalsh(data[valid]) > 0)
    assert np.all(np.linalg.eigvalsh(flat[:, 0]) > 0)


@pytest.mark.skipif(not os.path.isdir(BREIZHCROPS), reason="BreizhCrops not available")
def test_breizhcrops_quarters():
    caps = (60, 20, 30)
    with pytest.raises(InsufficientSamplesError):  # 3 clear dates per quarter for p = 4: derogation
        breizhcrops_series(caps=caps)
    _check(breizhcrops_series(caps=caps, allow_small_n=True),
           breizhcrops_series(caps=caps, pooled=True, allow_small_n=True), 4, 4)


@pytest.mark.skipif(not os.path.isdir(MVDOPPLER), reason="MVDoppler not available")
def test_mvdoppler_windows():
    caps = (40, 12, 20)
    series = mvdoppler_series(caps=caps, jobs=1)
    _check(series, mvdoppler_series(caps=caps, pooled=True, jobs=1), 7, 16)
    assert len(series[1]) <= sum(caps)  # floor of the per-class quotas


@pytest.mark.skipif(not os.path.isdir(f"{AUDIO_ROOT}/UrbanSound8K"), reason="UrbanSound8K not available")
def test_urbansound8k_resampled_windows():
    blob = open(f"{AUDIO_ROOT}/UrbanSound8K/audio/fold5/100852-0-0-25.wav", "rb").read()  # 4 s slice
    series = _series_job((blob, None, 1.0, 6, False, 22050))
    assert series.shape == (6, 20, 20) and (series != np.eye(20)).any((-1, -2)).all()
    assert np.all(np.linalg.eigvalsh(series) > 0)


@pytest.mark.skipif(not os.path.isdir(PASTIS), reason="PASTIS-R not available")
def test_pastis_fusion_periods():
    caps = (40, 12, 20)
    series = pastis_fusion_series(caps=caps, jobs=1)
    _check(series, pastis_fusion_series(caps=caps, pooled=True, jobs=1), 12, 14)
    assert series[2].min() >= 30  # pixels per covariance > 2p
