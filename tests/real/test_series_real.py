"""CovarianceSeries on the real copies, one dataset per family (small class subsets).

Checks, for the default mode: tokens SPD or masked identity, token = covariance of its
observations times the scale, splits of the loader, cache replayed.
"""

import numpy as np
import pytest

from spdnet_datasets.core import check_spd, estimate_covariance
from spdnet_datasets.series import CovarianceSeries
from tests.conftest import dataset_dir, skip_if_missing

pytestmark = pytest.mark.real_data

CASES = {  # family: (name, mode, data folder, loader arguments)
    "audio": ("esc50", None, ("AUDIO", "ESC-50"), {"max_samples_per_class": 2}),
    "radar": ("ci4r", None, ("RADAR_DOPPLER", "CI4R-MULTI3"), {"max_samples_per_class": 3}),
    "hsi": ("gsoff", "spatio_spectral", ("HYPERSPECTRAL", "GSOFF"),
            {"max_samples_per_class": 5}),
    "hsi_image": ("hyperleaf", None, ("HYPERSPECTRAL", "HyperLeaf2024"),
                  {"max_samples_per_class": 3}),
    "skeleton": ("ntu120", "frame", ("SKELETON", "NTU_RGBD_120"),
                 {"max_samples_per_class": 3}),
    "sits": ("pastis_r", None, ("SITS", "PASTIS-R"), {"max_samples_per_class": 2}),
    "sar": ("pol_insar_island", None, ("SAR", "Pol-InSAR-Island"),
            {"max_samples_per_class": 3}),
}


@pytest.mark.parametrize("family", sorted(CASES))
def test_default_mode_on_real_data(family, tmp_path):
    name, mode, folder, loader_kwargs = CASES[family]
    skip_if_missing(dataset_dir(*folder))
    series = CovarianceSeries(name, mode, loader_kwargs={**loader_kwargs, "verbose": False},
                              cache_dir=tmp_path, verbose=False)
    d = series.describe()
    assert len(series) > 0 and d["S"] >= 1 and d["spd_fraction"] > 0.5
    for i in (0, len(series) - 1):
        (tokens, mask), _ = series[i]
        tokens, mask = tokens.numpy(), mask.numpy()
        check_spd(tokens[mask])
        np.testing.assert_array_equal(tokens[~mask], np.broadcast_to(np.eye(d["p"]),
                                                                      tokens[~mask].shape))
        obs, _ = series.tokenize(i)
        for k in np.flatnonzero(mask)[:4]:
            ref = estimate_covariance(obs[k], series.estimator, check=False) * series.scale
            np.testing.assert_allclose(tokens[k], ref, rtol=1e-6, atol=1e-10 * np.abs(ref).max())
    for split, parts in series.loader.splits.items():
        for part, idx in parts.items():
            np.testing.assert_array_equal(series.splits[split][part], idx)
    again = CovarianceSeries(name, mode, loader_kwargs={**loader_kwargs, "verbose": False},
                             cache_dir=tmp_path, verbose=False)
    assert again.cache_path == series.cache_path and again.scale == series.scale


def test_hdm05_raises_a_clear_error(tmp_path):
    skip_if_missing(dataset_dir("SKELETON", "HDM05"))
    with pytest.raises(ValueError, match="covariances"):
        CovarianceSeries("hdm05", loader_kwargs={"verbose": False}, cache_dir=tmp_path)
