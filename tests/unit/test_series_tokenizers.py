"""Tokenizers of the covariance series (spdnet_datasets.series.tokenizers) and mode registry."""

import numpy as np
import pytest

from spdnet_datasets.real.series import (
    sliding_covariances,
    spectral_windows,
    window_starts,
)
from spdnet_datasets.series import SERIES_MODES, get_mode
from spdnet_datasets.series import tokenizers as tok


def test_window_starts_cover_and_regular():
    assert window_starts(10, 4, 2) == [0, 2, 4, 6]
    assert window_starts(11, 4, 2) == [0, 2, 4, 6]
    assert window_starts(11, 4, 2, cover=True) == [0, 2, 4, 6, 7]
    assert window_starts(10, 4, n_windows=3) == [0, 3, 6]
    assert window_starts(3, 4, 2) == []


def test_axis_windows_rules():
    assert tok.axis_windows(427, 86, 43)[-1] == (341, 427)  # last window flush with the end
    assert len(tok.axis_windows(427, 86, 43)) == 9
    assert tok.axis_windows(100, None, n_windows=4) == [(0, 40), (20, 60), (40, 80), (60, 100)]
    assert tok.axis_windows(30, 48, 24) == [(0, 30)]  # shorter than a window: one token
    assert tok.axis_windows(352, 58, n_windows=6)[-1] == (294, 352)
    with pytest.raises(ValueError):
        tok.axis_windows(10)


def test_temporal_shapes_coverage_and_order():
    x = np.arange(100 * 3, dtype=float).reshape(100, 3)
    obs, pos = tok.temporal(x, window=20, step=10)
    assert len(obs) == 9 and all(o.shape == (20, 3) for o in obs)
    assert [p[0] for p in pos] == [(s, s + 20) for s in range(0, 81, 10)]
    assert pos[0][1] == (0, 3)
    covered = np.zeros(100, bool)
    for (a, b), _ in pos:
        covered[a:b] = True
    assert covered.all()
    np.testing.assert_array_equal(obs[3], x[30:50])


def test_temporal_reuses_sliding_covariances():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 4))
    obs, _ = tok.temporal(x, window=40, step=20)
    ref = sliding_covariances(x, 40, 20, "scm", load=0.0)
    ours = np.stack([np.cov(o.T, bias=True) for o in obs])
    np.testing.assert_allclose(ours, ref)


def test_spectral_matches_spectral_windows():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 6, 64))
    obs, pos = tok.spectral(x, 16, 8)
    assert len(obs) == 7 and all(o.shape == (36, 16) for o in obs)
    ref = spectral_windows(x.reshape(-1, 64), 16, 8, "scm")
    ours = np.stack([np.cov(o.T, bias=True) for o in obs])
    np.testing.assert_allclose(ours, ref, rtol=1e-5)  # ref has a 1e-6 relative loading
    assert pos[-1] == [(0, 6), (0, 6), (48, 64)]


def test_spectral_adds_last_window_when_step_does_not_tile():
    obs, pos = tok.spectral(np.zeros((4, 4, 63)), 16, 8)
    assert [p[-1] for p in pos][-1] == (47, 63) and len(obs) == 7


def test_spatial_grid_row_major():
    x = np.arange(15 * 15 * 2, dtype=float).reshape(15, 15, 2)
    obs, pos = tok.spatial(x, (8, 8), (4, 4))
    assert len(obs) == 9 and all(o.shape == (64, 2) for o in obs)
    assert [p[:2] for p in pos[:3]] == [[(0, 8), (0, 8)], [(0, 8), (4, 12)], [(0, 8), (7, 15)]]
    np.testing.assert_array_equal(obs[4], x[4:12, 4:12].reshape(-1, 2))


def test_spatio_spectral_cartesian_product_spatial_major():
    x = np.random.default_rng(2).standard_normal((16, 16, 32))
    obs, pos = tok.spatio_spectral(x, {"window": (8, 8), "step": (8, 8)},
                                   {"band_window": 16, "band_step": 16})
    assert len(obs) == 4 * 2
    assert pos[0] == [(0, 8), (0, 8), (0, 16)] and pos[1] == [(0, 8), (0, 8), (16, 32)]
    assert pos[2][:2] == [(0, 8), (8, 16)]
    np.testing.assert_array_equal(obs[3], x[0:8, 8:16, 16:32].reshape(-1, 16))


def test_spectro_temporal_time_major():
    x = np.random.default_rng(3).standard_normal((100, 32))
    obs, pos = tok.spectro_temporal(x, {"n_windows": 4}, {"band_window": 16, "band_step": 16})
    assert len(obs) == 8 and obs[0].shape == (40, 16)
    assert pos[0] == [(0, 40), (0, 16)] and pos[1] == [(0, 40), (16, 32)]
    np.testing.assert_array_equal(obs[3], x[20:60, 16:32])


def test_spatio_temporal_and_parts():
    x = np.random.default_rng(4).standard_normal((10, 4, 4, 3))
    obs, pos = tok.spatio_temporal(x, {"window": 5, "step": 5}, {"window": (2, 2), "step": (2, 2)})
    assert len(obs) == 2 * 4 and obs[0].shape == (5 * 4, 3)
    ref = x[0:5, 0:2, 2:4].reshape(-1, 3)
    np.testing.assert_allclose(np.sort(obs[1], axis=0), np.sort(ref, axis=0))
    assert pos[1] == [(0, 5), (0, 2), (2, 4), (0, 3)]
    obs, groups = tok.parts(np.arange(12.0).reshape(4, 3), [[0, 2], [1]])
    assert groups == [(0, 2), (1,)] and obs[0].shape == (4, 2)


def test_pool_channels():
    x = np.arange(12.0).reshape(2, 6)
    np.testing.assert_allclose(tok.pool_channels(x, 3), [[0.5, 2.5, 4.5], [6.5, 8.5, 10.5]])
    assert tok.pool_channels(x, None) is x
    with pytest.raises(ValueError):
        tok.pool_channels(x, 7)


def test_every_registered_mode_meets_its_rules():
    for name, modes in SERIES_MODES.items():
        for mode in modes.values():
            S, p, n_t = mode.nominal()
            assert n_t >= mode.min_n_over_p * p, (name, mode.name, n_t, p)
            assert 4 <= S <= 32, (name, mode.name, S)


def test_registry_defaults_of_the_user_example():
    mode = get_mode("hyperleaf")
    assert mode.name == "spatio_spectral"
    assert mode.nominal() == (24, 48, 48 * 58)


def test_get_mode_errors():
    with pytest.raises(ValueError, match="No covariance-series mode"):
        get_mode("nope")
    with pytest.raises(ValueError, match="has no mode"):
        get_mode("esc50", "nope")
    with pytest.raises(ValueError, match="no parameter"):
        get_mode("esc50").resolve(windw=3)
