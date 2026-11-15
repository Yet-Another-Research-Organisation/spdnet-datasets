"""Audio covariance series (R10-C): window count, SPD tokens, silent windows as identity tokens."""

import numpy as np

from spdnet_datasets.real.series_audio import mfcc_window_series


def test_mfcc_window_series_spd_and_silence_as_identity():
    sr = 22050
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5 * sr)
    x[3 * sr:] = 0.0  # padding silence over the last two seconds
    series = mfcc_window_series(x, sr, window_s=1.0, t_max=10)
    assert series.shape == (10, 20, 20)
    identity = (series == np.eye(20)).all((-1, -2))
    assert identity[-1] and identity[-2]  # beyond the last window
    assert not identity[0] and identity[7]  # a sound window, a silent one (> 50 % zeros)
    assert np.all(np.linalg.eigvalsh(series[~identity]) > 0)
