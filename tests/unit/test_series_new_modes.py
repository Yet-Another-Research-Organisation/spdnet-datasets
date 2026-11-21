"""Modes of the SAR, EEG and fMRI covariance series: tokenizers, features, loader hooks."""

import numpy as np
import pytest

from spdnet_datasets.manager import DatasetManager
from spdnet_datasets.sar._common import log_amplitude, texture_features, tuzel_features
from spdnet_datasets.sar.cpazmal import pair_dates
from spdnet_datasets.series import CovarianceSeries
from spdnet_datasets.series import dataset as series_dataset
from spdnet_datasets.series.modes import (
    FILTER_BANK,
    SERIES_MODES,
    _date_windows,
    _filter_bank,
    _p_windows,
)


def test_date_windows_cover_the_calendar_channel_major():
    x = np.arange(30 * 5 * 2, dtype=float).reshape(30, 5, 2)  # dates, pixels, (HH, HV)
    obs, pos = _date_windows(x, window=12, step=6)
    assert [p[0] for p in pos] == [(0, 12), (6, 18), (12, 24), (18, 30)]
    assert all(o.shape == (5, 24) for o in obs)
    # pixel 3, window 2: HH on dates 6..17, then HV on the same dates
    np.testing.assert_array_equal(obs[1][3], np.r_[x[6:18, 3, 0], x[6:18, 3, 1]])
    obs, pos = _date_windows(np.zeros((21, 4, 2)), window=12, step=6)
    assert [p[0] for p in pos] == [(0, 12), (6, 18), (9, 21)]  # last window flush with the end


def test_cpazmal_modes_nominal_sizes():
    assert SERIES_MODES["cpazmal"]["dates"].nominal() == (4, 24, 144)
    assert SERIES_MODES["cpazmal"]["dates_both"].nominal() == (6, 24, 144)


def test_pair_dates_is_one_to_one_in_time_order():
    a = ["20200110", "20200121", "20200212"]
    b = ["20200108", "20200119", "20200130", "20200210", "20200221"]
    out = pair_dates(a, b)
    assert out == ["20200108", "20200119", "20200210"]
    assert len(set(out)) == len(out) and out == sorted(out)
    # nearest-date matching would use 20200119 twice here; the pairing keeps dates distinct
    assert pair_dates(["20200110", "20200118"], ["20200117", "20200130"]) == ["20200117",
                                                                             "20200130"]
    with pytest.raises(ValueError):
        pair_dates(b, a)


def test_texture_features():
    rng = np.random.default_rng(0)
    amp = rng.rayleigh(size=(16, 16, 2))
    amp[0, 0, 0] = 0.0  # a zero amplitude stays finite
    logs = log_amplitude(amp)
    assert np.isfinite(logs).all()
    f = texture_features(logs)
    assert f.shape == (16, 16, 4)
    np.testing.assert_allclose(f[..., :2], logs)
    ramp = np.tile(np.arange(8.0), (8, 1))[..., None]  # increases along columns (x)
    g = texture_features(ramp)
    np.testing.assert_allclose(g[2:-2, 2:-2, 1], 2.0)  # |[1, 0, -1] * ramp| = 2
    np.testing.assert_allclose(g[..., 2], 0.0)


def test_tuzel_features():
    H, W = 6, 8
    z = (np.arange(W)[None] * np.ones((H, 1)) + 1.0) * np.exp(1j * 0.3)
    f = tuzel_features(z)
    assert f.shape == (H, W, 9)
    np.testing.assert_array_equal(f[2, 5, :2], [5, 2])  # x = column, y = row
    np.testing.assert_allclose(f[..., 2], np.abs(z))
    np.testing.assert_allclose(f[:, 2:-2, 3], 2.0)  # |dI/dx| of a ramp
    np.testing.assert_allclose(f[..., 4], 0.0)
    np.testing.assert_allclose(f[:, 2:-2, 5], 0.0, atol=1e-12)  # second derivative of a ramp
    np.testing.assert_allclose(f[..., 7:], np.broadcast_to([np.cos(0.3), np.sin(0.3)], (H, W, 2)))


@pytest.mark.parametrize("name,shape", [("s1slc_cvdl", (5 * 5, 4)), ("atrnet_star", (16, 6)),
                                        ("air_polsar_seg", (64, 6)), ("mstar", (16, 9))])
def test_sar_grid_nominal(name, shape):
    mode = SERIES_MODES[name]["spatial"]
    S, p, n_t = mode.nominal()
    assert (S, p) == shape
    side = mode.defaults["window"][0]
    assert n_t == side * side and mode.loader_defaults["return_raw"] is True
    assert mode.standardize == "last"


def test_filter_bank_isolates_each_band():
    assert len(FILTER_BANK) == 9 and FILTER_BANK[0] == (4, 8) and FILTER_BANK[-1] == (36, 40)
    fs, t = 250.0, np.arange(1000) / 250.0
    x = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 30 * t)], -1)
    obs, _ = _filter_bank(x, sfreq=fs)
    power = np.array([(o ** 2).mean(0) for o in obs])  # (bands, channels)
    assert power[:, 0].argmax() == 1 and power[:, 1].argmax() == 6  # 8-12 and 28-32 Hz
    with pytest.warns(UserWarning, match="pass band"):
        _filter_bank(x, sfreq=fs, passband=(8.0, 32.0))


def test_p_windows_rules():
    # 4C when it gives at least 4 windows (BNCI2014_001: C = 22, 1001 samples)
    obs, pos = _p_windows(np.zeros((1001, 22)), window_over_p=4, min_windows=4)
    assert pos[0][0] == (0, 88) and len(obs) == 22
    # shrunk to 4 windows on a short epoch (PhysionetMI: C = 64, 481 samples) ...
    obs, _ = _p_windows(np.zeros((481, 64)), window_over_p=4, min_windows=4)
    assert len(obs) == 4 and len(obs[0]) == 193
    # ... never below 2C
    obs, _ = _p_windows(np.zeros((300, 64)), window_over_p=4, min_windows=4)
    assert len(obs[0]) == 128 and len(obs) == 4  # 0, 64, 128 and 172 (flush)
    # fMRI: 2p volumes at 50 % (NKI: p = 148, T = 895)
    obs, _ = _p_windows(np.zeros((895, 148)), window_over_p=2)
    assert len(obs) == 6 and all(len(o) == 296 for o in obs)


class _Epochs:
    """Fake adapter (named as the MOABB one) with a sampling rate and short epochs."""

    split, subset, task = None, None, "fine"
    classes = ["a", "b"]

    def __init__(self, n_times=400, sfreq=100.0, passband=(1.0, 45.0)):
        self.n_times, self.sfreq, self.passband = n_times, sfreq, passband
        self.labels = np.arange(8) % 2
        self.groups = np.repeat([0, 1], 4)
        idx = np.arange(8)
        self.splits = {"recommended": {"train": idx[:4], "val": idx[:0], "test": idx[4:]}}

    def __len__(self):
        return 8

    def structured_raw(self, i):
        return np.random.default_rng(i).standard_normal((self.n_times, 3))


_Epochs.__name__ = "MOABBCovariances"


def test_adapter_instance_sfreq_and_too_short_epochs(tmp_path):
    s = CovarianceSeries(_Epochs(), "filter_bank", cache_dir=tmp_path, verbose=False)
    assert s.name == "moabb" and s.split == "recommended"
    assert s.params["sfreq"] == 100.0 and s.params["passband"] == (1.0, 45.0)
    assert s.describe()["S"] == 9 and s.describe()["p"] == 3
    s = CovarianceSeries(_Epochs(), "temporal", cache_dir=tmp_path, verbose=False)
    assert s.describe()["S"] == 66  # 4C = 12 samples, step 6, over 400 samples
    with pytest.raises(ValueError, match="too short"):  # a single window of 5 samples
        CovarianceSeries(_Epochs(n_times=5), "temporal", cache_dir=tmp_path, verbose=False,
                         allow_small_n=True)
    with pytest.raises(ValueError, match="adapter"):
        series_dataset._build_loader("moabb", SERIES_MODES["moabb"]["temporal"], None)


def test_nilearn_needs_three_windows(tmp_path):
    class Scans(_Epochs):
        short = ()

        def structured_raw(self, i):
            n = 12 if i in self.short else self.n_times
            return np.random.default_rng(i).standard_normal((n, 5))

    Scans.__name__ = "NilearnConnectomes"
    s = CovarianceSeries(Scans(n_times=40), "temporal", cache_dir=tmp_path, verbose=False)
    assert s.name == "nilearn" and s.describe()["S"] == 7  # windows of 2p = 10 volumes
    assert s.describe()["n_short"] == 0
    scans = Scans(n_times=40)
    scans.short = (5,)  # one short scan (2 windows): kept and counted, not an error
    d = CovarianceSeries(scans, "temporal", cache_dir=tmp_path / "b", verbose=False).describe()
    assert d["n_short"] == 1 and d["S"] == 7
    with pytest.raises(ValueError, match="fewer than 3"):
        CovarianceSeries(Scans(n_times=14), "temporal", cache_dir=tmp_path, verbose=False)


def test_source_key_covers_the_mode_loader_settings():
    class L:
        def __init__(self):
            self.source, self.labels = "complex", np.zeros(2)

        def __len__(self):
            return 2

    a = series_dataset._source_key(L(), SERIES_MODES["mstar"]["spatial"])
    b = L()
    b.source = "jpg"
    assert a != series_dataset._source_key(b, SERIES_MODES["mstar"]["spatial"])


def test_sar_modes_are_registered_loaders():
    from spdnet_datasets import sar  # noqa: F401  (registration)

    for name in ("cpazmal", "s1slc_cvdl", "atrnet_star", "air_polsar_seg", "mstar"):
        assert name in DatasetManager.DATASETS and name in SERIES_MODES
    assert "fusar_ship" not in SERIES_MODES  # no series for FUSAR-Ship (user decision)


def test_ntu_part_windows_last_at_least_2p_frames():
    part = SERIES_MODES["ntu120"]["part"]
    x = np.random.default_rng(0).standard_normal((40, 25, 3))  # 40 frames: a third is 13
    obs, pos = part.tokenize(part.prepare(x), **part.defaults)
    assert len(obs) == 30 and all(len(o) == 24 for o in obs)  # 6 windows of 2p = 24 frames
    obs, _ = part.tokenize(part.prepare(x), **{**part.defaults, "min_window": 4})
    assert all(len(o) == 13 for o in obs)  # the former windows: masked tokens (13 < 24)
    pt = SERIES_MODES["ntu120"]["part_temporal"]
    obs, _ = pt.tokenize(x, **pt.defaults)
    assert len(obs) == 20 and all(len(o) == 30 for o in obs)


def test_hyperleaf_band_windows_cover_every_band():
    mode = SERIES_MODES["hyperleaf"]["spatio_spectral"]
    _, pos = mode.tokenize(np.zeros(mode.example_shape, np.float32), **mode.defaults)
    bands = sorted({tuple(p[2]) for p in pos})
    assert bands == [(0, 48), (39, 87), (78, 126), (117, 165), (156, 204)]
