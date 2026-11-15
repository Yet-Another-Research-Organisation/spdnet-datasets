"""Covariance-series builders on tiny synthetic file trees: estimation through ``core``, ``n >= 2p``
checks and derogations, train-only normalisations, and the regressions of the code review of the
work in progress (findings 1-10)."""

import io
import os

import numpy as np
import pandas as pd
import pytest

from spdnet_datasets.core import InsufficientSamplesError
from spdnet_datasets.real.series import (
    covariance,
    sliding_covariances,
    spectral_windows,
)


def is_spd(x):
    return np.allclose(x, np.swapaxes(x, -1, -2)) and np.linalg.eigvalsh(x).min() > 0


# --------------------------------------------------------------------------- shared helpers


def test_covariance_enforces_two_p_unless_allowed():
    x = np.random.default_rng(0).normal(size=(10, 8))
    with pytest.raises(InsufficientSamplesError):
        covariance(x)
    c = covariance(x, allow_small_n=True)
    assert c.shape == (8, 8) and is_spd(c)


def test_covariance_is_core_estimate_plus_relative_loading():
    from sklearn.covariance import ledoit_wolf

    x = np.random.default_rng(1).normal(size=(50, 4))
    expected = ledoit_wolf(x)[0]
    expected += 1e-6 * np.trace(expected) / 4 * np.eye(4)
    np.testing.assert_allclose(covariance(x), expected, rtol=1e-10)
    scm = np.cov(x.T, bias=True)
    np.testing.assert_allclose(covariance(x, "scm", load=0.0), scm, rtol=1e-12)


def test_sliding_covariances_rejects_series_shorter_than_the_window():  # finding 6
    x = np.random.default_rng(0).normal(size=(30, 3))
    with pytest.raises(ValueError, match="no complete window"):
        sliding_covariances(x, 40, 10)
    covs, means = sliding_covariances(x, 10, 10, return_mean=True)
    assert covs.shape == (3, 3, 3) and means.shape == (3, 3)
    np.testing.assert_allclose(means[1], x[10:20].mean(0))


def test_spectral_windows_mean():
    x = np.random.default_rng(0).normal(size=(64, 12))
    covs, means = spectral_windows(x, 4, 4, return_mean=True)
    assert covs.shape == (3, 4, 4) and means.shape == (3, 4) and is_spd(covs)
    np.testing.assert_allclose(means[2], x[:, 8:12].mean(0))


# --------------------------------------------------------------------------- CI4R (findings 1, 2)


def _jet_png(path, values):
    from matplotlib import colormaps
    from PIL import Image

    rgb = (colormaps["jet"](values)[..., :3] * 255).round().astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb).save(path)


@pytest.fixture
def ci4r_root(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "11_class_activity_data" / "77GHz"
    for name, cls in (("s01_a.png", "Away"), ("s02_b.png", "Bend")):
        values = rng.uniform(0.2, 1.0, (64, 120))
        values[:, :40] = 0.0  # first window entirely constant (zero after the jet inversion)
        values[:, 40:50] = 0.0  # zero columns inside the second window
        _jet_png(str(folder / cls / name), values)
    _jet_png(str(folder / "Away" / "s01_a (copy).png"), rng.uniform(0, 1, (64, 120)))
    return str(tmp_path)


def test_ci4r_n_obs_per_image_and_pooled(ci4r_root):  # finding 1
    from spdnet_datasets.real.series import ci4r_series

    with pytest.warns(UserWarning, match="constant windows"):
        data, labels, n_obs, groups = ci4r_series(ci4r_root, n_bands=4, window=40, step=20)
    assert data.shape == (2, 5, 4, 4) and list(labels) == [0, 1] and len(set(groups)) == 2
    assert n_obs.shape == (2,) and (n_obs == 40).all()
    pooled, _, n_pooled, _ = ci4r_series(ci4r_root, n_bands=4, window=40, step=20, pooled_cov=True)
    assert pooled.shape == (2, 1, 4, 4) and n_pooled.shape == (2,) and (n_pooled == 120).all()


def test_ci4r_keeps_zero_columns_and_masks_constant_windows(ci4r_root):  # finding 2
    from PIL import Image

    from spdnet_datasets.real.series import ci4r_series, jet_to_scalar

    with pytest.warns(UserWarning):
        data, *_ , mean = ci4r_series(ci4r_root, n_bands=4, window=40, step=20, return_mean=True)
    assert mean.shape == (2, 5, 4)
    np.testing.assert_array_equal(data[:, 0], np.broadcast_to(np.eye(4), (2, 4, 4)))  # constant window
    assert is_spd(data)
    path = os.path.join(ci4r_root, "11_class_activity_data", "77GHz", "Away", "s01_a.png")
    image = jet_to_scalar(np.asarray(Image.open(path).convert("RGB")))
    pooled = image[16:48].reshape(4, 8, -1).mean(1).T
    np.testing.assert_allclose(data[0, 2], covariance(pooled[40:80]))  # the 10 zero columns are kept


# --------------------------------------------------------------------------- Dop-NET (finding 7)


def test_dopnet_n_obs_is_frames_per_covariance(tmp_path):
    from scipy.io import savemat

    from spdnet_datasets.real.series import dopnet_series

    rng = np.random.default_rng(0)
    cells = np.empty((1, 4), dtype=object)
    for k in range(4):
        gestures = np.empty((2, 1), dtype=object)
        for j, frames in enumerate((40, 100)):
            gestures[j, 0] = rng.normal(size=(800, frames)) + 1j * rng.normal(size=(800, frames))
        cells[0, k] = gestures
    savemat(tmp_path / "Data_Training_Person_A.mat", {"Data_Training": {"Doppler_Signals": cells}})
    data, labels, n_obs, persons = dopnet_series(str(tmp_path))
    assert data.shape == (8, 3, 16, 16) and is_spd(data)
    assert list(labels) == [0, 0, 1, 1, 2, 2, 3, 3] and set(persons) == {0}
    assert list(n_obs) == [40, 48] * 4  # min(window, length), not the gesture length


# --------------------------------------------------------------------------- AVIRIS (finding 8, train-only)

AVIRIS_PROPORTIONS = ["p_urbain_dense", "p_urbain_faible", "p_urbain_moyen", "p_urbain_ouvert", "p_foret_coniferes",
                      "p_foret_mixte", "p_arbustes", "p_prairie_paturage", "p_amandiers", "p_vignes", "p_eau",
                      "p_sol_nu", "p_autre"]


def _aviris_root(tmp_path, test_factor=1.0):
    rng = np.random.default_rng(0)
    root = tmp_path / f"aviris{test_factor}"
    (root / "meta").mkdir(parents=True)
    (root / "windows").mkdir()
    ids = ["A", "B", "C"]
    table = pd.DataFrame({"window_id": ids, "dominant": "eau", "block10km": [0, 1, 2], "fold5": [1, 1, 0],
                          "n_ok_dates": 5, **dict.fromkeys(AVIRIS_PROPORTIONS, 0.0)})
    table["p_eau"] = 1.0
    table.to_csv(root / "meta" / "labels.csv", index=False)
    pd.DataFrame({"clean": [1] * 8 + [0, 0]}).to_csv(root / "meta" / "bands.csv", index=False)
    ok = np.zeros((3, 14), bool)
    ok[:, :3] = True
    np.savez(root / "meta" / "date_mask.npz", window_id=np.array(ids), ok=ok)
    for w in ids:
        cube = rng.uniform(0.1, 0.9, (14, 10, 32, 32))
        if w == "A":  # dates 1 and 2: three complete blocks of 200 pixels, the last block empty -> rejected
            for date in (1, 2):  # 4 rows of 16 pixels missing: 192 complete pixels per block
                cube[date, :, 0:4, :] = np.nan
                cube[date, :, 16:20, :16] = np.nan
                cube[date, :, 16:, 16:] = np.nan
        if w == "B":  # no accepted date
            cube[:3] = np.nan
        if w == "C":
            cube *= test_factor
        np.savez(root / "windows" / f"{w}_eau.npz", cube=cube.astype(np.float32))
    return str(root)


def test_aviris_drops_windows_without_dates_and_counts_accepted_dates_only(tmp_path):
    from spdnet_datasets.real.series_aviris import aviris_mix_series

    root = _aviris_root(tmp_path)
    with pytest.warns(UserWarning, match="1 windows without any accepted date"):
        data, labels, n_obs, groups = aviris_mix_series(root, width=4, n_groups=2, patches=2,
                                                        fit_mask=lambda s: s.fold5 != 0)
    assert data.shape == (2, 14 * 4 * 2, 4, 4) and list(groups) == [0, 2]
    assert list(n_obs) == [256, 256]  # the 192-pixel blocks of the rejected dates of A do not count
    valid = (data != np.eye(4)).any((-1, -2))
    assert valid[0].sum() == 8 and valid[1].sum() == 3 * 8 and is_spd(data[valid])


def test_aviris_scale_is_fitted_on_the_training_windows(tmp_path):
    from spdnet_datasets.real.series_aviris import aviris_mix_series

    kwargs = {"width": 4, "n_groups": 2, "patches": 2, "fit_mask": lambda s: s.fold5 != 0}
    root = _aviris_root(tmp_path)
    with pytest.warns(UserWarning):
        base = aviris_mix_series(root, **kwargs)[0]
        moved = aviris_mix_series(_aviris_root(tmp_path, test_factor=1.5), **kwargs)[0]
    np.testing.assert_array_equal(base[0], moved[0])  # training window unchanged
    assert not np.allclose(base[1], moved[1])
    with pytest.warns(UserWarning, match="fitted on all the samples"), pytest.warns(UserWarning, match="dropped"):
        aviris_mix_series(root, width=4, n_groups=2, patches=2)


def test_whiten_series_is_fitted_on_the_training_series():
    from spdnet_datasets.real.series_aviris import whiten_series

    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 4, 3, 20))
    data = a @ a.swapaxes(-1, -2) / 20
    fit = np.array([True, True, True, False, False, False])
    out = whiten_series(data, 2, fit_mask=fit)
    moved = data.copy()
    moved[~fit] *= 7.0
    np.testing.assert_allclose(whiten_series(moved, 2, fit_mask=fit)[fit], out[fit])
    np.testing.assert_allclose(whiten_series(data, 2, fit_mask=np.flatnonzero(fit)), out)
    with pytest.warns(UserWarning):
        whiten_series(data, 2)


# --------------------------------------------------------------------------- HyperLeaf (finding 9)


def test_hyperleaf_reuses_spectral_windows(tmp_path):
    import tifffile

    from spdnet_datasets.real.series import hyperleaf_series

    rng = np.random.default_rng(0)
    (tmp_path / "images").mkdir()
    cube = rng.integers(1000, 60000, (40, 20, 20)).astype(np.uint16)
    cube[:, :4] = 0  # background
    tifffile.imwrite(tmp_path / "images" / "0001.tiff", cube)
    pd.DataFrame({"ImageId": ["0001"], "Heerup": [0], "Kvium": [1], "Rembrandt": [0], "Sheriff": [0]}).to_csv(
        tmp_path / "train.csv", index=False)
    data, labels, n_pixels = hyperleaf_series(str(tmp_path), band_window=8, band_step=8)
    pixels = cube.reshape(40, -1).T.astype(np.float64) / 65535.0
    pixels = pixels[pixels.sum(1) > 0]
    np.testing.assert_allclose(data[0], spectral_windows(pixels, 8, 8, "scm"))
    assert list(labels) == [1] and list(n_pixels) == [len(pixels)]
    zones = hyperleaf_series(str(tmp_path), band_window=8, band_step=8, zones=(2, 1))[0]
    assert zones.shape == (1, 10, 8, 8) and is_spd(zones)


# --------------------------------------------------------------------------- GTZAN (finding 10)


def test_gtzan_decodes_each_clip_once(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sf = pytest.importorskip("soundfile")

    from spdnet_datasets.real import series_audio

    rng = np.random.default_rng(0)
    clips = [rng.normal(0, 0.1, 4 * 8000) for _ in range(2)]
    blobs = []
    for x in (clips[0], clips[1], clips[0]):  # the third clip duplicates the first
        buffer = io.BytesIO()
        sf.write(buffer, x, 8000, format="WAV", subtype="FLOAT")
        blobs.append(buffer.getvalue())
    table = pa.table({"audio": [{"bytes": b, "path": f"{i}.wav"} for i, b in enumerate(blobs)], "genre": [0, 1, 0]})
    pq.write_table(table, tmp_path / "train.parquet")
    calls = []
    decode = series_audio._decode
    monkeypatch.setattr(series_audio, "_decode", lambda *a: calls.append(1) or decode(*a))
    data, labels, n_obs, groups = series_audio.gtzan_series(str(tmp_path), t_max=2, jobs=1)
    assert len(calls) == 3  # once per clip
    assert data.shape == (3, 2, 20, 20) and groups[0] == groups[2] != groups[1]


# --------------------------------------------------------------------------- n < 2p derogations


def _uea_root(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "FingerMovements"
    folder.mkdir()
    for part, n in (("TRAIN", 10), ("TEST", 4)):
        lines = ["@problemName FingerMovements", "@data"]
        for i in range(n):
            channels = [",".join(f"{v:.4f}" for v in rng.normal(size=50)) for _ in range(28)]
            lines.append(":".join(channels + [("left", "right")[i % 2]]))
        (folder / f"FingerMovements_{part}.ts").write_text("\n".join(lines))
    return str(tmp_path)


def test_finger_movements_needs_allow_small_n(tmp_path):
    from spdnet_datasets.real.series_biosignal import uea_series

    root = _uea_root(tmp_path)
    with pytest.raises(InsufficientSamplesError):
        uea_series("FingerMovements", root)
    data, labels, n_obs, extra = uea_series("FingerMovements", root, allow_small_n=True)
    assert data.shape == (14, 8, 28, 28) and is_spd(data) and (n_obs == 20).all()
    assert set(extra["split"]) == {0, 1, 2}


def test_large_p_series_need_allow_small_n(tmp_path):
    from spdnet_datasets.real.series_large_p import (
        audio_mel_series,
        mvdoppler_window_series,
    )

    with pytest.raises(InsufficientSamplesError):
        audio_mel_series("esc50", root=str(tmp_path))
    with pytest.raises(InsufficientSamplesError):
        mvdoppler_window_series(str(tmp_path))


# --------------------------------------------------------------------------- NTU (train-only statistics)


def _ntu_root(tmp_path, test_factor=1.0, frames=100):
    rng = np.random.default_rng(0)
    root = tmp_path / f"ntu{test_factor}"
    for action in (1, 2):
        (root / f"A{action:03d}").mkdir(parents=True)
        for subject in (1, 2, 4, 5, 3, 6):  # 3 and 6 are test subjects of the cross-subject protocol
            x = rng.normal(size=(frames, 25, 3)) + np.arange(25)[None, :, None]
            if subject in (3, 6):
                x = x * test_factor
            name = f"S001C001P{subject:03d}R001A{action:03d}.npy"
            np.save(root / f"A{action:03d}" / name, {"skel_body0": x}, allow_pickle=True)
    return str(root)


@pytest.mark.parametrize("builder", ["ntu_part_series", "ntu_jointcov_series", "ntu_axis_series"])
def test_ntu_standardisation_is_fitted_on_the_training_split(tmp_path, builder):
    from spdnet_datasets.real import series_ntu

    function = getattr(series_ntu, builder)
    kwargs = {"n_classes": 2, "max_per_class": None, "val_fraction": 0.25}
    if builder == "ntu_axis_series":
        kwargs = {"classes": (1, 2), "val_fraction": 0.25}
    data, labels, n_obs, extra = function(root=_ntu_root(tmp_path), **kwargs)
    moved, *_ = function(root=_ntu_root(tmp_path, test_factor=3.0), **kwargs)
    train = extra["split"] == 0
    assert train.any() and (extra["split"] == 2).any() and is_spd(data)
    np.testing.assert_allclose(moved[train], data[train])
    assert not np.allclose(moved[~train & (extra["split"] == 2)], data[~train & (extra["split"] == 2)])


def test_ntu_part_default_min_frames_keeps_two_p(tmp_path):
    from spdnet_datasets.real.series_ntu import ntu_part_series

    root = _ntu_root(tmp_path, frames=60)  # 60 frames: windows of 20 frames < 2p = 24
    with pytest.raises(ValueError):  # no sequence long enough: nothing to fit on
        ntu_part_series(root=root, n_classes=2, max_per_class=None)
    with pytest.raises(InsufficientSamplesError):
        ntu_part_series(root=root, n_classes=2, max_per_class=None, min_frames=24)


def test_ntu_window_series_needs_allow_small_n(tmp_path):
    from spdnet_datasets.real.series_large_p import ntu_window_series

    root = _ntu_root(tmp_path)
    with pytest.raises(InsufficientSamplesError):
        ntu_window_series(root=root, n_classes=2, max_per_class=None)
    data, *_ = ntu_window_series(root=root, n_classes=2, max_per_class=None, allow_small_n=True)
    assert data.shape == (12, 8, 75, 75) and is_spd(data)


# --------------------------------------------------------------------------- Pol-InSAR-Island (train-only)


def _envi(path, array):
    array.tofile(path)
    with open(f"{path}.hdr", "w") as f:
        f.write(f"ENVI\nsamples = {array.shape[1]}\nlines = {array.shape[0]}\n"
                "map info = {UTM, 1, 1, 500000.0, 6000000.0, 1.0, 1.0}\n")


def _polinsar_root(tmp_path, test_factor=1.0):
    rng = np.random.default_rng(0)
    root = tmp_path / f"polinsar{test_factor}"
    for k, path in enumerate(("FP1", "FP2")):
        (root / "label" / path).mkdir(parents=True)
        label = np.repeat(np.repeat(np.array([[1, 2], [3, 4]], np.uint8) + 4 * k, 32, 0), 32, 1)
        _envi(str(root / "label" / path / "label_train.bin"), label)
        _envi(str(root / "label" / path / "label_test.bin"), np.zeros_like(label))
        factor = test_factor if path == "FP2" else 1.0
        for band in ("L", "S"):
            folder = root / "data" / path / band / "T6"
            folder.mkdir(parents=True)
            a = rng.normal(size=(64, 64, 6, 8)) + 1j * rng.normal(size=(64, 64, 6, 8))
            t6 = factor * (a @ a.conj().swapaxes(-1, -2)) / 8
            for i in range(6):
                for j in range(i, 6):
                    if i == j:
                        _envi(str(folder / f"T{i + 1}{j + 1}.bin"), t6[..., i, i].real.astype("<f4"))
                    else:
                        _envi(str(folder / f"T{i + 1}{j + 1}_real.bin"), t6[..., i, j].real.astype("<f4"))
                        _envi(str(folder / f"T{i + 1}{j + 1}_imag.bin"), t6[..., i, j].imag.astype("<f4"))
    return str(root)


@pytest.mark.parametrize("builder, kwargs", [("polinsar_island_series", {}),
                                             ("polinsar_logpower_series", {"mean": True})])
def test_polinsar_normalisation_is_fitted_on_the_training_cells(tmp_path, builder, kwargs):
    from spdnet_datasets.real import series_sar

    function = getattr(series_sar, builder)
    fit = np.array([True] * 4 + [False] * 4)  # FP1 cells
    data, labels, n_obs, groups = function(_polinsar_root(tmp_path), fit_mask=fit, **kwargs)
    moved, *_ = function(_polinsar_root(tmp_path, test_factor=10.0), fit_mask=lambda y, g: fit, **kwargs)
    assert data.shape[0] == 8 and sorted(labels) == list(range(8)) and is_spd(data)
    np.testing.assert_allclose(moved[fit], data[fit], rtol=1e-6)
    assert not np.allclose(moved[~fit], data[~fit])


def test_synthetic_imports_without_yetanotherspdnet(monkeypatch):
    import importlib
    import sys

    for name in [m for m in sys.modules if m.startswith("yetanotherspdnet")] + ["yetanotherspdnet"]:
        monkeypatch.setitem(sys.modules, name, None)  # import yetanotherspdnet... -> ImportError
    for name in [m for m in sys.modules if m.startswith("spdnet_datasets.synthetic")]:
        monkeypatch.delitem(sys.modules, name)
    synthetic = importlib.import_module("spdnet_datasets.synthetic")
    with pytest.raises(ImportError, match="spdnet-datasets\\[synthetic\\]"):
        _ = synthetic.generate_series


def test_polinsar_island_checks_pixels_per_token(tmp_path):
    from spdnet_datasets.real.series_sar import polinsar_island_series

    with pytest.raises(InsufficientSamplesError):
        polinsar_island_series(str(tmp_path), cell=8, patch=4)  # 16 pixels per 12 x 12 token
