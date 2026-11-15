"""Download registries (``FILES``) and simulated-network fetches of the downloadable datasets.

CPAZMaL (Hugging Face HDF5) and PASTIS-R (Zenodo zip) are fetched from a fake
``urlopen`` serving tiny synthetic copies (ESC-50 and GTZAN: ``test_fetch_audio.py``):
the first call downloads and checks the sums, later calls never download again,
including a replay after a failed preprocessing and new preprocessing parameters
(persistent ``downloads/`` folder).
"""

import importlib
import io
import re
import zipfile

import numpy as np
import pytest

from tests.unit.test_sar_cpazmal import cpaz_dir  # noqa: F401 (fixture)
from tests.unit.test_sits_pastis_r import REF, pastis_dir  # noqa: F401 (fixture)

MODULES = [
    "audio.esc50", "audio.gtzan", "audio.birdset_xcm", "skeleton.hdm05", "skeleton.ntu120",
    "radar_microdoppler.ci4r", "radar_microdoppler.mvdoppler", "radar_microdoppler.dopnet",
    "sits.pastis_r", "hyperspectral.gsoff", "hyperspectral.chikusei", "hyperspectral.uav",
    "hyperspectral.kaggle_wheat", "hyperspectral.placenta", "hyperspectral.rices90",
    "hyperspectral.hyperleaf", "hyperspectral.deephsfruit", "sar.cpazmal",
    "sar.pol_insar_island", "sar.s1slc_cvdl", "sar.air_polsar_seg", "sar.atrnet_star",
    "sar.fusar_ship", "sar.mstar",
]
DOWNLOADABLE = {"audio.esc50", "audio.gtzan", "sits.pastis_r", "sar.cpazmal"}


@pytest.mark.parametrize("name", MODULES)
def test_registry_format(name):
    mod = importlib.import_module(f"spdnet_datasets.{name}")
    assert isinstance(mod.FILES, dict)
    assert bool(mod.FILES) == (name in DOWNLOADABLE)
    for url, sha in mod.FILES.items():
        assert url.startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", sha)
        assert "/resolve/main/" not in url  # Hugging Face revisions are pinned
    if not mod.FILES:
        assert "Download (manual)" in mod.__doc__


def _register(monkeypatch, mod, folder, names, serve):
    files = {f"https://example.org/data/{n}": serve(f"https://example.org/data/{n}",
                                                     (folder / n).read_bytes()) for n in names}
    monkeypatch.setattr(mod, "FILES", files)
    return files


# ------------------------------------------------------------------------- CPAZMaL
def test_cpazmal_download_is_persistent(cpaz_dir, tmp_path, fake_net, monkeypatch):  # noqa: F811
    from spdnet_datasets.sar import cpazmal

    serve, calls = fake_net
    monkeypatch.setattr(cpazmal, "FILES", {
        cpazmal.HF_URL: serve(cpazmal.HF_URL, (cpaz_dir / "CPAZMaL.hdf5").read_bytes())})
    absent, cache = tmp_path / "absent", tmp_path / "cache"
    b = cpazmal.fetch_cpazmal(absent, cache_dir=cache)
    ref = cpazmal.fetch_cpazmal(cpaz_dir, cache_dir=tmp_path / "local")
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert calls == [cpazmal.HF_URL]
    assert (cache / "cpazmal" / "downloads" / cpazmal.HF_REVISION / "CPAZMaL.hdf5").exists()
    assert not (cache / "cpazmal" / "raw").exists()
    cpazmal.fetch_cpazmal(absent, cache_dir=cache, estimator="ledoit_wolf")
    cpazmal.fetch_cpazmal(absent, cache_dir=cache, n_dates=10, min_pixels=40,
                          keep="keep_covariance")  # new processed stage, same download
    assert len(calls) == 1
    # the registered class reaches the download path too (fresh cache folder)
    ds = cpazmal.CPAZMaLDataset(absent, cache_dir=tmp_path / "fresh", verbose=False)
    assert len(calls) == 2 and ds.covariances.shape[1:] == (40, 40)


# ------------------------------------------------------------------------ PASTIS-R
def test_pastis_r_download_extract_and_replay(pastis_dir, tmp_path, fake_net,  # noqa: F811
                                              monkeypatch):
    from spdnet_datasets.sits import pastis_r as pr

    serve, calls = fake_net
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted((pastis_dir / "PASTIS-R").rglob("*")):
            z.write(f, f.relative_to(pastis_dir))
    monkeypatch.setattr(pr, "FILES", {pr.ZENODO_URL: serve(pr.ZENODO_URL, buf.getvalue())})
    absent, cache = tmp_path / "absent", tmp_path / "cache"

    real = pr._process
    monkeypatch.setattr(pr, "_process", lambda *a: (_ for _ in ()).throw(RuntimeError))
    with pytest.raises(RuntimeError):
        pr.fetch_pastis_r(absent, cache, ref_date=REF)
    monkeypatch.setattr(pr, "_process", real)
    downloads = cache / "pastis_r" / "downloads"
    assert len(calls) == 1 and (downloads / "PASTIS-R" / "metadata.geojson").exists()
    assert not (downloads / pr.ARCHIVE).exists()  # archive deleted once extracted

    b = pr.fetch_pastis_r(absent, cache, ref_date=REF)
    ref = pr.fetch_pastis_r(pastis_dir, tmp_path / "local", ref_date=REF)
    np.testing.assert_allclose(b.covariances, ref.covariances)
    pr.fetch_pastis_r(absent, cache, ref_date=REF, scenario="B")
    assert len(calls) == 1
    ds = pr.PastisRDataset(absent, cache_dir=tmp_path / "fresh", ref_date=REF, verbose=False)
    assert len(calls) == 2 and len(ds) == len(b.covariances)


def test_pastis_r_interrupted_extraction_is_redone(pastis_dir, tmp_path, fake_net,  # noqa: F811
                                                   monkeypatch):
    from spdnet_datasets.sits import pastis_r as pr

    serve, _ = fake_net
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted((pastis_dir / "PASTIS-R").rglob("*")):
            z.write(f, f.relative_to(pastis_dir))
    monkeypatch.setattr(pr, "FILES", {pr.ZENODO_URL: serve(pr.ZENODO_URL, buf.getvalue())})
    stale = tmp_path / "cache" / "pastis_r" / "downloads" / "extract.part" / "PASTIS-R"
    stale.mkdir(parents=True)
    (stale / "half_written.npy").write_bytes(b"x")
    out = pr.download_pastis_r(tmp_path / "cache")
    assert (out / "metadata.geojson").exists() and not (out / "half_written.npy").exists()
