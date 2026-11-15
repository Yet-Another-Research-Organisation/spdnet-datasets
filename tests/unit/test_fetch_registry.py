"""Download registries (``FILES``) and simulated-network fetches of the downloadable datasets.

ESC-50 and GTZAN (Hugging Face parquet shards), CPAZMaL (Hugging Face HDF5) and
PASTIS-R (Zenodo zip) are fetched from a fake ``urlopen`` serving tiny synthetic
copies: the first call downloads and checks the sums, later calls never download
again, including a replay after a failed preprocessing and new preprocessing
parameters of the heavy datasets (persistent ``downloads/`` folder).
"""

import hashlib
import importlib
import io
import re
import zipfile

import numpy as np
import pytest

from spdnet_datasets.core import fetch as F
from tests.unit.test_audio_datasets import esc50_dir, gtzan_dir  # noqa: F401 (fixtures)
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


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_net(monkeypatch):
    """``serve(url -> bytes)``; returns the list of requested URLs."""
    served, calls = {}, []

    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Response(served[req.full_url])

    monkeypatch.setattr(F.urllib.request, "urlopen", urlopen)

    def serve(url, data):
        served[url] = data
        return hashlib.sha256(data).hexdigest()

    return serve, calls


def _register(monkeypatch, mod, folder, names, serve):
    files = {f"https://example.org/data/{n}": serve(f"https://example.org/data/{n}",
                                                     (folder / n).read_bytes()) for n in names}
    monkeypatch.setattr(mod, "FILES", files)
    return files


# --------------------------------------------------------------------------- audio
def test_esc50_download_checksum_idempotence_and_replay(esc50_dir, tmp_path, fake_net,  # noqa: F811
                                                        monkeypatch):
    from spdnet_datasets.audio import esc50

    serve, calls = fake_net
    _register(monkeypatch, esc50, esc50_dir, ["esc50_0.parquet", "esc50_1.parquet"], serve)
    absent, cache = tmp_path / "absent", tmp_path / "cache"
    ref = esc50.fetch_esc50(esc50_dir, cache_dir=tmp_path / "local")  # local copy

    real = esc50._process
    monkeypatch.setattr(esc50, "_process", lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    with pytest.raises(RuntimeError):
        esc50.fetch_esc50(absent, cache_dir=cache)
    assert len(calls) == 2 and len(list((cache / "esc50" / "raw").glob("*.parquet"))) == 2
    monkeypatch.setattr(esc50, "_process", real)

    b = esc50.fetch_esc50(absent, cache_dir=cache)  # replay: no new download
    assert len(calls) == 2
    np.testing.assert_allclose(b.covariances, ref.covariances)
    assert not (cache / "esc50" / "raw").exists()  # keep_processed (download default)
    esc50.fetch_esc50(absent, cache_dir=cache)
    esc50.fetch_esc50(absent, cache_dir=cache, estimator="ledoit_wolf")  # covariance stage only
    assert len(calls) == 2


def test_esc50_bad_checksum_is_refused(esc50_dir, tmp_path, fake_net, monkeypatch):  # noqa: F811
    from spdnet_datasets.audio import esc50

    serve, _ = fake_net
    files = _register(monkeypatch, esc50, esc50_dir, ["esc50_0.parquet"], serve)
    monkeypatch.setattr(esc50, "FILES", dict.fromkeys(files, "0" * 64))
    with pytest.raises(OSError, match="Checksum"):
        esc50.fetch_esc50(tmp_path / "absent", cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("*.parquet"))


def test_gtzan_download_keep_full(gtzan_dir, tmp_path, fake_net, monkeypatch):  # noqa: F811
    from spdnet_datasets.audio import gtzan

    serve, calls = fake_net
    _register(monkeypatch, gtzan, gtzan_dir, ["gtzan_0.parquet"], serve)
    b = gtzan.fetch_gtzan(tmp_path / "absent", cache_dir=tmp_path / "cache", keep="keep_full")
    assert len(calls) == 1 and "pop.00060.wav" not in set(b.meta["id"])
    assert (tmp_path / "cache" / "gtzan" / "raw" / "gtzan_0.parquet").exists()
    gtzan.fetch_gtzan(tmp_path / "absent", cache_dir=tmp_path / "cache", keep="keep_full",
                      n_mfcc=13)  # new preprocessing, raw kept: no new download
    assert len(calls) == 1


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
    assert (cache / "cpazmal" / "downloads" / "CPAZMaL.hdf5").exists()
    assert not (cache / "cpazmal" / "raw").exists()
    cpazmal.fetch_cpazmal(absent, cache_dir=cache, estimator="ledoit_wolf")
    cpazmal.fetch_cpazmal(absent, cache_dir=cache, n_dates=10, min_pixels=40,
                          keep="keep_covariance")  # new processed stage, same download
    assert len(calls) == 1


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
