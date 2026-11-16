"""Simulated-network fetches of ESC-50 and GTZAN (Hugging Face parquet shards)."""

import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("pyarrow")

from tests.unit.test_audio_datasets import esc50_dir, gtzan_dir  # noqa: E402,F401 (fixtures)
from tests.unit.test_fetch_registry import _register  # noqa: E402


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



def test_registered_classes_download_a_missing_copy(esc50_dir, gtzan_dir, tmp_path,  # noqa: F811
                                                    fake_net, monkeypatch):
    from spdnet_datasets.audio import esc50, gtzan

    serve, calls = fake_net
    _register(monkeypatch, esc50, esc50_dir, ["esc50_0.parquet", "esc50_1.parquet"], serve)
    _register(monkeypatch, gtzan, gtzan_dir, ["gtzan_0.parquet"], serve)
    ds = esc50.ESC50Dataset(tmp_path / "absent", cache_dir=tmp_path / "c1", verbose=False)
    assert len(calls) == 2 and len(ds) > 0
    with pytest.warns(UserWarning, match="artist"):
        ds = gtzan.GTZANDataset(tmp_path / "absent", cache_dir=tmp_path / "c2", verbose=False)
    assert len(calls) == 3 and len(ds) > 0
