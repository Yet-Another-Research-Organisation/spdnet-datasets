"""core.fetch: simulated downloads, checksums, manifest checks, staged cache, keep."""

import hashlib
import io
from pathlib import Path

import pytest

from spdnet_datasets.core import fetch as F


class _FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_net(monkeypatch):
    """Serve PAYLOAD for any URL and count the requests."""
    calls = []
    payload = b"spd" * 1000

    def urlopen(req, timeout=None):
        calls.append(req)
        rng = req.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return _FakeResponse(payload[start:], status=206)
        return _FakeResponse(payload)

    monkeypatch.setattr(F.urllib.request, "urlopen", urlopen)
    return payload, calls


def test_download_verifies_checksum_and_is_idempotent(tmp_path, fake_net):
    payload, calls = fake_net
    sha = hashlib.sha256(payload).hexdigest()
    dest = F.download("http://x/file.bin", tmp_path / "file.bin", checksum=sha)
    assert dest.read_bytes() == payload
    F.download("http://x/file.bin", dest, checksum=sha)
    assert len(calls) == 1  # second call served from disk


def test_download_resumes_partial_file(tmp_path, fake_net):
    payload, calls = fake_net
    part = tmp_path / "f.bin.part"
    part.write_bytes(payload[:100])
    F.download("http://x/f.bin", tmp_path / "f.bin", checksum=hashlib.sha256(payload).hexdigest())
    assert calls[0].headers["Range"] == "bytes=100-"
    assert (tmp_path / "f.bin").read_bytes() == payload


def test_download_bad_checksum_leaves_no_file(tmp_path, fake_net):
    with pytest.raises(OSError, match="Checksum"):
        F.download("http://x/f.bin", tmp_path / "f.bin", checksum="0" * 64)
    assert not list(tmp_path.iterdir())


def _make_local_dataset(root: Path):
    (root / "sub").mkdir(parents=True)
    (root / "a.npy").write_bytes(b"12345")
    (root / "sub" / "b.npy").write_bytes(b"xy")
    md5 = hashlib.md5(b"xy").hexdigest()
    (root / "MANIFEST_NAS.tsv").write_text(
        "path\tsize\tmtime\tmd5\na.npy\t5\t2026-01-01\t\nsub/b.npy\t2\t2026-01-01\t" + md5 + "\n")


def test_verify_local_against_manifest(tmp_path):
    _make_local_dataset(tmp_path)
    rep = F.verify_local(tmp_path, verify="md5", required=["a.npy"])
    assert rep.n_checked == 2 and not (rep.missing or rep.wrong_size or rep.wrong_md5)
    (tmp_path / "a.npy").write_bytes(b"1")
    (tmp_path / "sub" / "b.npy").write_bytes(b"zz")
    rep = F.verify_local(tmp_path, verify="md5")
    assert rep.wrong_size == ["a.npy"] and rep.wrong_md5 == ["sub/b.npy"]


def test_verify_local_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        F.verify_local(tmp_path / "nope")
    _make_local_dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="required"):
        F.verify_local(tmp_path, required=["absent.h5"])
    (tmp_path / "a.npy").unlink()
    assert F.verify_local(tmp_path).missing == ["a.npy"]


class _Pipeline:
    """Counts the calls of each stage; can fail once in the covariance stage."""

    def __init__(self, fail_cov=False):
        self.n = {"raw": 0, "process": 0, "cov": 0}
        self.fail_cov = fail_cov

    def raw(self, raw_dir):
        self.n["raw"] += 1
        (raw_dir / "src.bin").write_bytes(b"raw")
        return raw_dir

    def process(self, source, out):
        self.n["process"] += 1
        (out / "x.npy").write_bytes(b"proc")

    def cov(self, proc, out):
        self.n["cov"] += 1
        if self.fail_cov:
            self.fail_cov = False
            raise RuntimeError("boom")
        (out / "covs.npy").write_bytes((proc / "x.npy").read_bytes())

    def run(self, tmp_path, keep, raw=True, **kw):
        return F.run_pipeline("toy", params={"w": 15}, raw_fn=self.raw if raw else None,
                              process_fn=self.process, covariance_fn=self.cov,
                              cache_dir=tmp_path, keep=keep, **kw)


@pytest.mark.parametrize("keep,has_proc,has_raw", [
    ("keep_covariance", False, False),
    ("keep_processed", True, False),
    ("keep_full", True, True),
])
def test_keep_policies(tmp_path, keep, has_proc, has_raw):
    out = _Pipeline().run(tmp_path, keep)
    assert (out.covariance_dir / "covs.npy").exists()
    assert (out.processed_dir is not None) == has_proc
    assert (out.raw_dir is not None) == has_raw


def test_default_keep_depends_on_source(tmp_path):
    assert _Pipeline().run(tmp_path / "a", None).keep == "keep_processed"
    assert _Pipeline().run(tmp_path / "b", None, raw=False).keep == "keep_full"


def test_pipeline_is_idempotent_and_force_recomputes(tmp_path):
    p = _Pipeline()
    p.run(tmp_path, "keep_full")
    p.run(tmp_path, "keep_full")
    assert p.n == {"raw": 1, "process": 1, "cov": 1}
    p.run(tmp_path, "keep_full", force=True)
    assert p.n == {"raw": 2, "process": 2, "cov": 2}


def test_failed_covariance_replays_from_cache_without_download(tmp_path):
    p = _Pipeline(fail_cov=True)
    with pytest.raises(RuntimeError):
        p.run(tmp_path, "keep_full")
    out = p.run(tmp_path, "keep_full")
    assert p.n == {"raw": 1, "process": 1, "cov": 2}
    assert (out.covariance_dir / ".done").exists()


def test_params_change_the_stage_directory(tmp_path):
    a = F.Stage(tmp_path, "toy", "covariance", {"w": 15})
    b = F.Stage(tmp_path, "toy", "covariance", {"w": 16})
    assert a.path != b.path
    assert a.hash == F.params_hash({"w": 15, "_core_cache_version": F.CACHE_VERSION})


def test_bad_keep(tmp_path):
    with pytest.raises(ValueError, match="keep"):
        _Pipeline().run(tmp_path, "keep_everything")


def test_bunch_attribute_access():
    b = F.Bunch(covariances=1)
    b.labels = 2
    assert b.covariances == 1 and b["labels"] == 2
    with pytest.raises(AttributeError):
        _ = b.missing


def test_covariance_params_do_not_redo_preprocessing(tmp_path):
    p = _Pipeline()
    for est in ("scm", "tyler", "scm"):
        F.run_pipeline("toy", params={"w": 15}, covariance_params={"estimator": est},
                       raw_fn=p.raw, process_fn=p.process, covariance_fn=p.cov,
                       cache_dir=tmp_path, keep="keep_full")
    assert p.n == {"raw": 1, "process": 1, "cov": 2}
