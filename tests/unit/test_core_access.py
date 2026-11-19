"""core.access: credentials, authenticated downloads, manual sources, safe extraction."""

import base64
import io
import json
import tarfile
import urllib.request
import zipfile

import pytest

from spdnet_datasets.core import access as A
from spdnet_datasets.core import fetch as F


@pytest.fixture(autouse=True)
def _no_keyring(monkeypatch, tmp_path_factory):
    monkeypatch.setattr(A, "_keyring_get", lambda name: None)
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # never read the real ~/.kaggle of the machine running the tests
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path_factory.mktemp("kaggle_home")))


# --- secrets ---------------------------------------------------------------------------------

def test_secret_lookup_order(monkeypatch, tmp_path):
    f = tmp_path / "token"
    f.write_text("from-file\n")
    monkeypatch.setenv("MY_TOKEN", "from-env")
    assert A.get_secret("x", env=("MY_TOKEN",), files=(f,)) == "from-env"
    monkeypatch.delenv("MY_TOKEN")
    assert A.get_secret("x", env=("MY_TOKEN",), files=(f,)) == "from-file"
    monkeypatch.setattr(A, "_keyring_get", lambda name: "from-keyring")
    assert A.get_secret("x", env=("MY_TOKEN",)) == "from-keyring"


def test_secret_prompt_and_error(monkeypatch):
    monkeypatch.setattr(A.getpass, "getpass", lambda prompt: "typed")
    assert A.get_secret("x", prompt="token: ", interactive=True) == "typed"
    with pytest.raises(A.MissingCredentialsError, match="set MY_TOKEN"):
        A.get_secret("x", env=("MY_TOKEN",), prompt="token: ", interactive=False)


def test_kaggle_credentials_from_env_and_file(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "u", "key": "k"}))
    cfg.chmod(0o644)
    with pytest.warns(UserWarning, match="chmod 600"):
        assert A.kaggle_credentials(interactive=False) == ("u", "k")
    cfg.chmod(0o600)
    assert A.kaggle_credentials(interactive=False) == ("u", "k")
    monkeypatch.setenv("KAGGLE_USERNAME", "eu")
    monkeypatch.setenv("KAGGLE_KEY", "ek")
    assert A.kaggle_credentials(interactive=False) == ("eu", "ek")


def test_kaggle_credentials_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    with pytest.raises(A.MissingCredentialsError, match="access_token"):
        A.kaggle_credentials(interactive=False)
    with pytest.raises(A.MissingCredentialsError, match="Generate New Token"):
        A.kaggle_auth_header(interactive=False)


def test_kaggle_access_token_lookup_order(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "u", "key": "k"}))
    (tmp_path / "kaggle.json").chmod(0o600)
    legacy = "Basic " + base64.b64encode(b"u:k").decode()
    assert A.kaggle_auth_header(interactive=False) == legacy  # legacy file only
    tok = tmp_path / "access_token"
    tok.write_text("KGAT_file\n")
    tok.chmod(0o644)
    with pytest.warns(UserWarning, match="chmod 600"):
        assert A.kaggle_auth_header(interactive=False) == "Bearer KGAT_file"
    tok.chmod(0o600)
    assert A.kaggle_auth_header(interactive=False) == "Bearer KGAT_file"  # token first
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_env")
    assert A.kaggle_auth_header(interactive=False) == "Bearer KGAT_env"
    monkeypatch.delenv("KAGGLE_API_TOKEN")
    tok.unlink()
    monkeypatch.setattr(A, "_keyring_get",
                        lambda name: "KGAT_ring" if name == "kaggle_token" else None)
    assert A.kaggle_auth_header(interactive=False) == "Bearer KGAT_ring"


def test_kaggle_token_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(A.getpass, "getpass", lambda prompt: "KGAT_typed")
    assert A.kaggle_auth_header(interactive=True) == "Bearer KGAT_typed"


# --- authenticated downloads -------------------------------------------------------------------

class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_kaggle_download_sends_bearer_token_not_in_url(monkeypatch, tmp_path):
    seen = {}

    class _Opener:
        def open(self, req, timeout=None):
            seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
            return _Resp(b"zipbytes")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: _Opener())
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_secret")
    A.kaggle_download("comp-name", tmp_path, competition=True)
    assert seen["url"] == ("https://www.kaggle.com/api/v1/competitions/data/download-all/"
                           "comp-name")
    assert "KGAT_secret" not in seen["url"] and seen["auth"] == "Bearer KGAT_secret"


def test_kaggle_download_sends_basic_auth_not_in_url(monkeypatch, tmp_path):
    seen = {}

    class _Opener:
        def open(self, req, timeout=None):
            seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
            return _Resp(b"zipbytes")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: _Opener())
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    path = A.kaggle_download("owner/data", tmp_path)
    assert path.read_bytes() == b"zipbytes"
    assert seen["url"] == "https://www.kaggle.com/api/v1/datasets/download/owner/data"
    assert "secret" not in seen["url"]
    assert seen["auth"] == "Basic " + base64.b64encode(b"user:secret").decode()


def test_hf_download_public_and_gated(monkeypatch, tmp_path):
    calls = []

    def urlopen(req, timeout=None):
        calls.append(("plain", req.full_url))
        return _Resp(b"public")

    class _Opener:
        def open(self, req, timeout=None):
            calls.append(("auth", req.get_header("Authorization")))
            return _Resp(b"gated")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: _Opener())
    monkeypatch.setattr(A, "hf_token", lambda interactive=None, required=True:
                        "hf_x" if required else None)
    A.hf_download("org/ds", "a.txt", tmp_path / "pub")
    A.hf_download("org/ds", "b.txt", tmp_path / "gated", gated=True)
    assert calls[0] == ("plain", "https://huggingface.co/datasets/org/ds/resolve/main/a.txt")
    assert calls[1] == ("auth", "Bearer hf_x")


def test_redirect_to_other_host_drops_authorization():
    handler = F._NoCrossHostAuth()
    req = urllib.request.Request("https://www.kaggle.com/api/x",
                                 headers={"Authorization": "Basic abc"})
    same = handler.redirect_request(req, None, 302, "Found", {}, "https://www.kaggle.com/api/y")
    other = handler.redirect_request(req, None, 302, "Found", {},
                                     "https://storage.googleapis.com/bucket/x")
    assert same.get_header("Authorization") == "Basic abc"
    assert other.get_header("Authorization") is None


# --- manual sources and archives ----------------------------------------------------------------

def test_manual_source_opens_page_then_ingests_archive(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(A.webbrowser, "open", opened.append)
    with pytest.raises(A.ManualDownloadRequired, match="archive="):
        A.manual_source("Toy", page_url="https://example.org/toy", expected="toy.zip",
                        dest_dir=tmp_path / "raw", open_browser=True)
    assert opened == ["https://example.org/toy"]

    archive = tmp_path / "toy.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("toy/data.txt", "hello")
    out = A.manual_source("Toy", page_url="u", expected="toy.zip", dest_dir=tmp_path / "raw",
                          archive=archive, checksum=F.sha256sum(archive))
    assert (out / "toy" / "data.txt").read_text() == "hello"
    assert A.manual_source("Toy", page_url="u", expected="e", dest_dir=tmp_path / "x",
                           archive=out) == out  # a directory is used in place
    with pytest.raises(OSError, match="Checksum"):
        A.manual_source("Toy", page_url="u", expected="e", dest_dir=tmp_path / "y",
                        archive=archive, checksum="0" * 64)


def test_extract_archive_rejects_path_traversal(tmp_path):
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(OSError, match="Unsafe"):
        A.extract_archive(bad_zip, tmp_path / "out1")

    bad_tar = tmp_path / "bad.tar"
    with tarfile.open(bad_tar, "w") as t:
        info = tarfile.TarInfo("link")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        t.addfile(info)
    with pytest.raises(OSError, match="Unsafe"):
        A.extract_archive(bad_tar, tmp_path / "out2")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_archive_is_idempotent(tmp_path):
    archive = tmp_path / "a.tar"
    src = tmp_path / "f.txt"
    src.write_text("v")
    with tarfile.open(archive, "w") as t:
        t.add(src, arcname="f.txt")
    out = A.extract_archive(archive, tmp_path / "out")
    (out / "f.txt").write_text("kept")
    A.extract_archive(archive, tmp_path / "out")
    assert (out / "f.txt").read_text() == "kept"
