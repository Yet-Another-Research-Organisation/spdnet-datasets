"""utils.sources: dataset root lookup, order local -> archive -> cache -> download -> manual."""

import subprocess
import zipfile

import pytest

from spdnet_datasets.core.access import ManualDownloadRequired
from spdnet_datasets.utils import sources as S


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    monkeypatch.setenv("SPDNET_OPEN_BROWSER", "0")


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def test_find_root_is_breadth_first_and_skips_macosx(tmp_path):
    (tmp_path / "__MACOSX/pkg").mkdir(parents=True)
    (tmp_path / "__MACOSX/pkg/index.csv").write_text("x")
    (tmp_path / "a/b/pkg").mkdir(parents=True)
    (tmp_path / "a/b/pkg/index.csv").write_text("x")
    (tmp_path / "a/b/pkg/deep/pkg").mkdir(parents=True)
    (tmp_path / "a/b/pkg/deep/pkg/index.csv").write_text("x")
    assert S.find_root(tmp_path, ("index.csv",)) == tmp_path / "a/b/pkg"
    assert S.find_root(tmp_path, ("index.csv",), max_depth=2) is None
    assert S.find_root(tmp_path / "missing", ("index.csv",)) is None


def test_source_dir_order(tmp_path):
    kw = {"required": ("data/x.txt",), "page_url": "https://example.org", "expected": "pkg.zip",
          "cache_dir": tmp_path / "cache"}
    local = tmp_path / "local"
    (local / "data").mkdir(parents=True)
    (local / "data/x.txt").write_text("local")
    calls = []

    def download(d):
        calls.append(d)
        d.mkdir(parents=True, exist_ok=True)
        return _zip(d / "pkg.zip", {"pkg/data/x.txt": "downloaded"})

    # 1. local copy, never downloaded
    assert S.source_dir("toy", local, download=download, **kw) == local
    assert calls == []
    # 5. nothing and no automatic source: manual
    with pytest.raises(ManualDownloadRequired, match="pkg.zip"):
        S.source_dir("toy", tmp_path / "missing", **kw)
    # 4. download, extracted into <cache>/toy/downloads/
    root = S.source_dir("toy", tmp_path / "missing", download=download, **kw)
    assert (root / "data/x.txt").read_text() == "downloaded"
    assert root.is_relative_to(tmp_path / "cache/toy/downloads")
    # 3. idempotent: found in the cache
    assert S.source_dir("toy", tmp_path / "missing", download=download, **kw) == root
    assert len(calls) == 1


def test_source_dir_archive_checks_size_and_layout(tmp_path):
    kw = {"required": ("data/x.txt",), "page_url": "u", "expected": "e",
          "cache_dir": tmp_path / "cache"}
    good = _zip(tmp_path / "good.zip", {"top/data/x.txt": "a"})
    root = S.source_dir("toy", tmp_path / "missing", archive=good, size=good.stat().st_size, **kw)
    assert (root / "data/x.txt").read_text() == "a"
    with pytest.raises(OSError, match="size"):
        S.source_dir("toy", tmp_path / "missing", archive=good, size=1, **kw)
    bad = _zip(tmp_path / "bad.zip", {"other.txt": "b"})
    with pytest.raises(FileNotFoundError, match="data/x.txt"):
        S.source_dir("toy", tmp_path / "missing", archive=bad, **kw)
    # an extracted folder is read in place
    assert S.source_dir("toy", tmp_path / "missing", archive=root.parent, **kw) == root


def test_expose_as_links_a_folder_under_the_expected_name(tmp_path):
    (tmp_path / "source").mkdir()
    assert S.expose_as(tmp_path / "source", "source", tmp_path / "x") == tmp_path
    (tmp_path / "Data_v6").mkdir()
    root = S.expose_as(tmp_path / "Data_v6", "source", tmp_path / "root")
    assert (root / "source").resolve() == (tmp_path / "Data_v6").resolve()


def test_extract_rar_without_tool_asks_for_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(S.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="extract it by hand"):
        S.extract_rar(tmp_path / "x.rar", tmp_path / "out")


def test_extract_rar_uses_the_system_command(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(S.shutil, "which", lambda name: "/usr/bin/unrar" if name == "unrar" else None)

    def run(cmd, check, stdout):
        seen.append(cmd)
        (tmp_path / "out.part" / "City").mkdir()

    monkeypatch.setattr(subprocess, "run", run)
    out = S.extract_rar(tmp_path / "x.rar", tmp_path / "out")
    assert seen[0][:2] == ["unrar", "x"] and (out / "City").is_dir()
    assert S.extract_rar(tmp_path / "x.rar", tmp_path / "out") == out  # marker: not run again
    assert len(seen) == 1


def _seven_zip_volumes(tmp_path, files, volume=600):
    """7z cut in volumes ``.001``, ``.002``... (plain byte split, as 7-Zip does)."""
    py7zr = pytest.importorskip("py7zr")
    pytest.importorskip("multivolumefile")
    src = tmp_path / "src"
    for name, data in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
    whole = tmp_path / "whole.7z"
    with py7zr.SevenZipFile(whole, "w") as z:
        z.writeall(src, arcname="top")
    data = whole.read_bytes()
    (tmp_path / "vols").mkdir()
    for k in range(0, len(data), volume):
        (tmp_path / "vols" / f"arc.7z.{k // volume + 1:03d}").write_bytes(data[k:k + volume])
    return tmp_path / "vols" / "arc.7z.001"


def test_extract_7z_multivolume_with_py7zr(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(S.shutil, "which", lambda name: None)  # no 7z command
    files = {"SOC_40classes/train/A/x.mat": os.urandom(900),
             "SOC_40classes/test/A/y.mat": os.urandom(900),
             "EOC_band/train/A/z.mat": os.urandom(900)}
    first = _seven_zip_volumes(tmp_path, files)
    assert len(list(first.parent.glob("arc.7z.0*"))) >= 3  # really split
    out = S.extract_7z(first, tmp_path / "all")
    assert (out / "top/EOC_band/train/A/z.mat").read_bytes() == files["EOC_band/train/A/z.mat"]
    only = S.extract_7z(first, tmp_path / "soc", only="SOC_40classes")
    assert (only / "top/SOC_40classes/test/A/y.mat").exists()
    assert not (only / "top/EOC_band").exists()
