"""Locate the files of a dataset: local copy, archive, download or manual source.

Used by the ``fetch_<dataset>()`` functions and the dataset classes of the
hyperspectral and SAR modules, on top of :mod:`spdnet_datasets.core.access`:

- :func:`source_dir`: the order local copy -> ``archive=`` -> cache ->
  automatic download -> :func:`~spdnet_datasets.core.access.manual_source`;
- :func:`find_root`: the folder of an extracted archive that holds the
  expected files (archives rarely have the layout the loaders read);
- :func:`extract_7z` and :func:`extract_rar`: archives that the standard
  library cannot read (optional ``py7zr`` or system ``7z`` / ``unrar``).
"""

import shutil
import subprocess
from pathlib import Path

from ..core.access import extract_archive, manual_source
from ..core.config import get_cache_dir
from ..core.fetch import check_local

SKIP_DIRS = ("__MACOSX",)


def downloads_dir(name, cache_dir=None) -> Path:
    """``<cache_dir>/<name>/downloads``: downloads and extractions, outside the ``keep`` policy."""
    return Path(cache_dir or get_cache_dir()) / name / "downloads"


def existing_dir(root, cache_dir=None) -> Path:
    """``root`` if it exists, else the cache directory (created).

    :class:`~spdnet_datasets.base.BaseDataset` needs an existing folder; without
    a local copy, the dataset classes give it the cache, where their fetch
    downloads (or finds) the sources, then set ``data_dir`` to ``bunch.root``.
    """
    root = Path(root)
    if root.exists():
        return root
    cache = Path(cache_dir).expanduser() if cache_dir else get_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def find_root(base, required, max_depth=4):
    """First folder under ``base`` (breadth first) that holds every path of ``required``.

    ``__MACOSX`` folders and unfinished extractions (``*.part``) are skipped.
    Returns ``None`` when no folder matches.
    """
    base = Path(base)
    if not base.is_dir():
        return None
    level = [base]
    for depth in range(max_depth + 1):
        for d in level:
            if all((d / r).exists() for r in required):
                return d
        if depth == max_depth:
            break
        level = [p for d in level for p in sorted(d.iterdir())
                 if p.is_dir() and p.name not in SKIP_DIRS and not p.name.endswith(".part")]
    return None


def _ingest(path, dest_parent, required, what):
    path = Path(path)
    if path.is_file():
        path = extract_archive(path, Path(dest_parent) / path.name.rsplit(".", 1)[0])
    root = find_root(path, required)
    if root is None:
        raise FileNotFoundError(f"{what} {path} holds no folder with {list(required)}")
    return root


def source_dir(name, root, *, required, page_url, expected, cache_dir=None, archive=None,
               download=None, checksum=None, algo="md5", size=None, verify="size"):
    """Folder holding the files of a dataset (``required`` paths, relative to it).

    Tried in order:

    1. the local copy ``root`` (read only), checked with :func:`core.fetch.check_local`;
    2. ``archive`` (downloaded file or folder given by the user): a file is checked
       (``size``, ``checksum``) and extracted into ``<cache_dir>/<name>/downloads/``,
       a folder is read in place;
    3. a previous download or extraction in ``<cache_dir>/<name>/downloads/``;
    4. ``download(downloads_dir) -> path`` (file to extract, or folder) when the source
       can be downloaded automatically;
    5. otherwise :func:`core.access.manual_source`: opens ``page_url`` and raises
       :class:`~spdnet_datasets.core.access.ManualDownloadRequired` (``expected``
       describes what to download).
    """
    root = Path(root)
    if all((root / r).exists() for r in required):
        check_local(root, verify=verify, required=required)
        return root
    dl = downloads_dir(name, cache_dir)
    if archive is not None:
        path = Path(archive).expanduser()
        if path.is_file():
            # checks size / checksum; extraction into dl/<file name without extension>
            path = manual_source(name, page_url=page_url, expected=expected, archive=path,
                                 dest_dir=dl / path.name.rsplit(".", 1)[0], checksum=checksum,
                                 algo=algo, size=size)
        elif not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return _ingest(path, dl, required, "archive")
    cached = find_root(dl, required)
    if cached is not None:
        return cached
    if download is not None:
        return _ingest(download(dl), dl, required, "download")
    manual_source(name, page_url=page_url, expected=expected, dest_dir=dl)
    raise AssertionError("unreachable: manual_source raises without archive")  # pragma: no cover


def expose_as(path, sub, parent):
    """Folder ``root`` such that ``root / sub`` is ``path``.

    ``path.parent`` when ``path`` is already named ``sub``, otherwise ``parent``,
    where a symbolic link ``parent / sub -> path`` is created (loaders that
    read ``<root>/source/...`` from an extracted archive with another layout).
    """
    path, parent = Path(path), Path(parent)
    if path.name == sub:
        return path.parent
    link = parent / sub
    if not link.exists():
        parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(path.resolve(), target_is_directory=True)
    return parent


def check_size(path, size):
    """Raise ``OSError`` if ``path`` does not have ``size`` bytes (``None``: no check).

    Guard for files given by the user (``archive=``); downloads are checked by
    :func:`core.fetch.download` (``size=``) before they are renamed.
    """
    if size is not None and Path(path).stat().st_size != size:
        raise OSError(f"{path}: {Path(path).stat().st_size} bytes, expected {size}; delete it and retry")
    return Path(path)


# --- archives outside the standard library ---------------------------------------------------

def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def _finish(tmp, dest):
    (tmp / ".extracted").write_text("ok")
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


def _check_inside(dest, names):
    dest = dest.resolve()
    for n in names:
        target = (dest / n).resolve()
        if dest not in target.parents and target != dest:
            raise OSError(f"Unsafe path in archive: {n}")


def extract_7z(first_volume, dest_dir, *, only=None) -> Path:
    """Extract a (multi-volume ``.7z.001``, ``.7z.002``...) 7-Zip archive once.

    Uses the system ``7z`` / ``7za`` / ``7zz`` command when available, otherwise the
    optional ``py7zr`` package (``pip install 'spdnet-datasets[archives]'``, which
    also installs ``multivolumefile`` for split archives). ``only``: with py7zr,
    extract only the members having this folder name in their path (the system
    command extracts everything). Same ``.extracted`` marker as
    :func:`core.access.extract_archive`.
    """
    first, dest = Path(first_volume), Path(dest_dir)
    if (dest / ".extracted").exists():
        return dest
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tool = next((t for t in ("7z", "7za", "7zz") if shutil.which(t)), None)
    if tool is not None:
        _run([tool, "x", "-y", f"-o{tmp}", str(first)])
        return _finish(tmp, dest)
    try:
        import py7zr
    except ImportError as exc:
        raise ImportError(
            f"Extracting {first.name} needs the 7z command or the py7zr package: "
            "pip install 'spdnet-datasets[archives]' (or extract it by hand and pass the "
            "extracted folder as archive=)") from exc
    multi = first.suffix.lstrip(".").isdigit()
    if multi:
        import multivolumefile

        def opener():
            return multivolumefile.open(str(first.with_suffix("")), mode="rb")
    else:
        def opener():
            return open(first, "rb")
    with opener() as f, py7zr.SevenZipFile(f) as z:
        names = z.getnames()
    _check_inside(tmp, names)
    targets = [n for n in names if only in Path(n).parts] if only else None
    with opener() as f, py7zr.SevenZipFile(f) as z:
        z.extract(path=tmp, targets=targets) if targets is not None else z.extractall(path=tmp)
    return _finish(tmp, dest)


def extract_rar(archive, dest_dir) -> Path:
    """Extract a RAR archive once with ``unrar``, ``7z`` or ``bsdtar`` (system commands).

    Python has no RAR decoder (``rarfile`` itself calls these commands): without
    any of them, extract the archive by hand and pass the extracted folder.
    """
    archive, dest = Path(archive), Path(dest_dir)
    if (dest / ".extracted").exists():
        return dest
    cmds = {"unrar": lambda t: ["unrar", "x", "-o+", "-idq", str(archive), f"{t}/"],
            "7z": lambda t: ["7z", "x", "-y", f"-o{t}", str(archive)],
            "bsdtar": lambda t: ["bsdtar", "-xf", str(archive), "-C", str(t)]}
    tool = next((t for t in cmds if shutil.which(t)), None)
    if tool is None:
        raise RuntimeError(
            f"No RAR extractor (unrar, 7z or bsdtar) found to extract {archive.name}: "
            "extract it by hand and pass the extracted folder as archive=")
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _run(cmds[tool](tmp))
    return _finish(tmp, dest)


__all__ = ["check_size", "downloads_dir", "existing_dir", "expose_as", "extract_7z", "extract_rar",
           "find_root", "source_dir"]
