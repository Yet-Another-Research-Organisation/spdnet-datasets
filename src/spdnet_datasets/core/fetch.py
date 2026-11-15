"""Fetching, verification and staged preprocessing cache (see ADR-0004).

Inspired by :mod:`nilearn.datasets`: one ``fetch_<dataset>`` function per
dataset, an optional ``data_dir``, checksums and idempotent downloads.

A fetch runs three cached stages under ``<cache_dir>/<name>/``::

    raw/                       downloaded files (only for downloaded datasets)
    processed/<params_hash>/   preprocessed tensors used for the covariances
    covariance/<params_hash>/  unscaled covariances, labels, groups, metadata

Each stage writes a ``.done`` marker holding its parameters (JSON). A finished
stage is never recomputed unless ``force=True``; a failure leaves the earlier
stages untouched, so preprocessing can be replayed without downloading again.

Datasets already on the machine (``data_dir`` read-only) are never modified:
they are checked against their ``MANIFEST_NAS.tsv`` instead of downloaded.

``keep`` selects what stays on disk once the covariances exist:

- ``keep_covariance``: covariances only;
- ``keep_processed``: covariances and preprocessed tensors (default for
  downloaded datasets);
- ``keep_full``: raw downloads, preprocessed tensors and covariances (default
  for local datasets).
"""

import csv
import hashlib
import json
import shutil
import urllib.request
import warnings
from pathlib import Path

from .config import get_cache_dir

KEEP = ("keep_processed", "keep_covariance", "keep_full")
# Bumped when a change of the core invalidates cached results (part of every
# stage hash). 2: stricter SPD check (condition number <= 1e12).
CACHE_VERSION = 2


class Bunch(dict):
    """Dictionary with attribute access (as in scikit-learn / nilearn)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _hash_file(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5sum(path) -> str:
    return _hash_file(path, "md5")


def sha256sum(path) -> str:
    return _hash_file(path, "sha256")


def download(url, dest, checksum=None, algo="sha256", resume=True, timeout=60) -> Path:
    """Download ``url`` to ``dest`` once; verify ``checksum`` if given.

    The file is written to ``dest.part`` and renamed when complete, so an
    interrupted download is resumed (HTTP Range) or restarted, never mistaken
    for a finished one.
    """
    dest = Path(dest)
    if dest.exists():
        if checksum and _hash_file(dest, algo) != checksum:
            raise OSError(f"Checksum mismatch for existing file {dest}; delete it or use force")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if (resume and part.exists()) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        mode = "ab" if start and getattr(resp, "status", 200) == 206 else "wb"
        with open(part, mode) as out:
            shutil.copyfileobj(resp, out, 1 << 20)
    if checksum and _hash_file(part, algo) != checksum:
        part.unlink()
        raise OSError(f"Checksum mismatch for {url}")
    part.rename(dest)
    return dest


def verify_local(dataset_dir, manifest="MANIFEST_NAS.tsv", verify="size", required=()):
    """Check a read-only local dataset against its manifest.

    Parameters
    ----------
    dataset_dir : path
    manifest : str
        Tab-separated file with at least ``path`` and ``size`` columns (relative
        paths). ``MANIFEST_NAS.tsv`` has no checksum column; with
        ``verify="md5"`` an ``md5`` column is used when present.
    verify : {"size", "md5", None}
        ``None`` only checks that ``required`` files exist.
    required : iterable of str
        Relative paths that must exist (checked even without a manifest).

    Returns
    -------
    Bunch with ``missing``, ``wrong_size``, ``wrong_md5`` and ``n_checked``.
    Raises ``FileNotFoundError`` if the directory or a required file is missing.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    missing_req = [r for r in required if not (root / r).exists()]
    if missing_req:
        raise FileNotFoundError(f"Missing required files in {root}: {missing_req[:5]}")
    report = Bunch(missing=[], wrong_size=[], wrong_md5=[], n_checked=0)
    mpath = root / manifest
    if verify is None or not mpath.exists():
        return report
    with open(mpath, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        rel = row.get("path") or row.get("chemin")
        if not rel:
            continue
        p = root / rel
        report.n_checked += 1
        if not p.exists():
            report.missing.append(rel)
            continue
        size = row.get("size") or row.get("taille")
        if size not in (None, "") and p.stat().st_size != int(size):
            report.wrong_size.append(rel)
        if verify == "md5" and row.get("md5") and md5sum(p) != row["md5"]:
            report.wrong_md5.append(rel)
    return report


def check_local(dataset_dir, *, required=(), verify="size", manifest="MANIFEST_NAS.tsv",
                strict=True):
    """:func:`verify_local`, then raise or warn if a file differs from the manifest.

    Parameters
    ----------
    dataset_dir, required, verify, manifest
        See :func:`verify_local`.
    strict : bool
        ``True``: raise ``OSError`` on a missing, wrong-size or wrong-md5 file;
        ``False``: emit a ``UserWarning`` instead.

    Returns
    -------
    Bunch
        The report of :func:`verify_local`.
    """
    report = verify_local(dataset_dir, manifest=manifest, verify=verify, required=required)
    bad = report.missing + report.wrong_size + report.wrong_md5
    if bad:
        msg = (f"{dataset_dir}: {len(report.missing)} missing, {len(report.wrong_size)} wrong "
               f"size, {len(report.wrong_md5)} wrong md5 files w.r.t. {manifest} "
               f"(e.g. {bad[:5]})")
        if strict:
            raise OSError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    return report


def params_hash(params: dict) -> str:
    """Short stable hash of a JSON-serialisable parameter dict."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


class Stage:
    """One cached stage: ``<cache_dir>/<name>/<stage>/<params_hash>/``."""

    def __init__(self, cache_dir, name, stage, params):
        self.params = {**params, "_core_cache_version": CACHE_VERSION}
        self.hash = params_hash(self.params)
        self.path = Path(cache_dir) / name / stage / self.hash

    def done(self) -> bool:
        return (self.path / ".done").exists()

    def mark_done(self):
        (self.path / ".done").write_text(json.dumps(self.params, sort_keys=True, default=str))

    def reset(self):
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)


def run_pipeline(name, *, params, process_fn, covariance_fn, raw_fn=None, cache_dir=None,
                 keep=None, force=False, covariance_params=None):
    """Run the staged pipeline of one dataset.

    Parameters
    ----------
    name : str
        Cache sub-directory of the dataset.
    params : dict
        Parameters of the preprocessing (hashed into both stage paths).
    covariance_params : dict, optional
        Parameters that only change the covariances (estimator, split used to
        standardise...). They are hashed into the covariance stage only, so
        changing them does not redo the preprocessing.
    raw_fn : callable(raw_dir) -> Path, optional
        Downloads the sources into ``raw_dir`` and returns the source directory.
        ``None`` for local datasets: ``process_fn`` reads the read-only copy.
    process_fn : callable(source_dir | None, out_dir) -> None
        Writes preprocessed tensors in ``out_dir``.
    covariance_fn : callable(processed_dir, out_dir) -> None
        Writes covariances (and labels, groups, metadata) in ``out_dir``.
    keep : {"keep_processed", "keep_covariance", "keep_full"}, optional
        Default: ``keep_processed`` if ``raw_fn`` is given, else ``keep_full``.

    Returns
    -------
    Bunch with ``covariance_dir``, ``processed_dir`` (or None) and ``raw_dir``
    (or None) after the ``keep`` policy has been applied.
    """
    keep = keep or ("keep_processed" if raw_fn else "keep_full")
    if keep not in KEEP:
        raise ValueError(f"keep must be one of {KEEP}")
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir()

    raw_dir = cache_dir / name / "raw"
    source = None
    proc = Stage(cache_dir, name, "processed", params)
    cov = Stage(cache_dir, name, "covariance", {**params, **(covariance_params or {})})

    if force or not cov.done():
        if force or not proc.done():
            if raw_fn is not None:
                raw_dir.mkdir(parents=True, exist_ok=True)
                source = raw_fn(raw_dir)
            proc.reset()
            process_fn(source, proc.path)
            proc.mark_done()
        cov.reset()
        covariance_fn(proc.path, cov.path)
        cov.mark_done()

    if keep in ("keep_covariance",) and proc.path.exists():
        shutil.rmtree(proc.path)
    if keep in ("keep_covariance", "keep_processed") and raw_fn is not None and raw_dir.exists():
        shutil.rmtree(raw_dir)
    return Bunch(
        covariance_dir=cov.path,
        processed_dir=proc.path if proc.path.exists() else None,
        raw_dir=raw_dir if raw_dir.exists() else None,
        keep=keep,
    )
