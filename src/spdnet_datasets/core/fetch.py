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
import http.client
import json
import shutil
import time
import urllib.error
import urllib.parse
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


class _NoCrossHostAuth(urllib.request.HTTPRedirectHandler):
    """Redirect handler that drops credentials when the host changes.

    Kaggle and Hugging Face answer with a redirect to a storage host: the
    ``Authorization`` header must not follow it (it would leak the token, and
    signed storage URLs reject it).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(
                req.full_url).netloc:
            for key in list(new.headers):
                if key.lower() == "authorization":
                    del new.headers[key]
            new.unredirected_hdrs.pop("Authorization", None)
        return new


def _open(req, timeout):
    if any(k.lower() == "authorization" for k, _ in req.header_items()):
        opener = urllib.request.build_opener(_NoCrossHostAuth)
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


_RETRY_WAIT = 2.0  # seconds before the first retry, doubled at each new one


def _total_size(resp, start):
    """Total size of the file announced by a response (``None`` when unknown)."""
    headers = getattr(resp, "headers", None) or {}
    if getattr(resp, "status", 200) == 206:
        total = (headers.get("Content-Range") or "").rpartition("/")[2]
        if total.isdigit():
            return int(total)
        length = headers.get("Content-Length")
        return start + int(length) if length and length.isdigit() else None
    length = headers.get("Content-Length")
    return int(length) if length and length.isdigit() else None


def download(url, dest, checksum=None, algo="sha256", resume=True, timeout=60,
             headers=None, size=None, retries=3) -> Path:
    """Download ``url`` to ``dest`` once; verify ``size`` and ``checksum`` if given.

    The file is written to ``dest.part`` and renamed only when complete, so an
    interrupted download is resumed (HTTP Range) or restarted, never mistaken
    for a finished one. A ``416`` answer to a resume means that ``.part`` is
    already complete: it is checked and renamed.

    Completeness is checked against ``size`` (expected size in bytes) when
    given, else against the size announced by the server (``Content-Length``,
    or the total of ``Content-Range``). A connection closed early, or broken by
    a network error, keeps ``.part`` and is resumed after a short, growing
    wait, up to ``retries`` times after the first attempt; ``OSError`` is then
    raised with ``.part`` kept (a later call resumes it). A size that cannot be
    reached by resuming (``.part`` or ``dest`` larger than, or a complete file
    different from, ``size``) raises at once. HTTP errors other than ``416``
    (401, 403, 404...) are raised at once.

    ``headers`` (e.g. an ``Authorization`` header built by
    :mod:`spdnet_datasets.core.access`) are sent with the request but never
    forwarded to another host on redirects, and never logged.
    """
    dest = Path(dest)
    if dest.exists():
        if size is not None and dest.stat().st_size != size:
            raise OSError(f"{dest}: {dest.stat().st_size} bytes, expected {size}; delete it")
        if checksum and _hash_file(dest, algo) != checksum:
            raise OSError(f"Checksum mismatch for existing file {dest}; delete it or use force")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if not resume and part.exists():
        part.unlink()
    for attempt in range(retries + 1):
        start = part.stat().st_size if part.exists() else 0
        if size is not None and start > size:
            raise OSError(f"{part}: {start} bytes, more than the expected {size}; delete it")
        expected, error = size, None
        if size is None or start < size:
            hdrs = dict(headers or {})
            if start:
                hdrs["Range"] = f"bytes={start}-"
            req = urllib.request.Request(url, headers=hdrs)
            try:
                with _open(req, timeout) as resp:
                    resumed = start and getattr(resp, "status", 200) == 206
                    total = _total_size(resp, start if resumed else 0)
                    if size is not None and total is not None and total != size:
                        raise OSError(f"{url}: the server announces {total} bytes, "
                                      f"expected {size}")
                    expected = size if size is not None else total
                    with open(part, "ab" if resumed else "wb") as out:
                        shutil.copyfileobj(resp, out, 1 << 20)
            except urllib.error.HTTPError as exc:
                # 416 on a resume: the range starts at the end of the file, .part is complete
                if not (start and exc.code == 416):
                    raise
                if size is not None and start != size:
                    raise OSError(f"{part}: {start} bytes and the server has no more, "
                                  f"expected {size}") from exc
                expected = start
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    TimeoutError) as exc:  # network failure: resumed below
                error = exc
        got = part.stat().st_size if part.exists() else 0
        if expected is not None and got > expected:
            raise OSError(f"{part}: {got} bytes, more than the expected {expected}; delete it")
        if error is None and (expected is None or got == expected):
            break
        if attempt < retries:
            time.sleep(_RETRY_WAIT * 2 ** attempt)
    else:
        raise OSError(f"Incomplete download of {url} after {retries + 1} attempts: {got} bytes"
                      f"{f' of {expected}' if expected is not None else ''}"
                      f"{f' ({error})' if error is not None else ''}; {part} is kept and the "
                      "next call resumes it")
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
