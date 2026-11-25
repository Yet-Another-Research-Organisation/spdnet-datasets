"""Shared machinery of the datasets with one covariance per recording.

Used by the audio (:mod:`~spdnet_datasets.audio`) and skeleton
(:mod:`~spdnet_datasets.skeleton`) modules: their samples are variable-length
sequences of ``p`` features (frames x MFCC, frames x joint coordinates), and one
covariance is estimated per sequence with ``n`` = its number of frames.

Cache layout written by :func:`write_processed` and :func:`estimate_stage`
(see :func:`spdnet_datasets.core.fetch.run_pipeline`)::

    processed/<hash>/features.npy   (sum n_i, p) float32, sequences concatenated
                     offsets.npy    (B + 1,) int64, sequence i = features[o[i]:o[i+1]]
                     index.csv      one row per stored sequence (id, label, group, ...)
                     excluded.csv   sequences rejected while preprocessing (id, label, reason)
    covariance/<hash>/covariances.npy  (K, p, p) float64, unscaled
                      means.npy        (K, p) float64
                      meta.csv         index rows of the K kept sequences (+ ``row``, ``n_obs``)
                      excluded.csv     every rejected sequence and the reason

A sequence with ``n < 2p`` frames or whose estimate is not SPD is excluded and
counted (``excluded.csv``), never regularised silently.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import RANDOM, BaseDataset
from ..core import (
    Bunch,
    InsufficientSamplesError,
    NotSPDError,
    apply_task,
    balance_indices,
    channel_mean,
    class_weights,
    dataset_path,
    download,
    estimate_covariance,
    geometric_scale,
    get_cache_dir,
)
from ..core.fetch import sha256sum

EXCLUDED_COLUMNS = ["id", "label", "reason"]
CSV = {"keep_default_na": False, "na_values": [""]}  # "NA" is a valid group name
REASONS = {"InsufficientSamplesError": "n < 2p", "NotSPDError": "not SPD"}


def exclusion(rec_id, label, exc):
    """Row of ``excluded.csv`` for an estimation error (reason ``"n < 2p"`` or ``"not SPD"``)."""
    return {"id": rec_id, "label": label, "reason": f"{REASONS[type(exc).__name__]} ({exc})"}


def evenly_spaced(n, limit):
    """Indices of at most ``limit`` items evenly spread over ``range(n)`` (all if None)."""
    if limit is None or limit >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, int(limit)).round().astype(int))


def download_files(files, raw_dir):
    """Download every ``url -> sha256`` of ``files`` into ``raw_dir`` (``raw_fn`` of a fetch).

    Each file is named after the last component of its URL. A complete file is
    not downloaded again (only re-hashed, see :func:`~spdnet_datasets.core.fetch.download`).
    """
    raw_dir = Path(raw_dir)
    for url, sha256 in files.items():
        download(url, raw_dir / url.rsplit("/", 1)[-1], checksum=sha256)
    return raw_dir


def downloads_dir(cache_dir, name):
    """``<cache_dir>/<name>/downloads``: persistent downloads, outside the ``keep`` policy."""
    return (Path(cache_dir) if cache_dir else get_cache_dir()) / name / "downloads"


def ingest_archive(name, archive, dest_dir, pattern, *, page_url, expected, files=None):
    """Folder holding the ``pattern`` files of a user-provided archive or folder.

    ``archive`` (zip / tar file, or an extracted folder used in place) goes
    through :func:`~spdnet_datasets.core.access.manual_source`, which extracts
    it into ``dest_dir/<archive name>/`` (with ``archive=None`` it opens
    ``page_url`` and raises
    :class:`~spdnet_datasets.core.access.ManualDownloadRequired`). Files named
    like the last URL component of a ``url -> sha256`` entry of ``files`` are
    checked.
    """
    from ..core.access import manual_source

    if archive is not None:  # one extraction folder per archive
        dest_dir = Path(dest_dir) / Path(archive).expanduser().name
    root = manual_source(name, page_url=page_url, expected=expected, dest_dir=dest_dir,
                         archive=archive)
    found = sorted(Path(root).rglob(pattern))
    if not found:
        raise FileNotFoundError(f"No {pattern} file in {root} (expected: {expected})")
    folder = found[0].parent
    for url, sha256 in (files or {}).items():
        path = folder / url.rsplit("/", 1)[-1]
        if path.exists() and sha256sum(path) != sha256:
            raise OSError(f"Checksum mismatch for {path}")
    return folder


def write_processed(out_dir, features, index, excluded, p):
    """Write the processed stage.

    Parameters
    ----------
    out_dir : Path
    features : list of arrays (n_i, p)
        Sequences kept, in the order of ``index``.
    index : list of dict
        One row per kept sequence (at least ``id``, ``label``, ``group``).
    excluded : list of dict
        Rejected sequences (``id``, ``label``, ``reason``).
    p : int
        Number of features, used when ``features`` is empty.
    """
    out_dir = Path(out_dir)
    lengths = [len(x) for x in features]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    data = (np.concatenate(features).astype(np.float32) if features
            else np.zeros((0, p), np.float32))
    np.save(out_dir / "features.npy", data)
    np.save(out_dir / "offsets.npy", offsets)
    frame = pd.DataFrame(index) if index else pd.DataFrame(columns=["id", "label", "group"])
    frame["n_obs"] = lengths
    frame.to_csv(out_dir / "index.csv", index=False)
    pd.DataFrame(excluded, columns=EXCLUDED_COLUMNS).to_csv(out_dir / "excluded.csv",
                                                            index=False)


def estimate_stage(proc_dir, out_dir, estimator):
    """Covariance stage: one covariance per processed sequence (see module docstring)."""
    proc_dir, out_dir = Path(proc_dir), Path(out_dir)
    features = np.load(proc_dir / "features.npy", mmap_mode="r")
    offsets = np.load(proc_dir / "offsets.npy")
    index = pd.read_csv(proc_dir / "index.csv", dtype={"id": str, "label": str}, **CSV)
    excluded = pd.read_csv(proc_dir / "excluded.csv", dtype=str, **CSV).to_dict("records")
    covs, means, rows = [], [], []
    for i, rec in enumerate(index.itertuples(index=False)):
        X = np.asarray(features[offsets[i]:offsets[i + 1]], dtype=np.float64)
        try:
            covs.append(estimate_covariance(X, estimator))
        except (InsufficientSamplesError, NotSPDError) as exc:
            excluded.append(exclusion(rec.id, rec.label, exc))
            continue
        means.append(channel_mean(X))
        rows.append(i)
    p = features.shape[1]
    save_covariances(out_dir, covs, means, index.iloc[rows].assign(row=rows), excluded, p)


def save_covariances(out_dir, covs, means, meta, excluded, p):
    """Write the covariance stage files."""
    out_dir = Path(out_dir)
    np.save(out_dir / "covariances.npy",
            np.stack(covs) if covs else np.zeros((0, p, p)))
    np.save(out_dir / "means.npy", np.stack(means) if means else np.zeros((0, p)))
    meta.to_csv(out_dir / "meta.csv", index=False)
    pd.DataFrame(excluded, columns=EXCLUDED_COLUMNS).to_csv(out_dir / "excluded.csv",
                                                            index=False)


def load_result(run, splits_fn=None, **split_kwargs):
    """Assemble the ``Bunch`` returned by a ``fetch_<dataset>`` function.

    Parameters
    ----------
    run : Bunch
        Output of :func:`~spdnet_datasets.core.fetch.run_pipeline`.
    splits_fn : callable(meta, labels, groups, **split_kwargs) -> dict, optional
        Builds the named splits of the dataset.
    """
    cov_dir = Path(run.covariance_dir)
    meta = pd.read_csv(cov_dir / "meta.csv", dtype={"id": str, "label": str}, **CSV)
    labels = meta["label"].to_numpy()
    groups = meta["group"].to_numpy()
    out = Bunch(
        covariances=np.load(cov_dir / "covariances.npy"),
        means=np.load(cov_dir / "means.npy"),
        labels=labels,
        groups=groups,
        meta=meta,
        excluded=pd.read_csv(cov_dir / "excluded.csv", dtype=str, **CSV),
        processed_path=run.processed_dir,
        covariance_path=cov_dir,
    )
    out.splits = {}
    if splits_fn is not None and len(meta):
        try:
            out.splits = splits_fn(meta, labels, groups, **split_kwargs)
        except ValueError as exc:  # too few samples per class or group for the folds
            warnings.warn(f"No split available: {exc}", UserWarning, stacklevel=3)
    return out


def folds_to_split(fold_of, fold, n_folds):
    """Train / val / test indices from one fold number per sample.

    The test set is fold ``fold`` (1-based), the validation set the next fold
    (cyclically) and the training set the remaining folds.
    """
    fold_of = np.asarray(fold_of)
    if not 1 <= fold <= n_folds:
        raise ValueError(f"fold must be in 1..{n_folds}, got {fold}")
    val_fold = fold % n_folds + 1
    idx = np.arange(len(fold_of))
    return {"train": idx[(fold_of != fold) & (fold_of != val_fold)],
            "val": idx[fold_of == val_fold], "test": idx[fold_of == fold]}


def kfold_assignment(y, groups=None, n_folds=5, seed=0):
    """Fold number (1..n_folds) of each sample: stratified, grouped if ``groups``."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(y)
    fold_of = np.zeros(len(y), dtype=int)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        parts = splitter.split(np.zeros(len(y)), y)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        parts = splitter.split(np.zeros(len(y)), y, groups)
    with warnings.catch_warnings():  # rare classes: fewer members than folds
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, test) in enumerate(parts, start=1):
            fold_of[test] = k
    return fold_of


class SequenceCovarianceDataset(BaseDataset):
    """Dataset over the covariances produced by a ``fetch_<dataset>`` function.

    Subclasses set ``DOMAIN``, ``NAME``, ``COARSE_MAP`` and implement
    :meth:`_fetch` (returns the fetch ``Bunch``) and :meth:`_make_splits`.

    Parameters
    ----------
    data_dir : path, optional
        Read-only copy of the dataset (default ``<data root>/<DOMAIN>/<NAME>``).
    split : str
        Name of the split in :attr:`full_splits` that defines ``subset`` and the
        training set used to fit the normalisation.
    subset : {"train", "val", "test", None}
        Part of the split exposed by the dataset; ``None`` (default, or
        ``"all"``) exposes every sample, and :attr:`splits` then indexes the dataset (use
        it with :func:`~spdnet_datasets.base.create_dataloaders`).
    fold : int
        Test fold (1-based) of the k-fold protocols.
    cache_dir, keep, force, verify, limit
        Passed to the fetch function (``limit``: at most that many sequences,
        evenly spread, for quick tests).
    archive : path, optional
        Downloaded archive or folder, passed to the fetch (see its "Download"
        section) when ``data_dir`` does not exist.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments (``classes``,
        ``max_classes``, ``max_samples_per_class``, ``return_mean``,
        ``return_raw``, ``estimator``, ``task``, ``drop_classes``,
        ``merge_classes``, ``balance``, ``normalize``, ``seed``, ...).

    Notes
    -----
    - Splits are computed after the class selection, on the whole selection
      (:attr:`full_splits`). With ``subset=None`` they index the dataset and
      are exposed as :attr:`splits`; with a subset, :attr:`splits` is empty and
      :attr:`labels`, :attr:`groups` and :attr:`meta` describe the subset.
    - :attr:`full_splits` always holds ``"random"``, a sample-level split
      (``random_split_options``) that ignores the groups; ``split="random"``
      fits the normalisation on its training part and warns when a group holds
      several samples (leaky split).
    - ``balance="undersample" | "oversample"`` resamples the training indices
      of the selected split only; ``"weights"`` sets :attr:`class_weights`
      (computed on the training set).
    - With ``normalize=True`` the covariances are multiplied by the geometric
      scale ``s`` fitted on the training set of ``split``; raw data and means
      are multiplied by ``sqrt(s)`` so that they stay consistent.
    """

    DOMAIN = NAME = None
    COARSE_MAP = None
    DOWNLOADABLE = False  # True: a missing ``data_dir`` is downloaded by the fetch

    def __init__(self, data_dir=None, *, split="recommended", subset=None, fold=1,
                 cache_dir=None, keep=None, force=False, verify="size", limit=None,
                 archive=None, **kwargs):
        root = dataset_path(self.DOMAIN, self.NAME, data_dir)
        if root.exists() or not (self.DOWNLOADABLE or archive is not None):
            super().__init__(data_dir=root, **kwargs)
        else:  # the fetch downloads the sources into the cache
            cache = Path(cache_dir) if cache_dir else get_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            super().__init__(data_dir=cache, **kwargs)
            self.data_dir = root
        if subset == "all":
            subset = None
        if subset not in (None, "train", "val", "test"):
            raise ValueError(f"subset must be 'train', 'val', 'test' or None, got {subset!r}")
        self.split, self.subset, self.fold = split, subset, fold
        self.fetch_kwargs = {"cache_dir": cache_dir, "keep": keep, "force": force,
                             "verify": verify, "limit": limit, "estimator": self.estimator}
        if archive is not None:  # only the fetches with a manual / archive mode take it
            self.fetch_kwargs["archive"] = archive
        self._load_data()

    # -- to implement -----------------------------------------------------
    def _fetch(self):
        raise NotImplementedError

    def _make_splits(self, meta, labels, groups):
        raise NotImplementedError

    # -- shared logic -----------------------------------------------------
    def _load_data(self):
        b = self._fetch()
        self.excluded = b.excluded
        y, keep, names = apply_task(b.labels, task=self.task, coarse_map=self.COARSE_MAP,
                                    drop_classes=self.drop_classes,
                                    merge_classes=self.merge_classes)
        self.classes = self._limit_classes(list(names))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        task_class = np.full(len(keep), None, dtype=object)  # class after task, per sample
        task_class[keep] = np.asarray(names, dtype=object)[y]
        by_class = {c: np.flatnonzero(task_class == c).tolist() for c in self.classes}
        idx = np.sort(np.asarray(self._limit_samples_per_class(by_class), dtype=int))
        if len(idx) == 0:
            raise ValueError("No sample left after the class selection")

        meta = b.meta.iloc[idx].reset_index(drop=True)
        labels = np.array([self.class_to_idx[c] for c in task_class[idx]], dtype=np.int64)
        groups = np.asarray(b.groups)[idx]
        self._labels_all = labels
        self._covs = b.covariances[idx]
        self._means = b.means[idx]
        self._rows = meta["row"].to_numpy() if "row" in meta else None
        self._processed = b.processed_path
        if self.return_raw and (self._processed is None or self._rows is None):
            raise ValueError("return_raw=True needs the preprocessed sequences: fetch with "
                             "keep='keep_processed' or 'keep_full'")

        try:
            self.full_splits = self._make_splits(meta, labels, groups)
        except ValueError as exc:  # too few samples per class or group for the folds
            warnings.warn(f"No split available for this selection: {exc}", UserWarning,
                          stacklevel=3)
            self.full_splits = {}
        self.full_splits[RANDOM] = self._make_random_split(labels, groups)
        self._check_selected_split(self.split, groups)
        parts = self.full_splits.get(self.split)
        if parts is None and (self.subset is not None or self.normalize):
            raise ValueError(f"Unknown split {self.split!r}; "
                             f"available: {sorted(self.full_splits)}")
        self.class_weights = None
        if parts is not None and self.balance is not None:
            train = parts["train"]
            if self.balance == "weights":
                self.class_weights = class_weights(labels[train], n_classes=len(self.classes))
            else:
                parts["train"] = train[balance_indices(labels[train], self.balance,
                                                       seed=self.seed)]
        self.scale = (geometric_scale(self._covs[parts["train"]])
                      if self.normalize and len(parts["train"]) else 1.0)

        # Public attributes are aligned with the exposed samples; ``splits``
        # indexes them only when the whole selection is exposed.
        self._index = (np.arange(len(labels)) if self.subset is None
                       else np.asarray(parts[self.subset], dtype=int))
        self.labels = labels[self._index]
        self.groups = groups[self._index]
        self.meta = meta.iloc[self._index].reset_index(drop=True)
        self.splits = self.full_splits if self.subset is None else {}
        self.samples = [{"id": i, "class_name": self.classes[c]}
                        for i, c in zip(self.meta["id"], self.labels, strict=True)]
        if self.verbose:
            print(f"{type(self).__name__}: {len(self._index)} samples "
                  f"({self.subset or 'all'}, split {self.split!r}), {len(self.classes)} "
                  f"classes, {len(self.excluded)} sequences excluded")

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self._index)

    def _raw(self, j):
        proc = Path(self._processed)
        if not hasattr(self, "_features"):
            self._features = np.load(proc / "features.npy", mmap_mode="r")
            self._offsets = np.load(proc / "offsets.npy")
        r = self._rows[j]
        return np.asarray(self._features[self._offsets[r]:self._offsets[r + 1]])

    def __getitem__(self, i):
        j = self._index[i]
        cov = torch.from_numpy(self._covs[j] * self.scale)
        root = float(np.sqrt(self.scale))
        mean = torch.from_numpy(self._means[j] * root) if self.return_mean else None
        raw = (torch.from_numpy((self._raw(j) * root).astype(np.float32))
               if self.return_raw else None)
        x = self._pack(cov, mean, raw)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self._labels_all[j])


__all__ = ["CSV", "SequenceCovarianceDataset", "estimate_stage",
           "evenly_spaced", "folds_to_split", "kfold_assignment", "load_result",
           "save_covariances", "write_processed"]
