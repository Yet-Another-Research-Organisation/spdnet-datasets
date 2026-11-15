"""PyTorch dataset of SPD covariances built from a MOABB dataset and paradigm.

MOABB downloads, reads and epochs the recordings (``paradigm.get_data``); its
raw files stay in the MNE data directory (``MNE_DATA``) and are never touched
here. This module only caches what it derives from the epochs, under
``SPDNET_CACHE_DIR/moabb_<dataset code>/`` through
:func:`spdnet_datasets.core.fetch.run_pipeline`::

    processed/<hash>/   epochs.npy (float32), labels.npy, metadata.csv
    covariance/<hash>/  covariances.npy, means.npy, labels.npy, metadata.csv,
                        trial_index.npy, n_excluded.npy

The covariance stage is keyed on the estimator only on top of the epochs key,
so changing the estimator does not call ``paradigm.get_data`` again.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.config import get_cache_dir
from ..core.estimators import InsufficientSamplesError, NotSPDError
from ..core.fetch import run_pipeline
from ..core.normalize import geometric_scale
from .covariances import epochs_to_covariances

SPLIT_KINDS = ("within_session", "cross_session", "cross_subject")


def _key_value(v):
    """JSON-friendly, process-stable form of ``v`` for the cache key."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (bool, int, float, str, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_key_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _key_value(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, np.ndarray):
        return _key_value(v.tolist())
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(v))  # drop memory addresses


def _key_attributes(obj) -> dict:
    """Public attributes of ``obj`` (dataset or paradigm options) for the cache key."""
    return {k: _key_value(v) for k, v in sorted(vars(obj).items()) if not k.startswith("_")}


def _read_metadata(path):
    return pd.read_csv(path, dtype={"session": str, "run": str})


def moabb_splits(y, metadata, n_folds=5, seed=42) -> dict:
    """Train/test indices of the three MOABB evaluation protocols.

    The folds come from MOABB's own splitters
    (:class:`moabb.evaluations.splitters.WithinSessionSplitter`,
    ``CrossSessionSplitter`` and ``CrossSubjectSplitter``), so they match
    ``WithinSessionEvaluation``, ``CrossSessionEvaluation`` and
    ``CrossSubjectEvaluation``.

    Parameters
    ----------
    y : array of shape (n_trials,)
    metadata : DataFrame with ``subject`` and ``session`` columns and a
        positional ``RangeIndex``.
    n_folds : int
        Folds of the within-session protocol (MOABB default: 5).
    seed : int
        Random state of the within-session folds.

    Returns
    -------
    dict
        ``"<protocol>_<k>" -> {"train", "val", "test"}`` (``val`` is empty).
        ``cross_session`` needs at least two sessions per subject and
        ``cross_subject`` at least two subjects; otherwise they are absent.
    """
    from moabb.evaluations.splitters import (
        CrossSessionSplitter,
        CrossSubjectSplitter,
        WithinSessionSplitter,
    )

    y = np.asarray(y)
    metadata = metadata.reset_index(drop=True)
    splitters = {"within_session": WithinSessionSplitter(n_folds=n_folds, shuffle=True,
                                                         random_state=seed)}
    if metadata.groupby("subject")["session"].nunique().min() >= 2:
        splitters["cross_session"] = CrossSessionSplitter()
    if metadata["subject"].nunique() >= 2:
        splitters["cross_subject"] = CrossSubjectSplitter()
    empty = np.array([], dtype=np.int64)
    out = {}
    for kind, splitter in splitters.items():
        for k, (train, test) in enumerate(splitter.split(y, metadata)):
            out[f"{kind}_{k}"] = {"train": np.asarray(train, dtype=np.int64), "val": empty,
                                  "test": np.asarray(test, dtype=np.int64)}
    return out


class MOABBCovariances(BaseDataset):
    """SPD covariances of the epochs of a MOABB dataset, as a PyTorch dataset.

    Parameters
    ----------
    dataset : moabb.datasets.base.BaseDataset
        Any MOABB dataset, e.g. ``BNCI2014_001()``.
    paradigm : moabb.paradigms.base.BaseParadigm
        Any MOABB paradigm compatible with ``dataset``, e.g. ``LeftRightImagery()``.
    subjects : list of int, optional
        Subjects to load (default: ``dataset.subject_list``).
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
        Covariance estimator; ``n_times >= 2 * n_channels`` is enforced.
    keep : {"keep_processed", "keep_covariance", "keep_full"}, optional
        What stays in the cache. ``keep_covariance`` deletes the cached epochs,
        so ``return_raw`` is then unavailable. The MNE/MOABB raw files are never
        deleted. Default: ``keep_full`` (epochs and covariances kept).
    return_mean, return_raw : bool
        Optional outputs (:mod:`spdnet_datasets.core.outputs`): the channel mean
        (n_channels,) and the epoch (n_times, n_channels).
    split : str, optional
        Name of a split of :attr:`splits`; with ``normalize=True`` the geometric
        scale is fitted on its training indices. Without a split no scale is
        applied (there is no training set to fit it on).
    normalize : bool
        Apply the geometric scale fitted on ``splits[split]["train"]``.
    n_folds : int
        Folds of the within-session protocol.
    cache_dir : path, optional
        Default: ``$SPDNET_CACHE_DIR``.
    force : bool
        Recompute the cached epochs and covariances.
    seed : int
        Random state of the within-session folds.

    Attributes
    ----------
    metadata : DataFrame
        MOABB metadata (``subject``, ``session``, ``run``, ...), one row per trial.
    labels : ndarray of int
        Index into :attr:`classes`.
    groups : ndarray of str
        ``"<subject>-<session>"`` of each trial.
    subjects, sessions : ndarray
        Subject and session of each trial.
    splits : dict
        ``within_session_<k>``, ``cross_session_<k>``, ``cross_subject_<k>``
        (see :func:`moabb_splits`) and ``recommended``, an alias of the first
        fold of the strictest protocol available (cross-subject, then
        cross-session, then within-session).
    scale : float
        Geometric scale applied to the covariances.
    n_excluded : int
        Trials dropped because their covariance could not be estimated
        (too few time samples, or not SPD).
    trial_index : ndarray of int
        Index of each kept trial in the cached epochs.
    """

    def __init__(self, dataset, paradigm, subjects=None, *, estimator="scm", keep=None,
                 return_mean=False, return_raw=False, split=None, normalize=True,
                 n_folds=5, cache_dir=None, force=False, seed=42, verbose=False):
        cache_dir = get_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(cache_dir, seed=seed, verbose=verbose, return_mean=return_mean,
                         return_raw=return_raw, estimator=estimator, normalize=normalize)
        if return_raw and keep == "keep_covariance":
            raise ValueError("return_raw=True needs the cached epochs: use "
                             "keep='keep_processed' or 'keep_full', not 'keep_covariance'.")
        self.dataset, self.paradigm = dataset, paradigm
        self.subjects_requested = sorted(subjects or dataset.subject_list)
        self.keep, self.force, self.n_folds = keep, force, n_folds
        self.split = split
        self._load_data()

    def _params(self):
        return {"dataset": self.dataset.code, "subjects": self.subjects_requested,
                "dataset_params": _key_attributes(self.dataset),
                "paradigm": type(self.paradigm).__name__,
                "paradigm_params": _key_attributes(self.paradigm)}

    def _process(self, _source, out):
        X, y, metadata = self.paradigm.get_data(dataset=self.dataset,
                                                subjects=self.subjects_requested)
        np.save(out / "epochs.npy", np.asarray(X, dtype=np.float32))
        np.save(out / "labels.npy", np.asarray(y).astype(str))
        metadata.reset_index(drop=True).to_csv(out / "metadata.csv", index=False)

    def _covariances(self, processed, out):
        """Covariances of the epochs; trials whose estimate fails are excluded.

        A trial raising :class:`InsufficientSamplesError` or :class:`NotSPDError`
        is dropped from every array (``trial_index.npy`` maps the kept trials to
        the cached epochs). If no trial is left, the last error is raised.
        """
        X = np.load(processed / "epochs.npy")
        covs, kept, error = [], [], None
        for i, epoch in enumerate(X):
            try:
                covs.append(epochs_to_covariances(epoch[None], self.estimator)[0])
            except (InsufficientSamplesError, NotSPDError) as exc:
                error = exc
                continue
            kept.append(i)
        if not kept:
            raise error
        kept = np.asarray(kept, dtype=np.int64)
        np.save(out / "covariances.npy", np.stack(covs))
        np.save(out / "means.npy", X[kept].mean(axis=2, dtype=np.float64))
        np.save(out / "labels.npy", np.load(processed / "labels.npy")[kept])
        np.save(out / "trial_index.npy", kept)
        np.save(out / "n_excluded.npy", len(X) - len(kept))
        metadata = _read_metadata(processed / "metadata.csv").iloc[kept]
        metadata.reset_index(drop=True).to_csv(out / "metadata.csv", index=False)

    def _load_data(self):
        paths = run_pipeline(f"moabb_{self.dataset.code}", params=self._params(),
                             process_fn=self._process, covariance_fn=self._covariances,
                             cache_dir=self.data_dir, keep=self.keep, force=self.force,
                             covariance_params={"estimator": self.estimator})
        self.paths = paths
        d = paths.covariance_dir
        self.covs = np.load(d / "covariances.npy")
        self.trial_index = np.load(d / "trial_index.npy")
        self.n_excluded = int(np.load(d / "n_excluded.npy"))
        self.means = np.load(d / "means.npy")
        self.metadata = _read_metadata(d / "metadata.csv")
        self.classes, self.labels = np.unique(np.load(d / "labels.npy"), return_inverse=True)
        self.classes = self.classes.tolist()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # one entry per trial, as read by base.create_dataloaders (verbose split report)
        self.samples = [{"class_name": self.classes[k]} for k in self.labels]
        self.subjects = self.metadata["subject"].to_numpy()
        self.sessions = self.metadata["session"].to_numpy()
        self.groups = (self.metadata["subject"].astype(str) + "-"
                       + self.metadata["session"].astype(str)).to_numpy()
        self.epochs = None
        if self.return_raw:
            if paths.processed_dir is None:
                raise ValueError("The cached epochs were deleted (keep='keep_covariance'); "
                                 "rebuild with force=True and another keep to use return_raw.")
            self.epochs = np.load(paths.processed_dir / "epochs.npy", mmap_mode="r")

        self.splits = moabb_splits(self.labels, self.metadata, self.n_folds, self.seed)
        for kind in ("cross_subject", "cross_session", "within_session"):
            if f"{kind}_0" in self.splits:
                self.splits["recommended"] = self.splits[f"{kind}_0"]
                break
        self.scale = 1.0
        if self.split is not None:
            if self.split not in self.splits:
                raise ValueError(f"Unknown split {self.split!r}; available: {sorted(self.splits)}")
            if self.normalize:
                self.scale = geometric_scale(self.covs[self.splits[self.split]["train"]])

    @property
    def covariances(self) -> np.ndarray:
        """All covariances (n_trials, n_channels, n_channels), scaled by :attr:`scale`."""
        return self.covs * self.scale

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        cov = torch.from_numpy(self.covs[idx] * self.scale)
        mean = torch.from_numpy(self.means[idx]) if self.return_mean else None
        raw = (torch.from_numpy(np.array(self.epochs[self.trial_index[idx]].T, dtype=np.float32))
               if self.return_raw else None)
        return self._pack(cov, mean, raw), int(self.labels[idx])
