"""Shared pieces of the micro-Doppler datasets (CI4R, MVDoppler, DopNet).

Every dataset turns one recording into a preprocessed spectrogram ``X`` of
shape ``(T, p)`` (time steps x frequency channels) and estimates one ``p x p``
covariance over the ``T`` time steps. The cache layout written by the
``process`` and ``covariance`` stages of :func:`spdnet_datasets.core.run_pipeline`
is the same for the three datasets::

    processed/<hash>/raw.npy        (sum of T, p) float32, spectrograms concatenated
    processed/<hash>/offsets.npy    (B + 1,) int64, sample i is raw[offsets[i]:offsets[i+1]]
    processed/<hash>/meta.csv       one row per kept sample (label, group, source...)
    processed/<hash>/info.json      dataset-level counts (duplicates, rejections...)
    covariance/<hash>/covariances.npy, means.npy, meta.csv, info.json

Covariances are stored unscaled; the geometric scale is fitted on the training
part of the selected split when the dataset is instantiated.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.estimators import channel_mean, estimate_covariance
from ..core.fetch import Bunch
from ..core.normalize import geometric_scale
from ..core.tasks import apply_task, balance_indices, class_weights


def write_processed(out_dir, spectrograms, meta, info):
    """Write a list of ``(T_i, p)`` arrays, their metadata and dataset counts."""
    out_dir = Path(out_dir)
    lengths = np.array([len(x) for x in spectrograms], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    p = spectrograms[0].shape[1]
    raw = np.lib.format.open_memmap(out_dir / "raw.npy", mode="w+", dtype=np.float32,
                                    shape=(int(offsets[-1]), p))
    for x, a, b in zip(spectrograms, offsets[:-1], offsets[1:], strict=True):
        raw[a:b] = x
    raw.flush()
    del raw
    np.save(out_dir / "offsets.npy", offsets)
    finish_processed(out_dir, meta, info)


def finish_processed(out_dir, meta, info):
    """Write ``meta.csv`` and ``info.json`` of the processed stage."""
    meta = meta.reset_index(drop=True).copy()
    meta["n_obs"] = np.diff(np.load(Path(out_dir) / "offsets.npy"))
    meta.to_csv(Path(out_dir) / "meta.csv", index=False)
    (Path(out_dir) / "info.json").write_text(json.dumps(info, indent=1, default=str))


def covariance_stage(estimator):
    """Return the ``covariance_fn`` of :func:`run_pipeline` for ``estimator``.

    Every covariance goes through :func:`core.estimate_covariance` with
    ``check=True``: ``n >= 2p`` and the SPD property are enforced per sample.

    A spectrogram with a zero-variance channel (a frequency bin that is
    constant over time, e.g. clipped to the colour-map floor) has a singular
    SCM and M-estimate. Unless ``estimator="ledoit_wolf"`` (shrinkage keeps it
    SPD), such samples are rejected and counted in
    ``info["rejected_constant_channel"]``; any other non-SPD estimate raises.
    """

    def covariance_fn(processed_dir, out_dir):
        processed_dir, out_dir = Path(processed_dir), Path(out_dir)
        raw = np.load(processed_dir / "raw.npy", mmap_mode="r")
        offsets = np.load(processed_dir / "offsets.npy")
        meta = pd.read_csv(processed_dir / "meta.csv", dtype={"group": str, "participant": str})
        meta["raw_index"] = np.arange(len(meta))
        p = raw.shape[1]
        covs, means, kept = [], [], []
        for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            X = np.asarray(raw[a:b], dtype=np.float64)
            if estimator != "ledoit_wolf" and np.any(X.std(axis=0) == 0):
                continue
            try:
                covs.append(estimate_covariance(X, estimator))
            except ValueError as exc:
                raise type(exc)(f"sample {i} ({meta.iloc[i].to_dict()}): {exc}") from exc
            means.append(channel_mean(X))
            kept.append(i)
        np.save(out_dir / "covariances.npy", np.stack(covs) if covs else np.empty((0, p, p)))
        np.save(out_dir / "means.npy", np.stack(means) if means else np.empty((0, p)))
        meta.iloc[kept].to_csv(out_dir / "meta.csv", index=False)
        info = json.loads((processed_dir / "info.json").read_text())
        info["estimator"] = estimator
        info["rejected_constant_channel"] = len(meta) - len(kept)
        info["n_samples"] = len(kept)
        (out_dir / "info.json").write_text(json.dumps(info, indent=1, default=str))

    return covariance_fn


def load_bunch(paths, class_names):
    """Read the cached covariance stage into a :class:`Bunch`."""
    cov_dir = Path(paths.covariance_dir)
    meta = pd.read_csv(cov_dir / "meta.csv", dtype={"group": str, "participant": str})
    return Bunch(
        covariances=np.load(cov_dir / "covariances.npy"),
        means=np.load(cov_dir / "means.npy"),
        labels=meta["label"].to_numpy(dtype=np.int64),
        class_names=list(class_names),
        groups=meta["group"].to_numpy(),
        meta=meta,
        info=json.loads((cov_dir / "info.json").read_text()),
        covariance_path=cov_dir,
        processed_path=paths.processed_dir,
        keep=paths.keep,
    )


def load_raw(processed_path):
    """Memory-mapped spectrograms and offsets, or ``(None, None)``."""
    if processed_path is None or not (Path(processed_path) / "raw.npy").exists():
        return None, None
    return (np.load(Path(processed_path) / "raw.npy", mmap_mode="r"),
            np.load(Path(processed_path) / "offsets.npy"))


def group_folds(groups, n_splits, y=None):
    """``{"fold<k>": {"train", "val", "test"}}`` from a grouped K-fold (empty val)."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    folds = GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), y, groups)
    return {f"fold{k}": {"train": np.sort(tr), "val": np.array([], dtype=np.int64),
                         "test": np.sort(te)}
            for k, (tr, te) in enumerate(folds)}


class RadarCovarianceDataset(BaseDataset):
    """Common dataset logic on top of a ``fetch_<dataset>`` result.

    The instance holds **all** samples. ``dataset.splits`` indexes into it, so
    ``create_dataloaders(dataset, split=name)`` works directly; ``split`` names
    the split whose training part fits the normalisation scale and the class
    weights. ``subset="train"|"val"|"test"`` restricts the instance to one part
    of that split (``splits`` is then emptied).

    With ``normalize=True`` the covariances are multiplied by the geometric
    scale ``s`` of the training part, ``mean`` and ``raw`` by ``sqrt(s)``.
    """

    COARSE_MAP = None

    def __init__(self, bunch, *, split, subset=None, **kwargs):
        super().__init__(**kwargs)
        self._bunch = bunch
        self.split = split
        self.subset = subset
        self._load_data()

    @property
    def num_classes(self):
        return len(self.classes)

    def _load_data(self):
        b = self._bunch
        if self.split not in b.splits:
            raise ValueError(f"Unknown split {self.split!r}; available: {sorted(b.splits)}")
        names = np.array(b.class_names, dtype=object)[b.labels]

        present = set(names.tolist())
        selected = set(self._limit_classes([c for c in b.class_names if c in present]))
        keep = np.isin(names, list(selected))
        y, task_keep, classes = apply_task(
            names[keep], task=self.task, coarse_map=self.COARSE_MAP,
            drop_classes=self.drop_classes, merge_classes=self.merge_classes)
        idx = np.flatnonzero(keep)[task_keep]
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        by_class = {c: [int(i) for i, yy in zip(idx, y, strict=True) if self.classes[yy] == c]
                    for c in self.classes}
        idx = np.sort(np.array(self._limit_samples_per_class(by_class), dtype=np.int64))
        label_of = dict(zip(np.flatnonzero(keep)[task_keep].tolist(), y.tolist(), strict=True))

        position = {int(j): k for k, j in enumerate(idx)}
        splits = {}
        for name, parts in b.splits.items():
            splits[name] = {part: np.array([position[int(j)] for j in ix if int(j) in position],
                                           dtype=np.int64) for part, ix in parts.items()}
        self._index = idx
        self.labels = np.array([label_of[int(j)] for j in idx], dtype=np.int64)
        self.groups = np.asarray(b.groups)[idx]
        self.meta = b.meta.iloc[idx].reset_index(drop=True)

        train = splits[self.split]["train"]
        if self.balance in ("undersample", "oversample"):
            for parts in splits.values():
                tr = parts["train"]
                parts["train"] = tr[balance_indices(self.labels[tr], self.balance, self.seed)]
        train = splits[self.split]["train"]
        self.class_weights = (class_weights(self.labels[train], n_classes=len(self.classes))
                              if len(train) else None)
        self.scale = 1.0
        if self.normalize and len(train):
            self.scale = geometric_scale(b.covariances[idx[np.unique(train)]])

        if self.subset is not None:
            part = splits[self.split][self.subset]
            self._index, self.labels = self._index[part], self.labels[part]
            self.groups = self.groups[part]
            self.meta = self.meta.iloc[part].reset_index(drop=True)
            splits = {}
        self.splits = splits

        self._raw, self._offsets = (load_raw(b.processed_path) if self.return_raw
                                    else (None, None))
        if self.return_raw and self._raw is None:
            raise ValueError("return_raw=True needs the processed stage: fetch with "
                             "keep='keep_processed' or 'keep_full'")
        self.samples = [{"index": int(j), "class_name": self.classes[c]}
                        for j, c in zip(self._index, self.labels, strict=True)]
        if self.verbose:
            counts = np.bincount(self.labels, minlength=len(self.classes))
            print(f"{type(self).__name__}: {len(self)} samples, {len(self.classes)} classes, "
                  f"p={b.covariances.shape[-1]}, split={self.split!r}, scale={self.scale:.3e}")
            print("  " + ", ".join(f"{c}: {n}" for c, n in zip(self.classes, counts, strict=True)))

    def structured_raw(self, i):
        """Preprocessed spectrogram of sample ``i``, ``(T, p)`` unscaled (axes time x bins).

        Source of :class:`~spdnet_datasets.series.CovarianceSeries`; needs the processed stage.
        """
        if self._raw is None:
            self._raw, self._offsets = load_raw(self._bunch.processed_path)
            if self._raw is None:
                raise ValueError("structured_raw needs the processed stage: fetch with "
                                 "keep='keep_processed' or 'keep_full'")
        r = int(self._bunch.meta["raw_index"].iat[int(self._index[i])])
        return np.asarray(self._raw[self._offsets[r]:self._offsets[r + 1]])

    def __getitem__(self, i):
        j = int(self._index[i])
        b = self._bunch
        cov = torch.from_numpy(b.covariances[j] * self.scale)
        root = np.sqrt(self.scale)
        mean = torch.from_numpy(b.means[j] * root) if self.return_mean else None
        raw = None
        if self.return_raw:
            r = int(b.meta["raw_index"].iat[j])
            a, e = self._offsets[r], self._offsets[r + 1]
            raw = torch.from_numpy((np.asarray(self._raw[a:e]) * root).astype(np.float32))
        x = self._pack(cov, mean, raw)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[i])
