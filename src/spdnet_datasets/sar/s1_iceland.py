"""S1 Iceland SAR Dataset: temporal dual-pol amplitude covariances (30 x 30).

Local dataset of the LISTIC (not published yet: ``FILES`` is empty and the
fetch only checks the local copy against ``MANIFEST_NAS.tsv``). Directory:
``<data_root>/SAR/ICELAND_SAR`` with

- ``segmentation_amp_dataset.zarr/{train,val,test}/X_amp`` (N, T, 2, 256, 256)
  uint16: gamma0 amplitude of VV and VH times ``amp_scale`` (6000);
- ``segmentation_dataset.zarr/{train,val,test}/{X,y,r0,c0}`` and the dates in
  the group attributes (train/val: summer 2019, test: summer 2025); the complex
  ``X`` is only read for its validity mask (out of swath = 0 + 0j);
- ``splits/segmentation_blocks.shp``: the 10 km blocks of the official split.

Samples
-------
Each 256 x 256 patch is cut into non-overlapping 32 x 32 sub-patches, kept with
the rules of the dataset benchmark (atlas fiche ``iceland_sar``): at least 50 %
of the pixels labelled, at least 90 % valid, and a majority class covering at
least 60 % of the labelled valid pixels. A pixel is valid when VV and VH are
non-zero at every kept date (the ``nodata`` mask misses the swath gaps of
single 2025 dates, where the amplitude is 0). The covariance of a sub-patch is
estimated on its valid pixels (n >= 922 >= 2p = 60).

Channels (p = 30) are ``VV(t_1..t_15), VH(t_1..t_15)``: the amplitude at the 15
dates of one summer. 2019 has 15 dates; 2025 has 16, and the contiguous window
of 15 dates whose days of year are closest to the 2019 dates is kept
(:func:`select_dates`): this drops 2025-09-14, the last date, and leaves every
2025 date two days before its 2019 counterpart (dropping the first date
instead would give a four-day shift).

Split
-----
``"official"``: train and val are the 2019 patches, test the 2025 patches of a
disjoint band of 10 km blocks. ``groups`` is the 10 km block (1000 x 1000 px
cell of the common 10 m grid) holding the sub-patch centre.

Notes
-----
- Amplitudes below ``1 / amp_scale`` are stored as 0 (about 7 % of VV and 27 %
  of VH values). A sub-patch whose covariance is then not SPD (e.g. a channel
  constant over the window) is excluded and counted in ``n_excluded``.
- Overlapping 256 px patches of the same split make some training sub-patches
  overlap; windows never overlap across splits.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from ..base import BaseDataset
from ..core.config import dataset_path
from ..core.estimators import NotSPDError, channel_mean, estimate_covariance
from ..core.fetch import Bunch, run_pipeline, verify_local
from ..core.normalize import apply_scale, geometric_scale
from ..core.tasks import apply_task, balance_indices, class_weights
from ..manager import DatasetManager

DOMAIN, NAME = "SAR", "ICELAND_SAR"
FILES = {}  # not published yet: local copy only (URL to be added on release)
SEG, AMP = "segmentation_dataset.zarr", "segmentation_amp_dataset.zarr"
SPLITS = ("train", "val", "test")
REFERENCE_SPLIT = "train"  # its dates (2019) fix T and the date alignment
CLASS_NAMES = ("other_vegetation", "sealed", "periodic_herbaceous", "sparse_vegetated",
               "water", "snow_ice", "open_spaces")
COARSE_MAP = {"other_vegetation": "vegetation", "periodic_herbaceous": "vegetation",
              "sparse_vegetated": "vegetation", "sealed": "sealed", "water": "water",
              "snow_ice": "snow_ice", "open_spaces": "open_spaces"}
IGNORE = 255
BLOCK_PX = 1000  # 10 km blocks of the official split on the 10 m grid
VERSION = 1  # bump when the preprocessing changes (part of the cache key)


def _doy(d):
    return date(int(d[:4]), int(d[4:6]), int(d[6:8])).timetuple().tm_yday


def select_dates(dates, ref_dates):
    """Indices of the contiguous window of ``len(ref_dates)`` dates closest to them.

    Closeness is the mean absolute day-of-year difference between the i-th
    date of the window and the i-th reference date; ties keep the earliest
    window. Dates are ``"YYYYMMDD"`` strings in increasing order.
    """
    T = len(ref_dates)
    if len(dates) < T:
        raise ValueError(f"{len(dates)} dates, at least {T} are needed")
    ref = np.array([_doy(d) for d in ref_dates])
    doy = np.array([_doy(d) for d in dates])
    cost = [np.abs(doy[s:s + T] - ref).mean() for s in range(len(dates) - T + 1)]
    start = int(np.argmin(cost))
    return np.arange(start, start + T)


def _subpatches(y, valid, block, min_labeled, min_valid, min_purity):
    """Rows of ``(r, c, label)`` for the kept ``block x block`` sub-patches."""
    out = []
    H, W = y.shape
    for r in range(0, H - block + 1, block):
        for c in range(0, W - block + 1, block):
            yb, vb = y[r:r + block, c:c + block], valid[r:r + block, c:c + block]
            lab = yb != IGNORE
            if lab.mean() < min_labeled or vb.mean() < min_valid:
                continue
            cnt = np.bincount(yb[lab & vb], minlength=len(CLASS_NAMES))
            if cnt.sum() == 0:
                continue
            k = int(cnt.argmax())
            if cnt[k] / cnt.sum() >= min_purity:
                out.append((r, c, k))
    return out


def _process(root, out, *, block, min_labeled, min_valid, min_purity):
    """Write the uint16 sub-patch blocks (N, B, B, 2T), their masks and metadata."""
    import zarr

    seg = zarr.open_group(str(root / SEG), mode="r")
    amp = zarr.open_group(str(root / AMP), mode="r")
    amp_scale = float(amp.attrs["amp_scale"])
    ref = list(seg[REFERENCE_SPLIT].attrs["dates"])
    meta = {k: [] for k in ("split", "year", "patch", "row", "col", "label")}
    valid_all, info = [], {"amp_scale": amp_scale, "block": block, "dates": {}, "dropped": {}}
    with open(out / "raw.u16", "wb") as fraw:
        for sp in SPLITS:
            g, X = seg[sp], amp[sp]["X_amp"]
            dates = list(g.attrs["dates"])
            keep = select_dates(dates, ref)
            info["dates"][sp] = [dates[i] for i in keep]
            info["dropped"][sp] = [d for i, d in enumerate(dates) if i not in set(keep.tolist())]
            r0, c0 = g["r0"][:], g["c0"][:]
            for i in range(X.shape[0]):
                Z = g["X"][i][keep]  # (T, 4, H, W): re/im of VV, VH
                valid = (((Z[:, 0] != 0) | (Z[:, 1] != 0)) & ((Z[:, 2] != 0) | (Z[:, 3] != 0))).all(0)
                y = g["y"][i]
                rows = _subpatches(y, valid, block, min_labeled, min_valid, min_purity)
                if not rows:
                    continue
                A = X[i][keep]  # (T, 2, H, W)
                for r, c, k in rows:
                    b = A[:, :, r:r + block, c:c + block]  # (T, 2, B, B)
                    b = b.transpose(2, 3, 1, 0).reshape(block, block, -1)  # VV(t), VH(t)
                    fraw.write(np.ascontiguousarray(b, dtype="<u2").tobytes())
                    valid_all.append(valid[r:r + block, c:c + block])
                    meta["split"].append(sp)
                    meta["year"].append(int(dates[0][:4]))
                    meta["patch"].append(i)
                    meta["row"].append(int(r0[i]) + r)
                    meta["col"].append(int(c0[i]) + c)
                    meta["label"].append(k)
    info["n"] = len(valid_all)
    info["p"] = 2 * len(ref)
    np.save(out / "valid.npy", np.array(valid_all, dtype=bool).reshape(-1, block, block))
    np.savez(out / "meta.npz", **{k: np.asarray(v) for k, v in meta.items()})
    (out / "info.json").write_text(json.dumps(info, indent=1))


def _load_raw(proc):
    info = json.loads((Path(proc) / "info.json").read_text())
    B = info["block"]
    raw = np.memmap(Path(proc) / "raw.u16", dtype="<u2", mode="r",
                    shape=(info["n"], B, B, info["p"]))
    return raw, np.load(Path(proc) / "valid.npy"), info


def _covariances(proc, out, *, estimator):
    """Covariances and means on the valid pixels; non-SPD samples are flagged."""
    raw, valid, info = _load_raw(proc)
    n, p = info["n"], info["p"]
    covs, means = np.zeros((n, p, p)), np.zeros((n, p))
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        X = raw[i][valid[i]].astype(np.float64) / info["amp_scale"]
        means[i] = channel_mean(X)
        if np.any(np.ptp(X, axis=0) == 0):  # constant channel: singular covariance
            ok[i] = False
            continue
        try:
            covs[i] = estimate_covariance(X, estimator=estimator)
        except NotSPDError:
            ok[i] = False
    np.save(out / "covariances.npy", covs)
    np.save(out / "means.npy", means)
    np.save(out / "spd_ok.npy", ok)
    for name in ("meta.npz", "info.json"):
        (out / name).write_bytes((Path(proc) / name).read_bytes())


def fetch_s1_iceland(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                     verify="size", block=32, min_labeled=0.5, min_valid=0.9, min_purity=0.6):
    """Load (computing once) the 30 x 30 amplitude covariances of S1 Iceland.

    Parameters
    ----------
    data_dir : path, optional
        Dataset directory (default ``<data_root>/SAR/ICELAND_SAR``), read-only.
    cache_dir : path, optional
        Cache root (default :func:`~spdnet_datasets.core.config.get_cache_dir`).
    keep : {"keep_full", "keep_processed", "keep_covariance"}, optional
        ``keep_covariance`` drops the cached sub-patch blocks (no ``return_raw``).
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
    force : bool
        Recompute the cached stages.
    verify : {"size", "md5", None}
        Check of the local copy against ``MANIFEST_NAS.tsv``.
    block, min_labeled, min_valid, min_purity
        Sub-patch side and selection rules (defaults of the dataset benchmark).

    Returns
    -------
    Bunch
        ``covariances`` (N, 30, 30) unscaled, ``means`` (N, 30), ``labels`` (N,)
        indices into ``class_names``, ``groups`` (10 km block ids), ``meta``
        (DataFrame), ``splits`` ({"official": {"train", "val", "test"}}),
        ``dates`` and ``dropped_dates`` per split, ``n_excluded`` (non-SPD
        samples per split), ``amp_scale``, ``processed_path`` (or None) and
        ``covariance_path``.
    """
    import pandas as pd

    root = dataset_path(DOMAIN, NAME, data_dir)
    report = verify_local(root, verify=verify, required=(f"{SEG}/.zgroup", f"{AMP}/.zgroup"))
    bad = report.missing + report.wrong_size + report.wrong_md5
    if bad:
        raise OSError(f"{len(bad)} files of {root} differ from MANIFEST_NAS.tsv, e.g. {bad[:3]}")
    sel = {"block": block, "min_labeled": min_labeled, "min_valid": min_valid,
           "min_purity": min_purity}
    params = {"version": VERSION, "source": str(root), "estimator": estimator, **sel}
    res = run_pipeline(
        "s1_iceland", params=params, cache_dir=cache_dir, keep=keep, force=force,
        process_fn=lambda _src, out: _process(root, out, **sel),
        covariance_fn=lambda proc, out: _covariances(proc, out, estimator=estimator))
    cdir = res.covariance_dir
    ok = np.load(cdir / "spd_ok.npy")
    m = np.load(cdir / "meta.npz")
    info = json.loads((cdir / "info.json").read_text())
    meta = pd.DataFrame({k: m[k] for k in m.files})
    meta["raw_index"] = np.arange(len(meta))
    n_excluded = {sp: int((~ok[meta["split"].to_numpy() == sp]).sum()) for sp in SPLITS}
    meta = meta[ok].reset_index(drop=True)
    centre = block // 2
    groups = ((meta["row"].to_numpy() + centre) // BLOCK_PX) * 100_000 + (
        (meta["col"].to_numpy() + centre) // BLOCK_PX)
    split = meta["split"].to_numpy()
    return Bunch(
        covariances=np.load(cdir / "covariances.npy", mmap_mode="r")[ok],
        means=np.load(cdir / "means.npy")[ok],
        labels=meta["label"].to_numpy(np.int64),
        class_names=list(CLASS_NAMES),
        groups=groups,
        meta=meta,
        splits={"official": {sp: np.flatnonzero(split == sp) for sp in SPLITS}},
        dates=info["dates"],
        dropped_dates=info["dropped"],
        n_excluded=n_excluded,
        amp_scale=info["amp_scale"],
        processed_path=res.processed_dir,
        covariance_path=cdir,
    )


def _restrict_splits(splits, keep):
    """Re-index ``splits`` on the samples selected by the boolean mask ``keep``."""
    new = np.full(len(keep), -1)
    new[keep] = np.arange(keep.sum())
    return {name: {k: new[v][keep[v]] for k, v in parts.items()} for name, parts in splits.items()}


@DatasetManager.register_dataset("s1_iceland")
class S1IcelandDataset(BaseDataset):
    """S1 Iceland SAR sub-patches as 30 x 30 temporal VV/VH amplitude covariances.

    A sample is ``(x, y)``: ``x`` is the covariance (30, 30) float64, or a
    tuple with the channel mean (30,) float64 and/or the raw block
    (32, 32, 30) float32 (amplitudes, invalid pixels at 0); ``y`` the class.

    Parameters
    ----------
    data_dir : path, optional
        Dataset directory (default ``<data_root>/SAR/ICELAND_SAR``).
    split : str
        Named split used to fit the normalisation and the balancing
        (``"official"``; ``"recommended"`` is an alias).
    subset : {None, "train", "val", "test"}
        Keep only one part of ``split`` (default: all samples, use
        ``dataset.splits`` or ``create_dataloaders(split="official")``).
    cache_dir, keep, force, verify, block, min_labeled, min_valid, min_purity
        Passed to :func:`fetch_s1_iceland`.
    task : {"fine", "coarse"}
        ``coarse`` merges the three vegetation classes (``COARSE_MAP``).
    **kwargs
        ``return_mean``, ``return_raw``, ``estimator``, ``drop_classes``,
        ``merge_classes``, ``balance``, ``normalize``, ``seed``, ``verbose``
        (see :class:`~spdnet_datasets.base.BaseDataset`).
    """

    def __init__(self, data_dir=None, *, split="official", subset=None, cache_dir=None,
                 keep=None, force=False, verify="size", block=32, min_labeled=0.5,
                 min_valid=0.9, min_purity=0.6, **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(root, **kwargs)
        self.split = "official" if split == "recommended" else split
        self.subset = subset
        self._fetch_kw = {"cache_dir": cache_dir, "keep": keep, "force": force,
                          "verify": verify, "block": block, "min_labeled": min_labeled,
                          "min_valid": min_valid, "min_purity": min_purity}
        self._raw = None
        self._load_data()

    def _load_data(self):
        b = fetch_s1_iceland(self.data_dir, estimator=self.estimator, **self._fetch_kw)
        if self.return_raw and b.processed_path is None:
            raise ValueError("return_raw=True needs the cached sub-patches: use "
                             "keep='keep_full' or 'keep_processed'")
        if self.split not in b.splits:
            raise ValueError(f"Unknown split {self.split!r}; available: {sorted(b.splits)}")
        names = np.asarray(b.class_names)[b.labels]
        y, keep, self.classes = apply_task(
            names, task=self.task, coarse_map=COARSE_MAP, drop_classes=self.drop_classes,
            merge_classes=self.merge_classes)
        splits = _restrict_splits(b.splits, keep)
        covs = np.asarray(b.covariances[keep])
        if self.balance in ("undersample", "oversample"):
            for parts in splits.values():
                tr = parts["train"]
                parts["train"] = tr[balance_indices(y[tr], self.balance, seed=self.seed)]
        train = splits[self.split]["train"]
        self.scale = geometric_scale(covs[train]) if self.normalize else 1.0
        self.class_weights = class_weights(y[train]) if self.balance == "weights" else None
        idx = np.arange(len(y))
        if self.subset is not None:
            idx = splits[self.split][self.subset]
            splits = {self.split: {self.subset: np.arange(len(idx))}}
        self.covariances = apply_scale(covs[idx], self.scale)
        self.means = b.means[keep][idx]
        self.labels = y[idx]
        self.groups = b.groups[keep][idx]
        self.meta = b.meta[keep].iloc[idx].reset_index(drop=True)
        self.splits = splits
        self.samples = list(range(len(idx)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.dates, self.dropped_dates = b.dates, b.dropped_dates
        self.n_excluded, self.amp_scale = b.n_excluded, b.amp_scale
        self._processed_path = b.processed_path
        if self.verbose:
            print(f"S1 Iceland: {len(idx)} sub-patches, {len(self.classes)} classes, "
                  f"dropped dates {b.dropped_dates}, non-SPD excluded {b.n_excluded}")

    def _raw_block(self, i):
        if self._raw is None:
            self._raw = _load_raw(self._processed_path)[0]
        block = self._raw[int(self.meta["raw_index"].iloc[i])]
        return torch.from_numpy(block.astype(np.float32) / np.float32(self.amp_scale))

    def __getitem__(self, i):
        cov = torch.from_numpy(np.array(self.covariances[i]))
        mean = torch.from_numpy(self.means[i]) if self.return_mean else None
        raw = self._raw_block(i) if self.return_raw else None
        x = self._pack(cov, mean, raw)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[i])
