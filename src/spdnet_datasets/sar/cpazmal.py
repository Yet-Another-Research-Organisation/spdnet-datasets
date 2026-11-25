"""CPAZMaL: PAZ X-band dual-polarisation time series of the Mont-Blanc massif.

Data: ``SAR/PAZ_S1_proc_v2/CPAZMaL.hdf5`` (+ ``CPAZMaL.txt``), also published on
Hugging Face (``musmb/CPAZMaL``, Apache-2.0). Reference: Gallet M. et al.,
*CPAZMaL: A New Dataset of X-Band SAR Time Series in the French Alps for Machine
Learning*, IEEE JSTARS 19, 2026, doi:10.1109/JSTARS.2026.3713622.

Download: when ``CPAZMaL.hdf5`` is absent from ``data_dir``, the file of the
pinned Hugging Face revision (1.5 GB, sha256 = LFS identifier, equal to the
local copy) is downloaded once into ``<cache_dir>/cpazmal/downloads/<revision>/``. This
folder is outside the ``keep`` policy (never deleted by the fetch, remove it by
hand): the processed stage depends on ``orbit``, ``n_dates``, ``block``..., and
deleting the source after each run would download 1.5 GB again for every new
combination.

Download
--------
Direct, public Hugging Face repository (no token): ``dataset/CPAZMaL.hdf5`` of
https://huggingface.co/datasets/musmb/CPAZMaL at revision ``642e8c0f…`` (1 509 579 244
bytes, sha256 ``a3e58d77…`` = LFS identifier), licence Apache-2.0 (dataset card).
A copy already downloaded is used in place, after its sha256 is checked::

    fetch_cpazmal(archive="~/Downloads/CPAZMaL.hdf5")

Sample unit
-----------
The annotated *parcels* are the 68 polygons of the source shapefiles (one per
site, e.g. ``ABL001``); the HDF5 file stores them rasterised: pixels outside the
polygon have mask ``-127`` (``data/<SITE>/<ORBIT>/<POL>/masks``). Only pixels
inside the polygon are used, never a rectangular window around it. Each polygon
is a single connected component (checked on the 70 groups).

- ``block=None`` (default): one sample per parcel (its valid pixels).
- ``block=b``: the parcel is cut by a ``b x b`` pixel grid; every piece with at
  least ``min_pixels`` valid pixels is a sample (group = its parcel).

Covariance
----------
``p = 2 * n_dates`` channels: the amplitude ``sqrt(I)`` (``scale``) of HH then
HV on ``n_dates`` acquisition dates common to HH and HV (HV only exists until
2020-10-11 in ascending, 2021-01-27 in descending: 21 and 30 common dates). The
dates are spread evenly over the common dates, identical for every parcel, and
stored in ``dataset.dates``. With the default 20 dates, ``p = 40``.
Observations are the pixels valid on all ``p`` channels: inside the polygon
(mask ``!= -127``), mask code ``<= max_mask_value`` (0 none, 1 layover,
2 shadow, 3 both; Hugging Face README) and ``I > 0`` (-999 is nodata; about 1 %
of border pixels hold interpolated negative values). A sample needs
``min_pixels >= 2p`` such pixels (default 80; 120 recommended).

Outputs: covariance ``(p, p)``; mean = per-channel mean ``(p,)``; raw = the
valid pixels ``(n_i, p)`` of the sample (parcels are irregular, so ``n_i``
varies; there is no ``(N, M, p)`` grid). With ``normalize=True`` the
covariances are multiplied by the geometric scale ``s``, the mean and raw by
``sqrt(s)``.

Splits: no official split. ``"sgkf5_fold{k}"`` are ``StratifiedGroupKFold(5)``
folds grouped by parcel (site), with a grouped validation part;
``"recommended"`` is fold 0. Excluded: the two unlabelled study areas
(``STUDY``: ALLGLA, ARGTOP) and parcels lacking HH or HV for the orbit
(PLA006 in both orbits, PLA005 in descending), listed in ``dataset.excluded``.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import RANDOM, BaseDataset
from ..core.config import dataset_path, get_cache_dir
from ..core.estimators import (
    InsufficientSamplesError,
    NotSPDError,
    channel_mean,
    estimate_covariance,
)
from ..core.fetch import Bunch, check_local, download, run_pipeline, sha256sum
from ..core.normalize import apply_scale
from ..manager import DatasetManager
from ._common import (
    check_split_groups,
    finalize_labels,
    group_kfold_splits,
    train_scale,
)

DOMAIN, NAME = "SAR", "PAZ_S1_proc_v2"
FILENAME = "CPAZMaL.hdf5"
HF_REVISION = "642e8c0f82ec94016e56b33241677f5b8d4df228"
HF_URL = f"https://huggingface.co/datasets/musmb/CPAZMaL/resolve/{HF_REVISION}/dataset/CPAZMaL.hdf5"
# sha256 of the Hugging Face file (x-linked-etag), identical to the local copy
FILES = {HF_URL: "a3e58d77bf5473f8c409e06ba11c4a956cf2a0dcd7e2108e845d0ede84a9dc94"}
CLASSES = ("ABL", "ACC", "FOR", "HAG", "ICA", "LAC", "PLA", "ROC")
SCALES = ("amplitude", "intensity", "log10")
NODATA_MASK = -127
META_COLUMNS = ["site", "class", "n_pixels", "r0", "c0", "r1", "c1"]


def select_dates(common, n_dates):
    """``n_dates`` dates spread evenly over the sorted ``common`` dates."""
    common = sorted(common)
    if len(common) < n_dates:
        raise ValueError(f"Only {len(common)} dates common to HH and HV; n_dates={n_dates}")
    idx = np.unique(np.round(np.linspace(0, len(common) - 1, n_dates)).astype(int))
    return [common[i] for i in idx]


def _site_stack(group, orbit, dates, scale, max_mask_value):
    """``(H, W, 2T)`` features (HH dates then HV dates) and the valid-pixel mask."""
    feats, valid = [], None
    a, b = group[orbit]["HH"]["images"], group[orbit]["HV"]["images"]
    H, W = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    for pol in ("HH", "HV"):
        g = group[orbit][pol]
        ts = g["timestamps"][:].astype(str).tolist()
        idx = [ts.index(d) for d in dates]
        img = g["images"][:H, :W][..., idx].astype(np.float64)
        mask = g["masks"][:H, :W][..., idx]
        ok = (img > 0) & (mask != NODATA_MASK) & (mask <= max_mask_value)
        valid = ok.all(-1) if valid is None else valid & ok.all(-1)
        safe = np.where(img > 0, img, 1.0)
        feats.append({"amplitude": np.sqrt(safe), "intensity": safe,
                      "log10": np.log10(safe)}[scale])
    return np.concatenate(feats, -1), valid


def _process(path, out_dir, *, orbit, n_dates, scale, max_mask_value, min_pixels, block):
    import h5py

    with h5py.File(path, "r") as h:
        sites = sorted(h["data"])
        usable, excluded = [], {}
        for s in sites:
            g = h["data"][s]
            cls = str(g.attrs["class"])
            if cls not in CLASSES:
                excluded[s] = f"class {cls}"
            elif orbit not in g or not {"HH", "HV"} <= set(g[orbit]):
                excluded[s] = f"no HH+HV in {orbit}"
            else:
                usable.append(s)
        common = None
        for s in usable:
            g = h["data"][s][orbit]
            c = set(g["HH"]["timestamps"][:].astype(str)) & set(g["HV"]["timestamps"][:].astype(str))
            common = c if common is None else common & c
        dates = select_dates(common or [], n_dates)

        pixels, rows = [], []
        for s in usable:
            g = h["data"][s]
            X, valid = _site_stack(g, orbit, dates, scale, max_mask_value)
            H, W = valid.shape
            pieces = [(0, 0, H, W)] if block is None else [
                (r, c, min(r + block, H), min(c + block, W))
                for r in range(0, H, block) for c in range(0, W, block)]
            kept = 0
            for r0, c0, r1, c1 in pieces:
                v = valid[r0:r1, c0:c1]
                n = int(v.sum())
                if n < min_pixels:
                    continue
                pixels.append(X[r0:r1, c0:c1][v].astype(np.float32))
                rows.append({"site": s, "class": str(g.attrs["class"]), "n_pixels": n,
                             "r0": r0, "c0": c0, "r1": r1, "c1": c1})
                kept += 1
            if not kept:
                excluded[s] = f"fewer than {min_pixels} valid pixels"
    offsets = np.cumsum([0] + [len(x) for x in pixels])
    np.save(Path(out_dir) / "pixels.npy",
            np.concatenate(pixels) if pixels else np.zeros((0, 2 * len(dates)), np.float32))
    np.save(Path(out_dir) / "offsets.npy", offsets)
    pd.DataFrame(rows, columns=META_COLUMNS).to_csv(Path(out_dir) / "meta.csv", index=False)
    (Path(out_dir) / "info.json").write_text(json.dumps({"dates": dates, "excluded": excluded}))


def _covariances(proc_dir, out_dir, *, estimator):
    proc_dir, out_dir = Path(proc_dir), Path(out_dir)
    pixels = np.load(proc_dir / "pixels.npy", mmap_mode="r")
    off = np.load(proc_dir / "offsets.npy")
    p = pixels.shape[1]
    meta = pd.read_csv(proc_dir / "meta.csv", dtype={"site": str, "class": str})
    meta["proc_index"] = np.arange(len(meta))
    info = json.loads((proc_dir / "info.json").read_text())
    covs, means, kept, info["rejected"] = [], [], [], []
    for k, (a, b) in enumerate(zip(off[:-1], off[1:], strict=True)):
        x = np.asarray(pixels[a:b])
        try:
            covs.append(estimate_covariance(x, estimator))
        except (InsufficientSamplesError, NotSPDError) as exc:  # rejected and counted
            info["rejected"].append({"proc_index": k, "site": meta["site"].iat[k],
                                     "reason": f"{type(exc).__name__}: {exc}"})
            continue
        means.append(channel_mean(x))
        kept.append(k)
    np.save(out_dir / "covariances.npy", np.stack(covs) if covs else np.zeros((0, p, p)))
    np.save(out_dir / "means.npy", np.stack(means) if means else np.zeros((0, p)))
    meta.iloc[kept].to_csv(out_dir / "meta.csv", index=False)
    (out_dir / "info.json").write_text(json.dumps(info))


def fetch_cpazmal(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                  verify="size", *, orbit="DSC", n_dates=20, scale="amplitude",
                  max_mask_value=3, min_pixels=80, block=None, archive=None):
    """Load (and cache) the CPAZMaL parcel covariances.

    Parameters
    ----------
    data_dir : path, optional
        Folder holding ``CPAZMaL.hdf5`` (default ``<root>/SAR/PAZ_S1_proc_v2``).
        If the file is absent, it is downloaded from Hugging Face into the cache.
    archive : path, optional
        ``CPAZMaL.hdf5`` already downloaded (or its folder), used in place
        when ``data_dir`` has no copy; its sha256 is checked.
    cache_dir, keep, force
        See :func:`spdnet_datasets.core.fetch.run_pipeline`.
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
    verify : {"size", "md5", None}
        Check of the local copy against ``MANIFEST_NAS.tsv``.
    orbit : {"DSC", "ASC"}
    n_dates : int
        Number of dual-polarisation dates; ``p = 2 * n_dates``.
    scale : {"amplitude", "intensity", "log10"}
    max_mask_value : int
        Highest accepted distortion code (0 none, 1 layover, 2 shadow, 3 both).
    min_pixels : int
        Minimum number of valid pixels of a sample; must be ``>= 2p``.
    block : int, optional
        Cut parcels into ``block x block`` pieces (``None``: whole parcels).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p, unscaled), ``means`` (B, p), ``labels``
    (class names), ``groups`` (parcel), ``meta`` (DataFrame), ``dates``,
    ``excluded`` (parcel -> reason), ``rejected`` (samples whose estimate
    failed the ``n >= 2p`` or SPD check: list of ``proc_index``, ``site``,
    ``reason``), ``splits`` and ``processed_dir``.
    """
    if orbit not in ("ASC", "DSC"):
        raise ValueError("orbit must be 'ASC' or 'DSC'")
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    p = 2 * n_dates
    if min_pixels < 2 * p:
        raise ValueError(f"min_pixels={min_pixels} < 2p = {2 * p}: covariances would be ill-conditioned")
    root = dataset_path(DOMAIN, NAME, data_dir)
    local = root / FILENAME
    if local.exists():
        check_local(root, verify=verify, required=(FILENAME,))
        raw_fn = None
    elif archive is not None:
        local = Path(archive).expanduser()
        local = local / FILENAME if local.is_dir() else local
        if sha256sum(local) != FILES[HF_URL]:
            raise OSError(f"Checksum mismatch for {local} (expected the file of revision {HF_REVISION})")
        raw_fn = None
    else:
        def raw_fn(raw_dir):
            # persistent sibling of raw/, outside the keep policy (module docstring)
            dest = Path(raw_dir).parent / "downloads" / HF_REVISION / FILENAME
            return dest if dest.exists() else download(HF_URL, dest, checksum=FILES[HF_URL])
    opts = {"orbit": orbit, "n_dates": n_dates, "scale": scale, "max_mask_value": max_mask_value,
                "min_pixels": min_pixels, "block": block}
    params = dict(opts, source=str(local.resolve()) if local.exists() else HF_URL)

    def process_fn(source, out_dir):
        _process(source if source is not None else local, out_dir, **opts)

    res = run_pipeline("cpazmal", params=params,
                       covariance_params={"estimator": estimator, "version": 2},
                       raw_fn=raw_fn, process_fn=process_fn,
                       covariance_fn=lambda a, b: _covariances(a, b, estimator=estimator),
                       cache_dir=cache_dir, keep=keep, force=force)
    cdir = Path(res.covariance_dir)
    meta = pd.read_csv(cdir / "meta.csv", dtype={"site": str, "class": str})
    info = json.loads((cdir / "info.json").read_text())
    labels = meta["class"].to_numpy(str)
    groups = meta["site"].to_numpy(str)
    splits = group_kfold_splits(labels, groups) if len(np.unique(groups)) >= 5 else {}
    if splits:
        splits["recommended"] = splits["sgkf5_fold0"]
    return Bunch(covariances=np.load(cdir / "covariances.npy"), means=np.load(cdir / "means.npy"),
                 labels=labels, groups=groups, meta=meta, dates=info["dates"],
                 excluded=info["excluded"], rejected=info["rejected"], splits=splits, processed_dir=res.processed_dir,
                 covariance_dir=cdir)


@DatasetManager.register_dataset("cpazmal")
class CPAZMaLDataset(BaseDataset):
    """CPAZMaL parcels as ``2 n_dates x 2 n_dates`` amplitude covariances.

    See the module docstring for the sample unit, channels and splits.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/PAZ_S1_proc_v2``.
    split : str
        Split whose training part fits the normalisation (``"recommended"``,
        ``"sgkf5_fold{k}"`` or ``"random"``, sample-level and leaky: a site
        holds several parcels); all splits are in ``dataset.splits``.
    cache_dir, keep, orbit, n_dates, scale, max_mask_value, min_pixels, block
        See :func:`fetch_cpazmal`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments (``return_mean``,
        ``return_raw``, ``estimator``, ``task``, ``drop_classes``,
        ``merge_classes``, ``balance``, ``normalize``, ``seed``...).
    """

    OUTPUTS = ("cov", "mean", "raw")

    def __init__(self, data_dir=None, *, split="recommended", cache_dir=None, keep=None,
                 orbit="DSC", n_dates=20, scale="amplitude", max_mask_value=3,
                 min_pixels=80, block=None, archive=None, **kwargs):
        self._root = dataset_path(DOMAIN, NAME, data_dir)
        # the base class needs an existing folder: the cache when downloading
        base_dir = self._root if self._root.exists() else Path(cache_dir or get_cache_dir())
        base_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self._fetch_kwargs = {"cache_dir": cache_dir, "keep": keep, "orbit": orbit, "n_dates": n_dates,
                                  "scale": scale, "max_mask_value": max_mask_value,
                                  "min_pixels": min_pixels, "block": block, "archive": archive}
        super().__init__(data_dir=str(base_dir), **kwargs)
        self._load_data()

    def _load_data(self):
        b = fetch_cpazmal(self._root, estimator=self.estimator, **self._fetch_kwargs)
        if self.return_raw and b.processed_dir is None:
            raise ValueError("return_raw=True needs the preprocessed pixels (keep != 'keep_covariance')")
        keep = finalize_labels(self, b.labels, b.groups, b.splits)
        check_split_groups(self.splits, self.groups)
        self.meta = b.meta.iloc[keep].reset_index(drop=True)
        self.dates, self.excluded, self.rejected = b.dates, b.excluded, b.rejected
        self.covariances = b.covariances[keep]
        self.means = b.means[keep]
        self.scale = 1.0
        if self.normalize and (set(self.splits) - {RANDOM} or self.split == RANDOM):
            self.scale = train_scale(self.covariances, self.splits, self.split)
        elif self.normalize:
            warnings.warn("No split available: covariances are not normalised", UserWarning, stacklevel=2)
        self._processed = b.processed_dir
        self.samples = list(range(len(keep)))
        if self.verbose:
            print(f"CPAZMaL ({self._fetch_kwargs['orbit']}, p={self.covariances.shape[-1]}): "
                  f"{len(keep)} samples, {len(np.unique(self.groups))} parcels, "
                  f"excluded {sorted(self.excluded)}")

    def _raw(self, i):
        if not hasattr(self, "_pixels"):
            self._pixels = np.load(Path(self._processed) / "pixels.npy", mmap_mode="r")
            self._offsets = np.load(Path(self._processed) / "offsets.npy")
        j = int(self.meta["proc_index"].iat[i])
        return np.array(self._pixels[self._offsets[j]:self._offsets[j + 1]], dtype=np.float32)

    def __getitem__(self, idx):
        cov = torch.from_numpy(apply_scale(self.covariances[idx], self.scale))
        root = np.sqrt(self.scale)
        mean = torch.from_numpy(self.means[idx] * root)
        raw = torch.from_numpy((self._raw(idx) * root).astype(np.float32)) if self.return_raw else None
        x = self._pack(cov, mean, raw)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])
