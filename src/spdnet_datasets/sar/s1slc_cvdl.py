"""S1SLC_CVDL: Sentinel-1 SLC HH/HV patches of three cities (7 land-cover classes).

Reference: Asiyabi R. M. et al., S1SLC_CVDL, a Sentinel-1 SLC dataset for
complex-valued deep learning (IEEE DataPort, 2023; to be confirmed, no paper in
the folder). Manual download. Local copy: ``SAR/S1SLC_CVDL/complex_HHHV.zarr``
(zarr v3, needs ``pip install 'spdnet-datasets[sar]'``)::

    <City>/data    (N, 100, 100, 4) float32 = [Re HH, Im HH, Re HV, Im HV]
    <City>/labels  (N,) int32 in 1..7          cities: Houston, Chicago, Sao Paulo

The class names behind ids 1..7 are not documented in the folder (literature:
agricultural fields, forest, dense urban, high-rise, low-density urban,
industrial, water; order not verified): classes are named ``"1"``..``"7"``.

Outputs: raw or mean only (``OUTPUTS = ("raw", "mean")``).

- raw (``features="complex"``): ``(100, 100, 4)`` real and imaginary parts;
  (``"amplitude"``): ``(100, 100, 2)`` = ``|HH|, |HV|``;
- mean: mean amplitude ``(|HH|, |HV|)`` of the patch (the mean of the real and
  imaginary parts is zero).

Values are uncalibrated (int16 full scale): with ``normalize=True`` each city
is divided by its RMS amplitude, estimated on (at most ``n_stat_patches``)
patches of that city in the *training* part of ``split``; the held-out city
uses the RMS of the pooled training cities (no test statistics). Statistics are
cached in ``<cache_dir>/s1slc_cvdl/``.

Splits: ``"loco_<City>"`` (leave one city out; validation = 10 % of the
training cities, stratified), groups = city. ``"recommended"`` = ``"loco_Houston"``.
Patch positions are not provided: no spatial split within a city is possible
(a random split would put neighbouring patches on both sides).

Download
--------
Manual (IEEE DataPort, open access but a free IEEE account is required; the
download links are temporary signed URLs):
https://ieee-dataport.org/open-access/s1slccvdl-complex-valued-annotated-single-look-complex-sentinel-1-sar-dataset-complex
(doi:10.21227/nm4g-yd98, CC BY 4.0). The file is ``S1SLC_CVDL.rar`` (12.24 GB, no
checksum published; the page calls it ``S1SCL_CVDL.zip``). It holds one folder per
city with ``HH_Complex_Patches.npy``, ``HV_Complex_Patches.npy`` and ``Labels.npy``;
the zarr store read here is a conversion of them (:func:`npy_to_zarr`, port of the
local ``extract_real_imag_dataset.py``; checked on synthetic data only, the local
copy has no ``.npy``). Without a local copy, ``fetch_s1slc_cvdl()`` opens the page
and raises :class:`~spdnet_datasets.core.access.ManualDownloadRequired`; then::

    fetch_s1slc_cvdl(archive="~/Downloads/S1SLC_CVDL.rar")   # or the extracted folder

A ``.rar`` needs the ``unrar``, ``7z`` or ``bsdtar`` command (Python has no RAR
decoder); without one, extract it by hand and pass the folder. The store is
written into ``<cache_dir>/S1SLC_CVDL/downloads/complex_HHHV.zarr`` (about 22 GB).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.access import extract_archive, manual_source
from ..core.config import dataset_path, get_cache_dir
from ..core.fetch import Bunch, check_local
from ..manager import DatasetManager
from ..utils.sources import downloads_dir, existing_dir, extract_rar, find_root
from ._common import finalize_labels

DOMAIN, NAME = "SAR", "S1SLC_CVDL"
PAGE = ("https://ieee-dataport.org/open-access/s1slccvdl-complex-valued-annotated-single-"
        "look-complex-sentinel-1-sar-dataset-complex")
FILES = {}  # manual (IEEE account, temporary signed URLs), no checksum published
STORE = "complex_HHHV.zarr"
CITIES = ("Houston", "Chicago", "Sao Paulo")
NPY = ("HH_Complex_Patches.npy", "HV_Complex_Patches.npy", "Labels.npy")


def _open(root):
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("S1SLC_CVDL needs zarr: pip install 'spdnet-datasets[sar]'") from exc
    return zarr.open_group(str(Path(root) / STORE), mode="r")


def npy_to_zarr(src, store, chunk=100):
    """Write the zarr store read by :func:`fetch_s1slc_cvdl` from the DataPort ``.npy`` files.

    ``src`` holds one folder per city with :data:`NPY`; each city becomes a
    group with ``data`` (N, 100, 100, 4) float32 = [Re HH, Im HH, Re HV, Im HV]
    and ``labels`` (N,) int32, chunks of ``chunk`` patches (as the local
    conversion ``extract_real_imag_dataset.py``).
    """
    import shutil

    import zarr

    src, store = Path(src), Path(store)
    tmp = store.with_name(store.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    root = zarr.open_group(str(tmp), mode="w")
    for city in sorted(p.name for p in src.iterdir() if p.is_dir()):
        if not all((src / city / f).exists() for f in NPY):
            continue
        hh, hv = (np.load(src / city / f, mmap_mode="r") for f in NPY[:2])
        labels = np.atleast_1d(np.load(src / city / NPY[2])).ravel().astype(np.int32)
        if hh.shape != hv.shape or len(hh) != len(labels):
            raise ValueError(f"{city}: HH {hh.shape}, HV {hv.shape}, {len(labels)} labels")
        g = root.create_group(city)
        data = g.create_array("data", shape=(len(hh), *hh.shape[1:], 4), dtype=np.float32,
                              chunks=(chunk, *hh.shape[1:], 4), fill_value=0)
        g.create_array("labels", shape=labels.shape, dtype=np.int32, chunks=(chunk,),
                       fill_value=0)[:] = labels
        for i in range(0, len(hh), chunk):
            a, b = hh[i:i + chunk], hv[i:i + chunk]
            data[i:i + chunk] = np.stack([a.real, a.imag, b.real, b.imag], -1).astype(np.float32)
    tmp.rename(store)
    return store


def _download_root(cache_dir, archive):
    """Folder holding the zarr store converted in the cache from the DataPort archive."""
    out = downloads_dir(NAME, cache_dir)
    if (out / STORE).exists():
        return out
    if archive is None:
        manual_source(NAME, page_url=PAGE, dest_dir=out,
                      expected="S1SLC_CVDL.rar (12.24 GB; free IEEE account)")
    path = Path(archive).expanduser()
    if path.is_dir() and (path / STORE).exists():
        return path
    if path.is_file():
        stem = path.name.rsplit(".", 1)[0]
        path = (extract_rar(path, out / stem) if path.suffix.lower() == ".rar"
                else extract_archive(path, out / stem))
    elif not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    src = find_root(path, (f"{CITIES[0]}/{NPY[0]}",))
    if src is None:
        raise FileNotFoundError(f"{archive} holds no <City>/{NPY[0]} folder")
    npy_to_zarr(src, out / STORE)
    return out


def fetch_s1slc_cvdl(data_dir=None, *, verify="size", val_size=0.1, seed=0, cache_dir=None,
                     archive=None):
    """Index the S1SLC_CVDL patches.

    Parameters
    ----------
    cache_dir : path, optional
        Where an archive is extracted and converted.
    archive : path, optional
        ``S1SLC_CVDL.rar`` (or another archive, or a folder) holding the city
        ``.npy`` files, or a folder holding ``complex_HHHV.zarr`` (see
        "Download" in the module docstring).

    Returns
    -------
    Bunch with ``meta`` (``city``, ``index`` in the city array, ``label``),
    ``splits`` (``"loco_<City>"``, ``"recommended"``) and ``root``.
    """
    from sklearn.model_selection import train_test_split

    root = dataset_path(DOMAIN, NAME, data_dir)
    if not (root / STORE).exists():
        root = _download_root(cache_dir, archive)
    check_local(root, verify=verify, required=(STORE,))
    z = _open(root)
    frames = []
    for city in CITIES:
        if city in z:
            lab = np.asarray(z[city]["labels"][:])
            frames.append(pd.DataFrame({"city": city, "index": np.arange(len(lab)),
                                        "label": lab.astype(str)}))
    meta = pd.concat(frames, ignore_index=True)
    city, y = meta["city"].to_numpy(), meta["label"].to_numpy()
    splits = {}
    for c in np.unique(city):
        train, test = np.flatnonzero(city != c), np.flatnonzero(city == c)
        try:
            tr, va = train_test_split(train, test_size=val_size, stratify=y[train], random_state=seed)
        except ValueError:
            tr, va = train_test_split(train, test_size=val_size, random_state=seed)
        splits[f"loco_{c.replace(' ', '_')}"] = {"train": np.sort(tr), "val": np.sort(va), "test": test}
    splits["recommended"] = splits.get("loco_Houston", next(iter(splits.values())))
    return Bunch(meta=meta, splits=splits, root=root)


def city_rms(root, meta, train, *, n_stat_patches=2000, seed=0, cache_dir=None):
    """RMS amplitude per city on (a chunk-aligned sample of) its training patches."""
    key = hashlib.sha1(json.dumps([str(Path(root).resolve()), n_stat_patches, seed,
                                   hashlib.sha1(np.sort(train).tobytes()).hexdigest()]).encode()).hexdigest()[:12]
    cache = Path(cache_dir or get_cache_dir()) / "s1slc_cvdl" / f"rms_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    z = _open(root)
    rng = np.random.default_rng(seed)
    sums = {}
    for c in meta["city"].iloc[train].unique():
        idx = meta["index"].to_numpy()[train][meta["city"].to_numpy()[train] == c]
        arr = z[c]["data"]
        chunk = arr.chunks[0]
        chunks = np.unique(idx // chunk)
        rng.shuffle(chunks)
        s, n, used = 0.0, 0, 0
        for k in chunks:
            block = np.asarray(arr[k * chunk:(k + 1) * chunk], dtype=np.float64)
            sel = np.isin(np.arange(k * chunk, k * chunk + len(block)), idx)
            s += float(np.sum(block[sel] ** 2))
            n += int(sel.sum()) * block.shape[1] * block.shape[2] * 2  # complex values
            used += int(sel.sum())
            if used >= n_stat_patches:
                break
        sums[c] = (s, n)
    rms = {c: float(np.sqrt(s / n)) for c, (s, n) in sums.items()}
    pooled = float(np.sqrt(sum(s for s, _ in sums.values()) / sum(n for _, n in sums.values())))
    for c in meta["city"].unique():
        rms.setdefault(c, pooled)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rms))
    return rms


@DatasetManager.register_dataset("s1slc_cvdl")
class S1SLCCVDLDataset(BaseDataset):
    """S1SLC_CVDL patches (raw or mean only), leave-one-city-out.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/S1SLC_CVDL``.
    split : str
        Split whose training part fits the per-city normalisation.
    features : {"complex", "amplitude"}
    n_stat_patches : int
        Patches per city used for the RMS estimate.
    cache_dir, verify, val_size
        See :func:`fetch_s1slc_cvdl`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments; exactly one of
        ``return_raw`` / ``return_mean``.
    """

    OUTPUTS = ("raw", "mean")

    def __init__(self, data_dir=None, *, split="recommended", features="complex",
                 n_stat_patches=2000, cache_dir=None, verify="size", val_size=0.1, **kwargs):
        if features not in ("complex", "amplitude"):
            raise ValueError("features must be 'complex' or 'amplitude'")
        root = dataset_path(DOMAIN, NAME, data_dir)
        self.split, self.features = split, features
        self._opts = {"n_stat_patches": n_stat_patches, "cache_dir": cache_dir}
        self._fetch = {"verify": verify, "val_size": val_size, "cache_dir": cache_dir}
        self._chunk = {}
        self._root = root
        super().__init__(data_dir=str(existing_dir(root, cache_dir)), **kwargs)
        self._load_data()

    def _load_data(self):
        b = fetch_s1slc_cvdl(self._root, seed=self.seed, **self._fetch)
        self.data_dir = Path(b.root)
        keep = finalize_labels(self, b.meta["label"].to_numpy(), b.meta["city"].to_numpy(), b.splits)
        self.meta = b.meta.iloc[keep].reset_index(drop=True)
        if self.split not in self.splits:
            raise ValueError(f"Unknown split {self.split!r}; available: {sorted(self.splits)}")
        self.rms = dict.fromkeys(self.meta["city"].unique(), 1.0)
        if self.normalize:
            self.rms = city_rms(self.data_dir, self.meta, self.splits[self.split]["train"],
                                seed=self.seed, **self._opts)
        self.samples = list(range(len(self.meta)))
        self._z = None
        if self.verbose:
            print(f"S1SLC_CVDL: {len(self.meta)} patches, cities {dict(self.meta['city'].value_counts())}")

    def _patch(self, idx):
        if self._z is None:
            self._z = _open(self.data_dir)
        city, i = self.meta.at[idx, "city"], int(self.meta.at[idx, "index"])
        arr = self._z[city]["data"]
        size = arr.chunks[0]
        k = i // size
        if self._chunk.get("key") != (city, k):  # one chunk = 100 patches, decompressed once
            self._chunk = {"key": (city, k), "data": np.asarray(arr[k * size:(k + 1) * size])}
        return self._chunk["data"][i - k * size].astype(np.float32) / np.float32(self.rms[city])

    def __getitem__(self, idx):
        x = self._patch(idx)
        amp = np.stack([np.hypot(x[..., 0], x[..., 1]), np.hypot(x[..., 2], x[..., 3])], -1)
        if self.return_mean:
            out = self._pack(mean=torch.from_numpy(amp.reshape(-1, 2).mean(0, dtype=np.float64)))
        else:
            out = self._pack(raw=torch.from_numpy(np.ascontiguousarray(x if self.features == "complex" else amp)))
        if self.transform is not None:
            out = self.transform(out)
        return out, int(self.targets[idx])
