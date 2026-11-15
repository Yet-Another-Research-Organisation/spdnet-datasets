"""Hyperspectral placenta dataset (Puustinen et al., Data in Brief, 2023).

101 images 1024 x 1024 of four human placentas (surgical microscope, Senop
HSC-2), flat-field corrected reflectance, manual annotations. The local
derived version ``imagettes/`` holds 77 203 non-overlapping 24 x 24 windows
x 37 bands (515-697 nm), purity >= 0.9, from the 81 images P007-P087
(``placenta_windows_metadata.json``: ``sample_id`` = source image,
``dye_type`` red_blue / ICG, ``class``, ``original_class``, ``position``).
Five classes: Artery, Stroma, Suture, Umbilical, Vein.

Covariance: the 576 pixels of the window, one band in ``band_step``
(default 1: p = 37, n / 2p = 7.8, so no band subsampling is needed; QUESTIONS
A2). Windows with fewer than 2p valid pixels would be discarded and counted.

Splits:

- ``"by_image"`` = ``"recommended"``: StratifiedGroupKFold(5) by
  ``sample_id`` (test = fold ``fold``, val = fold ``fold + 1``). The 81 images
  come from only four placentas and no placenta identifier exists: a
  placenta is almost surely on both sides (leakage not measurable);
- ``"dye_transfer"``: train red_blue images, test ICG images (the ICG images
  were taken after the red/blue injection on the same placentas).

Download
--------
Direct (Zenodo record 8045940, doi:10.5281/zenodo.8045940, CC BY-NC-SA 4.0, no
account). Without a local ``imagettes/``, ``fetch_placenta`` downloads the four
zips of the annotated images P007-P101 (7.86 GB, md5 of the Zenodo record checked)
into ``<cache_dir>/Hyperspectral_Placenta_Dataset/downloads/``, extracts each into a
folder named after it and rebuilds the windows with
:func:`spdnet_datasets.utils.placenta_preprocess.build_windows` (same rule as the
LISTIC extraction; checked on BERTHA: same 77 203 windows of 81 images, and the 24 519
windows of P007-P030 are bit-identical to the local
``imagettes/``). The other files of the record (P001-P006 without dye, dye samples,
references, raw measurements: 16.7 GB) are not used. Already downloaded::

    fetch_placenta(archive="~/Downloads/placenta")   # folder with the four zips
                                                     # (or the four extracted folders)
"""

import json
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.access import extract_archive
from ..core.config import dataset_path
from ..core.estimators import check_sample_size
from ..core.fetch import Bunch, check_local, download, md5sum, run_pipeline
from ..manager import DatasetManager
from ..utils.sources import check_size, downloads_dir
from ._common import (
    HSIDataset,
    band_index,
    covariance_stage,
    group_kfold_split,
    load_covariance_dir,
    map_splits,
    valid_pixels,
    write_json,
)

DOMAIN, NAME = "HYPERSPECTRAL", "Hyperspectral_Placenta_Dataset"
ZENODO = "https://zenodo.org/records/8045940/files/"
ZIPS = {  # name: (md5 of the Zenodo record, size in bytes)
    "Placenta P007 - P030 red blue.zip": ("424092bd80e52b117b23fdb67daa5a47", 1_793_559_251),
    "Placenta P031 - P053 red blue.zip": ("44964f1b1cd8c928075a7b1d768f3b8d", 1_708_014_267),
    "Placenta P054 - P077 ICG.zip": ("de8e2a710ca67a1f86ec2be9888c1a5b", 1_791_566_682),
    "Placenta P078 - P101 ICG.zip": ("c32c51efd4726fabb7f6fbcf8639c4bd", 2_566_835_106),
}
FILES = {ZENODO + urllib.parse.quote(name) + "?download=1": md5 for name, (md5, _) in ZIPS.items()}
WINDOWS = "imagettes/placenta_windows_data.npy"
METADATA = "imagettes/placenta_windows_metadata.json"
N_BANDS = 37


def _download_root(cache_dir, archive):
    """Folder with ``imagettes/`` rebuilt in the cache from the Zenodo zips."""
    from ..utils.placenta_preprocess import build_windows

    out = downloads_dir(NAME, cache_dir)
    if (out / WINDOWS).exists() and (out / METADATA).exists():
        return out
    given = Path(archive).expanduser() if archive is not None else None
    if given is not None and not given.is_dir():
        raise NotADirectoryError(f"archive must be a folder holding {list(ZIPS)} "
                                 "or their extracted folders")
    if given is not None and all((given / z[:-4]).is_dir() for z in ZIPS):
        src = given  # extracted folders, read in place
    else:
        src = out
        for name, (md5, size) in ZIPS.items():
            if (out / name[:-4] / ".extracted").exists():
                continue
            if given is not None:
                z = check_size(given / name, size)
                if md5sum(z) != md5:
                    raise OSError(f"Checksum mismatch for {z}")
            else:
                z = download(ZENODO + urllib.parse.quote(name) + "?download=1", out / name,
                             checksum=md5, algo="md5")
            extract_archive(z, out / name[:-4])
    build_windows(src, out / "imagettes")
    return out


def fetch_placenta(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                   verify="size", *, band_step=1, n_splits=5, fold=0, seed=0, archive=None):
    """Window covariances of the placenta dataset.

    Parameters
    ----------
    archive : path, optional
        Folder holding the four Zenodo zips (or their extracted folders), used
        instead of downloading them (see "Download" in the module docstring).

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) unscaled, ``means``, ``labels``,
    ``groups`` (``sample_id``), ``meta``, ``info``, ``splits``
    (``"by_image"`` = ``"recommended"``, ``"dye_transfer"``) and ``root``
    (dataset folder, to pass as ``data_dir`` to :class:`PlacentaDataset`).
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    if not ((root / WINDOWS).exists() and (root / METADATA).exists()):
        root = _download_root(cache_dir, archive)
    check_local(root, verify=verify, required=(WINDOWS, METADATA))
    windows = np.load(root / WINDOWS, mmap_mode="r")
    if windows.shape[-1] != N_BANDS:
        raise ValueError(f"Expected {N_BANDS} bands, got {windows.shape}")
    bands = band_index(N_BANDS, band_step)
    check_sample_size(windows.shape[1] * windows.shape[2], len(bands))
    params = {"band_step": band_step, "source": str(root.resolve()), "version": 2}

    def process(_, out_dir):
        records = json.loads((root / METADATA).read_text())
        if len(records) != len(windows):
            raise ValueError("metadata and windows have different lengths")
        meta = pd.DataFrame({
            "window_index": np.arange(len(records)),
            "sample_id": [r["sample_id"] for r in records],
            "dye_type": [r["dye_type"] for r in records],
            "label": [r["class"] for r in records],
            "original_class": [r["original_class"] for r in records],
            "row": [r["position"][0] for r in records],
            "col": [r["position"][1] for r in records],
        })
        meta.to_csv(Path(out_dir) / "meta.csv", index=False)
        write_json(Path(out_dir) / "info.json", {"n_windows": len(meta), "p": len(bands)})

    def pixels(proc_dir, meta):
        for i in meta["window_index"]:
            w = np.asarray(windows[int(i)][..., bands])
            yield valid_pixels(w.reshape(-1, len(bands)))

    res = run_pipeline(NAME, params=params, process_fn=process,
                       covariance_fn=lambda p, o: covariance_stage(p, o, pixels, estimator),
                       covariance_params={"estimator": estimator},
                       cache_dir=cache_dir, keep=keep, force=force)
    out = load_covariance_dir(res.covariance_dir)
    meta = out.meta
    labels = meta["label"].to_numpy()
    groups = meta["sample_id"].to_numpy()
    full = out.meta_all
    by_image = map_splits(
        {"s": group_kfold_split(full["label"].to_numpy(), full["sample_id"].to_numpy(),
                                n_splits=n_splits, fold=fold, seed=seed)},
        meta["proc_index"], len(full))["s"]
    dye = meta["dye_type"].to_numpy()
    return Bunch(
        covariances=out.covariances, means=out.means, n_obs=out.n_obs, labels=labels,
        groups=groups, meta=meta, info=out.info, bands=bands, windows_path=root / WINDOWS,
        splits={"by_image": by_image, "recommended": by_image,
                "dye_transfer": {"train": np.flatnonzero(dye == "red_blue"),
                                 "val": np.zeros(0, int),
                                 "test": np.flatnonzero(dye == "ICG")}},
        covariance_path=res.covariance_dir, root=root,
    )


@DatasetManager.register_dataset("placenta")
class PlacentaDataset(HSIDataset):
    """Placenta 24 x 24 windows, 37 x 37 covariances, split grouped by source image.

    Parameters
    ----------
    band_step, n_splits, fold
        See :func:`fetch_placenta`.
    split : str
        ``"recommended"`` (by image) or ``"dye_transfer"``.
    subset
        See :class:`~spdnet_datasets.hyperspectral._common.HSIDataset`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options.
    """

    def __init__(self, data_dir=None, *, band_step=1, n_splits=5, fold=0,
                 split="recommended", subset=None, cache_dir=None, keep=None, force=False,
                 verify="size", **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        super().__init__(data_dir=str(root), **kwargs)
        bunch = fetch_placenta(root, cache_dir, keep, self.estimator, force, verify,
                               band_step=band_step, n_splits=n_splits, fold=fold)
        self._setup(bunch, split, subset)
        self._windows = np.load(bunch.windows_path, mmap_mode="r")

    def _load_raw(self, k):
        # numpy window (the former loader passed a torch tensor to the numpy estimator)
        i = int(self.meta_all["window_index"].iloc[k])
        return self._windows[i][..., self.bunch.bands]


__all__ = ["PlacentaDataset", "fetch_placenta"]
