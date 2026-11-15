"""FUSAR-Ship 1.0: Gaofen-3 C-band ship chips matched with AIS.

Reference: Hou X. et al., *FUSAR-Ship: building a high-resolution SAR-AIS
matchup dataset of Gaofen-3 for ship detection and recognition*, Science China
Information Sciences 63, 140303 (2020). Manual download; the data must be under
``<root>/SAR/FUSAR_Ship1.0``::

    <Category>/<SubCategory>/Ship_CxxSyyNzzzz.tiff   # 8-bit amplitude quick-looks
    meta.csv                                          # id, mmsi, ..., path

Classes: categories (``class_level="basic"``) or ``Category_SubCategory``
(``"fine"``). The dataset is very imbalanced (2 114 Cargo chips, 3
DiveVessel): classes with fewer than ``min_samples_per_class`` chips (30 by
default, 10 categories and 5 198 chips are left) are dropped, then
``exclude_classes`` / ``drop_classes`` are removed.

Groups and splits: one vessel can appear in several chips. Chips are grouped by
MMSI with a union-find over ``meta.csv`` (a chip can be matched with several
MMSI; rows with an empty MMSI are ignored; a chip without MMSI is its own
group). ``"sgkf5_fold{k}"`` are ``StratifiedGroupKFold(5)`` folds on these
groups with a grouped validation part; ``"recommended"`` is fold 0. The
original GF-3 scenes are not provided, so a scene-context leak remains possible.

Outputs: raw or mean only (``OUTPUTS = ("raw", "mean")``).

- raw: ``(image_size, image_size, 1)`` float32, ``log1p`` of the amplitude
  (heavy tailed), resized with antialiasing (chips are 512 x 512, 256 x 256 or
  slightly non-square) and, with ``standardize=True``, standardised per chip;
- mean: per-channel mean of the ``log1p`` resized chip (before
  standardisation, which would make it zero).

Download (manual): distributed by Fudan University (see the article and
``ReadMe.txt``; copyright Fudan University, no direct URL or checksum registered). Copy the class folders and ``meta`` files and place it under
``data_dir`` (default ``<data root>/SAR/FUSAR_Ship1.0``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..base import BaseDataset
from ..core.config import dataset_path
from ..core.fetch import Bunch, check_local
from ..manager import DatasetManager
from ._common import check_split_groups, finalize_labels, group_kfold_splits

DOMAIN, NAME = "SAR", "FUSAR_Ship1.0"
FILES = {}  # manual download


def mmsi_groups(chip_ids, meta_csv):
    """Group id of every chip: connected components of the chip-MMSI graph."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if meta_csv is not None and Path(meta_csv).exists():
        m = pd.read_csv(meta_csv, dtype=str, usecols=["id", "mmsi"]).dropna()
        for cid, mmsi in zip(m["id"], m["mmsi"], strict=True):
            a, b = find(("chip", cid.strip())), find(("mmsi", mmsi.strip()))
            parent[a] = b
    roots = [find(("chip", c)) for c in chip_ids]
    _, groups = np.unique([f"{k}:{v}" for k, v in roots], return_inverse=True)
    return groups


def fetch_fusar_ship(data_dir=None, *, class_level="basic", min_samples_per_class=30,
                     exclude_classes=(), verify="size", seed=0):
    """Index the FUSAR-Ship chips.

    Returns
    -------
    Bunch with ``meta`` (``path``, ``label``, ``chip_id``, ``group``), the
    dropped classes and ``splits``.
    """
    if class_level not in ("basic", "fine"):
        raise ValueError(f"class_level must be 'basic' or 'fine', got {class_level}")
    root = dataset_path(DOMAIN, NAME, data_dir)
    check_local(root, verify=verify)
    rows = []
    for path in sorted(root.glob("*/*/*.tif*")):
        cat, sub = path.relative_to(root).parts[:2]
        rows.append({"path": str(path.relative_to(root)), "chip_id": path.stem.split(".")[0],
                     "label": cat if class_level == "basic" else f"{cat}_{sub}"})
    if not rows:
        raise FileNotFoundError(f"No FUSAR TIFF chips found under {root}")
    meta = pd.DataFrame(rows)
    meta["group"] = mmsi_groups(meta["chip_id"], root / "meta.csv")
    counts = meta["label"].value_counts()
    keep = [c for c in counts.index if counts[c] >= min_samples_per_class and c not in set(exclude_classes)]
    dropped = sorted(set(counts.index) - set(keep))
    meta = meta[meta["label"].isin(keep)].reset_index(drop=True)
    splits = group_kfold_splits(meta["label"].to_numpy(), meta["group"].to_numpy(), seed=seed)
    splits["recommended"] = splits["sgkf5_fold0"]
    return Bunch(meta=meta, dropped=dropped, splits=splits, root=root)


@DatasetManager.register_dataset("fusar_ship")
@DatasetManager.register_dataset("fusar")
class FUSARShipDataset(BaseDataset):
    """FUSAR-Ship 1.0 chips (raw or mean only), grouped by vessel (MMSI).

    Registered as ``"fusar"`` (historical name) and ``"fusar_ship"``.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/FUSAR_Ship1.0``.
    class_level : {"basic", "fine"}
    min_samples_per_class : int
    exclude_classes : list of str, optional
    image_size : int
        Side of the square output chip.
    standardize : bool
        Standardise each raw chip (mean 0, std 1).
    preload : bool
        Load every chip into memory at construction.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments; exactly one of
        ``return_raw`` / ``return_mean``.
    """

    OUTPUTS = ("raw", "mean")

    def __init__(self, data_dir=None, *, class_level="basic", min_samples_per_class=30,
                 exclude_classes=None, image_size=224, standardize=True, preload=False,
                 verify="size", **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        self.class_level, self.image_size, self.standardize = class_level, image_size, standardize
        self._fetch = {"class_level": class_level, "min_samples_per_class": min_samples_per_class,
                       "exclude_classes": tuple(exclude_classes or ()), "verify": verify}
        self._cache = {}
        super().__init__(data_dir=str(root), **kwargs)
        self._load_data()
        if preload:
            for i in range(len(self)):
                self._cache[i] = self._load(i)

    def _load_data(self):
        b = fetch_fusar_ship(self.data_dir, seed=self.seed, **self._fetch)
        keep = finalize_labels(self, b.meta["label"].to_numpy(), b.meta["group"].to_numpy(), b.splits)
        check_split_groups(self.splits, self.groups)
        self.meta = b.meta.iloc[keep].reset_index(drop=True)
        self.dropped = b.dropped
        self.samples = list(range(len(self.meta)))
        if self.verbose:
            print(f"FUSAR-Ship ({self.class_level}): {len(self.meta)} chips, {len(self.classes)} classes, "
                  f"{len(np.unique(self.groups))} vessel groups; dropped {self.dropped}")

    def _load(self, idx):
        import tifffile

        a = tifffile.imread(self.data_dir / self.meta.at[idx, "path"])
        if a.ndim == 3:  # a few chips carry a trailing channel axis
            a = a[..., 0]
        img = torch.from_numpy(np.log1p(a.astype(np.float32)))[None, None]
        if img.shape[-2:] != (self.image_size, self.image_size):
            img = F.interpolate(img, size=(self.image_size, self.image_size), mode="bilinear",
                                align_corners=False, antialias=True)
        return img[0, 0, :, :, None]  # (H, W, 1)

    def __getitem__(self, idx):
        img = self._cache.get(idx)
        if img is None:
            img = self._load(idx)
        if self.return_mean:
            x = self._pack(mean=img.reshape(-1, 1).double().mean(0))
        else:
            x = (img - img.mean()) / (img.std() + 1e-6) if self.standardize else img
            x = self._pack(raw=x)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(self.classes)
