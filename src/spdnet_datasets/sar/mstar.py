"""MSTAR SOC: X-band HH chips of 10 military vehicles (public AFRL/SDMS release).

Reference: Keydel E. R. et al., *MSTAR extended operating conditions: a
tutorial*, Proc. SPIE 2757, 1996; 10-class SOC protocol. Copy used here:
a GitHub mirror (``SAR/MSTAR_SOC``) with ``soc/{train,test}/<class>/HBxxxxx.jpg``
(intensity, 8 bits) and ``soc_complex/{train,test}/<class>/HBxxxxx.0nn``
(Phoenix header + big-endian float32 amplitude then phase). Manual download.

Split ``"official"``: train = 17 deg depression (2 747 chips), test = 15 deg
(2 425 chips); validation = 10 % of train (stratified). **Inherent leakage of
the protocol**: the *same vehicles* (same serial number for every class) are
imaged at 17 and 15 deg, and 99.8 % of the test chips have a training chip of
the same class within 2 deg of azimuth (atlas). SOC measures a 2 deg
generalisation in depression, not recognition of new vehicles. There is no
exact duplicate (md5).

Chip sizes depend on the class (128 x 128 to 193 x 192): every chip is centre
cropped to ``crop x crop`` (default 128, the smallest size).

Outputs: raw or mean only (``OUTPUTS = ("raw", "mean")``).

- ``source="jpg"``: raw ``(crop, crop, 1)`` 8-bit intensity;
- ``source="amplitude"``: ``(crop, crop, 1)`` amplitude of the complex chip;
- ``source="complex"``: ``(crop, crop, 2)`` real and imaginary parts.

Mean = per-channel mean of the raw chip.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.config import dataset_path
from ..core.estimators import channel_mean
from ..core.fetch import Bunch, verify_local
from ..manager import DatasetManager
from ._common import finalize_labels

DOMAIN, NAME = "SAR", "MSTAR_SOC"
FILES = {}  # manual download
SOURCES = ("jpg", "amplitude", "complex")


def read_phoenix(path):
    """Complex chip ``(rows, cols)`` and header fields of an MSTAR Phoenix file."""
    raw = Path(path).read_bytes()
    head = raw[:6000].decode("ascii", "replace")

    def field(k):
        m = re.search(rf"^{k}\s*=\s*(\S+)", head, re.M)
        return m.group(1) if m else None

    nr, nc, hl = int(field("NumberOfRows")), int(field("NumberOfColumns")), int(field("PhoenixHeaderLength"))
    d = np.frombuffer(raw[hl:], ">f4")
    z = d[:nr * nc].reshape(nr, nc) * np.exp(1j * d[nr * nc:2 * nr * nc].reshape(nr, nc))
    return z, {"depression": field("MeasuredDepression"), "serial": field("TargetSerNum")}


def center_crop(x, crop):
    """Centre crop of the first two axes."""
    h, w = x.shape[:2]
    if crop > h or crop > w:
        raise ValueError(f"crop={crop} larger than the chip ({h} x {w})")
    r, c = (h - crop) // 2, (w - crop) // 2
    return x[r:r + crop, c:c + crop]


def fetch_mstar(data_dir=None, *, source="jpg", verify="size", val_size=0.1, seed=0):
    """Index the MSTAR SOC chips.

    Returns
    -------
    Bunch with ``meta`` (``path``, ``label``, ``part``, ``chip_id``) and
    ``splits`` (``"official"`` = ``"recommended"``).
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    root = dataset_path(DOMAIN, NAME, data_dir)
    folder = "soc" if source == "jpg" else "soc_complex"
    verify_local(root, verify=verify, required=(f"{folder}/train", f"{folder}/test"))
    rows = []
    for part in ("train", "test"):
        for cls_dir in sorted((root / folder / part).iterdir()):
            for f in sorted(cls_dir.iterdir()):
                if source == "jpg" and f.suffix.lower() != ".jpg":
                    continue
                if source != "jpg" and not re.fullmatch(r"\.\d{3}", f.suffix):
                    continue
                rows.append({"path": str(f.relative_to(root)), "label": cls_dir.name, "part": part,
                             "chip_id": f.stem})
    meta = pd.DataFrame(rows)
    from sklearn.model_selection import train_test_split

    part = meta["part"].to_numpy()
    train, test = np.flatnonzero(part == "train"), np.flatnonzero(part == "test")
    try:
        tr, va = train_test_split(train, test_size=val_size, stratify=meta["label"].to_numpy()[train],
                                  random_state=seed)
    except ValueError:
        tr, va = train_test_split(train, test_size=val_size, random_state=seed)
    splits = {"official": {"train": np.sort(tr), "val": np.sort(va), "test": test}}
    splits["recommended"] = splits["official"]
    return Bunch(meta=meta, splits=splits, root=root)


@DatasetManager.register_dataset("mstar")
class MSTARDataset(BaseDataset):
    """MSTAR SOC chips (raw or mean only), centre cropped.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/MSTAR_SOC``.
    source : {"jpg", "amplitude", "complex"}
    crop : int
        Side of the centre crop (``<= 128`` keeps every class).
    verify, val_size
        See :func:`fetch_mstar`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments; exactly one of
        ``return_raw`` / ``return_mean``.
    """

    OUTPUTS = ("raw", "mean")

    def __init__(self, data_dir=None, *, source="jpg", crop=128, verify="size", val_size=0.1, **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        self.source, self.crop = source, crop
        self._fetch = {"source": source, "verify": verify, "val_size": val_size}
        super().__init__(data_dir=str(root), **kwargs)
        self._load_data()

    def _load_data(self):
        b = fetch_mstar(self.data_dir, seed=self.seed, **self._fetch)
        keep = finalize_labels(self, b.meta["label"].to_numpy(), b.meta["chip_id"].to_numpy(), b.splits)
        self.meta = b.meta.iloc[keep].reset_index(drop=True)
        self.samples = list(range(len(self.meta)))
        if self.verbose:
            print(f"MSTAR SOC ({self.source}, crop {self.crop}): {len(self.meta)} chips, "
                  f"{len(self.classes)} classes")

    def _read(self, idx):
        path = self.data_dir / self.meta.at[idx, "path"]
        if self.source == "jpg":
            from PIL import Image

            x = np.asarray(Image.open(path).convert("L"), dtype=np.float32)[..., None]
        else:
            z, _ = read_phoenix(path)
            x = (np.abs(z)[..., None] if self.source == "amplitude"
                 else np.stack([z.real, z.imag], -1)).astype(np.float32)
        return np.ascontiguousarray(center_crop(x, self.crop))

    def __getitem__(self, idx):
        raw = self._read(idx)
        mean = channel_mean(raw.reshape(-1, raw.shape[-1])) if self.return_mean else None
        x = self._pack(raw=torch.from_numpy(raw) if self.return_raw else None,
                       mean=torch.from_numpy(mean) if mean is not None else None)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])
