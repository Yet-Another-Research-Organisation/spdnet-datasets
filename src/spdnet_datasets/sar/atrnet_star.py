"""ATRNet-STAR: airborne X/Ku-band quad-pol SAR chips of 40 civilian vehicles.

Reference: Liu Y. et al., *ATRNet-STAR: A Large Dataset and Benchmark Towards
Remote Sensing Object Recognition in the Wild*, arXiv:2501.13354 (v4 in the
dataset folder). Manual download (no checksum registry): the dataset must be
under ``<root>/SAR/ATRNet-STAR``.

Layout::

    {Slant_Range|Amplitude_8bit}/<protocol>/{train|test|test_<cond>}/<class>/
        <BAND>_<POL>_<DEP>_<AZ>_<id>.{mat|tif} (+ .xml)

Protocols (``split``) of the paper (Table 4): ``SOC_40classes`` (random 7:3
draw), ``EOC_scene`` (simple -> urban/factory/woodland), ``EOC_depression``
(15 deg -> 30/45/60 deg), ``EOC_azimuth`` (0-60 deg -> other azimuths),
``EOC_band`` (X -> Ku) and ``EOC_pol`` (HH -> HV/VH/VV, ``Amplitude_8bit``
only). ``test/`` is the union of the ``test_<cond>`` folders (checked); the
condition of each test chip is stored in ``meta["condition"]``.

Outputs: raw or mean only (``OUTPUTS = ("raw", "mean")``; asking for a
covariance raises ``ValueError``).

- ``Slant_Range``: one ``.mat`` per chip with the four complex channels
  ``data_hh, data_hv, data_vh, data_vv``. Raw = ``(128, 128, 4)`` amplitudes
  ``|S|`` (``slant_features="amplitude"``) or ``(128, 128, 8)`` real and
  imaginary parts (``"complex"``: hh.re, hh.im, hv.re, ...).
- ``Amplitude_8bit``: one 8-bit ``.tif`` per chip *and polarisation* (the
  polarisations of one acquisition have different ids, they are separate
  samples). Raw = ``(128, 128, 1)``.
- Mean = per-channel mean of the raw chip.

Leakage
-------
- Within each protocol, train and test share no file name and no file content
  (md5 of every chip, measured on BERTHA): ``assert_no_duplicates`` holds.
- Across protocols the chips are *reused*: every protocol is a re-partition of
  the same pool (e.g. Slant_Range: 13 112 test chips of ``EOC_depression`` are
  training chips of ``SOC_40classes``). Never pre-train on one protocol and
  test on another one.
- The same physical vehicles (one per class) appear in train and test: SOC
  measures instance recognition under close conditions, not generalisation to
  new vehicles.

Empty files: ``Amplitude_8bit/EOC_depression`` has 0-byte chips (all of
``test_30/45/60`` and 6 987 in ``train``). Each test chip is read from its
non-empty copy (``test/`` holds 30 085 of them); chips without a non-empty
copy are excluded and counted in ``dataset.n_empty``.

Index: the list of chips of one protocol comes from ``MANIFEST_NAS.tsv``
(path and size, no directory walk over the 2.1 M files) or, without manifest,
from a walk of the protocol folder only; it is cached as JSON in
``<cache_dir>/atrnet_star/index/``.
"""

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.config import dataset_path, get_cache_dir
from ..core.estimators import channel_mean
from ..core.fetch import Bunch, verify_local
from ..manager import DatasetManager
from ._common import finalize_labels

DOMAIN, NAME = "SAR", "ATRNet-STAR"
FILES = {}  # manual download (see the module docstring)
PROTOCOLS = ("EOC_band", "EOC_azimuth", "EOC_scene", "EOC_pol", "EOC_depression",
             "SOC_40classes")
REPRESENTATIONS = ("Slant_Range", "Amplitude_8bit")
EXT = {"Slant_Range": ".mat", "Amplitude_8bit": ".tif"}
POLS = ("hh", "hv", "vh", "vv")


def _scan_manifest(root, prefix, ext):
    rows = []
    with open(root / "MANIFEST_NAS.tsv", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            path = row["path"]
            if path.startswith(prefix) and path.endswith(ext):
                rows.append((path, int(row["size"])))
    return rows


def _scan_tree(root, prefix, ext):
    rows = []
    base = root / prefix
    for part in sorted(os.listdir(base)):
        for cls in sorted(os.listdir(base / part)):
            with os.scandir(base / part / cls) as it:
                for e in it:
                    if e.name.endswith(ext):
                        rows.append((f"{prefix}{part}/{cls}/{e.name}", e.stat().st_size))
    return rows


def build_index(root, representation, protocol, cache_dir=None):
    """Chips of one protocol: ``[(relative_path, size), ...]`` (cached JSON)."""
    root = Path(root)
    prefix = f"{representation}/{protocol}/"
    manifest = root / "MANIFEST_NAS.tsv"
    key = hashlib.sha1(f"{root.resolve()}|{prefix}|{manifest.exists() and manifest.stat().st_mtime}"
                       .encode()).hexdigest()[:12]
    cache = Path(cache_dir or get_cache_dir()) / "atrnet_star" / "index" / f"{representation}_{protocol}_{key}.json"
    if cache.exists():
        return [tuple(r) for r in json.loads(cache.read_text())]
    if not (root / prefix).is_dir():
        raise FileNotFoundError(f"{root / prefix} not found (protocol {protocol!r} absent from {representation})")
    rows = _scan_manifest(root, prefix, EXT[representation]) if manifest.exists() else []
    if not rows:
        rows = _scan_tree(root, prefix, EXT[representation])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_text(json.dumps(rows))
    tmp.rename(cache)
    return rows


def parse_name(name):
    """``X_HH_15_0_131281.tif`` -> band, pol, depression, azimuth, id."""
    band, pol, dep, az, ident = Path(name).stem.split("_")[:5]
    return {"band": band, "pol": pol, "depression": int(dep), "azimuth": int(az), "chip_id": ident}


def fetch_atrnet_star(data_dir=None, cache_dir=None, *, split="SOC_40classes",
                      representation="Slant_Range", verify="size", val_size=0.1, seed=0):
    """Index the chips of one ATRNet-STAR protocol.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/ATRNet-STAR``.
    cache_dir : path, optional
        Where the protocol index is cached.
    split : str
        Protocol, one of :data:`PROTOCOLS`.
    representation : {"Slant_Range", "Amplitude_8bit"}
    verify : {"size", None}
        ``"size"`` compares the size of every chip of the protocol with the
        manifest (when indexed from it).
    val_size : float
        Fraction of the training chips drawn (stratified) as validation.

    Returns
    -------
    Bunch with ``meta`` (one row per usable chip: ``path``, ``label``,
    ``part`` = train/test, ``condition``, band, pol, angles, ``chip_id``),
    ``splits``, ``n_empty`` (excluded 0-byte chips per part) and ``root``.
    """
    if split not in PROTOCOLS:
        raise ValueError(f"split must be one of {PROTOCOLS}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    if split == "EOC_pol" and representation == "Slant_Range":
        raise ValueError("EOC_pol only exists in Amplitude_8bit")
    root = dataset_path(DOMAIN, NAME, data_dir)
    verify_local(root, verify=None, required=(f"{representation}/{split}",))
    rows = build_index(root, representation, split, cache_dir)
    from_manifest = (root / "MANIFEST_NAS.tsv").exists()

    copies = {}  # (part, class, name) -> list of (path, size)
    condition = {}
    for path, size in rows:
        _, _, folder, cls, name = path.split("/")
        part = "train" if folder == "train" else "test"
        if folder.startswith("test_"):
            condition[(cls, name)] = folder[len("test_"):]
        copies.setdefault((part, cls, name), []).append((path, size))
    if verify == "size" and from_manifest:
        wrong = [p for p, s in rows if s and (root / p).stat().st_size != s]
        if wrong:
            raise OSError(f"{len(wrong)} chips differ from MANIFEST_NAS.tsv, e.g. {wrong[:3]}")

    records, empty = [], Counter()
    for (part, cls, name), cands in sorted(copies.items()):
        good = [p for p, s in cands if s > 0]
        if not good:
            empty[part] += 1
            continue
        records.append({"path": good[0], "label": cls, "part": part,
                        "condition": condition.get((cls, name), part), **parse_name(name)})
    meta = pd.DataFrame(records)
    train = np.flatnonzero(meta["part"].to_numpy() == "train")
    test = np.flatnonzero(meta["part"].to_numpy() == "test")
    from sklearn.model_selection import train_test_split

    y = meta["label"].to_numpy()
    if val_size and len(train) > 1:
        try:
            tr, va = train_test_split(train, test_size=val_size, stratify=y[train], random_state=seed)
        except ValueError:  # too few chips per class to stratify
            tr, va = train_test_split(train, test_size=val_size, random_state=seed)
    else:
        tr, va = train, train[:0]
    splits = {"official": {"train": np.sort(tr), "val": np.sort(va), "test": test}}
    cond = meta["condition"].to_numpy()
    for c in sorted(set(cond[test]) - {"test"}):
        splits[f"official_{c}"] = {"train": np.sort(tr), "val": np.sort(va),
                                   "test": test[cond[test] == c]}
    splits["recommended"] = splits["official"]
    if split == "EOC_depression" and "official_30" in splits:
        # fiche: validate on 30 deg, test on 45 and 60 deg
        splits["recommended"] = {"train": train, "val": test[cond[test] == "30"],
                                 "test": test[np.isin(cond[test], ["45", "60"])]}
    return Bunch(meta=meta, splits=splits, n_empty=dict(empty), root=root)


def protocol_overlap(data_dir=None, a="SOC_40classes", b="EOC_depression",
                     representation="Slant_Range", cache_dir=None):
    """Number of chips (same class and file name) shared by two protocols.

    Returns ``{(part_a, part_b): count}`` for parts ``train`` / ``test``. Within
    one protocol (``a == b``) ``("train", "test")`` must be 0.
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    sets = {}
    for prot in {a, b}:
        for path, _ in build_index(root, representation, prot, cache_dir):
            _, _, folder, cls, name = path.split("/")
            part = "train" if folder == "train" else "test"
            sets.setdefault((prot, part), set()).add((cls, name))
    return {(pa, pb): len(sets.get((a, pa), set()) & sets.get((b, pb), set()))
            for pa in ("train", "test") for pb in ("train", "test")}


def _read_chip(path, representation, slant_features):
    if representation == "Amplitude_8bit":
        import tifffile

        x = tifffile.imread(path).astype(np.float32)
        return x[..., None] if x.ndim == 2 else x
    from scipy.io import loadmat

    d = loadmat(path)
    s = np.stack([d[f"data_{p}"] for p in POLS], -1)
    if slant_features == "amplitude":
        return np.abs(s).astype(np.float32)
    return np.stack([s.real, s.imag], -1).reshape(*s.shape[:2], 8).astype(np.float32)


@DatasetManager.register_dataset("atrnet_star")
class ATRNetSTARDataset(BaseDataset):
    """ATRNet-STAR chips of one protocol, raw or mean outputs only.

    See the module docstring for protocols, outputs and leakage.

    Parameters
    ----------
    data_dir : path, optional
    split : str
        Protocol among :data:`PROTOCOLS` (``"SOC_40classes"`` by default).
        ``dataset.splits`` holds ``"official"`` (train/test of the protocol,
        validation = 10 % of train), ``"official_<cond>"`` (one test
        condition) and ``"recommended"``.
    representation : {"Slant_Range", "Amplitude_8bit"}
    slant_features : {"amplitude", "complex"}
    cache_dir, verify, val_size
        See :func:`fetch_atrnet_star`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments; exactly one of
        ``return_raw`` / ``return_mean`` must be True.
    """

    OUTPUTS = ("raw", "mean")

    def __init__(self, data_dir=None, *, split="SOC_40classes", representation="Slant_Range",
                 slant_features="amplitude", cache_dir=None, verify="size", val_size=0.1,
                 **kwargs):
        if slant_features not in ("amplitude", "complex"):
            raise ValueError("slant_features must be 'amplitude' or 'complex'")
        root = dataset_path(DOMAIN, NAME, data_dir)
        self.protocol, self.representation, self.slant_features = split, representation, slant_features
        self._fetch = {"cache_dir": cache_dir, "verify": verify, "val_size": val_size}
        super().__init__(data_dir=str(root), **kwargs)
        self._load_data()

    def _load_data(self):
        b = fetch_atrnet_star(self.data_dir, split=self.protocol, representation=self.representation,
                              seed=self.seed, **self._fetch)
        keep = finalize_labels(self, b.meta["label"].to_numpy(), b.meta["chip_id"].to_numpy(), b.splits)
        self.meta = b.meta.iloc[keep].reset_index(drop=True)
        self.n_empty = b.n_empty
        self.samples = list(range(len(self.meta)))
        if self.verbose:
            print(f"ATRNet-STAR {self.representation}/{self.protocol}: {len(self.meta)} chips, "
                  f"{len(self.classes)} classes, empty files excluded: {self.n_empty}")

    def __getitem__(self, idx):
        raw = _read_chip(self.data_dir / self.meta.at[idx, "path"], self.representation,
                         self.slant_features)
        mean = channel_mean(raw.reshape(-1, raw.shape[-1])) if self.return_mean else None
        x = self._pack(raw=torch.from_numpy(raw) if self.return_raw else None,
                       mean=torch.from_numpy(mean) if mean is not None else None)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])
