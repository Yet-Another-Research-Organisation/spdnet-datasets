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

Download
--------
Hugging Face, gated: https://huggingface.co/datasets/waterdisappear/ATRNet-STAR
(CC BY-SA 4.0 on the card, code Apache-2.0). Accept the conditions on the page
(automatic approval) and provide a read token (``HF_TOKEN``, ``huggingface-cli
login`` or the keyring, see :mod:`spdnet_datasets.core.access`). Without a local
copy, ``fetch_atrnet_star`` downloads only the archive of the requested protocol
into ``<cache_dir>/atrnet_star/downloads/`` (:data:`HF_ARCHIVES`; volumes of 4.7 GB,
sizes of the public file tree checked, the LFS checksums are hidden by the gate):

- ``Slant_Range``: ``Slant_Range/Complex_Float_Quad/<protocol>.7z.001...`` (the
  local ``Slant_Range`` matches ``Complex_Float_Quad``: same protocols, no
  ``EOC_pol``, ``*_quad_*.mat`` chips with the four polarisations), e.g. 14.9 GB
  for ``SOC_40classes``;
- ``Amplitude_8bit``: one archive for every protocol (10.2 GB,
  ``Ground_Range/Amplitude_8bit/Amplitude 8-bit data_...7z.001..003``).

The volumes are extracted with the ``7z`` command or the optional ``py7zr``
package (``pip install 'spdnet-datasets[archives]'``; with py7zr only the
requested protocol of the amplitude archive is extracted) and the protocol folder
is linked as ``downloads/root/<representation>/<protocol>``. The inner layout of
the archives could not be checked without a token: the protocol folder is looked
up by name. Already downloaded::

    fetch_atrnet_star(archive="~/Downloads/SOC_40classes.7z.001")  # first volume

(the first volume, the folder holding the volumes, an extracted folder or a copy
with the local layout).
"""

import csv
import hashlib
import json
import os
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.access import hf_download
from ..core.config import dataset_path, get_cache_dir
from ..core.estimators import channel_mean
from ..core.fetch import Bunch, verify_local
from ..manager import DatasetManager
from ..utils.sources import downloads_dir, extract_7z, find_root
from ._common import finalize_labels

DOMAIN, NAME = "SAR", "ATRNet-STAR"
HF_REPO = "waterdisappear/ATRNet-STAR"
HF_PAGE = f"https://huggingface.co/datasets/{HF_REPO}"
_VOL = 4_698_669_056
# (representation, protocol or None = all) -> (path of the 7z without ".001", volume sizes),
# from the public file tree of the repository (2026-11-15; LFS sha256 hidden by the gate)
HF_ARCHIVES = {
    ("Slant_Range", "SOC_40classes"): ("Slant_Range/Complex_Float_Quad/SOC_40classes.7z",
                                       (_VOL,) * 3 + (761_653_877,)),
    ("Slant_Range", "EOC_azimuth"): ("Slant_Range/Complex_Float_Quad/EOC_azimuth.7z",
                                     (_VOL,) * 5 + (3_372_524_403,)),
    ("Slant_Range", "EOC_band"): ("Slant_Range/Complex_Float_Quad/EOC_band.7z",
                                  (_VOL, 3_784_725_627)),
    ("Slant_Range", "EOC_depression"): ("Slant_Range/Complex_Float_Quad/EOC_depression.7z",
                                        (_VOL,) * 5 + (2_819_608_948,)),
    ("Slant_Range", "EOC_scene"): ("Slant_Range/Complex_Float_Quad/EOC_scene.7z",
                                   (_VOL,) * 5 + (3_229_012_282,)),
    ("Amplitude_8bit", None): ("Ground_Range/Amplitude_8bit/Amplitude 8-bit data_地距幅度8位数据.7z",
                               (4_697_620_480, 4_697_620_480, 798_202_909)),
}
FILES = {}  # gated Hugging Face volumes: sizes in HF_ARCHIVES, no visible checksum


def _hf_archive(representation, split):
    return HF_ARCHIVES[(representation, None if representation == "Amplitude_8bit" else split)]


def _protocol_dir(base, split):
    """Folder of one protocol (``train/``, ``test*/``) inside an extracted archive."""
    found = find_root(base, (f"{split}/train",), max_depth=5)
    return found / split if found is not None else find_root(base, ("train",), max_depth=5)


def _download_root(cache_dir, archive, representation, split):
    """Folder with ``<representation>/<split>`` built in the cache from the 7z volumes."""
    out = downloads_dir("atrnet_star", cache_dir)
    layout = out / "root"
    link = layout / representation / split
    if link.exists():
        return layout
    name, sizes = _hf_archive(representation, split)
    first = None
    if archive is not None:
        given = Path(archive).expanduser()
        if not given.exists():
            raise FileNotFoundError(f"{given} does not exist")
        if given.is_dir() and (given / representation / split).is_dir():
            return given  # copy with the local layout
        if given.is_file():
            first = given
        elif (given / f"{Path(name).name}.001").exists():
            first = given / f"{Path(name).name}.001"
        else:
            proto = _protocol_dir(given, split)
            if proto is None:
                raise FileNotFoundError(f"{given} holds no {split} folder nor {Path(name).name}.001")
    else:
        quoted = urllib.parse.quote(name)  # spaces and non-ASCII characters in the file name
        vols = [hf_download(HF_REPO, f"{quoted}.{k:03d}", out / "hf", gated=True, size=size)
                for k, size in enumerate(sizes, start=1)]
        first = vols[0]
    if first is not None:
        tag = split if representation == "Slant_Range" else "all"
        only = split if representation == "Amplitude_8bit" else None
        ext = extract_7z(first, out / "extracted" / f"{representation}_{tag}", only=only)
        proto = _protocol_dir(ext, split)
        if proto is None:
            raise FileNotFoundError(f"{first.name}: no {split} folder after extraction")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(proto.resolve(), target_is_directory=True)
    return layout
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
                      representation="Slant_Range", verify="size", val_size=0.1, seed=0,
                      archive=None):
    """Index the chips of one ATRNet-STAR protocol.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/ATRNet-STAR``; the protocol is downloaded into the
        cache when absent (Hugging Face token needed).
    cache_dir : path, optional
        Where the protocol index (and a download) is cached.
    archive : path, optional
        First ``.7z.001`` volume, folder of the volumes, extracted folder or
        copy with the local layout (see "Download" in the module docstring).
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
    if not (root / representation / split).is_dir():
        root = _download_root(cache_dir, archive, representation, split)
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
