"""PASTIS-R agricultural parcels as covariances of Sentinel-2 / Sentinel-1 pixels.

PASTIS-R (Sainte Fare Garnot & Landrieu, ICCV 2021; Sainte Fare Garnot, Landrieu &
Chehata, ISPRS JPRS 2022) holds 2433 patches of 128 x 128 pixels at 10 m from four
Sentinel-2 tiles in France (2018-09 to 2019-10), with Sentinel-2 (S2), Sentinel-1
ascending (S1A) and descending (S1D) image time series, semantic labels and parcel
identifiers.

Sample
------
One sample is one **parcel** of one patch (``ANNOTATIONS/ParcelIDs_<patch>.npy``).
Its observations are the pixels of the parcel (an irregular set of ``n`` pixels),
its channels the ``p`` variables of the chosen scenario, and its label the majority
class of ``ANNOTATIONS/TARGET_<patch>.npy[0]`` over the parcel. Background (0) and
void (19, parcels mostly outside the patch) are dropped: 18 crop classes remain.
The 35 patches that have images but no entry in ``metadata.geojson`` (no label, no
fold) are ignored. A parcel cut by a patch border appears in each patch it touches
(1024 identifiers); the copies are kept as separate samples, which never crosses
folds (no usable identifier is shared by two folds).

Scenarios (``scenario=``)
-------------------------
==== ============================================== === ==========
name channels per date                               p   min pixels
==== ============================================== === ==========
A    S2 10 bands, 1 date                             10  30
B    S2 10 bands + NDVI + NDWI, 1 date               12  30
C    S2 10 + S1A (VV, VH, VV/VH) + S1D (idem), 1 date 16  32
D    S2 10 bands, 3 consecutive dates                30  60
E    S2 10 + S1A 3 + S1D 3, 2 consecutive S2 dates   32  64
==== ============================================== === ==========

The minimum number of pixels is ``2p`` (``core.estimators.check_sample_size``).

Bands and indices
-----------------
S2 bands are stored as int16 reflectance x 1e4 in the order B02, B03, B04, B05,
B06, B07, B08, B8A, B11, B12 (the 13 L2A THEIA bands minus the atmospheric B01, B09
and B10, PASTIS documentation ``documentation/pastis-documentation.pdf`` of
github.com/VSainteuf/pastis-benchmark; the B08 < B8A, B11 > B12 band means agree).
NDVI = (B08 - B04) / (B08 + B04) and NDWI (McFeeters) = (B03 - B08) / (B03 + B08);
pixels with a non-positive denominator are invalid.

Sentinel-1 is stored as float16 backscatter in **dB** with channels (VV, VH, VV - VH):
the third channel is exactly the dB difference, a linear combination of the first
two, so ``[VV, VH, VV/VH]`` in dB has a singular covariance. With the default
``s1_domain="db"`` the channels are VV (dB), VH (dB) and the **linear** power ratio
VV/VH = 10 ** ((VV - VH) / 10), computed from the first two channels; with
``s1_domain="linear"`` all three are linear (VV, VH intensities and their ratio).

Dates (scenarios C, D and E; A and B use the same date selection without S1)
----------------------------------------------------------------------------
Acquisition dates are shared by all the patches of an S2 tile. For each patch, the
retained S2 dates (and the S1 dates paired with them) are the same for all its
parcels. Candidate windows are the ``k`` consecutive S2 acquisitions (k = 1, 1, 1,
3, 2) whose mean date lies within ``date_window`` days of ``ref_date`` (or the
window starting at index ``date_index`` of the patch's S2 series). A candidate is
valid if

- its span (last - first date) is at most ``max_span_days`` (D, E);
- every S2 date has an S1A and an S1D acquisition within ``max_gap_days``, the S1
  dates being distinct for distinct S2 dates (E: two S2 dates 5 days apart must
  not share one S1 image, which would duplicate channels);
- the patch cloud proxy (fraction of pixels with B02 > ``cloud_b02``, averaged
  over the window) is at most ``max_cloud_fraction`` (PASTIS has no cloud mask).

The valid candidate with the smallest maximal S1 gap, then the smallest offset to
``ref_date``, then the fewest clouds, is kept. **Homogeneity**: the gap of a
covariance is its largest absolute S2-S1 gap; a patch is dropped when this gap
differs by more than ``gap_tolerance`` days from its median over all the
covariances (default 1 day). Measured over all S2 dates of all
patches (nearest S1 acquisition), the absolute gaps have median 2 days and 90th
percentile 3 days for both S1A and S1D (95th: 5 days; max 15 / 19 days), hence
``max_gap_days=3``. Rejections are counted per reason, tile and fold in
``report``; the retained dates and signed gaps (S1 - S2, days) are in ``meta``,
and the full date-gap table and histogram are written to the processed cache
(``date_gaps.csv``, ``gap_histogram.csv``).

Normalisation
-------------
Channels have different units (reflectance x 1e4, indices in [-1, 1], dB, linear
ratios). Before estimation, each channel is standardised (mean 0, std 1) with
statistics pooled over the pixels of the **training folds of the chosen split**
only, so that shrinkage estimators (Ledoit-Wolf) are not dominated by the S2 bands
and no test statistic leaks. The covariance cache therefore depends on the split.
At load time, the scalar geometric normalisation of ``core.normalize`` is fitted on
the training covariances (``normalize=True``).

Splits
------
``"official"``: folds 1-3 train, 4 validation, 5 test (by patch, the folds being
spatially separated). ``"cv1"`` ... ``"cv5"``: test fold k, validation fold k - 1
(fold 5 for k = 1), training the three others; ``"cv5"`` and ``"recommended"``
equal ``"official"``.
The group of a sample is its patch; train, validation and test never share one.

``return_raw`` gives the standardised pixels of the parcel, an array ``(n, p)``
float32 whose ``n`` varies from parcel to parcel (the default PyTorch collate needs
``batch_size=1`` or a custom ``collate_fn``).

Download
--------
Zenodo record 5735646 (CC BY 4.0): one archive ``PASTIS-R.zip`` of 53.7 GB, whose
Zenodo md5 (``4887513d6c2d2b07fa935d325bd53e09``) and sha256 (:data:`FILES`) were
checked on the local copy. When ``data_dir`` holds no ``metadata.geojson``, the
archive is downloaded once into ``<cache_dir>/pastis_r/downloads/``, checked,
extracted there (about 73 GB; peak 127 GB with the archive) and deleted. The
extracted folder is outside the ``keep`` policy (never deleted by the fetch, remove
it by hand): every scenario and date setting has its own processed stage, and
deleting the source would mean a new 53.7 GB download for each of them.
"""

from __future__ import annotations

import ast
import datetime
import itertools
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core import (
    KEEP,
    Bunch,
    InsufficientSamplesError,
    NotSPDError,
    apply_scale,
    apply_task,
    assert_no_leakage,
    balance_indices,
    class_weights,
    dataset_path,
    estimate_covariance,
    geometric_scale,
    get_cache_dir,
    verify_local,
)
from ..core.fetch import Stage, download
from ..manager import DatasetManager

DOMAIN, NAME = "SITS", "PASTIS-R"
CACHE_NAME = "pastis_r"
ZENODO_URL = "https://zenodo.org/records/5735646/files/PASTIS-R.zip?download=1"
ARCHIVE = "PASTIS-R.zip"
FILES = {  # url -> sha256 (computed on the local copy, whose md5 equals Zenodo's)
    ZENODO_URL: "96eb6e3567348465ecb98d072d9c9a913272b05f49b7e0cac679408e50a4eeb3",
}

S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
_B02, _B03, _B04, _B08 = 0, 1, 2, 6
S1_CHANNELS = ("VV", "VH", "VV/VH")
S1_DOMAINS = ("db", "linear")
BACKGROUND, VOID = 0, 19
CLASSES = (
    "Meadow", "Soft winter wheat", "Corn", "Winter barley", "Winter rapeseed",
    "Spring barley", "Sunflower", "Grapevine", "Beet", "Winter triticale",
    "Winter durum wheat", "Fruits, vegetables, flowers", "Potatoes",
    "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal", "Sorghum",
)  # semantic codes 1..18
COARSE_MAP = {
    "Soft winter wheat": "Winter cereals", "Winter barley": "Winter cereals",
    "Winter triticale": "Winter cereals", "Winter durum wheat": "Winter cereals",
    "Mixed cereal": "Winter cereals",
    "Corn": "Summer cereals", "Spring barley": "Summer cereals",
    "Sorghum": "Summer cereals",
    "Winter rapeseed": "Oilseeds", "Sunflower": "Oilseeds", "Soybeans": "Oilseeds",
    "Meadow": "Grassland and fodder", "Leguminous fodder": "Grassland and fodder",
    "Grapevine": "Permanent crops", "Orchard": "Permanent crops",
    "Beet": "Root crops and vegetables", "Potatoes": "Root crops and vegetables",
    "Fruits, vegetables, flowers": "Root crops and vegetables",
}

SCENARIOS = {
    "A": {"n_dates": 1, "indices": False, "s1": False, "p": 10, "min_pixels": 30},
    "B": {"n_dates": 1, "indices": True, "s1": False, "p": 12, "min_pixels": 30},
    "C": {"n_dates": 1, "indices": False, "s1": True, "p": 16, "min_pixels": 32},
    "D": {"n_dates": 3, "indices": False, "s1": False, "p": 30, "min_pixels": 60},
    "E": {"n_dates": 2, "indices": False, "s1": True, "p": 32, "min_pixels": 64},
}
SPLITS = {"official": {"train": (1, 2, 3), "val": (4,), "test": (5,)}}
for _k in range(1, 6):
    _val = (_k - 2) % 5 + 1
    SPLITS[f"cv{_k}"] = {"train": tuple(f for f in range(1, 6) if f not in (_k, _val)),
                         "val": (_val,), "test": (_k,)}
SPLITS["recommended"] = SPLITS["official"]

DEFAULT_REF_DATE = "2019-07-15"
DEFAULTS = {
    "date_window": 15, "max_gap_days": 3, "gap_tolerance": 1, "max_span_days": 20,
    "max_cloud_fraction": 0.1, "cloud_b02": 1500, "s1_domain": "db",
}


# --------------------------------------------------------------------------- dates
def _day(value) -> int:
    """Ordinal day of a ``YYYYMMDD`` integer, an ISO string or a date."""
    if isinstance(value, datetime.date):
        return value.toordinal()
    if isinstance(value, str):
        return datetime.date.fromisoformat(value).toordinal()
    v = int(value)
    return datetime.date(v // 10000, v // 100 % 100, v % 100).toordinal()


def _iso(day: int) -> str:
    return datetime.date.fromordinal(int(day)).isoformat()


def _acquisitions(value) -> np.ndarray:
    """Dates of a ``dates-<sensor>`` property (a dict or its string form), in index order."""
    d = value if isinstance(value, dict) else ast.literal_eval(value)
    return np.array([_day(d[k]) for k in sorted(d, key=int)], dtype=np.int64)


def pair_s1(s2_days, s1_days, max_gap_days):
    """Pair each S2 date with a distinct S1 date within ``max_gap_days``.

    Among the three nearest S1 acquisitions of each S2 date, the assignment of
    distinct S1 dates with the smallest maximal absolute gap is returned as
    ``(indices, signed_gaps)`` (gap = S1 - S2 in days), or ``None``.
    """
    s2_days, s1_days = np.asarray(s2_days), np.asarray(s1_days)
    near = [np.argsort(np.abs(s1_days - d), kind="stable")[:3] for d in s2_days]
    best = None
    for js in itertools.product(*near):
        if len(set(js)) < len(js):
            continue
        gaps = s1_days[list(js)] - s2_days
        worst = int(np.abs(gaps).max())
        if worst <= max_gap_days and (best is None or worst < best[0]):
            best = (worst, tuple(int(j) for j in js), gaps.astype(int))
    return None if best is None else (best[1], best[2])


def select_dates(s2_days, s1a_days=None, s1d_days=None, *, n_dates=1, ref_day=None,
                 date_index=None, date_window=15, max_gap_days=3, max_span_days=20,
                 cloud=None, max_cloud_fraction=None):
    """Choose the S2 window (and paired S1 dates) of one patch.

    Parameters
    ----------
    s2_days, s1a_days, s1d_days : arrays of ordinal days
        ``s1a_days``/``s1d_days`` None for scenarios without Sentinel-1.
    n_dates : int
        Number of consecutive S2 acquisitions.
    ref_day : int, optional
        Reference ordinal day; candidates have their mean date within
        ``date_window`` days of it. Ignored when ``date_index`` is given.
    date_index : int, optional
        Use the window starting at this index of ``s2_days`` only.
    cloud : array, optional
        Cloud proxy of each S2 date; windows above ``max_cloud_fraction`` are invalid.

    Returns
    -------
    (selection, None) or (None, reason) with reason in ``"no_window"``,
    ``"span"``, ``"gap"``, ``"cloud"`` (the furthest check reached).
    """
    s2_days = np.asarray(s2_days)
    with_s1 = s1a_days is not None
    if date_index is not None:
        starts = [date_index] if 0 <= date_index <= len(s2_days) - n_dates else []
    else:
        starts = range(len(s2_days) - n_dates + 1)
    stage = {"no_window": 0, "span": 1, "gap": 2, "cloud": 3}
    reached, best = "no_window", None
    for start in starts:
        idx = np.arange(start, start + n_dates)
        days = s2_days[idx]
        offset = float(days.mean() - ref_day) if ref_day is not None else 0.0
        if date_index is None and abs(offset) > date_window:
            continue
        if n_dates > 1 and days[-1] - days[0] > max_span_days:
            reached = max(reached, "span", key=stage.get)
            continue
        pairs = {}
        if with_s1:
            for name, s1 in (("s1a", s1a_days), ("s1d", s1d_days)):
                pairs[name] = pair_s1(days, s1, max_gap_days)
            if None in pairs.values():
                reached = max(reached, "gap", key=stage.get)
                continue
        c = float(np.mean(np.asarray(cloud)[idx])) if cloud is not None else 0.0
        if max_cloud_fraction is not None and c > max_cloud_fraction:
            reached = max(reached, "cloud", key=stage.get)
            continue
        worst = max((int(np.abs(g).max()) for _, g in pairs.values()), default=0)
        key = (worst, abs(offset), c)
        if best is None or key < best[0]:
            sel = {"s2": tuple(int(i) for i in idx), "offset": offset, "cloud": c,
                   "max_gap": worst}
            for name, (js, gaps) in pairs.items():
                sel[name], sel[f"gap_{name}"] = js, tuple(int(g) for g in gaps)
            best = (key, sel)
    return (best[1], None) if best else (None, reached)


def homogeneity_mask(max_gaps, gap_tolerance, weights=None):
    """Keep the items whose gap lies within ``gap_tolerance`` days of the median gap.

    ``max_gaps``: maximal absolute S2-S1 gap of each patch (the gap of all its
    covariances); ``weights``: number of covariances of each patch, so that the
    median is taken over covariances. Returns ``(mask, median)``.
    """
    max_gaps = np.asarray(max_gaps, dtype=float)
    if not len(max_gaps) or gap_tolerance is None:
        return np.ones(len(max_gaps), bool), None
    reps = np.ones(len(max_gaps), int) if weights is None else np.asarray(weights, int)
    median = float(np.median(np.repeat(max_gaps, reps)))
    return np.abs(max_gaps - median) <= gap_tolerance, median


# ------------------------------------------------------------------ channels/pixels
def channel_names(scenario: str) -> list[str]:
    """Names of the ``p`` channels of a scenario, date by date."""
    cfg = SCENARIOS[scenario]
    names = []
    for t in range(cfg["n_dates"]):
        names += [f"t{t}:S2:{b}" for b in S2_BANDS]
        if cfg["indices"]:
            names += [f"t{t}:NDVI", f"t{t}:NDWI"]
        if cfg["s1"]:
            names += [f"t{t}:{s}:{c}" for s in ("S1A", "S1D") for c in S1_CHANNELS]
    return names


def _normalized_difference(a, b):
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (a - b) / den, np.nan).astype(np.float32)


def s1_channels(frame, s1_domain="db"):
    """(3, H, W) S1 channels from a stored (>=2, H, W) dB frame (see module docstring)."""
    if s1_domain not in S1_DOMAINS:
        raise ValueError(f"s1_domain must be one of {S1_DOMAINS}")
    vv, vh = frame[0].astype(np.float32), frame[1].astype(np.float32)
    ratio = np.power(np.float32(10), (vv - vh) / 10)
    if s1_domain == "linear":
        vv, vh = np.power(np.float32(10), vv / 10), np.power(np.float32(10), vh / 10)
    return np.stack([vv, vh, ratio])


def patch_channels(files_dir, patch, sel, scenario, s1_domain="db"):
    """(p, H, W) float32 channels of one patch for the selected dates."""
    cfg = SCENARIOS[scenario]
    s2 = np.load(Path(files_dir) / "DATA_S2" / f"S2_{patch}.npy", mmap_mode="r")
    s1 = {s: np.load(Path(files_dir) / f"DATA_{s}" / f"{s}_{patch}.npy", mmap_mode="r")
          for s in ("S1A", "S1D")} if cfg["s1"] else {}
    out = []
    for t, i in enumerate(sel["s2"]):
        bands = np.asarray(s2[i], dtype=np.float32)
        out.append(bands)
        if cfg["indices"]:
            out.append(_normalized_difference(bands[_B08], bands[_B04])[None])
            out.append(_normalized_difference(bands[_B03], bands[_B08])[None])
        for s in ("S1A", "S1D") if cfg["s1"] else ():
            out.append(s1_channels(np.asarray(s1[s][sel[s.lower()][t]]), s1_domain))
    return np.concatenate(out)


# -------------------------------------------------------------------------- paths
def _resolve(data_dir):
    """(dataset_dir holding MANIFEST_NAS.tsv, files_dir holding metadata.geojson)."""
    root = dataset_path(DOMAIN, NAME, data_dir)
    files = root if (root / "metadata.geojson").exists() else root / NAME
    return root, files


def download_pastis_r(cache_dir=None) -> Path:
    """Download and extract the Zenodo archive once; return the folder of ``metadata.geojson``.

    The archive goes to ``<cache_dir>/pastis_r/downloads/``, is checked against
    :data:`FILES`, extracted into a temporary folder renamed when complete (an
    interrupted extraction is redone, never used) and then deleted.
    """
    dest = (Path(cache_dir) if cache_dir else get_cache_dir()) / CACHE_NAME / "downloads"
    out = dest / NAME
    if (out / "metadata.geojson").exists():
        return out
    archive = download(ZENODO_URL, dest / ARCHIVE, checksum=FILES[ZENODO_URL])
    tmp = dest / "extract.part"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp)
    (tmp / NAME).rename(out)
    shutil.rmtree(tmp)
    archive.unlink()
    return out


def _read_metadata(files_dir):
    with open(Path(files_dir) / "metadata.geojson") as f:
        return sorted((feat["properties"] for feat in json.load(f)["features"]),
                      key=lambda p: int(p["ID_PATCH"]))


def _gap_tables(patches):
    """Signed S1 - S2 gaps (nearest acquisition) for every (tile, S2 date)."""
    rows = []
    for prop in patches:
        s2 = _acquisitions(prop["dates-S2"])
        row = {"patch": int(prop["ID_PATCH"]), "tile": prop["TILE"]}
        gaps = {}
        for s in ("S1A", "S1D"):
            s1 = _acquisitions(prop[f"dates-{s}"])
            j = np.abs(s1[None, :] - s2[:, None]).argmin(1)
            gaps[s] = (s1[j], s1[j] - s2)
        for k, d in enumerate(s2):
            rows.append({**row, "s2_date": _iso(d),
                         "s1a_date": _iso(gaps["S1A"][0][k]), "gap_s1a": int(gaps["S1A"][1][k]),
                         "s1d_date": _iso(gaps["S1D"][0][k]), "gap_s1d": int(gaps["S1D"][1][k])})
    df = pd.DataFrame(rows)
    table = (df.groupby(["tile", "s2_date", "s1a_date", "gap_s1a", "s1d_date", "gap_s1d"])
             .size().rename("n_patches").reset_index())
    hist = []
    for s in ("s1a", "s1d"):
        a = df[f"gap_{s}"].abs()
        per_patch = a.value_counts().sort_index()
        per_date = table[f"gap_{s}"].abs().value_counts().sort_index()
        for g in sorted(set(per_patch.index) | set(per_date.index)):
            hist.append({"sensor": s.upper(), "abs_gap_days": int(g),
                         "n_patch_dates": int(per_patch.get(g, 0)),
                         "n_tile_dates": int(per_date.get(g, 0))})
    q = {s.upper(): {f"p{p}": float(np.percentile(df[f"gap_{s}"].abs(), p))
                     for p in (50, 90, 95, 99, 100)} for s in ("s1a", "s1d")}
    return table, pd.DataFrame(hist), q


# --------------------------------------------------------------------- processing
def _process(files_dir, out, params):
    """Parcel table, date selection and parcel pixels (raw units) of one scenario."""
    cfg = SCENARIOS[params["scenario"]]
    patches = _read_metadata(files_dir)
    n_metadata = len(patches)
    if params["patches"] is not None:
        wanted = set(params["patches"])
        patches = [p for p in patches if int(p["ID_PATCH"]) in wanted]
    table, hist, quantiles = _gap_tables(patches)
    table.to_csv(out / "date_gaps.csv", index=False)
    hist.to_csv(out / "gap_histogram.csv", index=False)
    ref_day = _day(params["ref_date"]) if params["ref_date"] is not None else None
    n_files = len(list((Path(files_dir) / "DATA_S2").glob("S2_*.npy")))

    # Pass 1: candidate parcels and date selection per patch.
    rejected = {"patch": {}, "parcel": {}}
    chosen = []  # (prop, sel, parcels, labels, counts)

    def reject(kind, reason, prop, n=1):
        key = f"{reason}|{prop['TILE']}|fold{prop['Fold']}"
        rejected[kind][key] = rejected[kind].get(key, 0) + n

    for prop in patches:
        pid = int(prop["ID_PATCH"])
        ids = np.load(Path(files_dir) / "ANNOTATIONS" / f"ParcelIDs_{pid}.npy").ravel()
        target = np.load(Path(files_dir) / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0].ravel()
        uniq, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
        votes = np.bincount(inv * 256 + target.astype(np.int64),
                            minlength=len(uniq) * 256).reshape(len(uniq), 256)
        labels = votes.argmax(1)
        ok = (uniq != 0) & (labels != BACKGROUND) & (labels != VOID)
        small = ok & (counts < params["min_pixels"])
        if small.any():
            reject("parcel", "few_pixels", prop, int(small.sum()))
        ok &= counts >= params["min_pixels"]
        if not ok.any():
            reject("patch", "no_parcel", prop)
            continue
        s2 = np.load(Path(files_dir) / "DATA_S2" / f"S2_{pid}.npy", mmap_mode="r")
        cloud = (np.asarray(s2[:, _B02]) > params["cloud_b02"]).mean(axis=(1, 2))
        s1 = ({s: _acquisitions(prop[f"dates-{s}"]) for s in ("S1A", "S1D")}
              if cfg["s1"] else {"S1A": None, "S1D": None})
        sel, reason = select_dates(
            _acquisitions(prop["dates-S2"]), s1["S1A"], s1["S1D"],
            n_dates=cfg["n_dates"], ref_day=ref_day, date_index=params["date_index"],
            date_window=params["date_window"], max_gap_days=params["max_gap_days"],
            max_span_days=params["max_span_days"], cloud=cloud,
            max_cloud_fraction=params["max_cloud_fraction"])
        if sel is None:
            reject("patch", reason, prop)
            reject("parcel", f"patch_{reason}", prop, int(ok.sum()))
            continue
        chosen.append((prop, sel, uniq[ok], labels[ok], counts[ok]))

    median_gap = None
    if cfg["s1"]:
        mask, median_gap = homogeneity_mask([sel["max_gap"] for _, sel, *_ in chosen],
                                            params["gap_tolerance"],
                                            [len(c[2]) for c in chosen])
        for keep, (prop, _, parcels, *_) in zip(mask, chosen, strict=True):
            if not keep:
                reject("patch", "gap_heterogeneous", prop)
                reject("parcel", "patch_gap_heterogeneous", prop, len(parcels))
        chosen = [c for keep, c in zip(mask, chosen, strict=True) if keep]

    # Pass 2: pixels of the kept parcels, appended to one flat float32 file.
    p = cfg["p"]
    rows, offsets, total = [], [0], 0
    fold_sum, fold_sq, fold_n = np.zeros((6, p)), np.zeros((6, p)), np.zeros(6)
    with open(out / "pixels.f32", "wb") as fpix:
        for prop, sel, parcels, labels, counts in chosen:
            pid, fold = int(prop["ID_PATCH"]), int(prop["Fold"])
            cube = patch_channels(files_dir, pid, sel, params["scenario"], params["s1_domain"])
            flat = cube.reshape(p, -1).T
            ids = np.load(Path(files_dir) / "ANNOTATIONS" / f"ParcelIDs_{pid}.npy").ravel()
            s2_days = _acquisitions(prop["dates-S2"])
            dates = {"s2_dates": ";".join(_iso(s2_days[i]) for i in sel["s2"])}
            for s in ("s1a", "s1d") if cfg["s1"] else ():
                days = _acquisitions(prop[f"dates-{s.upper()}"])
                dates[f"{s}_dates"] = ";".join(_iso(days[j]) for j in sel[s])
                dates[f"gap_{s}"] = ";".join(str(g) for g in sel[f"gap_{s}"])
            for parcel, label, count in zip(parcels, labels, counts, strict=True):
                X = flat[ids == parcel]
                X = X[np.all(np.isfinite(X), axis=1)]
                if len(X) < params["min_pixels"]:
                    reject("parcel", "few_valid_pixels", prop)
                    continue
                X.astype(np.float32).tofile(fpix)
                total += len(X)
                offsets.append(total)
                fold_sum[fold] += X.sum(0, dtype=np.float64)
                fold_sq[fold] += np.square(X, dtype=np.float64).sum(0)
                fold_n[fold] += len(X)
                rows.append({"patch": pid, "parcel": int(parcel), "tile": prop["TILE"],
                             "fold": fold, "label": int(label) - 1,
                             "class_name": CLASSES[int(label) - 1], "n_pixels": len(X),
                             "n_pixels_parcel": int(count), **dates,
                             "max_abs_gap": sel["max_gap"],
                             "ref_offset_days": sel["offset"], "cloud_fraction": sel["cloud"]})
    report = {"params": params, "channels": channel_names(params["scenario"]),
              "n_patches_metadata": n_metadata, "n_patches_used": len(patches),
              "n_patches_without_metadata": n_files - n_metadata,
              "n_patches_kept": len(chosen), "n_parcels": len(rows),
              "rejected": rejected, "median_abs_gap": median_gap,
              "gap_quantiles_all_s2_dates": quantiles}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    if not rows:
        raise ValueError(f"No PASTIS-R parcel kept with {params}; see {out / 'report.json'}")
    pd.DataFrame(rows).to_csv(out / "parcels.csv", index=False)
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.savez(out / "fold_stats.npz", sum=fold_sum, sumsq=fold_sq, n=fold_n)


def _standardization(fold_stats, train_folds):
    s = sum(fold_stats["sum"][f] for f in train_folds)
    q = sum(fold_stats["sumsq"][f] for f in train_folds)
    n = sum(fold_stats["n"][f] for f in train_folds)
    if n == 0:
        raise ValueError(f"No training pixels in folds {train_folds}")
    mu = s / n
    sd = np.sqrt(np.maximum(q / n - mu ** 2, 0))
    if np.any(sd <= 0):
        raise ValueError("A channel is constant over the training pixels")
    return mu, sd


def _covariances(proc_dir, out, estimator, train_folds):
    meta = pd.read_csv(proc_dir / "parcels.csv")
    offsets = np.load(proc_dir / "offsets.npy")
    stats = np.load(proc_dir / "fold_stats.npz")
    mu, sd = _standardization(stats, train_folds)
    p = len(mu)
    pixels = np.memmap(proc_dir / "pixels.f32", dtype=np.float32, mode="r").reshape(-1, p) \
        if offsets[-1] else np.zeros((0, p), np.float32)
    covs, means, kept, failures = [], [], [], {}
    for i in range(len(meta)):
        X = (pixels[offsets[i]:offsets[i + 1]] - mu) / sd
        try:
            covs.append(estimate_covariance(X, estimator))
        except (InsufficientSamplesError, NotSPDError) as exc:
            reason = type(exc).__name__
            failures[reason] = failures.get(reason, 0) + 1
            continue
        means.append(X.mean(0))
        kept.append(i)
    np.save(out / "covariances.npy", np.asarray(covs, dtype=np.float64).reshape(-1, p, p))
    np.save(out / "means.npy", np.asarray(means, dtype=np.float64).reshape(-1, p))
    np.save(out / "kept.npy", np.asarray(kept, dtype=np.int64))
    np.savez(out / "standardization.npz", mean=mu, std=sd)
    meta.iloc[kept].to_csv(out / "parcels.csv", index=False)
    shutil.copy(proc_dir / "report.json", out / "report.json")
    (out / "covariance_report.json").write_text(json.dumps(
        {"estimator": estimator, "train_folds": list(train_folds), "failures": failures,
         "n_covariances": len(kept)}, indent=1))


def make_splits(folds, groups, split="official"):
    """Index split ``{"train", "val", "test"}`` of a named fold split, checked for leakage."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose among {sorted(SPLITS)}")
    folds = np.asarray(folds)
    parts = {k: np.flatnonzero(np.isin(folds, v)) for k, v in SPLITS[split].items()}
    groups = np.asarray(groups)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_leakage(groups[parts[a]], groups[parts[b]], what="patch")
    return parts


# -------------------------------------------------------------------------- fetch
def fetch_pastis_r(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                   verify="size", *, scenario="A", split="official", ref_date=None,
                   date_index=None, date_window=DEFAULTS["date_window"],
                   max_gap_days=DEFAULTS["max_gap_days"],
                   gap_tolerance=DEFAULTS["gap_tolerance"],
                   max_span_days=DEFAULTS["max_span_days"],
                   max_cloud_fraction=DEFAULTS["max_cloud_fraction"],
                   cloud_b02=DEFAULTS["cloud_b02"], s1_domain=DEFAULTS["s1_domain"],
                   min_pixels=None, patches=None) -> Bunch:
    """Parcel covariances of PASTIS-R for one scenario (see the module docstring).

    Parameters
    ----------
    data_dir : path, optional
        Dataset directory (holding ``metadata.geojson`` directly or in ``PASTIS-R/``);
        default ``<data root>/SITS/PASTIS-R``. Read only. Without ``metadata.geojson``
        the Zenodo archive is downloaded (:func:`download_pastis_r`).
    cache_dir : path, optional
        Default ``core.config.get_cache_dir()``.
    keep : {"keep_full", "keep_processed", "keep_covariance"}, optional
        Default ``keep_full`` (local dataset). ``keep_covariance`` deletes the parcel
        pixels, so ``return_raw`` is no longer available.
    estimator : {"scm", "ledoit_wolf", "tyler", "student"}
    force : bool
        Recompute both stages.
    verify : {"size", "md5", None}
        Check of the local copy against ``MANIFEST_NAS.tsv``.
    scenario : {"A", "B", "C", "D", "E"}
    split : {"official", "recommended", "cv1", ..., "cv5"}
        Its training folds fit the channel standardisation.
    ref_date : str or date, optional
        Reference date (default ``"2019-07-15"``); ignored with ``date_index``.
    date_index : int, optional
        Index of the first retained S2 acquisition in each patch's series.
    date_window, max_gap_days, gap_tolerance, max_span_days : int
        Date selection, in days (``gap_tolerance=None`` disables the homogeneity check).
    max_cloud_fraction : float or None
        Maximal fraction of pixels with ``B02 > cloud_b02`` (None disables it).
    s1_domain : {"db", "linear"}
    min_pixels : int, optional
        Minimal number of valid pixels per parcel; default ``2p`` of the scenario.
    patches : iterable of int, optional
        Restrict to these patch identifiers (quick runs); the homogeneity median is
        then computed on them only.

    Returns
    -------
    Bunch with ``covariances`` (B, p, p) float64 (standardised channels, without the
    geometric scale), ``means`` (B, p), ``labels`` (B,) in 0..17, ``class_names``,
    ``groups`` (patch identifiers), ``meta`` (DataFrame: patch, parcel, tile, fold,
    dates, gaps...), ``splits``, ``channels``, ``report``, ``processed_path``
    (None once deleted), ``covariance_path``.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose among {sorted(SCENARIOS)}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose among {sorted(SPLITS)}")
    if s1_domain not in S1_DOMAINS:
        raise ValueError(f"s1_domain must be one of {S1_DOMAINS}")
    keep = keep or "keep_full"
    if keep not in KEEP:
        raise ValueError(f"keep must be one of {KEEP}")
    cfg = SCENARIOS[scenario]
    min_pixels = cfg["min_pixels"] if min_pixels is None else int(min_pixels)
    if date_index is None and ref_date is None:
        ref_date = DEFAULT_REF_DATE
    if date_index is not None:
        ref_date = None
    root, files_dir = _resolve(data_dir)
    if not (files_dir / "metadata.geojson").exists():
        root = files_dir = download_pastis_r(cache_dir)
    rel = files_dir.relative_to(root) if files_dir != root else Path(".")
    report = verify_local(root, verify=verify, required=[str(rel / "metadata.geojson")])
    if report.missing or report.wrong_size or report.wrong_md5:
        raise OSError(f"Local PASTIS-R copy does not match its manifest: {dict(report)}")

    proc_params = {
        "scenario": scenario, "ref_date": None if ref_date is None else _iso(_day(ref_date)),
        "date_index": date_index, "date_window": date_window, "max_gap_days": max_gap_days,
        "gap_tolerance": gap_tolerance if cfg["s1"] else None,
        "max_span_days": max_span_days if cfg["n_dates"] > 1 else None,
        "max_cloud_fraction": max_cloud_fraction, "cloud_b02": cloud_b02,
        "s1_domain": s1_domain if cfg["s1"] else None, "min_pixels": min_pixels,
        "patches": None if patches is None else sorted(int(p) for p in patches),
        "version": 2,
    }
    train_folds = SPLITS[split]["train"]
    cov_params = {**proc_params, "estimator": estimator, "train_folds": list(train_folds)}
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir()
    proc = Stage(cache_dir, CACHE_NAME, "processed", proc_params)
    cov = Stage(cache_dir, CACHE_NAME, "covariance", cov_params)
    if force or not cov.done():
        if force or not proc.done():
            proc.reset()
            _process(files_dir, proc.path, proc_params)
            proc.mark_done()
        cov.reset()
        _covariances(proc.path, cov.path, estimator, train_folds)
        cov.mark_done()
    if keep == "keep_covariance" and proc.path.exists():
        shutil.rmtree(proc.path)
    processed = proc.path if proc.done() else None

    meta = pd.read_csv(cov.path / "parcels.csv")
    rep = json.loads((cov.path / "report.json").read_text())
    rep["covariance"] = json.loads((cov.path / "covariance_report.json").read_text())
    groups = meta["patch"].to_numpy()
    return Bunch(
        covariances=np.load(cov.path / "covariances.npy"),
        means=np.load(cov.path / "means.npy"),
        labels=meta["label"].to_numpy(dtype=np.int64),
        class_names=list(CLASSES), groups=groups, meta=meta,
        splits={split: make_splits(meta["fold"].to_numpy(), groups, split)},
        channels=rep["channels"], report=rep, processed_path=processed,
        covariance_path=cov.path, pixel_index=np.load(cov.path / "kept.npy"),
    )


# ------------------------------------------------------------------------ dataset
@DatasetManager.register_dataset("pastis_r")
class PastisRDataset(BaseDataset):
    """PASTIS-R parcels: covariance of the scenario channels over the parcel pixels.

    Each item is ``(x, y)``: ``x`` is the (p, p) float64 covariance, or a tuple
    with the (p,) mean and/or the (n, p) float32 raw pixels (``return_mean``,
    ``return_raw``; standardised channels); ``y`` the class index.

    Parameters
    ----------
    data_dir : path, optional
        See :func:`fetch_pastis_r`.
    scenario : {"A", "B", "C", "D", "E"}
    split : {"official", "cv1", ..., "cv5"}
        Split exposed in ``self.splits`` and used to fit the normalisations.
    subset : {"train", "val", "test"}, optional
        Keep only this part of the split (default: all samples).
    cache_dir, keep, force, verify, **date and channel options
        Passed to :func:`fetch_pastis_r`.
    task : {"fine", "coarse"}
        ``coarse`` uses :data:`COARSE_MAP` (6 groups: winter cereals, summer
        cereals, oilseeds, grassland and fodder, permanent crops, root crops and
        vegetables). ``drop_classes`` / ``merge_classes`` / ``classes`` take class names.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` options (estimator, normalize,
        balance, return_mean, return_raw, seed, ...).
    """

    OUTPUTS = ("cov", "mean", "raw")
    _FETCH_KEYS = ("ref_date", "date_index", "date_window", "max_gap_days", "gap_tolerance",
                   "max_span_days", "max_cloud_fraction", "cloud_b02", "s1_domain",
                   "min_pixels", "patches", "keep", "force", "verify")

    def __init__(self, data_dir=None, *, scenario="A", split="official", subset=None,
                 cache_dir=None, **kwargs):
        self._fetch_kwargs = {k: kwargs.pop(k) for k in self._FETCH_KEYS if k in kwargs}
        root, files_dir = _resolve(data_dir)
        if not (files_dir / "metadata.geojson").exists():  # downloaded into the cache
            root = Path(cache_dir) if cache_dir else get_cache_dir()
            root.mkdir(parents=True, exist_ok=True)
        super().__init__(str(root), **kwargs)
        if subset not in (None, "train", "val", "test"):
            raise ValueError("subset must be None, 'train', 'val' or 'test'")
        self.scenario, self.split, self.subset = scenario, split, subset
        self.cache_dir = cache_dir
        self._data_arg = data_dir
        self._load_data()

    def _load_data(self):
        b = fetch_pastis_r(self._data_arg, self.cache_dir, estimator=self.estimator,
                           scenario=self.scenario, split=self.split, **self._fetch_kwargs)
        self.report, self.channels = b.report, b.channels
        names = np.array([CLASSES[i] for i in b.labels], dtype=object)
        y, keep, class_names = apply_task(names, task=self.task, coarse_map=COARSE_MAP,
                                          drop_classes=self.drop_classes,
                                          merge_classes=self.merge_classes)
        idx = np.flatnonzero(keep)
        selected = self._limit_classes(class_names)
        if selected != class_names:
            sub = np.isin(np.asarray(class_names, dtype=object)[y], selected)
            idx, y = idx[sub], y[sub]
            remap = {class_names.index(c): i for i, c in enumerate(selected)}
            y = np.array([remap[v] for v in y.tolist()], dtype=np.int64)
            class_names = selected
        if self.max_samples_per_class:
            rng = np.random.default_rng(self.seed)
            take = []
            for c in np.unique(y):
                members = np.flatnonzero(y == c)
                n = min(len(members), self.max_samples_per_class)
                take.append(rng.choice(members, n, replace=False))
            take = np.sort(np.concatenate(take))
            idx, y = idx[take], y[take]

        parts = make_splits(b.meta["fold"].to_numpy()[idx], b.groups[idx], self.split)
        self.scale = 1.0
        if self.normalize and len(parts["train"]):  # fitted on the training part only
            self.scale = geometric_scale(b.covariances[idx[parts["train"]]])
        self.class_weights = None
        if self.balance in ("undersample", "oversample"):  # before the subset selection
            tr = parts["train"]
            parts["train"] = tr[balance_indices(y[tr], self.balance, seed=self.seed)]
        if self.balance == "weights":
            self.class_weights = class_weights(y[parts["train"]], n_classes=len(class_names))
        if self.subset is not None:
            idx, y = idx[parts[self.subset]], y[parts[self.subset]]
            parts = {k: (np.arange(len(idx)) if k == self.subset else np.zeros(0, int))
                     for k in parts}

        self.covariances = b.covariances[idx]
        self.means = b.means[idx]
        self.labels = y
        self.meta = b.meta.iloc[idx].reset_index(drop=True)
        self.groups = b.groups[idx]
        self.samples = list(range(len(idx)))
        self.classes = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.splits = {self.split: parts}
        self.covariances = apply_scale(self.covariances, self.scale)

        self._pixels = None
        if self.return_raw:
            if b.processed_path is None:
                raise ValueError("return_raw=True needs the parcel pixels: use keep='keep_full'")
            std = np.load(b.covariance_path / "standardization.npz")
            self._mu, self._sd = std["mean"], std["std"]
            self._offsets = np.load(b.processed_path / "offsets.npy")
            self._pixel_rows = b.pixel_index[idx]
            self._pixels = np.memmap(b.processed_path / "pixels.f32", dtype=np.float32,
                                     mode="r").reshape(-1, len(self._mu))
        if self.verbose:
            print(f"PASTIS-R scenario {self.scenario}: {len(self)} parcels, "
                  f"{len(self.classes)} classes, p = {self.covariances.shape[-1]}")

    def raw_pixels(self, i):
        """Standardised pixels (n, p) float32 of sample ``i``."""
        r = self._pixel_rows[i]
        X = self._pixels[self._offsets[r]:self._offsets[r + 1]]
        return ((X - self._mu) / self._sd).astype(np.float32)

    def __getitem__(self, i):
        cov = torch.from_numpy(self.covariances[i])
        mean = torch.from_numpy(self.means[i]) if self.return_mean else None
        raw = torch.from_numpy(self.raw_pixels(i)) if self.return_raw else None
        x = self._pack(cov, mean, raw)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[i])
