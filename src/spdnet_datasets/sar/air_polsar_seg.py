"""AIR-PolSAR-Seg: GaoFen-3 C-band quad-pol amplitude patches over Hangzhou.

Reference: Wang Z. et al., *AIR-PolSAR-Seg: A Large-Scale Data Set for Terrain
Segmentation in Complex-Scene PolSAR Images*, IEEE JSTARS 15, 2022 (PDF in the
dataset folder). Manual download; data under ``<root>/SAR/AIR-PolSAR-Seg``::

    {train_set,test_set}/AIR-PolarSAR-Seg-<k>_{HH,HV,VH,VV}.tiff   # uint16 512 x 512
    {train_set,test_set}/AIR-PolarSAR-Seg-<k>_gt.png               # RGB labels

The patches overlap massively (1 241 pairs share identical pixels, 43 of the
150 official test patches overlap a training patch) and the file names
``1..150`` exist in both folders for different images. This module therefore
**rebuilds the mosaic** from the exact overlaps (hashes of 16-pixel row
segments, probed on a regular grid, vote for displacements; every candidate is
checked on its whole overlap and the four polarisations; the rebuilt mosaic is
checked against every patch), then cuts it into non-overlapping
``cell x cell`` cells: a cell is kept if it is fully covered, at least
``min_annotated`` of its pixels are annotated and its majority class holds
``purity`` of them (atlas: 57 358 cells of 32 x 32 px).

Classes (gt colours): housing (255,255,0), industrial (0,0,255), natural
(0,255,0), land use (255,0,0), water (0,255,255), other (255,255,255); black
(0,0,0) is unannotated.

Splits: ``"macro1024_fold{k}"`` (``"recommended"`` = fold 0) are
``StratifiedGroupKFold(5)`` folds on 1024 x 1024 px macro-blocks of the mosaic
(component, row // 1024, col // 1024), with a grouped validation part; a guard
band removes the training / validation cells touching a test cell. The
official per-image split is not provided (overlapping train and test images).

Outputs: raw or mean only (``OUTPUTS = ("raw", "mean")``): raw = ``(cell,
cell, 4)`` float32 ``log10(1 + A)`` of HH, HV, VH, VV (4.9 % of the amplitudes
are 0 and some saturate at 65535); mean = per-channel mean of the raw cell.
"""

from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BaseDataset
from ..core.config import dataset_path
from ..core.fetch import Bunch, run_pipeline, verify_local
from ..manager import DatasetManager
from ._common import check_split_groups, finalize_labels, group_kfold_splits

DOMAIN, NAME = "SAR", "AIR-PolSAR-Seg"
FILES = {}  # manual download
POLS = ("HH", "HV", "VH", "VV")
COLOURS = {(255, 255, 0): "housing", (0, 0, 255): "industrial", (0, 255, 0): "natural",
           (255, 0, 0): "land use", (0, 255, 255): "water", (255, 255, 255): "other"}
CLASSES = tuple(COLOURS.values())
SEG = 16  # row-segment length of the overlap hashes


def _read_patches(root):
    import tifffile
    from PIL import Image

    names, amps, labels = [], [], []
    for folder in ("train_set", "test_set"):
        for hh in sorted((root / folder).glob("*_HH.tiff")):
            stem = hh.name[:-len("_HH.tiff")]
            names.append(f"{folder.split('_')[0]}/{stem}")
            amps.append(np.stack([tifffile.imread(root / folder / f"{stem}_{p}.tiff") for p in POLS]))
            rgb = np.asarray(Image.open(root / folder / f"{stem}_gt.png").convert("RGB"))
            lab = np.zeros(rgb.shape[:2], np.uint8)
            for k, col in enumerate(COLOURS, start=1):
                lab[(rgb == col).all(-1)] = k
            labels.append(lab)
    return names, np.stack(amps), np.stack(labels)


def _segment_hash(x):
    """Polynomial hash of every ``SEG``-long row segment of ``x`` (..., W) -> (..., W - SEG + 1)."""
    x = x.astype(np.uint64)
    w = x.shape[-1] - SEG + 1
    h = np.zeros((*x.shape[:-1], w), np.uint64)
    p = np.uint64(1_000_003)
    with np.errstate(over="ignore"):
        for i in range(SEG):
            h = h * p + x[..., i:i + w]
    return h


def find_overlaps(amps, *, step=8):
    """Exact overlaps between patches: list of ``(a, b, dy, dx)`` with ``a[y, x] == b[y - dy, x - dx]``.

    The segments of every ``step``-th row of each patch are indexed; each patch
    is probed at every row and every ``step``-th column, so any overlap at
    least ``2 step`` px high and ``SEG + 2 step`` px wide gets two votes or
    more (deterministic). Candidates are checked on their whole overlap and the
    four polarisations.
    """
    n, _, H, W = amps.shape
    hh = amps[:, 0]
    index = _segment_hash(hh[:, ::step])
    flat = index.ravel()
    order = np.argsort(flat, kind="stable")
    keys = flat[order]
    w = W - SEG + 1
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(0, w, step), indexing="ij")
    votes = Counter()
    for a in range(n):
        ys, xs = grid_y.ravel(), grid_x.ravel()
        seg = hh[a, ys[:, None], xs[:, None] + np.arange(SEG)]
        informative = seg.max(1) != seg.min(1)  # constant segments (zeros...) match anywhere
        q = _segment_hash(seg)[:, 0]
        lo, hi = np.searchsorted(keys, q, "left"), np.searchsorted(keys, q, "right")
        for j in np.flatnonzero(informative & (hi > lo) & (hi - lo <= 20)):
            for pos in order[lo[j]:hi[j]]:
                b, r = divmod(int(pos), index.shape[1] * w)
                yb, xb = divmod(r, w)
                if b != a:
                    votes[(a, b, int(ys[j] - yb * step), int(xs[j] - xb))] += 1
    pairs = []
    for (a, b, dy, dx), v in votes.items():
        if v < 2 or a > b:
            continue
        ya0, ya1, xa0, xa1 = max(0, dy), min(H, H + dy), max(0, dx), min(W, W + dx)
        if ya1 <= ya0 or xa1 <= xa0:
            continue
        if np.array_equal(amps[a, :, ya0:ya1, xa0:xa1], amps[b, :, ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]):
            pairs.append((a, b, dy, dx))
    return pairs


def place_patches(n, pairs):
    """Component and top-left position of every patch from the overlaps (BFS)."""
    adj = [[] for _ in range(n)]
    for a, b, dy, dx in pairs:  # origin(b) = origin(a) + (dy, dx)
        adj[a].append((b, dy, dx))
        adj[b].append((a, -dy, -dx))
    comp, pos = np.full(n, -1), np.zeros((n, 2), np.int64)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = c
        queue = deque([s])
        while queue:
            a = queue.popleft()
            for b, dy, dx in adj[a]:
                if comp[b] < 0:
                    comp[b], pos[b] = c, pos[a] + (dy, dx)
                    queue.append(b)
                elif tuple(pos[b]) != tuple(pos[a] + (dy, dx)):
                    raise ValueError(f"Inconsistent placement of patches {a} and {b}")
        c += 1
    for k in range(c):  # positions >= 0 inside each component
        pos[comp == k] -= pos[comp == k].min(0)
    return comp, pos


def _process(root, out_dir, *, cell, purity, min_annotated, macro):
    names, amps, labels = _read_patches(root)
    pairs = find_overlaps(amps)
    comp, pos = place_patches(len(names), pairs)
    S = amps.shape[-1]
    cells, rows, conflicts = [], [], 0
    for k in np.unique(comp):
        members = np.flatnonzero(comp == k)
        Hc, Wc = pos[members].max(0) + S
        mosaic = np.zeros((4, Hc, Wc), amps.dtype)
        lab = np.zeros((Hc, Wc), np.uint8)
        cover = np.zeros((Hc, Wc), bool)
        for i in members:
            y, x = pos[i]
            mosaic[:, y:y + S, x:x + S] = amps[i]
            win = lab[y:y + S, x:x + S]
            conflicts += int(((win > 0) & (labels[i] > 0) & (win != labels[i])).sum())
            lab[y:y + S, x:x + S] = np.where(labels[i] > 0, labels[i], win)
            cover[y:y + S, x:x + S] = True
        for i in members:  # every patch must be read back unchanged from the mosaic
            y, x = pos[i]
            if not np.array_equal(mosaic[:, y:y + S, x:x + S], amps[i]):
                raise ValueError(f"Patch {names[i]} disagrees with the rebuilt mosaic")
        for r in range(Hc // cell):
            for c in range(Wc // cell):
                sl = (slice(r * cell, (r + 1) * cell), slice(c * cell, (c + 1) * cell))
                if not cover[sl].all():
                    continue
                w = lab[sl]
                counts = np.bincount(w[w > 0], minlength=len(CLASSES) + 1)
                if counts.sum() < min_annotated * cell * cell or counts.max() < purity * counts.sum():
                    continue
                cells.append(np.moveaxis(mosaic[:, sl[0], sl[1]], 0, -1))
                rows.append({"component": int(k), "row": r * cell, "col": c * cell,
                             "label": CLASSES[int(counts.argmax()) - 1],
                             "block": f"{k}_{r * cell // macro}_{c * cell // macro}"})
    out_dir = Path(out_dir)
    np.save(out_dir / "cells.npy", np.stack(cells))
    pd.DataFrame(rows).to_csv(out_dir / "meta.csv", index=False)
    pd.DataFrame({"name": names, "component": comp, "py": pos[:, 0], "px": pos[:, 1]}).to_csv(
        out_dir / "positions.csv", index=False)
    pd.DataFrame(pairs, columns=["a", "b", "dy", "dx"]).to_csv(out_dir / "overlaps.csv", index=False)
    (out_dir / "label_conflicts.txt").write_text(str(conflicts))


def _guard(meta, keep, ref, gap):
    if not len(keep) or not len(ref):
        return keep
    from scipy.spatial import cKDTree

    xy = meta[["row", "col"]].to_numpy(float) + meta["component"].to_numpy()[:, None] * 1e7
    d, _ = cKDTree(xy[ref]).query(xy[keep], p=np.inf)
    return keep[d >= gap]


def fetch_air_polsar_seg(data_dir=None, cache_dir=None, keep=None, force=False, verify="size", *,
                         cell=32, purity=0.75, min_annotated=0.9, macro=1024, seed=0):
    """Rebuild the AIR-PolSAR-Seg mosaic and cut it into cells (cached).

    Returns
    -------
    Bunch with ``meta`` (component, row, col, label, block), ``positions``
    (patch placement in the mosaic), ``overlaps``, ``label_conflicts`` (pixels
    labelled differently by two overlapping patches), ``splits`` and
    ``processed_dir`` (``cells.npy``, uint16 amplitudes ``(N, cell, cell, 4)``).
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    verify_local(root, verify=verify, required=("train_set", "test_set"))
    opts = {"cell": cell, "purity": purity, "min_annotated": min_annotated, "macro": macro}
    params = dict(opts, source=str(root.resolve()), version=3)  # bump when the processing changes

    def covariance_fn(proc, out):  # raw/mean dataset: the stage only records the cell metadata
        for f in ("meta.csv", "positions.csv", "overlaps.csv", "label_conflicts.txt"):
            (Path(out) / f).write_bytes((Path(proc) / f).read_bytes())

    if keep == "keep_covariance":
        raise ValueError("AIR-PolSAR-Seg has no covariance: keep the processed cells")
    res = run_pipeline("air_polsar_seg", params=params, process_fn=lambda _s, out: _process(root, out, **opts),
                       covariance_fn=covariance_fn, cache_dir=cache_dir, keep=keep, force=force)
    cdir = Path(res.covariance_dir)
    meta = pd.read_csv(cdir / "meta.csv", dtype={"block": str})
    splits = group_kfold_splits(meta["label"].to_numpy(), meta["block"].to_numpy(), seed=seed,
                                prefix="macro1024_")
    splits = {name.replace("macro1024_5", "macro1024"): {
        "train": _guard(meta, p["train"], p["test"], 2 * cell),
        "val": _guard(meta, p["val"], p["test"], 2 * cell), "test": p["test"]} for name, p in splits.items()}
    splits["recommended"] = splits["macro1024_fold0"]
    return Bunch(meta=meta, positions=pd.read_csv(cdir / "positions.csv"),
                 overlaps=pd.read_csv(cdir / "overlaps.csv"), splits=splits,
                 label_conflicts=int((cdir / "label_conflicts.txt").read_text()),
                 processed_dir=res.processed_dir, root=root)


@DatasetManager.register_dataset("air_polsar_seg")
class AIRPolSARSegDataset(BaseDataset):
    """AIR-PolSAR-Seg mosaic cells (raw or mean only), macro-block folds.

    Parameters
    ----------
    data_dir : path, optional
        Default ``<root>/SAR/AIR-PolSAR-Seg``.
    cache_dir, keep, cell, purity, min_annotated, macro
        See :func:`fetch_air_polsar_seg`.
    **kwargs
        :class:`~spdnet_datasets.base.BaseDataset` arguments; exactly one of
        ``return_raw`` / ``return_mean``.
    """

    OUTPUTS = ("raw", "mean")

    def __init__(self, data_dir=None, *, cache_dir=None, keep=None, cell=32, purity=0.75,
                 min_annotated=0.9, macro=1024, verify="size", **kwargs):
        root = dataset_path(DOMAIN, NAME, data_dir)
        self._fetch = {"cache_dir": cache_dir, "keep": keep, "cell": cell, "purity": purity,
                       "min_annotated": min_annotated, "macro": macro, "verify": verify}
        super().__init__(data_dir=str(root), **kwargs)
        self._load_data()

    def _load_data(self):
        b = fetch_air_polsar_seg(self.data_dir, seed=self.seed, **self._fetch)
        keep = finalize_labels(self, b.meta["label"].to_numpy(), b.meta["block"].to_numpy(), b.splits)
        check_split_groups(self.splits, self.groups)
        self.meta = b.meta.iloc[keep].reset_index(drop=True)
        self.positions, self.overlaps, self.label_conflicts = b.positions, b.overlaps, b.label_conflicts
        self._cells = np.load(Path(b.processed_dir) / "cells.npy", mmap_mode="r")
        self._keep = keep
        self.samples = list(range(len(keep)))
        if self.verbose:
            print(f"AIR-PolSAR-Seg: {len(keep)} cells, {b.positions['component'].nunique()} mosaic "
                  f"components, {len(b.overlaps)} overlapping patch pairs")

    def __getitem__(self, idx):
        raw = np.log10(1.0 + np.asarray(self._cells[self._keep[idx]], dtype=np.float32))
        x = self._pack(raw=torch.from_numpy(raw) if self.return_raw else None,
                       mean=torch.from_numpy(raw.reshape(-1, 4).mean(0, dtype=np.float64))
                       if self.return_mean else None)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])
