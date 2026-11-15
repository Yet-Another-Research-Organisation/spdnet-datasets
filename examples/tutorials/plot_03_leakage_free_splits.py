"""
Leakage-free splits: groups, spatial blocks and the checks behind them
======================================================================

Samples cut from the same recording, subject, parcel or image region are
correlated. If they fall on both sides of a split, test accuracy measures
memorisation, not generalisation. :mod:`spdnet_datasets.core.splits` provides
the two splitters used by the loaders and the assertions that the tests run on
every dataset (:doc:`/guide/splits`). This tutorial uses synthetic indices only.
"""

# %%
# Group split
# -----------
# 600 samples from 30 subjects, 3 classes. A sample-level random split puts
# most subjects on both sides; :func:`~spdnet_datasets.core.splits.group_split`
# never separates a subject and keeps the class proportions close to the global
# ones.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from spdnet_datasets.core.splits import (
    assert_no_leakage,
    assert_no_window_overlap,
    group_split,
    spatial_block_split,
)

rng = np.random.default_rng(0)
subjects = np.repeat(np.arange(30), 20)
y = rng.integers(0, 3, size=len(subjects))

perm = rng.permutation(len(y))
random_train, random_test = perm[:480], perm[480:]
try:
    assert_no_leakage(subjects[random_train], subjects[random_test], what="subject")
except AssertionError as err:
    print("random split:", err)

parts = group_split(y, subjects, val_size=0.1, test_size=0.2, seed=0)
assert_no_leakage(subjects[parts["train"]], subjects[parts["test"]], what="subject")
assert_no_leakage(subjects[parts["train"]], subjects[parts["val"]], what="subject")
for name, idx in parts.items():
    share = np.bincount(y[idx], minlength=3) / len(idx)
    print(f"{name:5s} {len(idx):4d} samples, {len(np.unique(subjects[idx])):2d} subjects, "
          f"class shares {np.round(share, 2)}")

# %%
# Spatial block split
# -------------------
# Patch datasets (hyperspectral scenes, SAR images) cut overlapping windows
# from one large image. Neighbouring windows share pixels, so the image is cut
# into square blocks that serve as groups; test windows closer than one window
# side to a training window are then dropped, so that no test window overlaps
# a training window.

H = W = 240
window, stride, block = 16, 8, 60
rows, cols = np.meshgrid(np.arange(0, H - window + 1, stride),
                         np.arange(0, W - window + 1, stride), indexing="ij")
coords = np.column_stack([rows.ravel(), cols.ravel()])
labels = ((coords[:, 0] // 80) + (coords[:, 1] // 100)) % 3  # smooth land-cover map
boxes = np.column_stack([coords, coords + window])

naive = spatial_block_split(coords, block=block, y=labels, n_splits=5, fold=0)
try:
    assert_no_window_overlap(boxes[naive["train"]], boxes[naive["test"]])
except AssertionError as err:
    print("blocks without buffer:", err)

split = spatial_block_split(coords, block=block, window=window, y=labels, n_splits=5,
                            fold=0)
assert_no_window_overlap(boxes[split["train"]], boxes[split["test"]])
assert_no_leakage(split["groups"][split["train"]], split["groups"][split["test"]],
                  what="block")
dropped = np.setdiff1d(naive["test"], split["test"])
print(f"{len(coords)} windows: train {len(split['train'])}, test {len(split['test'])}, "
      f"{len(dropped)} test windows dropped by the buffer")

# %%
# Figure: windows of fold 0. Test blocks are surrounded by a ring of dropped
# windows, one window side wide.

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
centres = coords + window / 2
kinds = [("train", split["train"], "#2a78d6"), ("test", split["test"], "#eb6834"),
         ("dropped (overlap)", dropped, "#9a9893")]
for name, idx, color in kinds:
    axes[0].scatter(centres[idx, 1], centres[idx, 0], s=9, color=color, label=name)
for k in range(0, H + 1, block):
    axes[0].axhline(k, color="#52514e", lw=0.6)
    axes[0].axvline(k, color="#52514e", lw=0.6)
r0, c0 = coords[split["test"][0]]
axes[0].add_patch(Rectangle((c0, r0), window, window, fill=False, ec="#0b0b0b", lw=1.5))
axes[0].set_title("fold 0: blocks of 60 px, windows of 16 px", loc="left", fontsize=10)
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False,
               fontsize=9)

classes = ["#cde2fb", "#5a9be6", "#0d366b"]
for c, color in enumerate(classes):
    m = labels == c
    axes[1].scatter(centres[m, 1], centres[m, 0], s=9, color=color, label=f"class {c}")
axes[1].set_title("labels used to stratify the folds", loc="left", fontsize=10)
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False,
               fontsize=9)
for ax in axes:
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# %%
# In the loaders
# --------------
# Datasets store these splits in ``ds.splits`` (``"official"``,
# ``"recommended"``, ``"spatial"``, ``"fold0"``...) together with ``ds.groups``,
# and :func:`~spdnet_datasets.create_dataloaders` uses them with
# ``split="<name>"``. The test suite runs the same assertions on every
# registered dataset.
