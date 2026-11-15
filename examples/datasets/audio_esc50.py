"""
Audio: ESC-50 environmental sounds
==================================

2 000 five-second clips in 50 classes; each clip becomes the ``20 x 20``
covariance of its MFCC frames (or ``64 x 64`` log-mel). This example loads the
dataset, looks at its classes and covariances and checks the official folds.
Full description: :doc:`/datasets/esc50`.

.. note::

   This example reads the dataset under ``$SPDNET_DATA_ROOT`` and is not run
   when the documentation is built. The figures are those of the dataset page.
"""

# %%
# Fetching the dataset
# --------------------
# The first call extracts the MFCC frames and estimates the covariances into
# ``$SPDNET_CACHE_DIR``; later calls read the cache (:doc:`/guide/fetch`).
# Four silent clips are excluded because they have fewer than ``2p`` frames.

from spdnet_datasets import create_dataloaders
from spdnet_datasets.audio.esc50 import ESC50Dataset

# sphinx_gallery_thumbnail_path = "_static/thumbnails/esc50.png"
ds = ESC50Dataset(split="official", fold=1, return_raw=True)
print(len(ds), "clips,", ds.num_classes, "classes")
(cov, raw), y = ds[0]
print("cov", tuple(cov.shape), "raw (frames x p)", tuple(raw.shape), "class", ds.classes[y])

# %%
# Visualizing the dataset
# -----------------------
# The 20 MFCC of the non-silent frames of a few clips (coefficient x frame),
# then the class counts per split.
#
# .. raw:: html
#
#    <iframe src="../../_static/figures/esc50/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
#    <iframe src="../../_static/figures/esc50/class_hist.html" class="plotly-figure" loading="lazy"></iframe>

# %%
# Covariance matrices
# -------------------
# A few covariances (shown as correlations) and the spread of their
# eigenvalues (median and 10-90 % band of the log10 eigenvalues).
#
# .. raw:: html
#
#    <iframe src="../../_static/figures/esc50/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
#    <iframe src="../../_static/figures/esc50/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>

# %%
# Splitting the data
# ------------------
# ``official`` is the 5-fold protocol of the paper: test = fold ``fold``,
# validation = the next fold, train = the three others. Groups are the source
# Freesound recordings; four of them span two folds in the official protocol,
# which the check below reports.

parts = ds.splits["official"]
shared = set(ds.groups[parts["train"]]) & set(ds.groups[parts["test"]])
print({k: len(v) for k, v in parts.items()}, "-", len(shared), "recordings in train and test")

train, val, test = create_dataloaders(ds, split="official", batch_size=32, num_workers=0)

# %%
# .. raw:: html
#
#    <iframe src="../../_static/figures/esc50/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
