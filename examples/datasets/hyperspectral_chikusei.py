"""
Hyperspectral: Chikusei land cover
==================================

An airborne hyperspectral scene (128 bands, 19 land-cover classes) cut into
``15 x 15`` windows; each window becomes the ``64 x 64`` covariance of its
pixels (one band in two). Full description: :doc:`/datasets/chikusei`.

.. note::

   This example reads the dataset under ``$SPDNET_DATA_ROOT`` and is not run
   when the documentation is built. The figures are those of the dataset page.
"""

# %%
# Fetching the dataset
# --------------------
# ``n = 225`` pixels for ``p = 64`` bands is close to the ``n >= 2p`` floor:
# Ledoit-Wolf shrinkage, or ``window=23``, gives better-conditioned matrices
# (:doc:`/guide/covariance`).

from spdnet_datasets import create_dataloaders
from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.hyperspectral.chikusei import ChikuseiDataset

# sphinx_gallery_thumbnail_path = "_static/thumbnails/chikusei.png"
ds = ChikuseiDataset(estimator="ledoit_wolf")
cov, y = ds[0]
print(len(ds), "windows,", len(ds.classes), "classes; cov", tuple(cov.shape))

# %%
# Visualizing the dataset
# -----------------------
# The band-mean image of four windows, then the class counts per split.
#
# .. raw:: html
#
#    <iframe src="../../_static/figures/chikusei/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
#    <iframe src="../../_static/figures/chikusei/class_hist.html" class="plotly-figure" loading="lazy"></iframe>

# %%
# Covariance matrices
# -------------------
# A few covariances (shown as correlations) and the spread of their eigenvalues.
#
# .. raw:: html
#
#    <iframe src="../../_static/figures/chikusei/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
#    <iframe src="../../_static/figures/chikusei/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>

# %%
# Splitting the data
# ------------------
# ``spatial`` (= ``recommended``) assigns 320-pixel blocks to train, validation
# or test and discards the windows of the buffer zone between them, so that no
# two windows of different splits share a pixel (see
# :doc:`/auto_examples/tutorials/plot_03_leakage_free_splits`).

parts = ds.splits["spatial"]
assert_no_leakage(ds.groups[parts["train"]], ds.groups[parts["test"]], what="block")
print({k: len(v) for k, v in parts.items()})

train, val, test = create_dataloaders(ds, split="spatial", batch_size=64, num_workers=0)

# %%
# .. raw:: html
#
#    <iframe src="../../_static/figures/chikusei/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
