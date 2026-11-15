"""
Satellite image time series: PASTIS-R crop parcels
==================================================

Sentinel-2 and Sentinel-1 time series of agricultural parcels in 18 crop
classes; each parcel becomes the covariance of its pixels (``p`` = 10 to 32
depending on the scenario). Full description: :doc:`/datasets/pastis_r`.

.. note::

   This example reads the dataset under ``$SPDNET_DATA_ROOT`` and is not run
   when the documentation is built. The figures are those of the dataset page.
"""

# %%
# Fetching the dataset
# --------------------
# Scenario ``C`` gives ``16 x 16`` covariances; ``return_mean`` and
# ``return_raw`` add the mean pixel and the ``(n, p)`` pixels of the parcel
# (:doc:`/guide/outputs`).

from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.sits.pastis_r import PastisRDataset

# sphinx_gallery_thumbnail_path = "_static/thumbnails/pastis_r.png"
ds = PastisRDataset(scenario="C", split="official", return_mean=True, return_raw=True)
(cov, mean, raw), y = ds[0]
print(len(ds), "parcels; cov", tuple(cov.shape), "mean", tuple(mean.shape),
      "raw", tuple(raw.shape), "-", ds.classes[y])

# %%
# Visualizing the dataset
# -----------------------
# Four parcels as returned by ``return_raw=True`` (one column per pixel, one
# row per standardised channel), then the class counts per split.
#
# .. raw:: html
#
#    <iframe src="../../_static/figures/pastis_r/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
#    <iframe src="../../_static/figures/pastis_r/class_hist.html" class="plotly-figure" loading="lazy"></iframe>

# %%
# Covariance matrices
# -------------------
# .. raw:: html
#
#    <iframe src="../../_static/figures/pastis_r/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
#    <iframe src="../../_static/figures/pastis_r/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>

# %%
# Splitting the data
# ------------------
# ``official`` uses the five spatially separated folds of PASTIS-R (folds 1-3
# train, 4 validation, 5 test); the group of a parcel is its patch. Raw pixels
# have a variable ``n``: use ``batch_size=1`` or a custom ``collate_fn`` when
# ``return_raw=True``.

parts = ds.splits["official"]
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert_no_leakage(ds.groups[parts[a]], ds.groups[parts[b]], what="patch")
print({k: len(v) for k, v in parts.items()})

# %%
# .. raw:: html
#
#    <iframe src="../../_static/figures/pastis_r/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
