"""
Radar micro-Doppler: CI4R human activities
==========================================

Micro-Doppler spectrograms of 11 activities recorded with a 77 GHz FMCW radar
(and a Xethru UWB sensor); each spectrogram becomes the ``64 x 64`` covariance
of its time columns. Participants define the folds. Full description:
:doc:`/datasets/ci4r`.

.. note::

   This example reads the dataset under ``$SPDNET_DATA_ROOT`` and is not run
   when the documentation is built. The figures are those of the dataset page.
"""

# %%
# Fetching the dataset
# --------------------
# ``mode="77GHz"`` selects the FMCW radar (``"xethru"`` the UWB sensor). The
# covariances are cached in ``$SPDNET_CACHE_DIR`` (:doc:`/guide/fetch`).

from spdnet_datasets import create_dataloaders
from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.radar_microdoppler.ci4r import CI4RDataset

# sphinx_gallery_thumbnail_path = "_static/thumbnails/ci4r.png"
ds = CI4RDataset(mode="77GHz", fold=0)
cov, y = ds[0]
print(len(ds), "spectrograms,", len(ds.classes), "activities; cov", tuple(cov.shape))

# %%
# Visualizing the dataset
# -----------------------
# A few spectrograms (time x Doppler bin), then the class counts per split.
#
# .. raw:: html
#
#    <iframe src="../../_static/figures/ci4r_77ghz/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
#    <iframe src="../../_static/figures/ci4r_77ghz/class_hist.html" class="plotly-figure" loading="lazy"></iframe>

# %%
# Covariance matrices
# -------------------
# Covariances between Doppler bins, and the spread of their eigenvalues.
#
# .. raw:: html
#
#    <iframe src="../../_static/figures/ci4r_77ghz/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
#    <iframe src="../../_static/figures/ci4r_77ghz/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>

# %%
# Splitting the data
# ------------------
# ``fold0`` ... ``fold3`` are a participant-grouped 4-fold (leave-one-subject-out
# in 77 GHz mode), without validation part.

for k in range(4):
    parts = ds.splits[f"fold{k}"]
    assert_no_leakage(ds.groups[parts["train"]], ds.groups[parts["test"]], what="participant")
    print(f"fold{k}: train {len(parts['train'])}, test {len(parts['test'])}")

train, val, test = create_dataloaders(ds, split="fold0", batch_size=16, num_workers=0)

# %%
# .. raw:: html
#
#    <iframe src="../../_static/figures/ci4r_77ghz/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
