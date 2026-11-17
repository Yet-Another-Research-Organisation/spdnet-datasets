"""
SAR: MSTAR vehicle chips
========================

``128 x 128`` X-band SAR chips of 10 military vehicles, the standard benchmark
of automatic target recognition. Like most SAR datasets of the package, MSTAR
provides raw chips or channel means, not covariances: choose exactly one of
``return_raw=True`` or ``return_mean=True`` (:doc:`/guide/outputs`). Full
description: :doc:`/datasets/mstar`.

.. note::

   This example reads the dataset under ``$SPDNET_DATA_ROOT`` and is not run
   when the documentation is built. The figures are those of the dataset page.
"""

# %%
# Fetching the dataset
# --------------------

from spdnet_datasets import create_dataloaders
from spdnet_datasets.sar.mstar import MSTARDataset

# sphinx_gallery_thumbnail_path = "_static/previews/mstar/thumb.png"
ds = MSTARDataset(source="jpg", crop=128, return_raw=True)
x, y = ds[0]
print(len(ds), "chips; x", tuple(x.shape), "-", ds.classes[y])

# %%
# Visualizing the dataset
# -----------------------
# A few chips, the distribution of their pixel values, and the class counts
# per split.
#
# .. dataset-figure:: mstar/raw_samples
#
# .. dataset-figure:: mstar/amplitude_hist
#
# .. dataset-figure:: mstar/class_hist

# %%
# Splitting the data
# ------------------
# ``official`` is the SOC protocol: chips at 17 degrees of depression for
# training (10 % kept for validation), 15 degrees for test. ``ds.groups`` holds
# file stems, which are not physical units, so a group check says nothing here.

parts = ds.splits["official"]
print({k: len(v) for k, v in parts.items()})
train, val, test = create_dataloaders(ds, split="official", batch_size=64, num_workers=0)

# %%
# .. dataset-figure:: mstar/split_schema
