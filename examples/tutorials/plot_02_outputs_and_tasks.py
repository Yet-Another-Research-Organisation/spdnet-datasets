"""
Outputs and tasks on a toy dataset
==================================

What does ``dataset[i]`` contain, and how do ``task``, ``drop_classes``,
``merge_classes`` and ``balance`` change the labels? This tutorial builds a tiny
fake dataset on top of :class:`~spdnet_datasets.base.BaseDataset` that follows
the same contract as the real loaders (:doc:`/guide/outputs`,
:doc:`/guide/tasks`), so it runs without any data.
"""

# %%
# A toy dataset that follows the contract
# ---------------------------------------
# Each sample is a ``(T, p)`` sequence recorded in one of six sessions (its
# group). The loader estimates its covariance and channel mean, selects the
# task, builds a group split, balances the training part and fits the
# geometric normalisation on it, exactly as the real loaders do.

import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

from spdnet_datasets import create_dataloaders
from spdnet_datasets.base import BaseDataset
from spdnet_datasets.core.estimators import channel_mean, estimate_covariances
from spdnet_datasets.core.normalize import geometric_scale
from spdnet_datasets.core.splits import assert_no_leakage, group_split
from spdnet_datasets.core.tasks import apply_task, balance_indices, class_weights

SPECIES = np.array(["cat"] * 60 + ["dog"] * 40 + ["bird"] * 25 + ["frog"] * 15)
COARSE_MAP = {"cat": "mammal", "dog": "mammal", "bird": "bird", "frog": "amphibian"}


class ToyDataset(BaseDataset):
    """140 sequences of shape (T=32, p=4), 4 classes, 6 sessions."""

    COARSE_MAP = COARSE_MAP

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir=data_dir, verbose=False, **kwargs)
        self._load_data()

    def _load_data(self):
        rng = np.random.default_rng(0)
        n, T, p = len(SPECIES), 32, 4
        sessions = rng.integers(0, 6, size=n)
        scale = 1 + np.searchsorted(sorted(COARSE_MAP), SPECIES)  # class-dependent power
        raw = rng.standard_normal((n, T, p)) * scale[:, None, None] + scale[:, None, None]

        y, keep, self.classes = apply_task(
            SPECIES, task=self.task, coarse_map=self.COARSE_MAP,
            drop_classes=self.drop_classes, merge_classes=self.merge_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self._raw = raw[keep].astype(np.float32)
        self._covs = estimate_covariances(self._raw, self.estimator)
        self._means = np.stack([channel_mean(x) for x in self._raw])
        self.labels, self.groups = y, sessions[keep]
        self.samples = list(range(len(y)))

        parts = group_split(y, self.groups, val_size=0.2, test_size=0.2, seed=self.seed)
        self.class_weights = None
        if self.balance == "weights":
            self.class_weights = class_weights(y[parts["train"]], len(self.classes))
        elif self.balance is not None:
            parts["train"] = parts["train"][balance_indices(y[parts["train"]], self.balance)]
        self.splits = {"official": parts}
        self.scale = geometric_scale(self._covs[parts["train"]]) if self.normalize else 1.0

    def __getitem__(self, i):
        root = float(np.sqrt(self.scale))
        cov = torch.from_numpy(self._covs[i] * self.scale)
        mean = torch.from_numpy(self._means[i] * root) if self.return_mean else None
        raw = torch.from_numpy(self._raw[i] * root) if self.return_raw else None
        return self._pack(cov, mean, raw), int(self.labels[i])


data_dir = tempfile.mkdtemp()  # BaseDataset checks that data_dir exists
ds = ToyDataset(data_dir)
print(len(ds), "samples,", ds.get_num_classes(), "classes:", ds.classes)

# %%
# Outputs: ``cov``, ``(cov, mean)``, ``(cov, raw)``, ``(cov, mean, raw)``
# -----------------------------------------------------------------------
# By default a sample is its covariance. ``return_mean`` and ``return_raw``
# append the channel mean and the preprocessed data, in this fixed order.


def describe(x):
    if isinstance(x, tuple):
        return "(" + ", ".join(describe(t) for t in x) + ")"
    return f"{tuple(x.shape)} {str(x.dtype).removeprefix('torch.')}"


for flags in [{}, {"return_mean": True}, {"return_raw": True},
              {"return_mean": True, "return_raw": True}]:
    x, y = ToyDataset(data_dir, **flags)[0]
    print(f"{str(flags):45s} x = {describe(x)}")

# %%
# The default collate keeps the tuple, so a batch unpacks the same way. Use
# ``split="official"`` (the dataset's leakage-free split) rather than a random
# split: the toy dataset has groups, and a sample-level split would warn.

ds = ToyDataset(data_dir, return_mean=True)
train, val, test = create_dataloaders(ds, split="official", batch_size=16, num_workers=0,
                                      pin_memory=False, persistent_workers=False,
                                      verbose=False)
(cov, mean), y = next(iter(train))
print("batch:", tuple(cov.shape), tuple(mean.shape), tuple(y.shape))
parts = ds.splits["official"]
assert_no_leakage(ds.groups[parts["train"]], ds.groups[parts["test"]], what="session")
print("no session shared between train and test")

# %%
# Normalisation
# -------------
# With ``normalize=True`` (the default) covariances are multiplied by a scalar
# fitted on the training split, so that the geometric mean of the training
# eigenvalues is 1; means and raw data are multiplied by its square root.

train_covs = torch.stack([ds[i][0][0] for i in parts["train"]]).numpy()
print(f"scale s = {ds.scale:.4f}; geometric mean of the training eigenvalues = "
      f"{np.exp(np.log(np.linalg.eigvalsh(train_covs)).mean()):.3f}")

# %%
# Tasks: fine, coarse, dropping and merging classes
# -------------------------------------------------
# ``task="coarse"`` regroups classes with the dataset's ``COARSE_MAP``;
# ``drop_classes`` removes classes first, then ``merge_classes`` renames.

variants = {
    "fine": {},
    "coarse": {"task": "coarse"},
    "drop frog, merge\ncat+dog -> pet": {"drop_classes": ("frog",),
                                         "merge_classes": {"cat": "pet", "dog": "pet"}},
}
counts = {}
for name, kwargs in variants.items():
    d = ToyDataset(data_dir, **kwargs)
    counts[name] = dict(zip(d.classes, np.bincount(d.labels).tolist(), strict=True))
    print(f"{name.replace(chr(10), ' '):34s} {counts[name]}")

# %%
# Balancing the training split
# ----------------------------
# ``balance`` only touches the training part of the split: ``"undersample"``
# and ``"oversample"`` resample it, ``"weights"`` leaves it and exposes
# inverse-frequency ``class_weights`` for the loss.

for mode in [None, "undersample", "oversample"]:
    d = ToyDataset(data_dir, balance=mode)
    tr = d.splits["official"]["train"]
    counts[f"train, balance={mode}"] = dict(zip(d.classes, np.bincount(d.labels[tr]).tolist(),
                                                strict=True))
    print(f"balance={mode!s:12s} train counts {np.bincount(d.labels[tr]).tolist()}")
print("class weights:", np.round(ToyDataset(data_dir, balance="weights").class_weights, 2))

# %%
# Class counts of each variant (small multiples, one panel per selection).

fig, axes = plt.subplots(1, len(counts), figsize=(12, 2.8), layout="constrained")
for ax, (name, c) in zip(axes, counts.items(), strict=True):
    ax.barh(list(c), list(c.values()), color="#2a78d6", height=0.6)
    ax.set_title(name, loc="left", fontsize=9)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="#e7e6e2", lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_xlabel("samples")
plt.show()
