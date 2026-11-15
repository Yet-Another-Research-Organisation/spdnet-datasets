"""
The fetch cache: stages, ``.done`` markers, ``keep`` and replay
===============================================================

Every ``fetch_*`` function runs the same staged pipeline,
:func:`~spdnet_datasets.core.fetch.run_pipeline` (:doc:`/guide/fetch`)::

    raw (download)  ->  processed (preprocessing)  ->  covariance

Each stage is written under ``<cache_dir>/<name>/<stage>/<params hash>/`` and
marked complete by a ``.done`` file; a stage whose marker exists is not run
again. This tutorial runs the pipeline on a fake dataset whose "download" is a
local copy, so it needs neither the network nor ``/DATA``.
"""

# %%
# A fake dataset and its three stage functions
# --------------------------------------------
# The "remote archive" is a folder of 12 recordings of shape ``(64, 4)``. The
# stage functions record each call, so that we can see which ones run.

import json
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spdnet_datasets.core.estimators import estimate_covariances
from spdnet_datasets.core.fetch import run_pipeline

root = Path(tempfile.mkdtemp())
remote = root / "remote"
remote.mkdir()
rng = np.random.default_rng(0)
for i in range(12):
    np.save(remote / f"rec{i:02d}_class{i % 3}.npy", rng.standard_normal((64, 4)))

calls = []


def raw_fn(raw_dir):
    """Download the sources into ``raw_dir`` (here: a local copy)."""
    calls.append("raw")
    for f in sorted(remote.glob("*.npy")):
        shutil.copy(f, raw_dir / f.name)
    return raw_dir


def process_fn(source_dir, out_dir):
    """Preprocess every recording (here: standardise each channel)."""
    calls.append("processed")
    files = sorted(Path(source_dir).glob("*.npy"))
    X = np.stack([np.load(f) for f in files])
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    np.save(out_dir / "signals.npy", X)
    np.save(out_dir / "labels.npy", [int(f.stem[-1]) for f in files])


def covariance_fn(processed_dir, out_dir, estimator="scm"):
    """Estimate one covariance per recording."""
    calls.append("covariance")
    X = np.load(processed_dir / "signals.npy")
    np.save(out_dir / "covariances.npy", estimate_covariances(X, estimator))
    shutil.copy(processed_dir / "labels.npy", out_dir / "labels.npy")


def tree(path):
    """Files under ``path`` (relative to the temporary root), one line per folder."""
    folders = sorted({p.parent for p in Path(path).rglob("*") if p.is_file()})
    for folder in folders:
        names = sorted(p.name for p in folder.iterdir() if p.is_file())
        listing = ", ".join(names) if len(names) <= 4 else f"{len(names)} files"
        print(f"   {folder.relative_to(root)}/  {listing}")


def fetch(**kwargs):
    calls.clear()
    estimator = kwargs.pop("estimator", "scm")
    out = run_pipeline(
        "toy", params={"standardise": True}, covariance_params={"estimator": estimator},
        raw_fn=raw_fn, process_fn=process_fn,
        covariance_fn=lambda src, dst: covariance_fn(src, dst, estimator),
        cache_dir=root / "cache", **kwargs)
    print("stages run:", calls or "none (all cached)")
    return out


# %%
# First call: every stage runs
# ----------------------------
# The default policy for a downloaded dataset is ``keep="keep_processed"``: the
# raw download is deleted once the processed stage is complete.

out = fetch()
tree(root / "cache")
print("covariance_dir:", out.covariance_dir.relative_to(root))
print("raw_dir:", out.raw_dir)

# %%
# The ``.done`` marker holds the parameters hashed into the stage path, so a
# cache directory documents how it was built.

print(json.loads((out.covariance_dir / ".done").read_text()))

# %%
# Replay: nothing runs again
# --------------------------
# Same parameters, same hashes, markers present: ``run_pipeline`` returns the
# cached paths at once.

out = fetch()

# %%
# Changing only the covariance parameters
# ---------------------------------------
# ``covariance_params`` are hashed into the covariance stage only: changing the
# estimator re-estimates the covariances from the cached processed stage
# without downloading or preprocessing again. Both covariance stages now
# coexist in the cache.

out = fetch(estimator="ledoit_wolf")
tree(root / "cache" / "toy" / "covariance")

# %%
# The ``keep`` policy
# -------------------
# ``keep_covariance`` keeps only the covariances (smallest cache, but
# ``return_raw=True`` is no longer possible); ``keep_full`` also keeps the raw
# download. ``force=True`` reruns every stage.

out = fetch(keep="keep_covariance")
print("processed_dir after keep_covariance:", out.processed_dir)

out = fetch(keep="keep_full", force=True)
tree(root / "cache")

# %%
# What each policy leaves in the cache, after a fresh run:

policies = ["keep_covariance", "keep_processed", "keep_full"]
stages = ["raw", "processed", "covariance"]
kept = np.zeros((len(policies), len(stages)), bool)
for i, policy in enumerate(policies):
    out = fetch(keep=policy, force=True)
    kept[i] = [out.raw_dir is not None, out.processed_dir is not None,
               out.covariance_dir.exists()]

fig, ax = plt.subplots(figsize=(6, 2.4), layout="constrained")
for i in range(len(policies)):
    for j in range(len(stages)):
        ax.scatter(j, i, s=260, marker="s", color="#2a78d6" if kept[i, j] else "#e7e6e2")
        ax.annotate("kept" if kept[i, j] else "deleted", (j, i), xytext=(0, -17),
                    textcoords="offset points", ha="center", fontsize=8, color="#52514e")
ax.set_xticks(range(len(stages)), [f"{s} stage" for s in stages])
ax.set_yticks(range(len(policies)), [f'keep="{p}"' for p in policies])
ax.set_xlim(-0.6, len(stages) - 0.4)
ax.set_ylim(len(policies) - 0.4, -0.6)
ax.tick_params(length=0)
ax.spines[:].set_visible(False)
plt.show()

# %%
# Local datasets
# --------------
# Datasets read from the read-only copy under ``$SPDNET_DATA_ROOT`` pass
# ``raw_fn=None``: there is no raw stage, ``process_fn`` reads the copy itself
# and the default policy is ``keep_full``. Nothing is ever written next to the
# data; the cache lives in ``$SPDNET_CACHE_DIR`` (or ``cache_dir=``).

shutil.rmtree(root)
