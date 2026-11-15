"""NTU RGB+D 120 covariance series (per joint and per time window) on a small subset of real files."""

import glob
import os

import numpy as np
import pytest

from spdnet_datasets.core.config import dataset_path
from spdnet_datasets.real.series_ntu import (
    NTU120_TRAIN_SUBJECTS,
    ntu_frame_series,
    ntu_joint_series,
)

ROOT = str(dataset_path("SKELETON", "NTU_RGBD_120") / "npy_skeletons_by_class")
pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not os.path.isdir(ROOT), reason="NTU RGB+D 120 not available")]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Two classes x 10 files (one per subject, training and test subjects mixed), linked into a temporary root."""
    path = tmp_path_factory.mktemp("ntu")
    for c in (1, 2):
        os.makedirs(path / f"A{c:03d}")
        seen = set()
        for f in sorted(glob.glob(os.path.join(ROOT, f"A{c:03d}", "*.npy"))):
            subject = int(os.path.basename(f)[9:12])
            if subject not in seen and len(seen) < 10:
                seen.add(subject)
                os.symlink(f, path / f"A{c:03d}" / os.path.basename(f))
    return str(path)


@pytest.fixture(scope="module")
def both(root):
    kwargs = {"root": root, "n_classes": 2, "max_per_class": None, "val_fraction": 0.5}
    return ntu_joint_series(**kwargs), ntu_frame_series(n_windows=12, **kwargs)


def is_spd(x):
    return np.allclose(x, np.swapaxes(x, -1, -2)) and np.linalg.eigvalsh(x).min() > 0


def test_shapes_and_spd(both):
    (joint, labels, n_joint, extra), (frame, labels_f, n_frame, _) = both
    n = len(labels)
    assert n >= 15 and joint.shape == (n, 24, 3, 3) and frame.shape == (n, 12, 3, 3)
    assert joint.dtype == frame.dtype == np.float64
    assert is_spd(joint) and is_spd(frame)
    assert (n_joint >= 24).all() and (n_frame >= 2 * 24).all()  # n >= 2p (p = 3)


def test_same_samples(both):
    (_, labels, n_joint, extra), (_, labels_f, _, extra_f) = both
    assert np.array_equal(labels, labels_f) and set(labels.tolist()) == {0, 1}
    for key in extra:
        assert np.array_equal(extra[key], extra_f[key])


def test_split_by_subject(both):
    (_, _, _, extra), _ = both
    split, subject = extra["split"], extra["subject"]
    assert set(split.tolist()) == {0, 1, 2}
    groups = [set(subject[split == k].tolist()) for k in range(3)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert groups[2].isdisjoint(NTU120_TRAIN_SUBJECTS) and (groups[0] | groups[1]) <= set(NTU120_TRAIN_SUBJECTS)
