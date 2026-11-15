"""
Covariance series from NTU RGB+D 120 skeletons (data atlas, chapter 29 "Séries de covariances"): the same
skeletons tokenised two ways, to test whether an advantage of the SPD model depends on positions having a
fixed identity.

- :func:`ntu_joint_series`: one 3 x 3 token per joint (24 joints after centring on the root joint 0 and dropping
  it), covariance of the joint trajectory over all frames: positions = joint index (identified).
- :func:`ntu_frame_series`: ``n_windows`` equal consecutive windows of frames, one 3 x 3 token per window,
  covariance of all joint positions in the window (24 joints x frames): positions = time slots.

Both return the same samples in the same order: single-person actions ``NTU_CLASSES`` (A001-A010, the first daily
actions; drink / eat / brush teeth / brush hair are close to each other), body 0 only, sequences of at least
``min_frames`` frames, cross-subject protocol (official NTU-120 training subjects) with a validation split of
20 % of the training subjects, at most ``max_per_class`` sequences per class and split (random, seeded).
"""

import glob
import os

import numpy as np

from ..estimator.covariance import EstimateCovariance

NTU_CLASSES = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down",
               "stand up", "clapping"]  # A001-A010
NTU120_TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49,
                         50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93,
                         94, 95, 97, 98, 100, 103]  # cross-subject protocol (Liu et al., 2019)


def ntu_skeletons(
    root: str = "/DATA/NTU_RGBD_120/npy_skeletons_by_class",
    n_classes: int = 10,
    max_per_class: int | None = 300,
    min_frames: int = 24,
    val_fraction: float = 0.2,
    seed: int = 0,
    centre: bool = True,
    classes: tuple[int, ...] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Centred skeletons (T, 24, 3) (root joint subtracted then dropped), labels, {"split": 0 train / 1 val /
    2 test, "subject", "camera"}."""
    # ``classes``: action numbers Axxx (labels = their rank); default: the first ``n_classes`` actions
    actions = list(classes) if classes is not None else list(range(1, n_classes + 1))
    paths = [(label, path) for label, action in enumerate(actions)
             for path in sorted(glob.glob(os.path.join(root, f"A{action:03d}", "*.npy")))]
    subjects = [int(os.path.basename(path)[9:12]) for _, path in paths]  # SsssCcccPpppRrrrAaaa.npy
    # validation = a fraction of the training subjects present (A001-A060 are only filmed with subjects 1-40)
    train = sorted(set(subjects) & set(NTU120_TRAIN_SUBJECTS))
    rng = np.random.default_rng(seed)
    val_subjects = set(rng.choice(train, round(val_fraction * len(train)), replace=False).tolist())
    candidates = []  # (path, label, split, subject, camera)
    for (label, path), subject in zip(paths, subjects):
        split = (1 if subject in val_subjects else 0) if subject in NTU120_TRAIN_SUBJECTS else 2
        candidates.append((path, label, split, subject, int(os.path.basename(path)[5:8])))
    kept, taken = [], {}
    for i in rng.permutation(len(candidates)):  # at most max_per_class valid sequences per (class, split)
        path, label, split, subject, camera = candidates[i]
        if max_per_class is not None and taken.get((label, split), 0) >= max_per_class:
            continue
        record = np.load(path, allow_pickle=True).item()
        if "skel_body0" not in record:  # files without a body (atlas: 356 in the whole dataset)
            continue
        x = np.asarray(record["skel_body0"], np.float64)
        x = x[np.any(x != 0, axis=(1, 2))]  # frames where the body was lost
        if len(x) < min_frames:
            continue
        taken[(label, split)] = taken.get((label, split), 0) + 1
        kept.append((path, (x - x[:, :1])[:, 1:] if centre else x, label, split, subject, camera))
    kept.sort(key=lambda k: k[0])
    skeletons = [k[1] for k in kept]
    labels, split, subjects, cameras = (np.array([k[i] for k in kept]) for i in range(2, 6))
    return skeletons, labels, {"split": split, "subject": subjects, "camera": cameras}


def load(covariances: np.ndarray, value: float = 1e-6) -> np.ndarray:
    """Diagonal loading C + value * tr(C) / p * I of a stack of covariances (..., p, p)."""
    p = covariances.shape[-1]
    trace = np.trace(covariances, axis1=-2, axis2=-1)[..., None, None] / p
    return covariances + value * trace * np.eye(p)


def ntu_joint_series(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(n, 24, 3, 3) covariances of each joint trajectory over all frames, labels, frames per sequence (n
    observations of every token), {"split", "subject", "camera"}. Arguments of :func:`ntu_skeletons`."""
    skeletons, labels, extra = ntu_skeletons(**kwargs)
    estimator = EstimateCovariance("scm", remove_mean=True, exclude_zero=False)
    data = np.stack([load(np.stack([estimator(x[:, j]) for j in range(x.shape[1])])) for x in skeletons])
    return data, labels, np.array([len(x) for x in skeletons]), extra


def ntu_frame_series(n_windows: int = 12, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(n, n_windows, 3, 3) covariances of all joint positions in equal consecutive windows of frames, labels,
    observations of the smallest window (24 joints x its frames), {"split", "subject", "camera"}."""
    skeletons, labels, extra = ntu_skeletons(**kwargs)
    estimator = EstimateCovariance("scm", remove_mean=True, exclude_zero=False)
    data, n_obs = [], []
    for x in skeletons:
        windows = np.array_split(x, n_windows)  # lengths differ by at most one frame
        data.append(load(np.stack([estimator(w.reshape(-1, 3)) for w in windows])))
        n_obs.append(min(w.shape[0] * w.shape[1] for w in windows))
    return np.stack(data), labels, np.array(n_obs), extra


NTU_PARTS = {"trunk": (1, 20, 2, 3), "left arm": (5, 6, 7, 21), "right arm": (9, 10, 11, 23),
             "left leg": (12, 13, 14, 15), "right leg": (16, 17, 18, 19)}
"""Five body parts of four joints (0-based indices of the 25 NTU joints; the root joint 0, the shoulders, the hips
and the thumbs are left out): 12 coordinates per part."""
NTU_HALVES = {"arms": NTU_PARTS["left arm"] + NTU_PARTS["right arm"],
              "legs": NTU_PARTS["left leg"] + NTU_PARTS["right leg"]}
"""Two groups of eight joints (24 coordinates): keeps the correlations between the two arms / the two legs."""


def ntu_part_series(parts: dict | None = None, n_windows: int = 6, width: float = 1 / 3, mean: bool = False,
                    **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(n, n_windows * len(parts), p, p) with p = 3 x joints per part (+ 1 when ``mean``): one token per time
    window and per body part, in time order (window, then part). The token is the Ledoit-Wolf covariance, over the
    frames of the window, of the root-centred coordinates of the joints of the part; coordinates are standardised
    with the statistics of the whole dataset (no label used). ``n_windows`` windows of ``width`` x the sequence
    length, evenly spread (they overlap). ``mean=True``: Calvo-Oller embedding of (window mean, covariance), which
    keeps the mean pose that a covariance removes. Other arguments: :func:`ntu_skeletons`."""
    from .series_breizhcrops import _ledoit_wolf
    from .series_sar import calvo_oller

    groups = [list(group) for group in (parts or NTU_PARTS).values()]
    skeletons, labels, extra = ntu_skeletons(centre=False, **kwargs)
    skeletons = [x - x[:, :1] for x in skeletons]
    sample = np.concatenate([x[::4] for x in skeletons])
    location, scale = sample.mean(0), sample.std(0) + 1e-12
    data, n_obs = [], []
    for x in skeletons:
        z = (x - location) / scale
        w = max(round(len(z) * width), 4)
        tokens = []
        for start in np.round(np.linspace(0, len(z) - w, n_windows)).astype(int):
            for group in groups:
                v = z[start:start + w][:, group].reshape(w, -1)
                covariance = _ledoit_wolf(v)
                tokens.append(calvo_oller(v.mean(0), covariance) if mean else covariance)
        data.append(np.stack(tokens))
        n_obs.append(w)
    return np.stack(data), labels, np.array(n_obs), extra


NTU_JOINTS_16 = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18)
"""Neck, head, both arms (shoulder, elbow, wrist, hand), both legs (hip, knee, ankle): 0-based NTU indices."""
NTU_JOINTS_24 = tuple(range(1, 25))
"""All joints but the root (it is the origin)."""


def ntu_jointcov_series(joints: tuple[int, ...] = NTU_JOINTS_16, n_windows: int = 6, width: float = 1 / 3,
                        **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(n, n_windows, J, J): covariance BETWEEN JOINTS, estimated over time, one token per time window (user
    request, 14/11). Variables = the J joints, observations = the frames of the window; the three axes are pooled
    as three observations of the same J-dimensional variable, C_jk = mean over frames and axes of the centred
    coordinates of joints j and k (model Sigma_joints (x) I_3, invariant to a rotation of the body or the camera):
    n = 3 x frames per window. Coordinates are centred on the root joint, each joint is scaled by its global standard
    deviation (no label used), the mean over the window is removed per joint and axis (no mean embedding).
    ``n_windows`` windows of ``width`` x the sequence length, evenly spread (they overlap). Ledoit-Wolf."""
    from .series_breizhcrops import _ledoit_wolf

    skeletons, labels, extra = ntu_skeletons(centre=False, **kwargs)
    skeletons = [(x - x[:, :1])[:, list(joints)] for x in skeletons]
    sample = np.concatenate([x[::4] for x in skeletons])
    scale = np.sqrt(sample.var(0).mean(-1))[:, None] + 1e-12  # (J, 1): one isotropic scale per joint
    data, n_obs = [], []
    for x in skeletons:
        z = x / scale
        w = max(round(len(z) * width), 4)
        tokens = []
        for start in np.round(np.linspace(0, len(z) - w, n_windows)).astype(int):
            v = z[start:start + w]
            v = (v - v.mean(0)).transpose(0, 2, 1).reshape(-1, len(joints))  # (3 w, J)
            tokens.append(_ledoit_wolf(v))
        data.append(np.stack(tokens))
        n_obs.append(3 * w)
    return np.stack(data), labels, np.array(n_obs), extra


NTU_LONG_ACTIONS = (4, 11, 12, 14, 15, 16, 17, 29, 30, 37, 42)
"""The 11 actions (numbers Axxx) with at least 500 sequences of at least 96 frames."""
NTU_LONG_CLASSES = ["brush hair", "reading", "writing", "put on jacket", "take off jacket", "put on a shoe",
                    "take off a shoe", "play with phone/tablet", "type on a keyboard", "wipe face", "staggering"]
NTU_SINGLE_60 = tuple(range(1, 50))
"""A001-A049: the single-person actions of NTU RGB+D 60."""


def body_frame(x: np.ndarray) -> np.ndarray:
    """Root-centred skeleton (T, 25, 3) in camera axes -> body axes (lateral, vertical, sagittal): one rotation per
    sequence, from the mean spine (root -> spine at the shoulders) and the mean shoulder line (right -> left)."""
    up = (x[:, 20] - x[:, 0]).mean(0)
    up = up / (np.linalg.norm(up) + 1e-12)
    side = (x[:, 4] - x[:, 8]).mean(0)
    side = side - (side @ up) * up
    side = side / (np.linalg.norm(side) + 1e-12)
    return x @ np.stack([side, up, np.cross(side, up)]).T


def ntu_axis_series(classes: tuple[int, ...] = NTU_LONG_ACTIONS, min_frames: int = 96, n_windows: int = 4,
                    width: float = 1 / 3, joints: tuple[int, ...] = NTU_JOINTS_16, **kwargs
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(n, 3 n_windows, J, J): covariance between joints, estimated over the frames of a time window, separately
    for each body axis; tokens in the order (window 1: lateral, vertical, sagittal), (window 2: ...), ... (user
    design, 14/11). Only sequences of at least ``min_frames`` frames are kept, and a window has
    floor(``width`` x length) frames, so that frames >= 2 J in every token (96 frames, a third: 32 frames for
    J = 16). Windows evenly spread (they overlap when n_windows x width > 1). All sequences of the actions
    ``classes`` (numbers Axxx), no cap. Skeleton centred on the root joint and rotated into the body frame
    (:func:`body_frame`), each coordinate scaled by its global standard deviation, window mean removed, Ledoit-Wolf.
    Extra output "frames": length of each sequence."""
    from .series_breizhcrops import _ledoit_wolf

    skeletons, labels, extra = ntu_skeletons(centre=False, classes=classes, min_frames=min_frames,
                                             max_per_class=None, **kwargs)
    skeletons = [body_frame(x - x[:, :1])[:, list(joints)] for x in skeletons]
    scale = np.concatenate([x[::4] for x in skeletons]).std(0) + 1e-12  # (J, 3)
    data, n_obs = [], []
    for x in skeletons:
        z = x / scale
        w = int(len(z) * width)
        assert w >= 2 * len(joints), (len(z), w)
        tokens = []
        for start in np.round(np.linspace(0, len(z) - w, n_windows)).astype(int):
            tokens += [_ledoit_wolf(z[start:start + w, :, axis]) for axis in range(3)]
        data.append(np.stack(tokens))
        n_obs.append(w)
    return np.stack(data), labels, np.array(n_obs), {**extra, "frames": np.array([len(x) for x in skeletons])}
