"""Task selection on integer or string labels.

- ``task="fine"``: all classes as given.
- ``task="coarse"``: classes regrouped with a dataset-specific ``coarse_map``
  (``{fine_class: coarse_class}``).
- ``task="regression"``: labels are continuous targets and are returned as is.
- ``task="multilabel"``: each sample has a 0/1 vector over the classes (several
  classes can be present); handled by the dataset classes, not by
  :func:`apply_task`. Balancing: :func:`pos_weights` (per-label positive weights).
- ``drop_classes``: classes removed before anything else.
- ``merge_classes``: ``{old_class: new_class}`` applied after dropping.
- Balancing: :func:`balance_indices` (under/over-sampling) or
  :func:`class_weights` (inverse-frequency weights).
"""

import numpy as np

TASKS = ("fine", "coarse", "regression")
MULTILABEL = "multilabel"
BALANCE = (None, "undersample", "oversample", "weights")


def apply_task(y, *, task="fine", coarse_map=None, drop_classes=(), merge_classes=None):
    """Return ``(y_new, keep_mask, class_names)``.

    ``y_new`` holds integer indices into ``class_names`` (classification) or the
    untouched targets (regression); ``keep_mask`` selects the kept samples.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose among {TASKS}")
    y = np.asarray(y)
    if task == "regression":
        if drop_classes or merge_classes:
            raise ValueError("drop_classes / merge_classes do not apply to regression")
        return y, np.ones(len(y), dtype=bool), None

    keep = ~np.isin(y, list(drop_classes)) if len(drop_classes) else np.ones(len(y), bool)
    labels = y[keep]
    mapping = dict(merge_classes or {})
    if task == "coarse":
        if coarse_map is None:
            raise ValueError("task='coarse' requires a coarse_map for this dataset")
        missing = set(np.unique(labels).tolist()) - set(coarse_map)
        if missing:
            raise ValueError(f"coarse_map has no entry for classes {sorted(missing)}")
        labels = np.array([coarse_map[v] for v in labels.tolist()])
    if mapping:
        labels = np.array([mapping.get(v, v) for v in labels.tolist()])
    class_names = sorted(np.unique(labels).tolist())
    index = {c: i for i, c in enumerate(class_names)}
    return np.array([index[v] for v in labels.tolist()], dtype=np.int64), keep, class_names


def balance_indices(y, mode=None, seed=0):
    """Indices of a class-balanced resampling of ``y`` (or all indices if ``mode`` is None)."""
    if mode not in (None, "undersample", "oversample"):
        raise ValueError("mode must be None, 'undersample' or 'oversample'")
    y = np.asarray(y)
    idx = np.arange(len(y))
    if mode is None:
        return idx
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.min() if mode == "undersample" else counts.max()
    out = [rng.choice(idx[y == c], size=target, replace=mode == "oversample" and n < target)
           for c, n in zip(classes, counts, strict=True)]
    return np.sort(np.concatenate(out))


def class_weights(y, n_classes=None) -> np.ndarray:
    """Inverse-frequency weights ``n / (k * n_c)``, indexed by class.

    The result has ``max(n_classes, max(y) + 1)`` entries; a class absent from
    ``y`` gets weight 0.
    """
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes or 0)
    k = np.count_nonzero(counts)
    with np.errstate(divide="ignore"):
        w = np.where(counts > 0, len(y) / (k * counts), 0.0)
    return w


def pos_weights(Y) -> np.ndarray:
    """Per-label positive weights ``n_negative / n_positive`` of a 0/1 matrix ``Y`` (n, k).

    The ``pos_weight`` of :class:`torch.nn.BCEWithLogitsLoss`; a label without
    positive sample gets weight 0.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Expected a (n, k) label matrix, got shape {Y.shape}")
    pos = Y.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pos > 0, (len(Y) - pos) / pos, 0.0)
