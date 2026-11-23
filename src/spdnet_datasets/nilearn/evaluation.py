"""Cross-validation over the fold splits of :class:`NilearnConnectomes`.

Small fMRI cohorts give wide score intervals: one test site of ADHD-200 holds
6 to 8 subjects. :func:`cross_validate_folds` scores every fold of a split
family (``site_fold_<k>`` or ``stratified_fold_<k>``), so that every subject is
tested once, and reports a bootstrap interval and a permutation p-value.
"""

from __future__ import annotations

import re

import numpy as np


def _fold_names(splits, prefix):
    names = [n for n in splits if re.fullmatch(rf"{prefix}_\d+", n)]
    if not names:
        raise ValueError(f"no split named {prefix}_<k>; available: {sorted(splits)}")
    return sorted(names, key=lambda n: int(n.rsplit("_", 1)[1]))


def _permute_within(y, groups, rng):
    """``y`` shuffled inside each group."""
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[rng.permutation(idx)]
    return out


def cross_validate_folds(ds, fit_predict, prefix=None, *, score=None, n_permutations=1000,
                         n_bootstrap=1000, seed=0):
    """Score ``fit_predict`` on every fold of a split family of ``ds``.

    Parameters
    ----------
    ds : NilearnConnectomes
        Dataset with fold splits (``ds.splits``).
    fit_predict : callable
        ``fit_predict(X_train, y_train, X_test) -> y_pred``. ``X`` holds the
        unscaled covariances ``ds.covs`` (no scale fitted on another fold); the
        training part of a fold is its ``train`` plus ``val`` subjects.
    prefix : {"site_fold", "stratified_fold"}, optional
        Split family. Default: ``"site_fold"`` (leave-one-site-out) when the
        dataset has it, else ``"stratified_fold"``.
    score : callable, optional
        ``score(y_true, y_pred) -> float``. Default: balanced accuracy for
        classification (chance level ``1 / n_classes`` whatever the class
        balance), :math:`R^2` for regression.
    n_permutations : int
        Label permutations for the p-value; labels are shuffled within each
        site (``ds.sites``), so a site effect cannot pass for a label effect.
        ``0`` skips the test.
    n_bootstrap : int
        Bootstrap resamples of the subjects for the confidence interval.
    seed : int
        Random state of the permutations and the bootstrap.

    Returns
    -------
    dict
        ``folds`` (split names), ``scores`` (one per fold), ``mean`` (their
        mean), ``ci`` (2.5 and 97.5 percentiles of the score of the pooled
        out-of-fold predictions over subjects resampled with replacement),
        ``p_value`` (``(1 + #{null >= mean}) / (1 + n_permutations)``, ``nan``
        without permutations), ``null`` (mean fold score of each permutation)
        and ``y_pred`` (out-of-fold prediction of every subject).
    """
    from sklearn.metrics import balanced_accuracy_score, r2_score

    regression = ds.task == "regression"
    score = score or (r2_score if regression else balanced_accuracy_score)
    if prefix is None:
        prefix = ("site_fold" if any(n.startswith("site_fold_") for n in ds.splits)
                  else "stratified_fold")
    names = _fold_names(ds.splits, prefix)
    folds = [ds.splits[n] for n in names]
    tested = np.sort(np.concatenate([f["test"] for f in folds]))
    if not np.array_equal(tested, np.arange(len(ds))):
        raise ValueError(f"the {prefix}_<k> test parts do not cover every subject once")
    X, y = ds.covs, np.asarray(ds.labels)

    def run(labels):
        pred = np.empty(len(labels), dtype=np.float64 if regression else labels.dtype)
        scores = []
        for f in folds:
            train = np.concatenate([f["train"], f["val"]])
            pred[f["test"]] = fit_predict(X[train], labels[train], X[f["test"]])
            scores.append(score(labels[f["test"]], pred[f["test"]]))
        return np.asarray(scores), pred

    scores, pred = run(y)
    rng = np.random.default_rng(seed)
    boot = [score(y[b], pred[b]) for b in rng.integers(0, len(y), (n_bootstrap, len(y)))]
    null = np.array([run(_permute_within(y, ds.sites, rng))[0].mean()
                     for _ in range(n_permutations)])
    mean = float(scores.mean())
    p_value = ((1 + np.sum(null >= mean)) / (1 + n_permutations) if n_permutations
               else float("nan"))
    return {"folds": names, "scores": scores, "mean": mean,
            "ci": tuple(np.percentile(boot, [2.5, 97.5])), "p_value": float(p_value),
            "null": null, "y_pred": pred}
