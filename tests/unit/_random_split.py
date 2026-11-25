"""Shared checks of the ``"random"`` split for the per-family loader tests."""

import warnings

import numpy as np
import pytest

from spdnet_datasets.base import create_dataloaders


def _random_warnings(caught):
    return [w for w in caught if "split='random'" in str(w.message)]


def assert_random_split(make, *, leaky, cov=True, subset=True):
    """Check ``make(split="random")`` against ``make()`` (default split).

    ``make(**kw)`` builds the dataset. ``cov``: items are ``(covariance, y)``,
    the scale is then checked on the training part; ``subset``: the loader
    takes ``subset=``. Returns the dataset built with ``split="random"``.
    """
    default = make()
    assert "random" in default.splits and default.split_leaky is False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = make(split="random")
    assert len(_random_warnings(caught)) == (1 if leaky else 0), [str(w.message) for w in caught]
    assert ds.split == "random" and ds.split_leaky is leaky
    assert ds.random_split_leaky is leaky

    parts = ds.splits["random"]
    n = len(ds)
    cat = np.concatenate([parts["train"], parts["val"], parts["test"]])
    assert sorted(cat.tolist()) == list(range(n))  # disjoint and covering
    assert len(parts["test"]) == round(0.2 * n) and len(parts["val"]) == round(0.1 * n)
    # same split whatever the selected split (indices only, drawn after loading)
    for k in ("train", "val", "test"):
        np.testing.assert_array_equal(default.splits["random"][k], parts[k])

    if cov:
        train = parts["train"]
        covs = np.stack([ds[int(i)][0].numpy() for i in train])
        eig = np.linalg.eigvalsh(covs)
        # the scale is fitted on the training part of the random split
        assert np.exp(np.mean(np.log(eig))) == pytest.approx(1.0, rel=1e-8)
        assert ds.scale != pytest.approx(default.scale, rel=1e-12)

    tr, va, te = create_dataloaders(ds, split="random", batch_size=4, num_workers=0,
                                    verbose=False)
    assert (len(tr.dataset), len(va.dataset), len(te.dataset)) == tuple(
        len(parts[k]) for k in ("train", "val", "test"))
    # statistics fitted on the default split: the random split is refused
    with pytest.raises(ValueError, match="split='random'"):
        create_dataloaders(default, split="random", num_workers=0, verbose=False)

    if subset:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sub = make(split="random", subset="train")
        assert len(sub) == len(parts["train"]) and sub.scale == ds.scale
    return ds
