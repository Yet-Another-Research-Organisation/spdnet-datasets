"""Deprecated: use :mod:`spdnet_datasets.skeleton.hdm05`.

``spdnet_datasets.real.hdm05.HDM05Dataset`` keeps the former defaults (every
sample, no normalisation, ``scaling_factor``) on top of the new loader; it is
not registered in the :class:`~spdnet_datasets.manager.DatasetManager` (the
name ``"hdm05"`` points to the new class).
"""

import warnings

from spdnet_datasets.skeleton.hdm05 import HDM05Dataset as _HDM05Dataset
from spdnet_datasets.skeleton.hdm05 import fetch_hdm05

__all__ = ["HDM05Dataset", "fetch_hdm05"]


class HDM05Dataset(_HDM05Dataset):
    """Former HDM05 loader (deprecated).

    Parameters
    ----------
    scaling_factor : float
        Multiplies every covariance (former behaviour; the new loader uses the
        geometric normalisation fitted on the training split instead).
    **kwargs
        :class:`spdnet_datasets.skeleton.hdm05.HDM05Dataset` arguments; by
        default ``subset=None`` and ``normalize=False``.
    """

    def __init__(self, scaling_factor: float = 1.0, **kwargs):
        warnings.warn("spdnet_datasets.real.hdm05 is deprecated; use "
                      "spdnet_datasets.skeleton.hdm05.HDM05Dataset", DeprecationWarning,
                      stacklevel=2)
        kwargs.setdefault("subset", None)
        kwargs.setdefault("normalize", False)
        super().__init__(**kwargs)
        self.scale = self.scale * scaling_factor
        self.scaling_factor = scaling_factor
