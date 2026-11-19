"""Covariance series: one sequence of SPD tokens per sample of a main loader.

- :mod:`.tokenizers`: pure functions that cut a structured sample into token observations;
- :mod:`.modes`: registry :data:`SERIES_MODES` of the modes and default parameters per dataset;
- :mod:`.dataset`: :class:`CovarianceSeries`, the PyTorch dataset, and :func:`collate_series`.

See the guide page "Covariance series".
"""

from .dataset import CovarianceSeries, collate_series
from .modes import SERIES_MODES, SeriesMode, get_mode

__all__ = ["SERIES_MODES", "CovarianceSeries", "SeriesMode", "collate_series", "get_mode"]
