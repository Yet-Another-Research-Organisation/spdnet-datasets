"""Skeleton datasets: one covariance of joint features per motion sequence.

- :mod:`.hdm05`: HDM05, provided 93 x 93 covariances, stratified folds (no actor id).
- :mod:`.ntu120`: NTU RGB+D 120, body-part or joint features, official X-Sub / X-Setup.
"""

from .hdm05 import HDM05Dataset, fetch_hdm05
from .ntu120 import NTURGBD120Dataset, fetch_ntu120

__all__ = ["HDM05Dataset", "NTURGBD120Dataset", "fetch_hdm05", "fetch_ntu120"]
