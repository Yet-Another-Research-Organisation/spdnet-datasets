"""
Estimator module for SPDNet benchmarks.
Provides covariance estimation methods.
"""

from .covariance import EstimateCovariance, EstimateCovarianceTorch

__all__ = [
    'EstimateCovariance',
    'EstimateCovarianceTorch',
]
