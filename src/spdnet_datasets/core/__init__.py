"""Shared building blocks of the dataset modules.

- :mod:`.config`: data root and cache directory (environment variables).
- :mod:`.estimators`: covariance estimators, ``n >= 2p`` and SPD checks.
- :mod:`.normalize`: geometric normalisation fitted on the training split.
- :mod:`.outputs`: output contract (covariance, mean, raw).
- :mod:`.tasks`: fine / coarse / regression tasks, class dropping and balancing.
- :mod:`.splits`: grouped and spatial splits, leakage checks.
- :mod:`.fetch`: downloads, checksums, manifest checks and staged cache.
"""

from .config import dataset_path, get_cache_dir, get_data_root
from .estimators import (
    ESTIMATORS,
    InsufficientSamplesError,
    NotSPDError,
    channel_mean,
    check_sample_size,
    check_spd,
    estimate_covariance,
    estimate_covariances,
)
from .fetch import KEEP, Bunch, check_local, download, run_pipeline, verify_local
from .normalize import apply_scale, geometric_scale
from .outputs import pack_outputs, validate_output_request
from .splits import (
    assert_no_duplicates,
    assert_no_leakage,
    assert_no_window_overlap,
    group_split,
    spatial_block_split,
)
from .tasks import apply_task, balance_indices, class_weights

__all__ = [
    "ESTIMATORS", "KEEP", "Bunch", "InsufficientSamplesError", "NotSPDError",
    "apply_scale", "apply_task", "assert_no_duplicates", "assert_no_leakage",
    "assert_no_window_overlap", "balance_indices", "channel_mean", "check_local",
    "check_sample_size", "check_spd", "class_weights", "dataset_path", "download",
    "estimate_covariance",
    "estimate_covariances", "geometric_scale", "get_cache_dir", "get_data_root",
    "group_split", "pack_outputs", "run_pipeline", "spatial_block_split",
    "validate_output_request", "verify_local",
]
