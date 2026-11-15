"""Synthetic SPD data generation for simulation experiments.

The generators need the optional dependency ``yetanotherspdnet`` (extra ``synthetic``). The names below are
imported lazily, on first access, so that ``import spdnet_datasets.synthetic`` works without it and the
``ImportError`` only appears when a generator is actually used.
"""

import importlib

_EXPORTS = {
    # Configuration
    'ExperimentConfig': 'config',
    'BATCHNORM_METHODS': 'config',
    'BATCHNORM_METHODS_WITH_AFFINE': 'config',
    'METHOD_SHORT_NAMES': 'config',
    'EXPERIMENT_SETS': 'config',
    'GRID_REGISTRY': 'config',
    'generate_experiment_configs': 'config',
    'get_configs_for_grid': 'config',
    'list_grid_info': 'config',
    'list_experiment_sets': 'config',

    # Generators
    'ScaleMatrixGeneratorDiagonal': 'data_generator',
    'ScaleMatrixGeneratorBlock': 'data_generator',
    'WishartGenerator': 'data_generator',
    'generate_eigenvalues': 'data_generator',
    'shuffle_data': 'data_generator',

    # Dataset
    'SimulationDataset': 'dataset',
    'create_dataloaders': 'dataset',

    # Covariance series
    'SeriesConfig': 'series',
    'SeriesData': 'series',
    'generate_series': 'series',
    'sample_wishart': 'series',
    'validate_series': 'series',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
    except ImportError as exc:
        if "yetanotherspdnet" not in str(exc):
            raise
        raise ImportError(f"spdnet_datasets.synthetic.{name} needs yetanotherspdnet "
                          "(pip install 'spdnet-datasets[synthetic]')") from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
