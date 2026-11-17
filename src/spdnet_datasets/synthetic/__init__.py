"""Synthetic SPD data generation for simulation experiments."""

from .config import (
    BATCHNORM_METHODS,
    BATCHNORM_METHODS_WITH_AFFINE,
    EXPERIMENT_SETS,
    GRID_REGISTRY,
    METHOD_SHORT_NAMES,
    ExperimentConfig,
    generate_experiment_configs,
    get_configs_for_grid,
    list_experiment_sets,
    list_grid_info,
)
from .data_generator import (
    ScaleMatrixGeneratorBlock,
    ScaleMatrixGeneratorDiagonal,
    WishartGenerator,
    generate_eigenvalues,
    shuffle_data,
)
from .dataset import SimulationDataset, create_dataloaders

__all__ = [
    # Configuration
    'ExperimentConfig',
    'BATCHNORM_METHODS',
    'BATCHNORM_METHODS_WITH_AFFINE',
    'METHOD_SHORT_NAMES',
    'EXPERIMENT_SETS',
    'GRID_REGISTRY',
    'generate_experiment_configs',
    'get_configs_for_grid',
    'list_grid_info',
    'list_experiment_sets',

    # Generators
    'ScaleMatrixGeneratorDiagonal',
    'ScaleMatrixGeneratorBlock',
    'WishartGenerator',
    'generate_eigenvalues',
    'shuffle_data',

    # Dataset
    'SimulationDataset',
    'create_dataloaders',
]
