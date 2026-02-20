"""Synthetic SPD data generation for simulation experiments."""

from .config import (
    ExperimentConfig,
    BATCHNORM_METHODS,
    METHOD_SHORT_NAMES,
    generate_experiment_configs,
    get_configs_for_grid,
)
from .data_generator import (
    ScaleMatrixGeneratorDiagonal,
    ScaleMatrixGeneratorBlock,
    WishartGenerator,
    generate_eigenvalues,
    shuffle_data,
)
from .dataset import SimulationDataset, create_dataloaders

__all__ = [
    # Configuration
    'ExperimentConfig',
    'BATCHNORM_METHODS',
    'METHOD_SHORT_NAMES',
    'generate_experiment_configs',
    'get_configs_for_grid',

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
