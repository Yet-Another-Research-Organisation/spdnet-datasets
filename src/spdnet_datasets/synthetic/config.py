"""
Experiment configuration and parameter grids.

Defines ExperimentConfig dataclass and all experiment grid presets
(quick, medium, full, filtered, filtered-extended, wishart-inverse).
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal, Iterator
from itertools import product
import hashlib
import numpy as np


# =============================================================================
# BatchNorm method definitions
# =============================================================================

BATCHNORM_METHODS = [
    "arithmetic",
    "harmonic",
    "geometric_arithmetic_harmonic",
    "adaptive_geometric_arithmetic_harmonic",
]

BATCHNORM_METHODS_WITH_AFFINE = BATCHNORM_METHODS + ["affine_invariant"]

METHOD_SHORT_NAMES = {
    "arithmetic": "arith",
    "harmonic": "harm",
    "geometric_arithmetic_harmonic": "gah",
    "adaptive_geometric_arithmetic_harmonic": "adaptive",
    "affine_invariant": "affine_inv",
}


# =============================================================================
# Experiment configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single simulation experiment."""

    # Data generation
    generation_mode: Literal["scale", "wishart"] = "scale"
    structure: Literal["diagonal", "block", "full"] = "diagonal"
    eigenvalue_mode: Literal["constant", "linspace", "geomspace", "random"] = "geomspace"

    matrix_size: int = 16
    n_classes: int = 3
    n_samples_per_class: int = 120

    # Eigenvalue parameters
    max_value: float = 100.0
    conditioning: float = 100.0
    n_discriminant: int = 2
    discriminant_position: Literal["large", "small", "mixed"] = "large"
    class_separation_ratio: float = 0.01

    # Wishart-specific
    df: Optional[int] = 50

    # Model
    batchnorm_method: str = "arithmetic"

    # Training
    batch_size: int = 16
    max_epochs: int = 150
    early_stopping_patience: int = 50
    lr: float = 1e-6
    target_lr: float = 0.05

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_config_hash(self) -> str:
        """6-character hash of configuration (excluding seed and method)."""
        config = {
            k: v
            for k, v in self.to_dict().items()
            if k not in ["seed", "batchnorm_method"]
        }
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:6]

    def get_full_hash(self) -> str:
        """6-character hash of the full configuration."""
        config_str = str(sorted(self.to_dict().items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:6]


# =============================================================================
# Parameter grids
# =============================================================================

FULL_GRID = {
    "generation_mode": ["scale", "wishart"],
    "structure": ["diagonal", "block"],
    "eigenvalue_mode": ["constant", "linspace", "geomspace"],
    "matrix_size": [4, 16],
    "n_discriminant": [1, 4],
    "conditioning": [10, 500],
    "max_value": [10, 500],
    "df": [20, 100],
    "class_separation_ratio": [0.01, 0.10],
    "discriminant_position": ["large", "small", "mixed"],
    "seeds": [42, 123],
    "methods": BATCHNORM_METHODS,
}

QUICK_GRID = {
    "generation_mode": ["scale"],
    "structure": ["diagonal"],
    "eigenvalue_mode": ["geomspace"],
    "matrix_size": [4],
    "n_discriminant": [2],
    "conditioning": [100],
    "max_value": [100],
    "df": [50],
    "class_separation_ratio": [0.02],
    "discriminant_position": ["large", "small"],
    "seeds": [42],
    "methods": BATCHNORM_METHODS,
}

MEDIUM_GRID = {
    "generation_mode": ["scale", "wishart"],
    "structure": ["diagonal"],
    "eigenvalue_mode": ["geomspace"],
    "matrix_size": [4, 16],
    "n_discriminant": [2],
    "conditioning": [10, 100],
    "max_value": [100],
    "df": [50],
    "class_separation_ratio": [0.02, 0.10],
    "discriminant_position": ["large", "small", "mixed"],
    "seeds": [42, 123],
    "methods": BATCHNORM_METHODS,
}

# Filtered v1: scale only, 7 seeds, 4 methods
FILTERED_GRID = {
    "generation_mode": ["scale"],
    "structure": ["block", "diagonal"],
    "eigenvalue_mode": ["constant", "geomspace", "linspace"],
    "matrix_size": [4, 16],
    "n_discriminant": [1, 4],
    "discriminant_position": ["large", "small"],
    "conditioning": [10, 500],
    "max_value": [10, 500],
    "class_separation_ratio": [0.01, 0.1],
    "df": [None],
    "seeds": [42, 123, 456, 789, 1011, 1213, 1415],
    "methods": BATCHNORM_METHODS,
}

# Filtered v2 (extended): diagonal only, 5 seeds, 5 methods (+ affine_invariant)
FILTERED_EXTENDED_GRID = {
    "generation_mode": ["scale"],
    "structure": ["diagonal"],
    "eigenvalue_mode": ["geomspace", "linspace"],
    "matrix_size": [4, 16, 32],
    "n_discriminant": [1, 4],
    "discriminant_position": ["large", "small"],
    "conditioning": [500, 2500, 15000],
    "max_value": [10, 500, 2500],
    "class_separation_ratio": [0.01, 0.005],
    "df": [None],
    "seeds": [42, 123, 456, 789, 1011],
    "methods": BATCHNORM_METHODS_WITH_AFFINE,
}

# Wishart/Inverse-Wishart: random SPD scales, 5 seeds, 5 methods
WISHART_INVERSE_SEEDS = [42, 123, 456, 789, 1011]

WISHART_INVERSE_GRID = {
    "generation_mode": ["wishart"],
    "structure": ["full"],
    "eigenvalue_mode": ["random"],
    "matrix_size": [64],
    "n_discriminant": [0],
    "discriminant_position": ["large", "small"],
    "conditioning": [100],
    "max_value": [1.0],
    "class_separation_ratio": [0.0],
    "df": [64],
    "seeds": WISHART_INVERSE_SEEDS,
    "methods": BATCHNORM_METHODS_WITH_AFFINE,
}

# Named grid registry
GRID_REGISTRY = {
    "quick": QUICK_GRID,
    "medium": MEDIUM_GRID,
    "full": FULL_GRID,
    "filtered": FILTERED_GRID,
    "filtered-extended": FILTERED_EXTENDED_GRID,
    "wishart-inverse": WISHART_INVERSE_GRID,
}


# =============================================================================
# Config generation
# =============================================================================

def generate_experiment_configs(grid: Dict) -> Iterator[ExperimentConfig]:
    """
    Generate experiment configurations from a parameter grid.

    Args:
        grid: Parameter grid with lists of values for each parameter.
              Must contain 'methods' and 'seeds' keys.

    Yields:
        ExperimentConfig instances.
    """
    methods = grid.get("methods", BATCHNORM_METHODS)
    seeds = grid.get("seeds", [42, 123, 456])

    # All remaining parameter keys (exclude meta-keys)
    param_keys = [
        k for k in grid if k not in ("methods", "seeds")
    ]
    param_values = [grid[k] for k in param_keys]

    for values in product(*param_values):
        base_params = dict(zip(param_keys, values))

        # Skip invalid combinations
        n_disc = base_params.get("n_discriminant", 2)
        mat_size = base_params.get("matrix_size", 16)
        disc_pos = base_params.get("discriminant_position", "large")

        if n_disc > mat_size:
            continue
        if disc_pos == "mixed" and n_disc < 2:
            continue

        for method in methods:
            for seed in seeds:
                yield ExperimentConfig(
                    **base_params,
                    batchnorm_method=method,
                    seed=seed,
                )


def get_configs_for_grid(grid_name: str) -> List[ExperimentConfig]:
    """
    Get all experiment configs for a named grid.

    Args:
        grid_name: One of the keys in GRID_REGISTRY.

    Returns:
        List of ExperimentConfig.
    """
    if grid_name not in GRID_REGISTRY:
        raise ValueError(
            f"Unknown grid '{grid_name}'. Available: {list(GRID_REGISTRY.keys())}"
        )
    return list(generate_experiment_configs(GRID_REGISTRY[grid_name]))


def list_grid_info(grid_name: str) -> None:
    """Print summary information about a named grid."""
    grid = GRID_REGISTRY[grid_name]
    configs = list(generate_experiment_configs(grid))
    n_methods = len(grid.get("methods", []))
    n_seeds = len(grid.get("seeds", []))
    n_unique = len(configs) // max(n_methods * n_seeds, 1)

    print(f"Grid: {grid_name}")
    print("=" * 60)
    print(f"Unique configurations: {n_unique}")
    print(f"Seeds: {n_seeds}")
    print(f"Methods: {n_methods} {grid.get('methods', [])}")
    print(f"Total experiments: {len(configs)}")
    print()
    print("Parameters:")
    for key in sorted(grid.keys()):
        if key not in ("methods", "seeds"):
            print(f"  {key}: {grid[key]}")


# =============================================================================
# Predefined experiment sets (numbered, for --set N)
# =============================================================================

def _make_set_grid(**overrides) -> Dict:
    """Create a grid based on defaults with overrides."""
    base = {
        "generation_mode": ["scale"],
        "structure": ["diagonal"],
        "eigenvalue_mode": ["geomspace"],
        "matrix_size": [16],
        "n_discriminant": [2],
        "conditioning": [100],
        "max_value": [100],
        "df": [50],
        "class_separation_ratio": [0.02],
        "discriminant_position": ["large", "small", "mixed"],
        "seeds": [42, 123, 456],
        "methods": BATCHNORM_METHODS,
    }
    base.update(overrides)
    return base


EXPERIMENT_SETS = {
    1: ("eigenvalue_mode", _make_set_grid(
        eigenvalue_mode=["constant", "linspace", "geomspace"])),
    2: ("conditioning", _make_set_grid(
        conditioning=[10, 100, 1000])),
    3: ("n_discriminant", _make_set_grid(
        n_discriminant=[1, 2, 4])),
    4: ("matrix_size", _make_set_grid(
        matrix_size=[4, 16])),
    5: ("structure", _make_set_grid(
        structure=["diagonal", "block"])),
    6: ("wishart_vs_scale", _make_set_grid(
        generation_mode=["scale", "wishart"])),
    7: ("wishart_df", _make_set_grid(
        generation_mode=["wishart"], df=[20, 50, 100])),
    8: ("class_separation", _make_set_grid(
        class_separation_ratio=[0.01, 0.02, 0.10])),
    9: ("max_value", _make_set_grid(
        max_value=[10, 100, 1000])),
}


def list_experiment_sets() -> str:
    """Return a formatted list of all numbered experiment sets."""
    lines = ["Available Experiment Sets:", "=" * 50]
    for num, (name, grid) in EXPERIMENT_SETS.items():
        count = sum(1 for _ in generate_experiment_configs(grid))
        lines.append(f"  {num}. {name}: {count} experiments")
    return "\n".join(lines)
