"""
Data generators for SPD matrix classification experiments.

Provides three generation strategies:
1. ScaleMatrixGeneratorDiagonal: Direct diagonal eigenvalue control with rotation.
2. ScaleMatrixGeneratorBlock: Block-structured SPD matrices.
3. WishartGenerator: Wishart / Inverse-Wishart with random SPD scale matrices.
"""

import numpy as np
import torch
from scipy.stats import wishart, invwishart, ortho_group
from typing import Tuple, Optional, Dict, Literal
from pathlib import Path
import sys

# Import random_SPD from the spdnet submodule
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "spdnet" / "src"))
from yetanotherspdnet.random.spd import random_SPD


# =============================================================================
# Eigenvalue generation
# =============================================================================

def generate_eigenvalues(
    matrix_size: int,
    max_value: float,
    conditioning: float,
    mode: Literal["constant", "linspace", "geomspace"] = "geomspace",
    n_discriminant: int = 2,
    discriminant_position: Literal["large", "small", "mixed"] = "large",
    class_idx: int = 0,
    class_separation_ratio: float = 0.01,
) -> np.ndarray:
    """
    Generate an eigenvalue vector for a given class.

    Args:
        matrix_size: Number of eigenvalues.
        max_value: Maximum eigenvalue.
        conditioning: Ratio max / min eigenvalue.
        mode: Distribution of eigenvalues:
            - 'constant': n_discriminant at max, rest at max/conditioning
            - 'linspace': linearly spaced from max/conditioning to max
            - 'geomspace': geometrically spaced from max/conditioning to max
        n_discriminant: Number of discriminant eigenvalues.
        discriminant_position: Which eigenvalues to modify for class separation.
        class_idx: Class index (0-based).
        class_separation_ratio: Multiplicative ratio per class.

    Returns:
        Eigenvalue array in descending order.
    """
    min_value = max_value / conditioning

    if mode == "constant":
        eigenvalues = np.full(matrix_size, min_value)
        eigenvalues[:n_discriminant] = max_value
    elif mode == "linspace":
        eigenvalues = np.linspace(max_value, min_value, matrix_size)
    elif mode == "geomspace":
        eigenvalues = np.geomspace(max_value, min_value, matrix_size)
    else:
        raise ValueError(f"Unknown eigenvalue mode: {mode}")

    # Apply class-specific perturbation to discriminant eigenvalues
    if class_idx > 0:
        multiplier = (1 + class_separation_ratio) ** class_idx

        if discriminant_position == "large":
            eigenvalues[:n_discriminant] *= multiplier
        elif discriminant_position == "small":
            eigenvalues[-n_discriminant:] *= multiplier
        elif discriminant_position == "mixed":
            n_large = n_discriminant // 2
            n_small = n_discriminant - n_large
            eigenvalues[:n_large] *= multiplier
            if n_small > 0:
                eigenvalues[-n_small:] *= multiplier
        else:
            raise ValueError(f"Unknown discriminant_position: {discriminant_position}")

    return eigenvalues


# =============================================================================
# Utility
# =============================================================================

def shuffle_data(
    data: np.ndarray,
    labels: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle data and labels together, preserving correspondence."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))
    return data[perm], labels[perm]


# =============================================================================
# Scale matrix generator (diagonal)
# =============================================================================

class ScaleMatrixGeneratorDiagonal:
    """
    Generate SPD matrices from specified eigenvalues with random orthogonal rotation.

    Each sample in a class shares the same eigenvalue spectrum;
    variation comes from different random rotations per sample.
    """

    def __init__(
        self,
        matrix_size: int = 16,
        n_classes: int = 3,
        seed: Optional[int] = None,
    ):
        self.matrix_size = matrix_size
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

    def generate_data(
        self,
        n_samples_per_class: int,
        max_value: float,
        conditioning: float,
        mode: Literal["constant", "linspace", "geomspace"] = "geomspace",
        n_discriminant: int = 2,
        discriminant_position: Literal["large", "small", "mixed"] = "large",
        class_separation_ratio: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a dataset of diagonal-based SPD matrices.

        Returns:
            data: (n_samples, matrix_size, matrix_size)
            labels: (n_samples,)
            info: dict with generation parameters
        """
        data_list = []
        labels_list = []
        eigenvalues_per_class = []

        for class_idx in range(self.n_classes):
            eigenvalues = generate_eigenvalues(
                matrix_size=self.matrix_size,
                max_value=max_value,
                conditioning=conditioning,
                mode=mode,
                n_discriminant=n_discriminant,
                discriminant_position=discriminant_position,
                class_idx=class_idx,
                class_separation_ratio=class_separation_ratio,
            )
            eigenvalues_per_class.append(eigenvalues.copy())

            samples = np.zeros((n_samples_per_class, self.matrix_size, self.matrix_size))
            for i in range(n_samples_per_class):
                Q = ortho_group.rvs(self.matrix_size, random_state=self.rng)
                samples[i] = Q @ np.diag(eigenvalues) @ Q.T

            data_list.append(samples)
            labels_list.append(np.full(n_samples_per_class, class_idx))

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        info = {
            "generator": "diagonal",
            "matrix_size": self.matrix_size,
            "n_classes": self.n_classes,
            "n_samples_per_class": n_samples_per_class,
            "max_value": max_value,
            "conditioning": conditioning,
            "mode": mode,
            "n_discriminant": n_discriminant,
            "discriminant_position": discriminant_position,
            "class_separation_ratio": class_separation_ratio,
            "eigenvalues_per_class": eigenvalues_per_class,
        }
        return data, labels, info


# =============================================================================
# Scale matrix generator (block)
# =============================================================================

class ScaleMatrixGeneratorBlock:
    """
    Generate SPD matrices with a two-block structure.

    The matrix is split into two diagonal blocks (half each), each with its
    own rotation, then a global rotation is applied.
    """

    def __init__(
        self,
        matrix_size: int = 16,
        n_classes: int = 3,
        seed: Optional[int] = None,
    ):
        self.matrix_size = matrix_size
        self.block_size = matrix_size // 2
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self._global_ortho = ortho_group.rvs(matrix_size, random_state=self.rng)

    def _generate_block(self, eigenvalues: np.ndarray) -> np.ndarray:
        size = len(eigenvalues)
        Q = ortho_group.rvs(size, random_state=self.rng)
        return Q @ np.diag(eigenvalues) @ Q.T

    def generate_data(
        self,
        n_samples_per_class: int,
        max_value: float,
        conditioning: float,
        mode: Literal["constant", "linspace", "geomspace"] = "geomspace",
        n_discriminant: int = 2,
        discriminant_position: Literal["large", "small", "mixed"] = "large",
        class_separation_ratio: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a dataset of block-structured SPD matrices.

        Returns:
            data, labels, info
        """
        data_list = []
        labels_list = []
        eigenvalues_per_class = []

        for class_idx in range(self.n_classes):
            eigenvalues = generate_eigenvalues(
                matrix_size=self.matrix_size,
                max_value=max_value,
                conditioning=conditioning,
                mode=mode,
                n_discriminant=n_discriminant,
                discriminant_position=discriminant_position,
                class_idx=class_idx,
                class_separation_ratio=class_separation_ratio,
            )
            eigenvalues_per_class.append(eigenvalues.copy())

            block1_eigvals = eigenvalues[: self.block_size]
            block2_eigvals = eigenvalues[self.block_size :]

            samples = np.zeros((n_samples_per_class, self.matrix_size, self.matrix_size))
            for i in range(n_samples_per_class):
                a1 = self._generate_block(block1_eigvals)
                a2 = self._generate_block(block2_eigvals)
                X = np.zeros((self.matrix_size, self.matrix_size))
                X[: self.block_size, : self.block_size] = a1
                X[self.block_size :, self.block_size :] = a2
                X = self._global_ortho @ X @ self._global_ortho.T
                samples[i] = (X + X.T) / 2

            data_list.append(samples)
            labels_list.append(np.full(n_samples_per_class, class_idx))

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        info = {
            "generator": "block",
            "matrix_size": self.matrix_size,
            "block_size": self.block_size,
            "n_classes": self.n_classes,
            "n_samples_per_class": n_samples_per_class,
            "max_value": max_value,
            "conditioning": conditioning,
            "mode": mode,
            "n_discriminant": n_discriminant,
            "discriminant_position": discriminant_position,
            "class_separation_ratio": class_separation_ratio,
            "eigenvalues_per_class": eigenvalues_per_class,
        }
        return data, labels, info


# =============================================================================
# Wishart / Inverse-Wishart generator (merged)
# =============================================================================

class WishartGenerator:
    """
    Generate Wishart or Inverse-Wishart distributed SPD matrices.

    Uses random SPD matrices (via random_SPD) as scale matrices, with small
    perturbations to create class separation.

    - discriminant_position='large': sample from Wishart(df, scale)
    - discriminant_position='small': sample from InverseWishart(df, scale)
    """

    def __init__(
        self,
        matrix_size: int = 32,
        n_classes: int = 3,
        seed: int = 42,
    ):
        self.matrix_size = matrix_size
        self.n_classes = n_classes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _generate_scales(
        self,
        conditioning: float,
        perturbation_strength: float = 0.25,
    ) -> np.ndarray:
        """
        Generate per-class scale matrices by perturbing a shared base SPD matrix.

        Returns:
            Array of shape (n_classes, matrix_size, matrix_size).
        """
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)

        base_scale_torch = random_SPD(
            n_features=self.matrix_size,
            n_matrices=1,
            cond=conditioning,
            dtype=torch.float64,
            generator=generator,
        )
        base_scale = base_scale_torch.numpy()
        if base_scale.ndim == 3:
            base_scale = base_scale[0]

        scales = []
        for i in range(self.n_classes):
            torch.manual_seed(self.seed + i + 1000)
            gen = torch.Generator().manual_seed(self.seed + i + 1000)
            pert_torch = random_SPD(
                n_features=self.matrix_size,
                n_matrices=1,
                cond=5.0,
                dtype=torch.float64,
                generator=gen,
            )
            pert = pert_torch.numpy()
            if pert.ndim == 3:
                pert = pert[0]

            pert = pert / np.linalg.norm(pert, "fro")
            pert = pert * np.linalg.norm(base_scale, "fro") * perturbation_strength

            perturbed = base_scale + pert
            perturbed = (perturbed + perturbed.T) / 2
            perturbed += np.eye(self.matrix_size) * 1e-6
            scales.append(perturbed)

        return np.array(scales)

    def generate_data(
        self,
        n_samples_per_class: int,
        conditioning: float,
        df: int,
        discriminant_position: str,
        perturbation_strength: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate Wishart or Inverse-Wishart distributed data.

        Args:
            n_samples_per_class: Number of samples per class.
            conditioning: Condition number for random SPD scale matrices.
            df: Degrees of freedom (must be > matrix_size - 1).
            discriminant_position: 'large' for Wishart, 'small' for Inverse-Wishart.
            perturbation_strength: Strength of inter-class perturbations.

        Returns:
            data, labels, info
        """
        assert df > self.matrix_size - 1, (
            f"df ({df}) must be > matrix_size - 1 ({self.matrix_size - 1})"
        )

        scales = self._generate_scales(conditioning, perturbation_strength)

        data_list = []
        labels_list = []

        for class_idx in range(self.n_classes):
            if discriminant_position == "small":
                samples = invwishart(df=df, scale=scales[class_idx]).rvs(
                    size=n_samples_per_class, random_state=self.rng,
                )
            else:
                samples = wishart(df=df, scale=scales[class_idx]).rvs(
                    size=n_samples_per_class, random_state=self.rng,
                )

            if samples.ndim == 2:
                samples = samples[np.newaxis, :]

            data_list.append(samples)
            labels_list.append(np.full(n_samples_per_class, class_idx))

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # Eigenvalue statistics (sample first 30 matrices)
        eigvals_sample = np.array(
            [np.linalg.eigvalsh(m) for m in data[:30]]
        )

        info = {
            "generator": "wishart" if discriminant_position == "large" else "invwishart",
            "matrix_size": self.matrix_size,
            "n_classes": self.n_classes,
            "n_samples_per_class": n_samples_per_class,
            "df": df,
            "conditioning": conditioning,
            "perturbation_strength": perturbation_strength,
            "discriminant_position": discriminant_position,
            "eigenvalue_mean": float(eigvals_sample.mean()),
            "eigenvalue_std": float(eigvals_sample.std()),
            "eigenvalue_min": float(eigvals_sample.min()),
            "eigenvalue_max": float(eigvals_sample.max()),
        }
        return data, labels, info
