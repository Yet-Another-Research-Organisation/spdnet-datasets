# Synthetic data

`spdnet_datasets.synthetic` generates SPD datasets whose class differences
are known, to study what a network or a batch normalization can separate.

## Generators

`ScaleMatrixGeneratorDiagonal`
: Each class has its own spectrum $\lambda^{(c)}$; a sample is
  $X = Q \operatorname{diag}(\lambda^{(c)}) Q^\top$ with a random rotation $Q$
  per sample. Classes differ only through their eigenvalues.

`ScaleMatrixGeneratorBlock`
: Same, with block-structured matrices.

`WishartGenerator`
: Each class has a scale matrix $\Sigma_c$ (a shared random SPD matrix,
  perturbed per class); samples are drawn from $\mathcal{W}(\Sigma_c, \nu)$
  (`discriminant_position="large"`) or the inverse Wishart (`"small"`).

## Controlling the spectrum

`generate_eigenvalues` builds the spectrum of each class:

- `max_value` and `conditioning` set the range
  $[\lambda_{\max} / \kappa, \lambda_{\max}]$;
- `mode` spaces the eigenvalues: `"geomspace"`, `"linspace"`, or
  `"constant"` (`n_discriminant` eigenvalues at the maximum, the others at the
  minimum);
- `n_discriminant` eigenvalues, at the large end, the small end or both
  (`discriminant_position`), are multiplied by
  $(1 + \texttt{class\_separation\_ratio})^{c}$ for class $c$.

```python
from spdnet_datasets.synthetic import ScaleMatrixGeneratorDiagonal, SimulationDataset

generator = ScaleMatrixGeneratorDiagonal(matrix_size=16, n_classes=3, seed=42)
data, labels, info = generator.generate_data(
    n_samples_per_class=120, max_value=100.0, conditioning=100.0,
    mode="geomspace", n_discriminant=2, discriminant_position="small",
)
dataset = SimulationDataset(data, labels)
```

`ExperimentConfig` and the grids of `synthetic.config` (`GRID_REGISTRY`,
`get_configs_for_grid`) describe the parameter sweeps of the batch
normalization study.
