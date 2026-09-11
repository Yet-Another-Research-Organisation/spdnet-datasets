# Copilot Instructions — spdnet-datasets

## Build & Test

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file or test
pytest tests/test_foo.py
pytest tests/test_foo.py::test_bar

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```

Ruff is configured with 88-char line length, targeting Python 3.11+.

## Architecture

This is a **shared infrastructure library** for SPDNet research. It provides dataset loaders, covariance estimators, and synthetic data generators for Symmetric Positive Definite (SPD) matrix classification.

### Core layer (`spdnet_datasets/`)

- **`base.py`** — `BaseDataset(Dataset, ABC)` defines the template for all datasets: subclasses implement `_load_data()` and `__getitem__()`. Also provides `create_dataloaders()` with stratified/random splitting.
- **`manager.py`** — `DatasetManager` is a class-level registry. Datasets register via `@DatasetManager.register_dataset('name')` decorator and are instantiated through `DatasetManager.create_dataloaders(config_dict)`.

### Real datasets (`real/`)

Nine dataset classes, each decorated with `@DatasetManager.register_dataset(...)`:

| Registration name | Class | Domain |
|---|---|---|
| `rices90` | `Rices90Dataset` | Rice varieties (pre-computed .pt) |
| `hyperleaf` | `HyperLeafDataset` | Barley leaf hyperspectral |
| `hdm05` | `HDM05Dataset` | Human motion capture |
| `uav` | `UAVDataset` | UAV crop classification |
| `gsoff` | `GSOffDataset` | Tree species hyperspectral |
| `chikusei` | `ChikuseiDataset` | Land cover classification |
| `deephsfruit` | `DeepHSFruitDataset` | Fruit classification |
| `placenta` | `PlacentaDataset` | Tissue hyperspectral |
| `kaggle_wheat` | `KaggleWheatDataset` | Wheat disease |

Most datasets support two modes: `'cov'` (pre-computed covariance matrices) and `'raw'` (compute from source images on the fly).

### Synthetic data (`synthetic/`)

- **`config.py`** — `ExperimentConfig` dataclass with parameter grids and presets for systematic experiments.
- **`data_generator.py`** — Three SPD matrix generation strategies: `ScaleMatrixGeneratorDiagonal`, `ScaleMatrixGeneratorBlock`, `WishartGenerator`.
- **`dataset.py`** — `SimulationDataset` wraps generated data as a PyTorch `Dataset`.

### Covariance estimation (`estimator/`)

- `EstimateCovariance` — NumPy-based (supports SCM and Ledoit-Wolf methods).
- `EstimateCovarianceTorch` — PyTorch-based (GPU-compatible, SCM only).

Both provide `from_image()` and `from_batch()` convenience methods.

### Utilities (`utils/`)

Preprocessing scripts for extracting windows from raw imagery and precomputing covariance matrices. These run standalone, separate from the dataset loading pipeline.

## Conventions

### Adding a new dataset

1. Create a new file in `src/spdnet_datasets/real/`.
2. Subclass `BaseDataset` and implement `_load_data()` and `__getitem__()`.
3. Decorate the class with `@DatasetManager.register_dataset('name')`.
4. Export the class from `real/__init__.py` and `spdnet_datasets/__init__.py`.
5. The dataset is automatically available via `DatasetManager.create_dataloaders({'name': '...', ...})`.

### Dataset config dictionaries

Datasets are instantiated through config dicts passed to `DatasetManager`. The manager maps `'path'` → `'data_dir'` and strips dataloader-specific keys (`batch_size`, `val_ratio`, etc.) before passing to the dataset constructor.

```python
config = {
    'name': 'rices90',
    'path': '/data/rices90',
    'max_classes': 50,
    'batch_size': 32,
    'stratified': True,
    'seed': 42,
}
train_loader, val_loader, test_loader, num_classes = DatasetManager.create_dataloaders(config)
```

### Type hints

Use type hints on all public functions and methods. Use `typing` module types (`Tuple`, `Optional`, `List`, `Dict`, `Literal`).

### External dependency on sibling package

The synthetic Wishart generator imports `yetanotherspdnet.random.spd.random_SPD` from the sibling `spdnet` package. This is the only cross-package dependency.


## Code Quality Standards

- Write clear, compact, and human-readable code without emojis
- Avoid code duplication; reuse functions and modules whenever possible
- Prioritize execution speed, memory efficiency, and CPU/GPU optimization
- Make all modifications using the minimum amount of code possible
- Ensure all changes are easily explainable and understandable by humans
- Include comments for complex logic or non-obvious optimizations
- Favor readability over clever or overly concise implementations
