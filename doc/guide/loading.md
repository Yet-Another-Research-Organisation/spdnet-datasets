# Loading a dataset

`DatasetManager.create_dataloaders(config)` builds the dataset registered
under `config["name"]`, splits it into train / validation / test, and
returns `(train_loader, val_loader, test_loader, num_classes)`.
`DatasetManager.create_dataset(config)` returns the dataset alone.

## Configuration keys

The keys are grouped by role. Keys that belong to neither the split nor the
loaders are passed to the dataset constructor, so every dataset-specific
option ({doc}`datasets`) goes in the same dictionary.

| Role | Key | Default | Meaning |
|---|---|---|---|
| **Which dataset** | `name` | required | Name under which the loader is registered (`"hyperleaf"`, `"hdm05"`, …) |
| **Where** | `path` | required | Data root; becomes the `data_dir` of the dataset |
| | `subdir` | none | Joined to `path` (`/DATA` + `HyperLeaf2024`) |
| **Subset** | `max_classes`, `classes` | all | `max_classes` classes drawn at random (fixed seed 42, independent of `seed`), or an explicit list |
| | `max_samples_per_class` | all | Cap per class (drawn with `seed`) |
| **Split** | `val_ratio`, `test_ratio` | 0.1, 0.2 | Fractions of the whole dataset |
| | `stratified` | `False` | Keep the class proportions in every split (recommended with many classes) |
| | `seed` | 42 | Seed of the subset draw and of the split |
| **Loaders** | `batch_size` | 32 | Same for the three loaders |
| | `shuffle` | `True` | Shuffle the training loader |
| | `num_workers`, `pin_memory`, `persistent_workers` | 4, `True`, `True` | Passed to `torch.utils.data.DataLoader` |
| **Output** | `verbose` | `True` | Print the dataset summary and the split distribution |
| **Dataset-specific** | anything else | | Forwarded to the dataset constructor |

## How the split is made

The test set is taken first (`test_ratio` of the dataset), then the
validation set from the rest, so that `val_ratio` and `test_ratio` are both
fractions of the whole dataset. With `stratified=True` both draws are
stratified by label (`sklearn.model_selection.train_test_split`); otherwise
they are random permutations. The split depends only on `seed`: two
configurations with the same seed and dataset get the same test set, whatever
their batch size.

## What a batch contains

Each item is `(x, label)`. In the covariance modes, `x` is an SPD matrix of
shape `(n, n)`, so a batch is `(B, n, n)`, the input of the
`yetanotherspdnet` models; with `mode="image"` (HyperLeaf) it is an image
`(C, H, W)` for a CNN backbone. `num_classes` comes from the dataset after the
class subset.

## From a Hydra configuration

In `spdnet-training`, the `dataset` config group holds exactly these keys:

```yaml
# configs/dataset/hdm05.yaml
name: hdm05
path: /DATA
subdir: HDM05
scaling_factor: 190.0
batch_size: 16
val_ratio: 0.15
test_ratio: 0.15
stratified: true
```

and any key can be overridden on the command line
(`spdnet-train dataset=hdm05 dataset.batch_size=32`).
