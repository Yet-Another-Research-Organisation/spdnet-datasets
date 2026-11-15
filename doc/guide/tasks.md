# Tasks

The same samples can serve several tasks, chosen in the constructor
({mod}`spdnet_datasets.core.tasks`).

| Argument | Values | Effect |
|---|---|---|
| `task` | `"fine"` (default) | every class of the dataset |
| | `"coarse"` | classes regrouped with the dataset's coarse map (documented on each dataset page) |
| | `"regression"` | continuous targets, when the dataset has some (e.g. AVIRIS MIX class proportions, HyperLeaf traits) |
| `drop_classes` | iterable of class names or codes | samples of these classes are removed |
| `merge_classes` | `{old: new}` | classes renamed or merged, after dropping |
| `balance` | `None`, `"undersample"`, `"oversample"`, `"weights"` | class balancing of the **training** split; `"weights"` only exposes `dataset.class_weights` (inverse frequency) for a weighted loss |

Order of operations: drop, then coarse map, then merge, then re-indexing
(`0..k-1`, sorted class names) and balancing.

```python
ds = PastisRDataset(scenario="A", drop_classes=["Meadow"], balance="weights")
loss = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(ds.class_weights))
```

Balancing never touches the validation and test splits, so the reported
scores keep the natural class distribution.
