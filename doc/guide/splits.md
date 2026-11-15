# Splits and data leakage

A random split of samples is optimistic as soon as two samples share
something the model can memorise: the same subject, the same recording, the
same image tile, overlapping windows, or the same physical target. Every dataset
therefore exposes:

- `dataset.groups`: one group identifier per sample;
- `dataset.splits`: named splits `{"train": idx, "val": idx, "test": idx}`.
  `"official"` is the split of the reference paper when there is one;
  `"recommended"` is a leakage-free split otherwise.

```python
from spdnet_datasets import create_dataloaders
train, val, test = create_dataloaders(ds, split="official", batch_size=32)
```

`create_dataloaders` without `split=` still performs the former sample-level
split, and warns when the dataset provides groups.

## Strategies

| Risk | Datasets | Strategy |
|---|---|---|
| same subject, speaker, person or session | audio, skeletons, micro-Doppler | split by subject / recordist / session (official folds when available) |
| same patch or neighbouring parcels | PASTIS-R | official folds by patch |
| overlapping sliding windows | hyperspectral, SAR, CPAZMaL | spatial blocks + buffer: test windows closer than one window to a training window are removed |
| same target in different conditions | ATRNet-STAR, MSTAR | official SOC/EOC protocols; overlap between protocols checked; the remaining same-vehicle effect is documented |
| dates mixed between train and test | S1 Iceland, CPAZMaL, PASTIS-R C–E | temporal or geographic separation, dates recorded per sample |
| exact duplicates | DopNet, GTZAN | duplicates (MD5 of the signal) removed or kept on one side only |

Each dataset page states the split used, its sizes, whether it comes from the
paper, and the leakage risks that remain.

## Checks

{mod}`spdnet_datasets.core.splits` provides the assertions used in the test
suite: {func}`~spdnet_datasets.core.assert_no_leakage` (no shared group),
{func}`~spdnet_datasets.core.assert_no_window_overlap` (no intersecting windows)
and {func}`~spdnet_datasets.core.assert_no_duplicates` (no shared content hash).
