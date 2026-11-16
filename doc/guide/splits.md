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

## Per dataset

*Origin*: **paper** = split of the reference publication; **proposed** = no
usable split in the paper (or it leaks), a grouped split is provided instead.

| Dataset | Default split | Origin | Group kept together | Checked in the tests | Remaining risk |
|---|---|---|---|---|---|
| ESC-50 | 5 folds | paper | source recording (fold column) | no shared fold | 4 `src_file` appear in two folds (from the source) |
| GTZAN | stratified 5 folds | proposed | — (no artist metadata) | 14 exact duplicate pairs removed | artist effect: the artist-filtered split cannot be rebuilt; a `UserWarning` is raised |
| BirdSet_XCM | StratifiedGroupKFold(5) | proposed | recordist | no shared recordist | partial copy (21 shards) |
| HDM05 | stratified 5 folds | proposed | — (no actor id in the copy) | — | **actor leakage cannot be removed**; documented |
| NTU RGB+D 120 | X-Sub, X-Setup | paper | subject / setup | no shared subject / setup | — |
| CI4R | GroupKFold(4) by participant (leave-one-participant-out for 77 GHz; xethru folds group 7 participants) | proposed | participant (inferred from file names) | no shared participant, MD5 duplicates removed | three codes confound class and participant |
| MVDoppler | 4 official folds | paper | subject | no shared subject in test | 50 % overlapping snapshots within a recording |
| DopNet | official train / test, `person0..5` | paper / proposed | person (LOPO) | MD5 test∈train duplicates removed | the official split is intra-person |
| PASTIS-R | folds 1–3 / 4 / 5, `cv1..cv5` | paper | patch | no shared patch | neighbouring patches across folds (official design) |
| GSOFF, Chikusei, UAV-HSI-Crop | spatial blocks (96, 320, 480 px) | proposed | spatial block, buffer of one window | no overlapping windows | GSOFF: unknown provenance |
| Kaggle wheat | stratified 5 folds | proposed | — (no group available) | all-zero patches removed | patches of one field may fall on both sides |
| Placenta | grouped by image | proposed | image (`sample_id`) | no shared image | 4 placentas without identifier |
| Rices_90 | bundle 1 → bundle 2 | proposed | cube (variety–session) | grains of a cube kept together | variety and session confounded for 73 varieties |
| HyperLeaf2024 | grouped split | proposed | plot | no shared plot | official test set unlabelled |
| DeepHS-Fruit ⚠️ | grouped by physical fruit | proposed | fruit (both faces) | no shared fruit | the v2 official split leaks (46 / 140 test cubes) |
| Pol-InSAR-Island | 5 folds on 512 m UTM blocks, guard band | proposed | spatial block | no shared block, guard band | official checkerboard is optimistic (warning) |
| CPAZMaL | StratifiedGroupKFold(5) | proposed | site | no shared site | only 66–67 parcels by default (no parcel shapefile) |
| S1SLC_CVDL | leave-one-city-out | proposed | city | no shared city | radiometric shift between cities |
| AIR-PolSAR-Seg | 1 024 px macro-blocks of the rebuilt mosaic | proposed | macro-block | no shared cell (MD5) | the official image split overlaps and is not provided |
| ATRNet-STAR | `SOC_40classes`, `EOC_*` | paper | — | no shared file name or MD5 within a protocol | same vehicles across train / test; heavy chip reuse **between** protocols (`protocol_overlap()`) |
| MSTAR | SOC 17° / 15° | paper | — | — | same vehicles in train and test (protocol design) |
| FUSAR-Ship 1.0 | StratifiedGroupKFold(5) | proposed | MMSI (ship identity) | no shared MMSI | — |

Datasets only available on the `local` branch (S1 Iceland SAR, AVIRIS) document
their splits on their own pages.

## Checks

{mod}`spdnet_datasets.core.splits` provides the assertions used in the test
suite: {func}`~spdnet_datasets.core.assert_no_leakage` (no shared group),
{func}`~spdnet_datasets.core.assert_no_window_overlap` (no intersecting windows)
and {func}`~spdnet_datasets.core.assert_no_duplicates` (no shared content hash).
