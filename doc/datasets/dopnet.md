# DopNet

```{list-table}
:stub-columns: 1

* - Domain
  - micro-Doppler radar (hand gestures)
* - Task(s)
  - fine classification (4 gestures: Click, Pinch, Swipe, Wave)
* - Covariance
  - 48×48, n = T frames of the gesture (96 to 540, median 155; n / 2p ≥ 1)
* - Outputs
  - cov · mean · raw (T × 48, variable T)
* - Samples
  - 2 145 gestures (1 753 training file, 392 test file) of 3 076, after removing duplicates and gestures with T < 96
* - Splits
  - `official` (training files / test file: 1 753 / 0 / 392, default of the class) · `person0`…`person5` leave-one-person-out; `recommended` = `person<fold>` (default `person0`)
* - Module
  - `spdnet_datasets.radar_microdoppler.dopnet` (`fetch_dopnet`, `DopNetDataset`, registry name `dopnet`)
```

## Description and source

Dop-NET is a micro-Doppler radar data challenge: complex spectrograms of 4 hand
gestures (Wave, Pinch, Swipe, Click) made by 6 persons (A to F) about 30 cm in
front of an Ancortek 24 GHz FMCW (frequency-modulated continuous wave) radar.

- Reference: M. Ritchie, R. Capraru and F. Fioranelli, "Dop-NET: a micro-Doppler
  radar data challenge", *Electronics Letters* 56(11), 568–570, 2020,
  doi:[10.1049/el.2019.4153](https://doi.org/10.1049/el.2019.4153).
- Source: <https://dop-net.com> (registration; manual download). Licence: not
  stated in the copy (public challenge data).
- The paper announces 3 452 files from 10 persons; the copy holds 6 persons
  and 2 433 + 643 gestures.
- Local copy: `<data root>/RADAR_DOPPLER/Dop-NET/Training Data/*.mat` and
  `Test Data/Data_For_Test_Random.mat`.

## Raw data

Each gesture is a complex spectrogram (800 Doppler bins × T frames, complex128),
T variable (43 to 540).

- `Training Data/Data_Per_PersonData_Training_Person_<X>.mat`:
  `Data_Training.Doppler_Signals` is a 1 × 4 cell array, one cell per gesture.
  **The cells are ordered Wave, Pinch, Swipe, Click** (cells 0, 1, 2, 3).
  The order is checked on the exact copies shared by the labelled test file
  and the training files: every copy agrees (Click = cell 3, Pinch = 1,
  Swipe = 2, Wave = 0).
- `Test Data/Data_For_Test_Random.mat`: `Data_rand` (643, 2), the signal and a
  label string such as `"Click 20 Person F"` (same six persons).

Class indices of the dataset follow the alphabetical order
`CLASSES = ["Click", "Pinch", "Swipe", "Wave"]`, which differs from the cell
order. The figure shows the preprocessed (frame × 48) log10 band power of four
gestures.

```{raw} html
<iframe src="../_static/figures/dopnet/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

## Classes

Kept gestures in the official split: training file Click 375, Pinch 556,
Swipe 394, Wave 428; test file Click 76, Pinch 126, Swipe 91, Wave 99. After
deduplication and the T ≥ 96 rule, 451 of the 1 000 Click gestures remain.
Gesture durations differ by class: median T of the kept gestures is 131
(Click), 118 (Pinch), 241 (Swipe) and 209 (Wave) frames.

```{raw} html
<iframe src="../_static/figures/dopnet/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/dopnet/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

1. **Duplicates.** Exact duplicates (MD5, Message-Digest 5, hash of the complex array) are
   removed, keeping the first copy, training files first: 222 copies inside
   the training files, 133 on the test side (111 test gestures identical to a
   training gesture, 22 repeated inside the test file).
2. **Length.** Gestures with T < 2p = 96 frames are rejected and counted:
   458 in the training files, 118 in the test file.
3. **Band.** The smallest band centred on zero Doppler (bin 400) that holds at
   least `energy = 0.95` of the mean normalised power profile of the kept
   training gestures: bins [132, 668), 536 bins (95.0 % of the energy).
4. **Decimation.** The band power `|S|²` is averaged into p = 48 contiguous
   blocks (a box anti-aliasing filter), then `log10(· + 1e-12)`.
5. One symmetric positive definite (SPD) covariance over the T frames
   (`"scm"`, the sample covariance matrix, by default).

With `normalize=True` (default), the covariances (not the means and raw data)
are multiplied by the scalar fitted on the training part of the chosen split
(geometric mean of the eigenvalues = 1).

```{raw} html
<iframe src="../_static/figures/dopnet/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/dopnet/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

- `"official"`: training files against the test file, after deduplication, so
  no test gesture equals a training one. The same six persons are on both
  sides: it is an intra-person split. 1 753 / 0 / 392.
- `"person<k>"` (LOPO, leave-one-person-out, over all gestures of both files):
  test = person k (0 = A … 5 = F). `"recommended"` is `person<fold>`.

| Split | official | person0 (A) | person1 (B) | person2 (C) | person3 (D) | person4 (E) | person5 (F) |
|---|---|---|---|---|---|---|---|
| train | 1 753 | 1 852 | 1 681 | 1 819 | 1 730 | 1 812 | 1 831 |
| test | 392 | 293 | 464 | 326 | 415 | 333 | 314 |

Validation parts are empty. Groups are persons. The figures show `official`.

```{raw} html
<iframe src="../_static/figures/dopnet/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| 111 test gestures (17 %) identical to training gestures | global MD5 deduplication, training files first (tested: no shared hash) | none for exact copies |
| same person in train and test (official split) | `person<k>` splits for inter-person generalisation (tested) | official split stays intra-person |
| band fitted on the training files | — | in `person<k>`, the band (one hyperparameter) is fitted on data that include the held-out person |
| duration bias: Swipe and Wave are long (median T 241 and 209), Click and Pinch short (131 and 118) | none (T is the covariance sample size, not an input) | a classifier can exploit the duration through the raw output or the estimation noise; compare with a duration-only baseline |

## Supported tasks

- `task="fine"`: 4 gestures. No coarse grouping is defined.
- `drop_classes`, `merge_classes`, `classes`; `balance="undersample" |
  "oversample"` (training part of every split); class weights on the
  training part.
- `energy`, `p` and `min_frames` (≥ 2p) change the construction.
- `raw` has a variable length T: use `batch_size=1` or a custom collate
  function with `return_raw=True`.

## Usage

```python
from spdnet_datasets import create_dataloaders
from spdnet_datasets.radar_microdoppler.dopnet import DopNetDataset

ds = DopNetDataset()                                    # split="official", every gesture
train, val, test = create_dataloaders(ds, split="official", batch_size=32)
lopo = DopNetDataset(split="recommended", fold=2)       # test = person C
```

## Migration notes

- **Label permutation fixed.** The former `cov/` precompute named the training
  cells Click, Swipe, Pinch, Wave; they are Wave, Pinch, Swipe, Click, so its
  training labels were permuted.
- `DopNetDataset(split="all")` now means the `official` split (every gesture,
  `dataset.splits` indexed); `split="train"` / `"test"` select that part of
  `official`.
- Duplicates are removed and gestures with T < 96 are rejected (2 145
  gestures instead of 3 076).
- `scaling_factor` is replaced by `normalize=True`.
