# MVDoppler

```{list-table}
:stub-columns: 1

* - Domain
  - micro-Doppler radar (gait patterns, two views)
* - Task(s)
  - fine classification (4 walking patterns)
* - Covariance
  - 64×64 joint two-view covariance ("C2"), n = 256 frames (n / 2p = 2)
* - Outputs
  - cov · mean · raw (256 × 64)
* - Samples
  - 54 834 snapshots (54 836 in `design_table.csv`, 2 unreadable files) from 4 210 recordings of 13 subjects
* - Splits
  - official 4-fold, disjoint in subjects: `fold0`…`fold3`; `official` = `fold<fold>` (default `fold0`: 39 362 / 4 069 / 11 403)
* - Module
  - `spdnet_datasets.radar_microdoppler.mvdoppler` (`fetch_mvdoppler`, `MVDopplerDataset`, registry name `mvdoppler`)
```

## Description and source

MVDoppler records 13 subjects walking in front of two millimetre-wave (mmWave)
FMCW (frequency-modulated continuous wave) radars with orthogonal views (x and
y), with 4 patterns: normal walk, phone call, hands in pockets, texting. The
recordings are cut into snapshots of 1.28 s (256 Doppler frames) with a step
of 0.64 s, i.e. a 50 % overlap.

- Reference: S. Hor, S. Yang, J. Choi and A. Arbabian, "MVDoppler: Unleashing
  the Power of Multi-View Doppler for MicroMotion-Based Gait Classification",
  *NeurIPS (Neural Information Processing Systems) 2023 Datasets and
  Benchmarks*, vol. 36, pp. 58064–58074.
- Source: <https://github.com/soheilhr/MVDoppler> (data on the authors' Google
  Drive; manual download). Licence: not stated in the copy.
- The paper mainly evaluates two binary tasks (hands detection, distraction
  detection); this loader classifies the 4 patterns, as the data atlas.
- Local copy: `<data root>/RADAR_DOPPLER/MVDoppler/Data/<pattern>_<ID>.h5` and
  `Labels_and_metadata/design_table.csv`.

## Raw data

Each HDF5 (Hierarchical Data Format 5) file holds `radar_dat` of shape
(128 Doppler bins, 256 frames, 2) in float64, plus attributes (subject,
pattern, trajectory, `snr_db_x`, `snr_db_y`).

**The last axis holds the two radar views, not I/Q (in-phase / quadrature)
components.** All values are positive amplitudes (0.42 to 43.8), the
attributes give one signal-to-noise ratio (SNR) per view, `meta_data.json`
sets `merge_radar_spectrograms: true`, and the two channels are almost
uncorrelated (correlation 0.09) with different Doppler peaks. The figure shows
the preprocessed (frame × 64) data of four snapshots: bins 0–31 are view x,
32–63 view y.

```{raw} html
<iframe src="../_static/figures/mvdoppler/raw_samples.html" class="plotly-figure" loading="lazy"></iframe>
```

Skipped and counted (`dataset._bunch.info`): 2 files without the HDF5
signature (`phone_call_02515`, `texting_12478`) and 24 `.h5` files absent from
`design_table.csv`.

## Classes

normal 14 570, phone_call 13 151, pockets 13 266, texting 13 847 snapshots.
The histogram shows `fold0`.

```{raw} html
<iframe src="../_static/figures/mvdoppler/class_hist.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/mvdoppler/class_means.html" class="plotly-figure" loading="lazy"></iframe>
```

## Covariance construction

"C2" is the name of the joint x–y covariance in the data atlas sheet
`mvdoppler` (C1 being the 32×32 single-view one). The paper itself defines no
"C2" protocol; the definition below follows the atlas.

1. Per view, log amplitude `log(max(a, 1e-12))`.
2. The 128 Doppler bins are averaged by blocks of 4, giving 32 bins per view.
3. The two views are stacked into p = 64 channels, so the covariance holds the
   cross-view correlations.
4. One symmetric positive definite (SPD) covariance over the 256 frames
   (n = 256 = 4p) with the chosen estimator (`"scm"`, the sample covariance
   matrix, by default; with n / p = 4, `"ledoit_wolf"` is better conditioned).

With `normalize=True` (default), the covariances (not the means and raw data)
are multiplied by the scalar fitted on the training part of the chosen split
(geometric mean of the eigenvalues = 1).

```{raw} html
<iframe src="../_static/figures/mvdoppler/cov_samples.html" class="plotly-figure" loading="lazy"></iframe>
<iframe src="../_static/figures/mvdoppler/eigen_spectrum.html" class="plotly-figure" loading="lazy"></iframe>
```

## Splits

The official 4-fold cross-validation of `design_table.csv` (column `fold`) is
disjoint in subjects: fold 0 = subjects 4, 6, 11; fold 1 = 2, 8, 10; fold 2 =
1, 7, 9; fold 3 = 0, 3, 5, 12. For `fold<k>`: test = fold k; validation = the
rows flagged `val_set` in the other folds (the flag is constant per
recording, so no overlapping snapshot crosses train and validation);
train = the rest. Groups are subjects.

| Split | train | val | test |
|---|---|---|---|
| `fold0` (= `official` with `fold=0`) | 39 362 | 4 069 | 11 403 |
| `fold1` | 37 222 | 3 871 | 13 741 |
| `fold2` | 38 237 | 3 910 | 12 687 |
| `fold3` | 34 129 | 3 702 | 17 003 |

```{raw} html
<iframe src="../_static/figures/mvdoppler/split_schema.html" class="plotly-figure" loading="lazy"></iframe>
```

## Leakage risks and measures

| Risk | Measure | Residual |
|---|---|---|
| snapshots of one recording overlap by 50 % (median 13 per recording) | official folds by subject; validation flag constant per recording | none; never split snapshots at random |
| same subject in train and test | official folds disjoint in subjects (tested) | 13 subjects only: high variance between folds |
| 10 `exp_fname` appear in two folds (timestamps shared between subjects, atlas) | none needed: folds are defined by subject | documented |

## Supported tasks

- `task="fine"`: 4 patterns. No coarse grouping is defined; the binary tasks
  of the paper can be built with `merge_classes` (e.g. hands busy = phone_call
  + texting).
- `drop_classes`, `classes`; `balance="undersample" | "oversample"`
  (training part of every split); class weights on the training part.

## Usage

```python
from spdnet_datasets import create_dataloaders
from spdnet_datasets.radar_microdoppler.mvdoppler import MVDopplerDataset

ds = MVDopplerDataset(fold=0, estimator="ledoit_wolf")       # every snapshot, splits fold0..fold3
train, val, test = create_dataloaders(ds, split="official", batch_size=64)
```

Changing the estimator recomputes the preprocessing stage (about 1 min,
3.6 GB of cache).

## Migration notes

- **Size change: 96×96 → 64×64.** The former precompute
  (`utils/mvdoppler_covariance_precompute.py`) took `sqrt(c0² + c1²)` as an
  I/Q modulus, which mixes the two views; the new covariance keeps the views
  separate.
- `scaling_factor` is replaced by `normalize=True`.
- `subset=None` is the default: the instance holds every snapshot.
