# nilearn adapter (fMRI connectomes)

[nilearn](https://nilearn.github.io) downloads and reads public functional
magnetic resonance imaging (fMRI) datasets. `spdnet_datasets.nilearn` turns
their region time series into one symmetric positive definite (SPD) covariance
matrix per subject: a *connectome*, ready for SPDNet models or a Riemannian
scikit-learn pipeline. nilearn does everything else: download, atlases and
signal extraction.

Five nilearn datasets are supported:

| `dataset` | nilearn fetcher | Samples | Label | Groups |
|---|---|---|---|---|
| `"abide"` | `fetch_abide_pcp` | 871 subjects (quality-checked) | autism (1) vs control (0) | acquisition site (20) |
| `"adhd"` | `fetch_adhd` | 40 subjects | ADHD (1) vs control (0) | acquisition site (5) |
| `"development"` | `fetch_development_fmri` | 155 subjects | child vs adult | subject (single site) |
| `"nki"` | `fetch_surf_nki_enhanced` | 102 subjects | sex (F / M), or age | subject (single site) |
| `"ds000030"` | `fetch_ds000030_urls` + `fetch_openneuro_dataset` | 261 subjects with a preprocessed resting state (of 272) | control, ADHD, bipolar, schizophrenia; or one of them vs control | subject (single site, 2 scanners) |

- ABIDE: Autism Brain Imaging Data Exchange.
- PCP: the Preprocessed Connectomes Project, which shares ABIDE time series
  already averaged over regions of interest (ROI). See Craddock et al., 2013.
- ADHD: attention-deficit/hyperactivity disorder. nilearn ships a 40-subject
  sample of ADHD-200.
- MSDL: multi-subject dictionary learning, a probabilistic atlas of 39 regions.
- NKI: the Nathan Kline Institute Rockland Sample. nilearn ships the resting
  state of 102 adults projected on the fsaverage5 cortical surface; the adapter
  averages it in the 148 regions of the Destrieux atlas.
- ds000030: the UCLA Consortium for Neuropsychiatric Phenomics LA5c study
  (Poldrack et al., 2016), release R1.0.4 on OpenNeuro, with its fMRIPrep 1.0.4
  derivatives. Only the resting-state run is used.

```{dataset-preview} nilearn
```

## Install

```bash
pip install "spdnet-datasets[nilearn]"
```

The `nilearn` extra pulls in `nilearn>=0.14.1`. nilearn stores the downloads in
its own data directory. Set that directory before the first run, or pass
`data_dir=`:

```bash
export NILEARN_DATA=/path/to/nilearn_data     # default: ~/nilearn_data
```

For ABIDE, only the ROI time series (`.1D` text files, about 150 kB per subject
for the TT atlas) are downloaded, not the images. `adhd` and `development`
download preprocessed 4D images (tens of MB per subject) and extract the MSDL
signals with a `NiftiMapsMasker`, regressing out the confounds that ship with
the data (`confounds=True`).

## Quick start: ABIDE

```python
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from spdnet_datasets.base import create_dataloaders
from spdnet_datasets.nilearn import NilearnConnectomes

ds = NilearnConnectomes("abide", atlas="tt", split="site_grouped")
print(len(ds), ds.n_excluded_short, ds.n_excluded_not_spd)

# SPDNet: no acquisition site is shared between train, val and test
train, val, test = create_dataloaders(ds, batch_size=32, num_workers=0,
                                      split="site_grouped")
x, y = next(iter(train))      # x: (32, 97, 97) float64, y: (32,) in {0, 1}

# scikit-learn baseline on the same split
part = ds.splits["site_grouped"]
X, y = ds.covariances, ds.labels
pipe = make_pipeline(TangentSpace(), LogisticRegression(C=0.1, max_iter=2000))
pipe.fit(X[part["train"]], y[part["train"]])
print(pipe.score(X[part["test"]], y[part["test"]]))
```

`fetch_abide_pcp(n_subjects=N)` keeps the *first* N rows of the phenotypic
table, and that table is grouped by site, starting with PITT. A small
`n_subjects` therefore gives a single site. To select sites or subjects, pass nilearn's phenotypic filters:
`fetcher_kwargs={"SITE_ID": ["NYU", "UM_1"]}` or `{"SUB_ID": [...]}`.
`fetcher_kwargs` also takes the PCP options (`pipeline`,
`band_pass_filtering`, `global_signal_regression`, `quality_checked`) and, for
`development`, `age_group`.

To work on raw time series instead, the `Covariances` transformer maps a list of
`(T_i, p)` arrays to `(n_subjects, p, p)`. It replaces
`nilearn.connectome.ConnectivityMeasure(kind="covariance")` and enforces the
same checks:

```python
from spdnet_datasets.nilearn import Covariances

pipe = make_pipeline(Covariances("scm"), TangentSpace(), LogisticRegression())
```

## Output contract

Each item is `(x, y)`. `y` indexes `ds.classes`: `["control", "autism"]`,
`["control", "adhd"]`, `["adult", "child"]`, `["F", "M"]` or, for
`ds000030`, `["control", "adhd", "bipolar", "schz"]`. With `task="regression"`, `y`
is the age in years, as a float.

| Flags | `x` |
|---|---|
| default | `cov`, shape `(p, p)`, float64 |
| `return_mean=True` | `(cov, mean)`: mean of each region signal *before* standardization, `(p,)`, float64 |
| `return_raw=True` | `(cov, raw)`: the time series `(T, p)`, float32 (z-scored if `standardize=True`) |
| both | `(cov, mean, raw)` |

`T` differs between sites. To batch `return_raw`, use `batch_size=1` or write a
custom `collate_fn`.

With `standardize=True` (the default), each region is z-scored over time before
the covariance, so the SCM (sample covariance matrix) is a correlation matrix.
`estimator` is one of `"scm"`, `"ledoit_wolf"`, `"tyler"` or `"student"`.

Other attributes:

- `ds.metadata`, one row per kept subject: `subject`, `site`, `label`, `age`,
  `sex` and `n_timepoints`;
- `ds.phenotypic`: the full nilearn phenotypic rows of the kept subjects;
- `ds.groups`: the site for `abide` and `adhd`, the subject for the
  single-site datasets (`development`, `nki`, `ds000030`);
- `ds.sites`: the acquisition site (the scanner for `ds000030`).

One subject gives exactly one sample. A duplicated subject identifier raises an
error.

## The n ≥ 2p rule and the choice of atlas

`spdnet-datasets` estimates a covariance only when there are at least twice as
many observations as dimensions. Here that means `T ≥ 2p` time points for `p`
regions. In ABIDE, `T` is fixed by each site's scanning protocol: the median is
176, with a range of 78 (OHSU) to 316. The rule therefore drops whole sites,
not random subjects.

The table below was measured on BERTHA on 2026-11-15, on the 871
quality-checked subjects (PCP `cpac`, `nofilt_noglobal`).

- "T ≥ 2p" counts all 871 subjects. Its median T is that of the subjects
  passing the rule; the median over all 871 is 176 for every atlas.
- "Constant ROI" comes from a sample of 5 subjects per site (100 subjects). A
  region outside a subject's field of view has a constant (zero) signal. Its
  SCM is singular, so the subject is excluded as not SPD.
- `T` was checked to be the same for every derivative of a subject.

For the default atlas, the full cohort was run through `NilearnConnectomes`.
It keeps **375 / 871 subjects (43.1 %)** from 12 sites: 447 are excluded for
`T < 194` and 49 for a constant ROI. 194 are controls and 181 have autism, and
the median T is 236.

| Atlas (`atlas=`) | p | 2p | T ≥ 2p (871) | Median T (kept) | Sites left | Constant ROI (sample) |
|---|---|---|---|---|---|---|
| `tt` (Talairach) **default** | 97 | 194 | 424 (48.7 %) | 236 | 12 | 15 % |
| `ho` (Harvard-Oxford) | 111 | 222 | 266 (30.5 %) | 246 | 7 | 36 % |
| `aal` | 116 | 232 | 266 (30.5 %) | 246 | 7 | 10 % |
| `ez` (Eickhoff-Zilles) | 116 | 232 | 266 (30.5 %) | 246 | 7 | 9 % |
| `dosenbach160` | 161 | 322 | 0 | – | 0 | 9 % |
| `cc200` (Craddock 200) | 200 | 400 | 0 | – | 0 | 6 % |
| `cc400` (Craddock 400) | 392 | 784 | 0 | – | 0 | 12 % |

T per site: OHSU 78; MAX_MUN, UCLA_1 and UCLA_2 116 (a few MAX_MUN at 196);
KKI 152; CALTECH and TRINITY 146; NYU and SDSU 176; PITT, SBL and YALE 196;
OLIN 206; CMU 202–316; USM 232–236; STANFORD 176–236; LEUVEN_1 and LEUVEN_2
246; UM_1 and UM_2 296.

The ABIDE default is therefore `tt`. It keeps the most subjects (43 %) and
the most sites (12) of all ROI derivatives. The kept cohort is *site-selected*:
the short-scan sites (NYU, UCLA, KKI, OHSU, ...) are gone. Results on it are
not comparable with results on the full ABIDE.

For `adhd` and `development`, MSDL (p = 39) needs T ≥ 78. The length of these
series was not measured here, because their images were not downloaded. Some
ADHD-200 sites have short scans, so check `n_excluded_short`.

Subjects that fail the rule are excluded and counted:

- `n_excluded_short`: `T < 2p`;
- `n_excluded_not_spd`: the covariance is not SPD, typically because of a
  constant ROI;
- `n_excluded`: their sum.

If no subject is left, an error is raised.

### Large atlases: `allow_small_n=True`

The spdlearn example
["Classifying fMRI data with SPDNet"](https://spdlearn.org/generated/auto_examples/applied_examples/plot_fmri.html)
uses CC200 (p = 200) with `ConnectivityMeasure(kind="covariance")`. That class
uses Ledoit-Wolf shrinkage by default (`cov_estimator=None`) and standardizes
the series, so it never checks `T` against `p`. To reproduce that setup:

```python
ds = NilearnConnectomes("abide", atlas="cc200", allow_small_n=True,
                        fetcher_kwargs={"SUB_ID": ids})
```

`allow_small_n=True` warns, ignores `estimator`, and estimates *every*
covariance with Ledoit-Wolf, without the `T ≥ 2p` check. The result is still
checked to be SPD. With `T` around 176 and p = 200, the SCM has rank below `p`,
so the SPD structure of these matrices comes from the shrinkage, not from the
data. Shrinkage also hides constant ROIs, because the identity term makes
them SPD. Treat such matrices as regularized estimates.

Other differences with the spdlearn example:

- here the split is grouped by site;
- here the series are z-scored with `ddof=1`;
- spdlearn takes the first 400 quality-checked subjects. They come from
  10 of the 20 sites, and none of them has T ≥ 400.

## Site leakage and splits

ABIDE and ADHD-200 are multi-site datasets. Scanner and protocol effects are
strong enough for a classifier to learn the site instead of the diagnosis
(Dadi et al., 2019). Every split is `{"train", "val", "test"}` at the subject
level:

| Split | Construction | Use |
|---|---|---|
| `stratified` | stratified on the label, 10 % test (as in spdlearn), 10 % val | ⚠️ sites mixed between train and test: optimistic; a warning is raised |
| `site_grouped` | `StratifiedGroupKFold` on sites: test = fold 0 of `min(5, n_sites)`, val = fold 0 of the remaining sites; `assert_no_leakage` on every pair | `abide`, `adhd` |
| `site_fold_<k>` | leave-one-site-out: test = every subject of the k-th site (sites sorted by name), val = the next site, train = the others | `abide`, `adhd`: one fold per site |
| `stratified_fold_<k>` | fold k of a shuffled 5-fold `StratifiedKFold` on the label (`KFold` for regression); val = fold k + 1 | single-site datasets: `development`, `nki`, `ds000030` |
| `recommended` | `site_grouped`, or `stratified` for the single-site datasets (one sample per subject) | default |

Each split is built on its own. On a small cohort, the `stratified` test and
validation parts are enlarged to hold at least one subject per class; a split
that still cannot be built (too few subjects per class or per site) is left
out with a warning, and the others remain available. Asking for it explicitly
raises a `ValueError` that gives the reason.

For `adhd`, the cohorts `Peking_1`, `Peking_2` and `Peking_3` count as one site,
`Peking`. With `normalize=True`, the geometric scale (see
{doc}`/guide/outputs`) is fitted on the training indices of `split`; the means
and the time series are then multiplied by `sqrt(s)`, so that they stay consistent
with the covariances. Pass the same name to `create_dataloaders(split=...)`.

### Cross-validation over the folds

Over all the folds of a family, every subject is tested exactly once.
`cross_validate_folds` scores a model on each fold and summarises the result:

```python
from spdnet_datasets.nilearn import NilearnConnectomes, cross_validate_folds

def fit_predict(X_train, y_train, X_test):
    pipe = make_pipeline(TangentSpace(), StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced"))
    return pipe.fit(X_train, y_train).predict(X_test)

ds = NilearnConnectomes("adhd", split="site_fold_0")
res = cross_validate_folds(ds, fit_predict)          # prefix="site_fold" by default
print(res["scores"], res["mean"], res["ci"], res["p_value"])
```

- `X` is the unscaled covariances `ds.covs`, so no scale fitted on another
  fold leaks into the test fold; each fold trains on its `train` and `val`
  subjects.
- The score is the balanced accuracy by default (R² for regression): its
  chance level is 1 / n_classes whatever the class balance.
- `ci` is the 95 % bootstrap interval (1 000 resamples of the subjects) of the
  score of the pooled out-of-fold predictions.
- `p_value` comes from a permutation test: the labels are shuffled *within each
  site* (1 000 permutations by default) and the whole cross-validation is run
  again, so a site effect cannot pass for a diagnosis effect.
  p = (1 + #{null ≥ observed}) / (1 + n_permutations).

````{admonition} ADHD: how to read a score
:class: warning

The nilearn ADHD-200 sample has 40 subjects; 2 KKI subjects (T = 77 < 2p = 78)
are excluded, which leaves 38, 19 with ADHD and 19 controls, in 5 sites:

| Site | ADHD | Control |
|---|---|---|
| KKI | 3 | 3 |
| NYU | 4 | 4 |
| NeuroImage | 4 | 4 |
| OHSU | 4 | 4 |
| Peking (Peking_1–3) | 4 | 4 |

- A single test site holds 6 to 8 subjects. On 8 subjects, the 95 % interval
  of an accuracy of 0.5 is about ±0.35: one `site_grouped` score says nothing.
- Use the 5 `site_fold_<k>` folds with `cross_validate_folds`, and report the
  mean, the interval and the permutation p-value.
- Measured on BERTHA (2026-11-19), MSDL, tangent space + standard scaling +
  logistic regression (C = 0.1, balanced class weights): balanced accuracy
  per left-out site KKI 0.33, NYU 0.75, NeuroImage 0.38, OHSU 0.63,
  Peking 0.13; mean **0.44**, 95 % CI [0.29, 0.61], permutation p = 0.75
  (1 000 permutations within sites). This is chance level.
- For a diagnosis effect on more subjects, use `ds000030` (see below), a
  single-centre cohort with 40 ADHD subjects and 122 controls.
````

### Indicative score

These numbers are only indicative: 85 subjects, a single seed, no tuning.

- Data: ABIDE, `tt`, 100 subjects drawn from the 12 sites with T ≥ 194. 85
  were kept; 15 were excluded for a constant ROI.
- Pipeline: `TangentSpace()` → `StandardScaler` → `LogisticRegression(C=0.1)`.

| Protocol | Accuracy | Chance |
|---|---|---|
| `site_grouped` (3 test sites, 21 subjects) | 0.476 | 0.506 |
| 5-fold leave-sites-out (`StratifiedGroupKFold`) | 0.531 ± 0.109 | 0.506 |
| 5-fold stratified, sites mixed | 0.647 ± 0.037 | 0.506 |

The gap between the last two rows is the site effect described above.

## NKI enhanced (surface)

```python
from spdnet_datasets.nilearn import NilearnConnectomes

ds = NilearnConnectomes("nki")                      # sex, 148 x 148 covariances
age = NilearnConnectomes("nki", task="regression")  # age in years
```

- Regions: the 74 Destrieux labels of each hemisphere, `Unknown` and
  `Medial_wall` excluded (p = 148); a region signal is the mean of its vertices.
  With 895 time points per subject, the n ≥ 2p rule holds.
- The covariances are very ill-conditioned (κ around 1e10, measured on the
  first subjects): neighbouring regions of the smoothed surface data are highly
  correlated. Ledoit-Wolf (`estimator="ledoit_wolf"`) improves this.
- Single site: the groups are the subjects and only the `stratified` split
  exists.

```{note}
nilearn 0.14.1 reads the NKI phenotypic file with `header=1`, so its first
subject (A00028185) becomes the header row and has no age or sex. The adapter
reads the downloaded file again with its real header, so every subject keeps
its phenotype.
```

## ds000030 (UCLA CNP, OpenNeuro)

```python
from spdnet_datasets.nilearn import NilearnConnectomes

ds = NilearnConnectomes("ds000030", split="stratified_fold_0")   # 4 diagnoses
adhd = NilearnConnectomes("ds000030", diagnosis="adhd")          # ADHD vs control
age = NilearnConnectomes("ds000030", task="regression")          # age in years
```

The UCLA Consortium for Neuropsychiatric Phenomics LA5c study (Poldrack et
al., 2016) scanned 272 adults: healthy controls and patients with
schizophrenia, bipolar disorder or ADHD. The adapter uses release R1.0.4 on
OpenNeuro and its fMRIPrep 1.0.4 derivatives.

- **Download.** `fetch_ds000030_urls()` gives the index of the release (about
  411 000 files). Only the files matched by `DS000030_FILES` are passed to
  `fetch_openneuro_dataset`: `participants.tsv` and, per subject,
  `derivatives/fmriprep/sub-<id>/func/sub-<id>_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz`
  and `sub-<id>_task-rest_bold_confounds.tsv`. On BERTHA (2026-11-19) these 523 files
  took **25.8 GB** and 28 minutes to download (about 99 MB per subject).
  `fetch_openneuro_dataset` also adds symbolic links with the newer BIDS
  derivative names; the adapter reads the original files.
- **Subjects.** 261 of the 272 participants have a preprocessed resting state
  (122 controls, 40 ADHD, 49 bipolar, 50 schizophrenia). The 11 others
  (8 controls, 3 ADHD) are excluded and counted in `n_excluded_missing`.
- **Signals.** MSDL atlas (p = 39) with a `NiftiMapsMasker`, TR = 2 s,
  T = 152 volumes (128 for one subject), so T ≥ 2p = 78 holds for every
  subject. The confounds are
  the 6 motion parameters and the white-matter signal, with their first
  differences (14 regressors, `DS000030_CONFOUNDS`). The fMRIPrep 1.0.4
  tables of this release have no CSF column, and the global signal is not
  regressed out. `confounds=False` keeps the raw MSDL signals.
- **Labels.** `task="fine"`: `["control", "adhd", "bipolar", "schz"]`.
  `diagnosis="adhd"` (or `"bipolar"`, `"schz"`) keeps the controls and that
  diagnosis, a binary task. `diagnosis` is applied after the cache, so
  switching it does not extract the signals again. `task="regression"`: age.
- **Groups and splits.** One centre, so the groups are the subjects:
  `stratified` and the 5 `stratified_fold_<k>` folds. The participants were
  scanned on two Siemens Trio scanners, and the controls mostly on one of them
  (106 of 130). `ds.sites` holds the scanner (`scanner_35343`,
  `scanner_35426`), and `cross_validate_folds` permutes the labels within each
  scanner.

Measured on BERTHA (2026-11-19), 261 subjects, MSDL, `cross_validate_folds`
on the 5 `stratified_fold_<k>` folds, tangent space + standard scaling +
logistic regression (C = 0.1, balanced class weights), 1 000 permutations
within scanners. κ (condition number) of the covariances: median 544, range
161–2 969.

| Task | Subjects | Balanced accuracy per fold | Mean | 95 % CI | p | Chance |
|---|---|---|---|---|---|---|
| 4 diagnoses | 261 (122 / 40 / 49 / 50) | 0.36, 0.39, 0.39, 0.37, 0.44 | **0.39** | [0.33, 0.45] | 0.001 | 0.25 |
| ADHD vs control | 162 (40 / 122) | 0.46, 0.51, 0.50, 0.52, 0.65 | **0.53** | [0.46, 0.59] | 0.25 | 0.50 |
| Schizophrenia vs control | 172 (50 / 122) | 0.63, 0.63, 0.81, 0.69, 0.67 | **0.69** | [0.61, 0.76] | 0.001 | 0.50 |
| Bipolar vs control | 171 (49 / 122) | 0.59, 0.64, 0.71, 0.59, 0.49 | **0.60** | [0.52, 0.67] | 0.004 | 0.50 |

Schizophrenia is the most separable diagnosis from resting-state connectomes
here, bipolar disorder is weaker, and ADHD is not distinguished from controls
in this cohort: the same conclusion as the leave-one-site-out ADHD-200 result
above, on a cohort four times larger.

## Cache and `keep`

nilearn manages its files under `data_dir`, and the adapter never deletes them.
It uses `run_pipeline` without a download stage. Under
`SPDNET_CACHE_DIR/nilearn_<dataset>/` it caches:

- `processed/<hash>/`: the time series (all subjects stacked, float32), their
  lengths, the region means, the metadata and the phenotypic table;
- `covariance/<hash>/`: the unscaled covariances, the means, the kept-subject
  index, the exclusion counts and the metadata.

The `processed` hash covers `dataset`, `atlas`, `n_subjects`, `standardize`,
`confounds` (not for `abide`), `fetcher_kwargs` and a pipeline version. The
`covariance` hash adds the effective estimator and `allow_small_n`, so
changing the estimator does not extract the series again. `force=True`
recomputes both.

| `keep` | Cache content |
|---|---|
| `keep_full` (default) or `keep_processed` | time series and covariances |
| `keep_covariance` | covariances only; `return_raw` is then unavailable |

## Tested versions

| Package | Version |
|---|---|
| nilearn | 0.14.1 |
| nibabel | 5.5.0 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.1 |
| torch | 2.14.1 |
| pyriemann | 0.12 |
| Python | 3.11 |

The real-data test (`tests/unit/test_nilearn_real.py`, `-m "slow and network"`)
covers ABIDE, `tt`, with 20 subjects from PITT, USM, UM_1 and YALE.
`adhd`, `development`, `nki` and `ds000030` are covered by unit tests only,
because their images are too large for CI; they were run on real data in the
end-to-end validation.

## References

- Craddock, C., Benhajali, Y., Chu, C., et al. (2013). The Neuro Bureau
  Preprocessing Initiative: open sharing of preprocessed neuroimaging data and
  derivatives. *Frontiers in Neuroinformatics* (ABIDE Preprocessed,
  <http://preprocessed-connectomes-project.org/abide/>).
- Dadi, K., Rahim, M., Abraham, A., et al. (2019). Benchmarking functional
  connectome-based predictive models for resting-state fMRI. *NeuroImage*, 192,
  115–134.
- Nooner, K. B., et al. (2012). The NKI-Rockland Sample: a model for
  accelerating the pace of discovery science in psychiatry. *Frontiers in
  Neuroscience*, 6, 152.
- Poldrack, R. A., Congdon, E., Triplett, W., et al. (2016). A phenome-wide
  examination of neural and cognitive function. *Scientific Data*, 3, 160110.
- Destrieux, C., Fischl, B., Dale, A., Halgren, E. (2010). Automatic
  parcellation of human cortical gyri and sulci using standard anatomical
  nomenclature. *NeuroImage*, 53, 1–15.
- spdlearn, [Classifying fMRI data with SPDNet](https://spdlearn.org/generated/auto_examples/applied_examples/plot_fmri.html).

## How to cite

````{admonition} If you use this adapter in a publication, cite
:class: tip

```{include} _cite/nilearn.md
```
````
