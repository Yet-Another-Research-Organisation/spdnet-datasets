# nilearn adapter (fMRI connectomes)

[nilearn](https://nilearn.github.io) downloads and reads public functional
magnetic resonance imaging (fMRI) datasets. `spdnet_datasets.nilearn` turns
their region time series into one symmetric positive definite (SPD) covariance
matrix per subject: a *connectome*, ready for SPDNet models or a Riemannian
scikit-learn pipeline. nilearn does everything else: download, atlases and
signal extraction.

Three nilearn datasets are supported:

| `dataset` | nilearn fetcher | Subjects | Label | Groups |
|---|---|---|---|---|
| `"abide"` | `fetch_abide_pcp` | 871 (quality-checked) | autism (1) vs control (0) | acquisition site (20) |
| `"adhd"` | `fetch_adhd` | 40 | ADHD (1) vs control (0) | acquisition site (5) |
| `"development"` | `fetch_development_fmri` | 155 | child vs adult | subject (single site) |

- ABIDE: Autism Brain Imaging Data Exchange.
- PCP: the Preprocessed Connectomes Project, which shares ABIDE time series
  already averaged over regions of interest (ROI). See Craddock et al., 2013.
- ADHD: attention-deficit/hyperactivity disorder. nilearn ships a 40-subject
  sample of ADHD-200.
- MSDL: multi-subject dictionary learning, a probabilistic atlas of 39 regions.

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
`["control", "adhd"]` or `["adult", "child"]`. With `task="regression"`, `y`
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
- `ds.groups`: the site for `abide` and `adhd`, the subject for `development`.

One subject gives exactly one sample. A duplicated subject identifier raises an
error.

## The n ≥ 2p rule and the choice of atlas

`spdnet-datasets` estimates a covariance only when there are at least twice as
many observations as dimensions. Here that means `T ≥ 2p` time points for `p`
regions. In ABIDE, `T` is fixed by each site's scanning protocol: the median is
176, with a range of 78 (OHSU) to 316. The rule therefore drops whole sites,
not random subjects.

The table below was measured on BERTHA on 2026-11-15. "T ≥ 2p" counts all 871
quality-checked subjects (PCP `cpac`, `nofilt_noglobal`). "Constant ROI" and
"Kept" come from a sample of 5 subjects per site (100 subjects). A region
outside a subject's field of view has a constant (zero) signal, so its SCM is
singular and the subject is excluded as not SPD. `T` was checked to be the same
for every derivative of a subject.

| Atlas (`atlas=`) | p | 2p | T ≥ 2p (871) | Sites left | Constant ROI (sample) | Kept (sample) |
|---|---|---|---|---|---|---|
| `tt` (Talairach) **default** | 97 | 194 | 424 (48.7 %) | 12 | 15 % | 40 / 100 |
| `ho` (Harvard-Oxford) | 111 | 222 | 266 (30.5 %) | 7 | 36 % | 21 / 100 |
| `aal` | 116 | 232 | 266 (30.5 %) | 7 | 10 % | 27 / 100 |
| `ez` (Eickhoff-Zilles) | 116 | 232 | 266 (30.5 %) | 7 | 9 % | 27 / 100 |
| `dosenbach160` | 161 | 322 | 0 | 0 | 9 % | 0 / 100 |
| `cc200` (Craddock 200) | 200 | 400 | 0 | 0 | 6 % | 0 / 100 |
| `cc400` (Craddock 400) | 392 | 784 | 0 | 0 | 12 % | 0 / 100 |

T per site: OHSU 78; MAX_MUN, UCLA_1 and UCLA_2 116 (a few MAX_MUN at 196);
KKI 152; CALTECH and TRINITY 146; NYU and SDSU 176; PITT, SBL and YALE 196;
OLIN 206; CMU 202–316; USM 232–236; STANFORD 176–236; LEUVEN_1 and LEUVEN_2
246; UM_1 and UM_2 296.

The ABIDE default is therefore `tt`: it is the only ROI derivative that keeps
about half the cohort, across 12 sites. The kept cohort is *site-selected*:
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
| `recommended` | `site_grouped`, or `stratified` for `development` (single site, one sample per subject) | default |

For `adhd`, the cohorts `Peking_1`, `Peking_2` and `Peking_3` count as one site,
`Peking`. With `normalize=True`, the geometric scale (see
{doc}`outputs`) is fitted on the training indices of `split`. Pass the same
name to `create_dataloaders(split=...)`.

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
`adhd` and `development` are covered by unit tests only, because their images
are too large for CI.

## References

- Craddock, C., Benhajali, Y., Chu, C., et al. (2013). The Neuro Bureau
  Preprocessing Initiative: open sharing of preprocessed neuroimaging data and
  derivatives. *Frontiers in Neuroinformatics* (ABIDE Preprocessed,
  <http://preprocessed-connectomes-project.org/abide/>).
- Dadi, K., Rahim, M., Abraham, A., et al. (2019). Benchmarking functional
  connectome-based predictive models for resting-state fMRI. *NeuroImage*, 192,
  115–134.
- spdlearn, [Classifying fMRI data with SPDNet](https://spdlearn.org/generated/auto_examples/applied_examples/plot_fmri.html).
