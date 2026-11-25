"""nilearn adapter: ``NilearnConnectomes`` with fake nilearn fetchers (no network)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

from spdnet_datasets.core.estimators import check_spd
from spdnet_datasets.core.splits import assert_no_leakage
from spdnet_datasets.nilearn import ABIDE_ATLASES, Covariances, NilearnConnectomes
from spdnet_datasets.nilearn import dataset as nl_mod

nl_datasets = pytest.importorskip("nilearn.datasets")

SITES = ["NYU", "UM_1", "USM", "Pitt", "Yale", "KKI"]
P_TT = ABIDE_ATLASES["tt"]  # 97 regions


def _series(rng, T, p, label):
    x = rng.standard_normal((T, p))
    x[:, 0] += (2.0 if label == 1 else 0.0) * x[:, 1]  # class-dependent correlation
    return x * 3.0 + 10.0


class FakeABIDE:
    """Stand-in for ``nilearn.datasets.fetch_abide_pcp``: 6 sites x 4 subjects."""

    def __init__(self, T_short_site="KKI", T=None, T_short=None, constant=None):
        self.calls, self.kwargs = 0, []
        self.T, self.T_short, self.T_short_site, self.constant = T, T_short, T_short_site, constant

    def __call__(self, data_dir=None, n_subjects=None, derivatives=None, verbose=1, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        rng = np.random.default_rng(0)
        rows = [{"SUB_ID": 50000 + 10 * i + k, "SITE_ID": s, "DX_GROUP": 1 + k % 2,
                 "AGE_AT_SCAN": 10.0 + i + k, "SEX": 1 + (k // 2), "FILE_ID": f"{s}_{k}"}
                for i, s in enumerate(SITES) for k in range(4)]
        ph = pd.DataFrame(rows)
        if "SITE_ID" in kwargs:
            ph = ph[ph["SITE_ID"].isin(kwargs["SITE_ID"])]
        if n_subjects is not None:
            ph = ph[:n_subjects]
        (deriv,) = derivatives
        p = ABIDE_ATLASES[deriv.removeprefix("rois_")]
        series = []
        for j, r in enumerate(ph.itertuples()):
            T = ((self.T_short or p) if r.SITE_ID == self.T_short_site
                 else (self.T or 2 * p + 10))  # KKI below the 2p rule
            ts = _series(rng, T, p, 2 - r.DX_GROUP)
            if self.constant is not None and j == self.constant:
                ts[:, 3] = 7.0
            series.append(ts)
        return Bunch(phenotypic=ph, **{deriv: series})


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def abide(monkeypatch):
    fake = FakeABIDE()
    monkeypatch.setattr(nl_datasets, "fetch_abide_pcp", fake)
    return fake


def _make(cache, **kw):
    kw.setdefault("dataset", "abide")
    return NilearnConnectomes(cache_dir=cache, **kw)


# ---------------------------------------------------------------- ABIDE
def test_abide_shapes_labels_groups(cache, abide):
    ds = _make(cache)
    # KKI (T = p < 2p) is excluded whole
    assert len(ds) == 20 and ds.n_excluded == ds.n_excluded_short == 4
    assert "KKI" not in set(ds.groups)
    x, y = ds[0]
    assert tuple(x.shape) == (P_TT, P_TT) and str(x.dtype) == "torch.float64"
    assert ds.classes == ["control", "autism"] and y in (0, 1)
    check_spd(ds.covariances)
    assert ds.metadata.columns.tolist() == ["subject", "site", "label", "age", "sex",
                                            "n_timepoints"]
    assert set(ds.metadata["sex"]) == {"M", "F"} and "DX_GROUP" in ds.phenotypic
    # DX_GROUP 1 (autism) -> 1, 2 (control) -> 0
    assert (ds.labels == (ds.phenotypic["DX_GROUP"] == 1)).all()
    assert ds.metadata["subject"].is_unique


def test_standardized_series_give_correlations(cache, abide):
    ds = _make(cache, normalize=False)
    np.testing.assert_allclose(np.diagonal(ds.covariances, axis1=1, axis2=2),
                               (ds.metadata["n_timepoints"].to_numpy()[:, None] - 1)
                               / ds.metadata["n_timepoints"].to_numpy()[:, None]
                               * np.ones(P_TT), rtol=1e-5)
    raw = _make(cache, normalize=False, standardize=False)
    assert raw.covariances[0, 0, 0] > 5  # scale 3 -> variance ~ 9


@pytest.mark.parametrize("return_mean,return_raw", [(False, False), (True, False),
                                                    (False, True), (True, True)])
def test_output_tuples(cache, abide, return_mean, return_raw):
    ds = _make(cache, return_mean=return_mean, return_raw=return_raw)
    root = np.sqrt(ds.scale)  # mean and raw are scaled by sqrt(s), like cov by s
    x, _ = ds[1]
    parts = x if isinstance(x, tuple) else (x,)
    assert len(parts) == 1 + return_mean + return_raw
    if return_mean:
        assert tuple(parts[1].shape) == (P_TT,)
        assert abs(float(parts[1].mean()) / root - 10.0) < 1.0  # mean before standardization
    if return_raw:
        raw = parts[-1]
        assert tuple(raw.shape) == (2 * P_TT + 10, P_TT) and str(raw.dtype) == "torch.float32"
        np.testing.assert_allclose(raw.numpy().std(axis=0, ddof=1), root, rtol=1e-4)


def test_raw_series_match_covariance(cache, abide):
    ds = _make(cache, return_raw=True, normalize=False)
    cov, raw = ds[2][0]
    np.testing.assert_allclose(Covariances().fit_transform([raw.numpy().astype(float)])[0],
                               cov.numpy(), atol=1e-5)


def test_normalised_raw_matches_covariance(cache, abide):
    """With a split, cov is scaled by s and raw / mean by sqrt(s)."""
    plain = _make(cache, return_raw=True, return_mean=True, normalize=False, split="stratified")
    ds = _make(cache, return_raw=True, return_mean=True, split="stratified")
    assert ds.scale != 1.0
    cov, mean, raw = ds[2][0]
    np.testing.assert_allclose(Covariances().fit_transform([raw.numpy().astype(float)])[0],
                               cov.numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(mean.numpy(), plain[2][0][1].numpy() * np.sqrt(ds.scale),
                               rtol=1e-6)


def test_allow_small_n_keeps_everyone_with_ledoit_wolf(cache, abide):
    with pytest.warns(UserWarning, match="allow_small_n"):
        ds = _make(cache, allow_small_n=True, estimator="scm")
    assert len(ds) == 24 and ds.n_excluded == 0 and ds.estimator == "ledoit_wolf"
    check_spd(ds.covariances)


def test_not_spd_subject_is_excluded_and_counted(cache, monkeypatch):
    monkeypatch.setattr(nl_datasets, "fetch_abide_pcp", FakeABIDE(constant=0, T_short_site=None))
    ds = _make(cache, standardize=False)
    assert ds.n_excluded_not_spd == 1 and ds.n_excluded_short == 0 and len(ds) == 23
    assert 0 not in ds.subject_index


def test_no_subject_left_raises(cache, monkeypatch):
    monkeypatch.setattr(nl_datasets, "fetch_abide_pcp", FakeABIDE(T=P_TT))
    with pytest.raises(ValueError, match="No subject left"):
        _make(cache)


def test_duplicate_subject_raises(cache, monkeypatch):
    fake = FakeABIDE()

    def dup(**kw):
        b = fake(**kw)
        b.phenotypic["SUB_ID"] = 1
        return b

    monkeypatch.setattr(nl_datasets, "fetch_abide_pcp", dup)
    with pytest.raises(ValueError, match="Duplicate"):
        _make(cache)


def test_estimator_change_reuses_the_time_series(cache, abide):
    scm = _make(cache)
    lw = _make(cache, estimator="ledoit_wolf")
    assert abide.calls == 1
    assert scm.paths.processed_dir == lw.paths.processed_dir
    assert scm.paths.covariance_dir != lw.paths.covariance_dir
    _make(cache)
    assert abide.calls == 1
    _make(cache, force=True)
    assert abide.calls == 2


def test_cache_key_covers_atlas_and_options(cache, abide):
    tt, ho = _make(cache, atlas="tt"), _make(cache, atlas="ho")
    assert tt.paths.processed_dir != ho.paths.processed_dir and ho.covs.shape[1] == 111
    _make(cache, fetcher_kwargs={"SITE_ID": ["NYU", "UM_1", "USM"]})
    _make(cache, standardize=False)
    _make(cache, n_subjects=12)
    assert abide.calls == 5
    assert abide.kwargs[2] == {"SITE_ID": ["NYU", "UM_1", "USM"]}
    assert len(list((cache / "nilearn_abide" / "processed").iterdir())) == 5
    # a NumPy array of IDs is keyed by value, not by its truncated repr
    ids = np.arange(50000, 50000 + 10 * len(SITES), 10)
    _make(cache, fetcher_kwargs={"SUB_ID": ids})
    _make(cache, fetcher_kwargs={"SUB_ID": ids.tolist()})
    assert abide.calls == 6


def test_keep_covariance_forbids_raw(cache, abide):
    ds = _make(cache, keep="keep_covariance")
    assert ds.paths.processed_dir is None
    with pytest.raises(ValueError, match="return_raw"):
        _make(cache, keep="keep_covariance", return_raw=True)
    with pytest.raises(ValueError, match="deleted"):
        _make(cache, return_raw=True)


def test_regression_on_age(cache, abide):
    ds = _make(cache, task="regression")
    x, y = ds[0]
    assert isinstance(y, float) and ds.classes == []
    np.testing.assert_allclose(ds.labels, ds.metadata["age"])
    assert_no_leakage(ds.sites[ds.splits["site_grouped"]["train"]],
                      ds.sites[ds.splits["site_grouped"]["test"]], what="site")


def test_bad_arguments(cache, abide):
    with pytest.raises(ValueError, match="dataset"):
        _make(cache, dataset="hcp")
    with pytest.raises(ValueError, match="atlas"):
        _make(cache, atlas="msdl")
    with pytest.raises(ValueError, match="task"):
        _make(cache, task="coarse")


# ---------------------------------------------------------------- splits
def test_site_grouped_split_shares_no_site(cache, abide):
    ds = _make(cache)
    sg = ds.splits["site_grouped"]
    assert ds.splits["recommended"] is sg
    parts = [sg["train"], sg["val"], sg["test"]]
    assert sum(map(len, parts)) == len(ds) and len(np.unique(np.concatenate(parts))) == len(ds)
    for a in range(3):
        for b in range(a + 1, 3):
            assert_no_leakage(ds.sites[parts[a]], ds.sites[parts[b]], what="site")
    assert len(sg["test"]) > 0 and len(sg["val"]) > 0


def test_stratified_split_warns_and_mixes_sites(cache, abide):
    with pytest.warns(UserWarning, match="optimistic"):
        ds = _make(cache, split="stratified")
    st = ds.splits["stratified"]
    assert len(np.intersect1d(st["train"], st["test"])) == 0
    assert sum(len(v) for v in st.values()) == len(ds)


def test_normalization_is_fitted_on_the_split_train(cache, abide):
    ds = _make(cache)
    train = ds.splits["recommended"]["train"]
    assert abs(np.mean(np.log(np.linalg.eigvalsh(ds.covariances[train])))) < 1e-10
    assert _make(cache, normalize=False).scale == 1.0
    with pytest.raises(ValueError, match="is not available"):
        _make(cache, split="nope")


def test_create_dataloaders_site_grouped(cache, abide):
    from spdnet_datasets.base import create_dataloaders

    ds = _make(cache, split="site_grouped")
    tr, va, te = create_dataloaders(ds, batch_size=4, num_workers=0, split="site_grouped",
                                    verbose=False)
    assert len(tr.dataset) + len(va.dataset) + len(te.dataset) == len(ds)
    x, y = next(iter(tr))
    assert tuple(x.shape) == (4, P_TT, P_TT) and y.dtype.is_floating_point is False


# ---------------------------------------------------------------- ADHD / development
def _fake_extract(func, confounds, atlas, data_dir, t_r):
    rng = np.random.default_rng(1)
    _fake_extract.confounds = confounds
    return [rng.standard_normal((176, 39)) for _ in func]


def test_adhd_sites_and_labels(cache, monkeypatch):
    n = 20
    ph = pd.DataFrame({"Subject": [f"{i:07d}" for i in range(n)],
                       "site": ["KKI", "NYU", "Peking_1", "Peking_2", "OHSU"] * 4,
                       "adhd": [0, 1] * 10, "age": np.linspace(8, 15, n), "sex": ["M", "F"] * 10})
    monkeypatch.setattr(nl_datasets, "fetch_adhd", lambda **kw: Bunch(
        func=[f"f{i}" for i in range(n)], confounds=[f"c{i}" for i in range(n)],
        phenotypic=ph, t_r=2.0))
    monkeypatch.setattr(nl_mod, "_extract_timeseries", _fake_extract)
    ds = _make(cache, dataset="adhd")
    assert ds.atlas == "msdl" and len(ds) == n and tuple(ds[0][0].shape) == (39, 39)
    assert ds.classes == ["control", "adhd"] and ds.labels.tolist() == [0, 1] * 10
    assert set(ds.sites) == {"KKI", "NYU", "Peking", "OHSU"}  # Peking_* merged
    assert ds.metadata["subject"].iloc[1] == "0000001"  # leading zeros kept
    assert ds.phenotypic["Subject"].iloc[1] == "0000001"
    assert _fake_extract.confounds == [f"c{i}" for i in range(n)]
    sg = ds.splits["site_grouped"]
    assert_no_leakage(ds.sites[sg["train"]], ds.sites[sg["test"]], what="site")
    _make(cache, dataset="adhd", confounds=False)
    assert _fake_extract.confounds is None


def test_development_child_adult(cache, monkeypatch):
    ph = pd.DataFrame({"participant_id": [f"sub-pixar{i:03d}" for i in range(20)],
                       "Age": np.linspace(4, 30, 20), "Child_Adult": ["child", "adult"] * 10,
                       "Gender": ["M", "F"] * 10})
    monkeypatch.setattr(nl_datasets, "fetch_development_fmri", lambda **kw: Bunch(
        func=[f"f{i}" for i in range(20)], confounds=[None] * 20, phenotypic=ph, t_r=2))
    monkeypatch.setattr(nl_mod, "_extract_timeseries", _fake_extract)
    ds = _make(cache, dataset="development")
    assert ds.classes == ["adult", "child"] and len(ds) == 20
    assert "site_grouped" not in ds.splits and ds.splits["recommended"] is ds.splits["stratified"]
    assert len(set(ds.groups)) == 20  # one subject per sample


def test_nilearn_resolves_to_the_pypi_package():
    import nilearn

    assert hasattr(nilearn, "datasets") and "spdnet_datasets" not in nilearn.__file__


# ------------------------------------------------------- review regressions
def test_eight_subject_cohort_builds_every_split(cache, abide):
    ds = _make(cache, fetcher_kwargs={"SITE_ID": ["NYU", "UM_1"]})  # 8 subjects, 4 + 4
    assert len(ds) == 8
    strat = ds.splits["stratified"]
    for part in ("val", "test"):  # at least one subject per class
        assert set(ds.labels[strat[part]]) == {0, 1}
    assert ds.splits["recommended"] is ds.splits["site_grouped"]
    assert_no_leakage(ds.sites[ds.splits["site_grouped"]["train"]],
                      ds.sites[ds.splits["site_grouped"]["test"]])


def test_infeasible_stratified_split_does_not_block_site_grouped(cache, abide):
    with pytest.warns(UserWarning, match="'stratified' cannot be built"):
        ds = _make(cache, n_subjects=5, fetcher_kwargs={"SITE_ID": ["NYU", "UM_1"]})
    assert "stratified" not in ds.splits and "site_grouped" in ds.splits
    assert ds.split == "recommended" and ds.splits["recommended"] is ds.splits["site_grouped"]
    with pytest.raises(ValueError, match="'stratified' is not available"):
        with pytest.warns(UserWarning):
            _make(cache, n_subjects=5, split="stratified",
                  fetcher_kwargs={"SITE_ID": ["NYU", "UM_1"]})


# ---------------------------------------------------------------- NKI enhanced
N_VERT, T_NKI = 60, 80


class FakeNKI:
    """Stand-in for ``fetch_surf_nki_enhanced`` and ``fetch_atlas_surf_destrieux``:
    3 labels per hemisphere plus Unknown and Medial_wall, 6 subjects."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "nki"
        self.surfaces = {}
        rng = np.random.default_rng(1)
        ids = [f"A000{k}" for k in range(6)]
        for k, sid in enumerate(ids):
            for hemi in ("left", "right"):
                path = self.root / sid / f"{sid}_{hemi}_preprocessed_fwhm6.gii"
                self.surfaces[str(path)] = rng.standard_normal((N_VERT, T_NKI)) + k
        # phenotypic table in another order than the files, with an extra subject
        self.ph = pd.DataFrame({"Subject": ids[::-1] + ["A000X"],
                                "Age": [20.0 + k for k in range(7)],
                                "Dominant Hand": "R",
                                "Sex": ["F", "M"] * 3 + ["F"]})
        self.ids = ids

    def fetch(self, n_subjects=10, data_dir=None, verbose=1, **kwargs):
        ids = self.ids[:n_subjects]
        return Bunch(func_left=[str(self.root / s / f"{s}_left_preprocessed_fwhm6.gii")
                                for s in ids],
                     func_right=[str(self.root / s / f"{s}_right_preprocessed_fwhm6.gii")
                                 for s in ids],
                     phenotypic=self.ph, description="")

    def atlas(self, data_dir=None, verbose=1, **kwargs):
        # vertices: 10 Unknown (0), 10 Medial_wall (4), 40 split over labels 1..3
        labels = np.array([0] * 10 + [4] * 10 + [1, 2, 3] * 13 + [1])
        return Bunch(labels=["Unknown", "a", "b", "c", "Medial_wall"],
                     map_left=labels, map_right=labels[::-1].copy())


@pytest.fixture
def nki(monkeypatch, tmp_path):
    fake = FakeNKI(tmp_path)
    monkeypatch.setattr(nl_datasets, "fetch_surf_nki_enhanced", fake.fetch)
    monkeypatch.setattr(nl_datasets, "fetch_atlas_surf_destrieux", fake.atlas)
    monkeypatch.setattr(nl_mod, "_read_surface", lambda path: fake.surfaces[str(path)])
    return fake


def test_nki_regions_labels_and_phenotypes(cache, nki):
    ds = _make(cache, dataset="nki", normalize=False, return_raw=True, standardize=False)
    assert len(ds) == 6 and ds.covariances.shape == (6, 6, 6)  # 3 labels x 2 hemispheres
    assert ds.classes == ["F", "M"]
    # phenotypic rows follow the files, not the table order
    assert ds.metadata["subject"].tolist() == nki.ids
    assert ds.metadata["age"].tolist() == [25.0, 24.0, 23.0, 22.0, 21.0, 20.0]
    # region 0 = mean of the left vertices of label 1
    labels = nki.atlas().map_left
    left = nki.surfaces[nki.fetch()["func_left"][0]]
    np.testing.assert_allclose(ds[0][0][1].numpy()[:, 0], left[labels == 1].mean(0), rtol=1e-5)
    assert set(ds.groups) == set(nki.ids)  # single site: subjects as groups
    age = _make(cache, dataset="nki", task="regression")
    assert age.labels.dtype == np.float32 and "site_grouped" not in age.splits


def test_nki_phenotypic_first_row_is_kept(cache, nki):
    """nilearn drops the first subject of the CSV (header=1): it is read again."""
    nki.root.mkdir(parents=True, exist_ok=True)
    (nki.root / nl_mod.NKI_PHENOTYPIC).write_text(
        ",Age,Dominant Hand,Sex\n"
        + "".join(f"{s},{30 + k},R,{'MF'[k % 2]}\n" for k, s in enumerate(nki.ids)))
    nki.ph = nki.ph[nki.ph["Subject"] != nki.ids[0]]  # what nilearn returns
    ds = _make(cache, dataset="nki", split=None)
    assert ds.metadata["subject"].tolist() == nki.ids
    assert ds.metadata["age"].tolist() == [30.0 + k for k in range(6)]


# ---------------------------------------------------------------- ds000030
DS30_URL = "https://s3.amazonaws.com/openneuro/ds000030/ds000030_R1.0.4/uncompressed/"
DS30_DIAG = ["CONTROL"] * 8 + ["ADHD"] * 6 + ["BIPOLAR"] * 5 + ["SCHZ"] * 6
DS30_MISSING = "sub-10013"  # an ADHD participant without a preprocessed resting state


class FakeDS30:
    """Stand-in for ``fetch_ds000030_urls`` and ``fetch_openneuro_dataset``: 25
    participants, one without fMRIPrep resting-state files."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "ds000030"
        self.ids = [f"sub-{10000 + k}" for k in range(len(DS30_DIAG))]
        self.urls = [DS30_URL + "participants.tsv", DS30_URL + "README"]
        for s in self.ids:
            self.urls.append(DS30_URL + f"{s}/func/{s}_task-rest_bold.nii.gz")  # raw BIDS
            if s == DS30_MISSING:
                continue
            d = DS30_URL + f"derivatives/fmriprep/{s}/func/{s}_task-"
            self.urls += [d + "rest_bold_confounds.tsv",
                          d + "rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz",
                          d + "rest_bold_space-T1w_preproc.nii.gz",
                          d + "bart_bold_confounds.tsv"]
        self.requested = []

    def index(self, data_dir=None, verbose=1):
        return "urls.json", list(self.urls)

    def fetch(self, urls=None, data_dir=None, verbose=1, **kwargs):
        self.requested += urls
        rng = np.random.default_rng(0)
        for u in urls:
            path = self.root / u.split("/uncompressed/", 1)[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            if u.endswith("participants.tsv"):
                pd.DataFrame({"participant_id": self.ids, "diagnosis": DS30_DIAG,
                              "age": np.arange(len(self.ids)) + 20.0,
                              "gender": ["F", "M"] * 12 + ["F"], "rest": 1,
                              "ScannerSerialNumber": [35343.0, 35426.0] * 12 + [35343.0]}
                             ).to_csv(path, sep="\t", index=False, na_rep="n/a")
            elif u.endswith("confounds.tsv"):
                c = pd.DataFrame(rng.standard_normal((152, 9)),
                                 columns=["WhiteMatter", "GlobalSignal", "X", "Y", "Z",
                                          "RotX", "RotY", "RotZ", "FramewiseDisplacement"])
                c.loc[0, "FramewiseDisplacement"] = np.nan
                c.to_csv(path, sep="\t", index=False, na_rep="n/a")
        return str(self.root), []


def _fake_extract_ds30(func, confounds, atlas, data_dir, t_r):
    _fake_extract_ds30.calls.append((list(func), confounds, t_r))
    rng = np.random.default_rng(2)
    return [rng.standard_normal((152, 39)) for _ in func]


@pytest.fixture
def ds30(monkeypatch, tmp_path):
    fake = FakeDS30(tmp_path)
    monkeypatch.setattr(nl_datasets, "fetch_ds000030_urls", fake.index)
    monkeypatch.setattr(nl_datasets, "fetch_openneuro_dataset", fake.fetch)
    _fake_extract_ds30.calls = []
    monkeypatch.setattr(nl_mod, "_extract_timeseries", _fake_extract_ds30)
    return fake


def test_ds000030_files_labels_and_missing_subject(cache, ds30):
    ds = _make(cache, dataset="ds000030")
    # only participants.tsv, the MNI resting-state images and their confounds
    assert all(nl_mod.DS000030_FILES.search(u) for u in ds30.requested)
    assert len(ds30.requested) == 1 + 2 * 24
    assert not any("T1w" in u or "bart" in u or "fmriprep" not in u
                   for u in ds30.requested[1:])
    assert len(ds) == 24 and ds.n_excluded_missing == 1 and ds.n_excluded == 1
    assert DS30_MISSING not in set(ds.metadata["subject"])
    assert ds.classes == ["control", "adhd", "bipolar", "schz"]
    assert np.bincount(ds.labels).tolist() == [8, 5, 5, 6]
    assert set(ds.sites) == {"scanner_35343", "scanner_35426"}
    assert len(set(ds.groups)) == 24 and "site_grouped" not in ds.splits
    (func, conf, t_r), = _fake_extract_ds30.calls
    assert t_r == 2.0 and all("MNI152NLin2009cAsym_preproc" in str(f) for f in func)
    # 7 confounds and their first differences, the first difference set to 0
    assert conf[0].shape == (152, 14) and not np.isnan(conf[0]).any()
    np.testing.assert_array_equal(conf[0][0, 7:], 0)
    assert ds.covariances.shape == (24, 39, 39)


def test_ds000030_diagnosis_reuses_the_cache(cache, ds30):
    full = _make(cache, dataset="ds000030")
    adhd = _make(cache, dataset="ds000030", diagnosis="adhd")
    assert len(_fake_extract_ds30.calls) == 1
    assert adhd.paths.covariance_dir == full.paths.covariance_dir
    assert adhd.classes == ["control", "adhd"] and len(adhd) == 13
    assert np.bincount(adhd.labels).tolist() == [8, 5]
    assert set(adhd.metadata["label"]) == {"control", "adhd"}
    assert len(adhd.phenotypic) == len(adhd.covs) == len(adhd.subject_index) == 13
    age = _make(cache, dataset="ds000030", task="regression", diagnosis="schz")
    assert len(age) == 14 and age.labels.dtype == np.float32
    _make(cache, dataset="ds000030", confounds=False)
    assert len(_fake_extract_ds30.calls) == 2 and _fake_extract_ds30.calls[1][1] is None


def test_diagnosis_is_a_ds000030_option(cache, abide, ds30):
    with pytest.raises(ValueError, match="diagnosis"):
        _make(cache, diagnosis="adhd")
    with pytest.raises(ValueError, match="diagnosis"):
        _make(cache, dataset="ds000030", diagnosis="ADHD")


# ---------------------------------------------------------------- fold splits
def _assert_folds(ds, prefix, n_folds):
    names = [n for n in ds.splits if n.startswith(prefix)]
    assert len(names) == n_folds
    tested = np.concatenate([ds.splits[n]["test"] for n in names])
    assert np.array_equal(np.sort(tested), np.arange(len(ds)))  # each subject tested once
    for n in names:
        sp = ds.splits[n]
        assert sum(map(len, sp.values())) == len(ds)
        assert len(np.unique(np.concatenate(list(sp.values())))) == len(ds)
    return names


def test_site_folds_leave_one_site_out(cache, abide):
    ds = _make(cache)  # 5 sites left (KKI excluded)
    names = _assert_folds(ds, "site_fold_", 5)
    sites = sorted(set(ds.sites))
    for k, n in enumerate(sorted(names, key=lambda s: int(s.rsplit("_", 1)[1]))):
        sp = ds.splits[n]
        assert set(ds.sites[sp["test"]]) == {sites[k]}
        assert set(ds.sites[sp["val"]]) == {sites[(k + 1) % 5]}
        for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
            assert_no_leakage(ds.sites[sp[a]], ds.sites[sp[b]], what="site")
    assert not any(n.startswith("stratified_fold") for n in ds.splits)
    assert ds.splits["recommended"] is ds.splits["site_grouped"]


def test_two_sites_give_folds_without_val(cache, abide):
    ds = _make(cache, fetcher_kwargs={"SITE_ID": ["NYU", "UM_1"]})
    _assert_folds(ds, "site_fold_", 2)
    assert len(ds.splits["site_fold_0"]["val"]) == 0


def test_stratified_folds_on_single_site(cache, ds30):
    ds = _make(cache, dataset="ds000030")
    _assert_folds(ds, "stratified_fold_", nl_mod.N_FOLDS)
    for k in range(nl_mod.N_FOLDS):
        sp = ds.splits[f"stratified_fold_{k}"]
        assert set(ds.labels[sp["test"]]) == {0, 1, 2, 3}  # stratified on the diagnosis
    reg = _make(cache, dataset="ds000030", task="regression")
    _assert_folds(reg, "stratified_fold_", nl_mod.N_FOLDS)


# ---------------------------------------------------------------- cross-validation
def _lr_on(i, j):
    from sklearn.linear_model import LogisticRegression

    def fit_predict(X_train, y_train, X_test):
        model = LogisticRegression().fit(X_train[:, i, j, None], y_train)
        return model.predict(X_test[:, i, j, None])
    return fit_predict


def test_cross_validate_folds_signal_and_noise(cache, monkeypatch):
    from spdnet_datasets.nilearn import cross_validate_folds

    monkeypatch.setattr(nl_datasets, "fetch_abide_pcp", FakeABIDE(T_short_site=None))
    ds = _make(cache)  # 6 sites x 4 subjects, class carried by cov[0, 1]
    res = cross_validate_folds(ds, _lr_on(0, 1), n_permutations=99, n_bootstrap=200)
    assert res["folds"] == [f"site_fold_{k}" for k in range(6)] and len(res["scores"]) == 6
    assert res["mean"] > 0.9 and res["p_value"] == 0.01
    assert res["ci"][0] <= res["mean"] <= res["ci"][1] + 1e-12
    noise = cross_validate_folds(ds, _lr_on(2, 3), n_permutations=99, n_bootstrap=200)
    assert noise["p_value"] > 0.05 and noise["ci"][0] < 0.5 < noise["ci"][1]


def test_cross_validate_folds_constant_predictor_has_p_one(cache, abide):
    from spdnet_datasets.nilearn import cross_validate_folds

    ds = _make(cache)
    res = cross_validate_folds(ds, lambda Xtr, ytr, Xte: np.zeros(len(Xte), dtype=int),
                               n_permutations=20, n_bootstrap=50)
    # labels shuffled within each site: every left-out site keeps its classes
    assert res["mean"] == 0.5 and res["p_value"] == 1.0
    with pytest.raises(ValueError, match="stratified_fold"):
        cross_validate_folds(ds, _lr_on(0, 1), prefix="stratified_fold", n_permutations=0)


def test_random_split_multi_site_is_leaky(cache, abide):
    from ._random_split import assert_random_split

    assert_random_split(lambda **kw: _make(cache, **kw), leaky=True, subset=False)


def test_random_split_one_subject_per_group_is_not_leaky(cache, ds30):
    from ._random_split import assert_random_split

    assert_random_split(lambda **kw: _make(cache, dataset="ds000030", **kw), leaky=False,
                        subset=False)
