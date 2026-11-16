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
    x, _ = _make(cache, return_mean=return_mean, return_raw=return_raw)[1]
    parts = x if isinstance(x, tuple) else (x,)
    assert len(parts) == 1 + return_mean + return_raw
    if return_mean:
        assert tuple(parts[1].shape) == (P_TT,)
        assert abs(float(parts[1].mean()) - 10.0) < 1.0  # mean before standardization
    if return_raw:
        raw = parts[-1]
        assert tuple(raw.shape) == (2 * P_TT + 10, P_TT) and str(raw.dtype) == "torch.float32"
        np.testing.assert_allclose(raw.numpy().std(axis=0, ddof=1), 1.0, rtol=1e-4)


def test_raw_series_match_covariance(cache, abide):
    ds = _make(cache, return_raw=True, normalize=False)
    cov, raw = ds[2][0]
    np.testing.assert_allclose(Covariances().fit_transform([raw.numpy().astype(float)])[0],
                               cov.numpy(), atol=1e-5)


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
