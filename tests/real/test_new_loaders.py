"""Smoke tests for the CI4R/DopNet/MVDoppler/NTU120 loaders against real data."""

import torch

from spdnet_datasets.real import (
    CI4RDataset,
    DopNetDataset,
    MVDopplerDataset,
    NTU120Dataset,
    Rices90Dataset,
)
from tests.conftest import dataset_dir, skip_if_missing


def _check_spd_sample(dataset, expected_dim: int | None = None) -> None:
    assert len(dataset) > 0
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float64
    assert x.ndim == 2 and x.shape[0] == x.shape[1]
    if expected_dim is not None:
        assert x.shape[0] == expected_dim
    assert torch.allclose(x, x.T, atol=1e-6), "sample is not symmetric"
    eigvals = torch.linalg.eigvalsh(x)
    assert (eigvals > -1e-6).all(), "sample is not (numerically) SPD"
    assert isinstance(int(y), int)
    assert 0 <= int(y) < dataset.num_classes


class TestCI4RDataset:
    def test_77ghz_modality(self):
        data_dir = dataset_dir("CI4R-MULTI3")
        skip_if_missing(data_dir)
        ds = CI4RDataset(data_dir=str(data_dir), modality="77GHz", verbose=False)
        # Actual precomputed matrices are 236x236, not the 471x471 the
        # module docstring claims -- docstring is stale, not asserting it.
        _check_spd_sample(ds)

    def test_xethru_modality(self):
        data_dir = dataset_dir("CI4R-MULTI3")
        skip_if_missing(data_dir)
        ds = CI4RDataset(data_dir=str(data_dir), modality="Xethru", verbose=False)
        _check_spd_sample(ds)

    def test_ci4r_explicit_classes_and_limits(self):
        """The `classes` and `max_samples_per_class` kwargs are honoured."""
        data_dir = dataset_dir("CI4R-MULTI3")
        skip_if_missing(data_dir)
        full = CI4RDataset(data_dir=str(data_dir), modality="77GHz", verbose=False)
        subset = sorted(full.classes)[:2]
        restricted = CI4RDataset(
            data_dir=str(data_dir),
            modality="77GHz",
            classes=subset,
            max_samples_per_class=3,
            verbose=False,
        )
        assert restricted.classes == subset
        assert {s["class_name"] for s in restricted.samples} == set(subset)
        assert len(restricted.samples) <= 3 * len(subset)


class TestDopNetDataset:
    def test_train_split(self):
        data_dir = dataset_dir("Dop-NET")
        skip_if_missing(data_dir)
        ds = DopNetDataset(data_dir=str(data_dir), split="train", verbose=False)
        _check_spd_sample(ds)

    def test_test_split(self):
        data_dir = dataset_dir("Dop-NET")
        skip_if_missing(data_dir)
        ds = DopNetDataset(data_dir=str(data_dir), split="test", verbose=False)
        _check_spd_sample(ds)


class TestMVDopplerDataset:
    def test_load(self):
        data_dir = dataset_dir("MVDoppler")
        skip_if_missing(data_dir)
        ds = MVDopplerDataset(data_dir=str(data_dir), verbose=False)
        _check_spd_sample(ds, expected_dim=96)


class TestNTU120Dataset:
    def test_load(self):
        data_dir = dataset_dir("NTU_RGBD_120")
        skip_if_missing(data_dir)
        ds = NTU120Dataset(data_dir=str(data_dir), verbose=False)
        _check_spd_sample(ds, expected_dim=75)


class TestExplicitClasses:
    def test_explicit_classes_subset(self):
        """BaseDataset._limit_classes: an explicit `classes` list is honored
        (currently wired only into Rices90Dataset, see CI4R test above)."""
        data_dir = dataset_dir("Rices_90")
        skip_if_missing(data_dir)
        full = Rices90Dataset(data_dir=str(data_dir), verbose=False)
        two_classes = sorted(full.classes)[:2]
        subset = Rices90Dataset(
            data_dir=str(data_dir), classes=two_classes, verbose=False
        )
        assert sorted(subset.classes) == two_classes
        assert subset.num_classes == 2
        assert len(subset) < len(full)

    def test_explicit_classes_rejects_unknown_class(self):
        data_dir = dataset_dir("Rices_90")
        skip_if_missing(data_dir)
        import pytest

        with pytest.raises(ValueError, match="not found"):
            Rices90Dataset(
                data_dir=str(data_dir), classes=["not_a_real_class"], verbose=False
            )
