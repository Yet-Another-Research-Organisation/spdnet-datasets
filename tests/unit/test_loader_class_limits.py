"""Class selection and per-class limits of the radar/skeleton loaders.

These loaders read one ``.pt`` covariance per file under ``cov/<class>/``
(``cov/<modality>/<class>/`` for CI4R, ``cov/<split>/<class>/`` for DopNet).
The selection logic does not depend on the file contents, so it is tested on a
tiny synthetic tree; the real-data tests live in ``tests/real``.
"""

from pathlib import Path

import pytest
import torch

from spdnet_datasets.real import (
    CI4RDataset,
    DopNetDataset,
    MVDopplerDataset,
    NTU120Dataset,
)

CLASSES = ["a", "b", "c", "d"]
FILES_PER_CLASS = 5


def _make_tree(root: Path, prefix: tuple[str, ...]) -> None:
    for class_name in CLASSES:
        class_dir = root.joinpath("cov", *prefix, class_name)
        class_dir.mkdir(parents=True)
        for i in range(FILES_PER_CLASS):
            torch.save(torch.eye(3, dtype=torch.float64), class_dir / f"{i}.pt")


LOADERS = {
    "ci4r": (CI4RDataset, ("77GHz",), {"modality": "77GHz"}),
    "dopnet": (DopNetDataset, ("train",), {"split": "train"}),
    "mvdoppler": (MVDopplerDataset, (), {}),
    "ntu120": (NTU120Dataset, (), {}),
}


@pytest.fixture(params=sorted(LOADERS))
def loader(request, tmp_path):
    cls, prefix, kwargs = LOADERS[request.param]
    _make_tree(tmp_path, prefix)
    return lambda **extra: cls(data_dir=str(tmp_path), verbose=False, **kwargs, **extra)


def test_all_classes_by_default(loader):
    ds = loader()
    assert ds.classes == CLASSES
    assert len(ds) == len(CLASSES) * FILES_PER_CLASS


def test_explicit_classes(loader):
    ds = loader(classes=["d", "b"])
    assert ds.classes == ["b", "d"]
    assert {s["class_name"] for s in ds.samples} == {"b", "d"}
    assert ds.class_to_idx == {"b": 0, "d": 1}
    labels = {ds[i][1] for i in range(len(ds))}
    assert labels == {0, 1}


def test_unknown_class_raises(loader):
    with pytest.raises(ValueError):
        loader(classes=["a", "zzz"])


def test_max_classes(loader):
    ds = loader(max_classes=2)
    assert len(ds.classes) == 2
    assert {s["class_name"] for s in ds.samples} == set(ds.classes)


def test_max_samples_per_class(loader):
    ds = loader(max_samples_per_class=2)
    counts = dict.fromkeys(CLASSES, 0)
    for sample in ds.samples:
        counts[sample["class_name"]] += 1
    assert counts == dict.fromkeys(CLASSES, 2)
