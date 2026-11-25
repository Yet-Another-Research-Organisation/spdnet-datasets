"""Smoke test of doc/tools/make_figures.py on a toy registered dataset."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from spdnet_datasets import DatasetManager
from spdnet_datasets.base import BaseDataset

pytest.importorskip("plotly")
TOOL = Path(__file__).resolve().parents[2] / "doc" / "tools" / "make_figures.py"


class _ToyFigures(BaseDataset):
    def __init__(self, data_dir, **kw):
        super().__init__(data_dir, verbose=False, **kw)
        rng = np.random.default_rng(0)
        self.raw = rng.standard_normal((30, 6, 6, 4))
        self.labels = np.repeat([0, 1, 2], 10)
        self.classes = ["a", "b", "c"]
        self.samples = list(range(30))
        self.groups = np.repeat(np.arange(15), 2)
        self.splits = {"official": {"train": np.arange(20), "test": np.arange(20, 30)}}

    def _load_data(self):
        pass

    def __getitem__(self, i):
        X = self.raw[i].reshape(-1, 4)
        return self._pack(np.cov(X.T), X.mean(0), self.raw[i]), int(self.labels[i])


def test_make_figures_writes_every_kind(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.parent))  # make_figures imports figures_io
    spec = importlib.util.spec_from_file_location("make_figures", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setitem(DatasetManager.DATASETS, "_toy_figures", _ToyFigures)
    summary = mod.build("_toy_figures", {"data_dir": str(tmp_path)}, "official", "toy")
    kinds = {"raw_samples", "class_hist", "cov_samples", "class_means", "split_schema",
             "eigen_spectrum"}
    # interactive kinds as JSON, the others as a light and a dark PNG
    import figures_io

    static = kinds - set(figures_io.INTERACTIVE)
    out = tmp_path / "toy"
    assert {f.stem for f in out.glob("*.json")} - {"summary"} == kinds - static
    assert {f.stem for f in out.glob("*.png")} == static | {f"{k}_dark" for k in static}
    assert summary["p"] == 4 and summary["counts"]["train"] == [10, 10, 0]
    assert json.loads((tmp_path / "toy" / "summary.json").read_text())["n_samples"] == 30


class _ToyMultilabel(_ToyFigures):
    task = "multilabel"

    def __init__(self, data_dir, **kw):
        super().__init__(data_dir, **kw)
        self.task = "multilabel"
        self.labels = (np.random.default_rng(1).random((30, 3)) < 0.5).astype(np.float32)

    def __getitem__(self, i):
        X = self.raw[i].reshape(-1, 4)
        return self._pack(np.cov(X.T), X.mean(0), self.raw[i]), self.labels[i]


def test_make_figures_multilabel(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.parent))
    spec = importlib.util.spec_from_file_location("make_figures", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setitem(DatasetManager.DATASETS, "_toy_ml", _ToyMultilabel)
    ds = _ToyMultilabel(str(tmp_path))
    summary = mod.build("_toy_ml", {"data_dir": str(tmp_path)}, "official", "toy_ml")
    # label counts per split part are the column sums of the 0/1 labels
    assert summary["counts"]["train"] == ds.labels[:20].sum(0).astype(int).tolist()
    assert summary["counts"]["test"] == ds.labels[20:].sum(0).astype(int).tolist()
