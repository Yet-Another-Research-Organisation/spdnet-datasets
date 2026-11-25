"""Hugging Face download of AVIRIS-C California multilabel (private: needs a token).

``pytest -m "network and not slow"``: the small files (bands, classes, window table)
of the pinned revision; ``-m network`` also downloads one 732 MB shard. Skipped without a
Hugging Face token (``HF_TOKEN`` or ``~/.cache/huggingface/token``).
"""

import numpy as np
import pandas as pd
import pytest

from spdnet_datasets.core.access import hf_token
from spdnet_datasets.hyperspectral import aviris_ca_ml as A

pytestmark = [pytest.mark.network,
              pytest.mark.skipif(hf_token(interactive=False, required=False) is None,
                                 reason="no Hugging Face token")]
SMALL = ["bands.csv", "classes.csv", "metadata/windows.parquet"]


def test_small_files_of_the_pinned_revision(tmp_path):
    pytest.importorskip("pyarrow")
    root = A.download_files(SMALL, cache_dir=tmp_path, interactive=False)
    assert root == tmp_path / "aviris_ca_ml" / "downloads" / A.HF_REVISION
    for rel in SMALL:  # sizes and sha256 were checked by the download
        assert (root / rel).stat().st_size == A.HF_FILES[rel][0]
    W = pd.read_parquet(root / "metadata/windows.parquet")
    assert np.bincount(W["fold"], minlength=5).tolist() == [413, 402, 405, 398, 416]
    assert len(A.read_bands(root)) == 155


@pytest.mark.slow
def test_one_shard_decodes(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    rel = A.shard_paths([0])[0]
    root = A.download_files(["bands.csv", rel], cache_dir=tmp_path, interactive=False)
    f = pq.ParquetFile(root / rel)
    cubes = A.decode_cubes(f.read_row_group(0, columns=["cube"]).column("cube"), 14, 155)
    assert cubes.shape[1:] == (14, 155, 32, 32) and cubes.dtype == np.float16
