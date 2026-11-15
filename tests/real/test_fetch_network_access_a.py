"""Real downloads of small files of the audio, skeleton and micro-Doppler sources.

Needs Internet (``pytest -m "slow and network"``). Only files below 50 MB are
downloaded; for the large archives only the first bytes are read (HTTP Range).
"""

import urllib.parse
import urllib.request

import pytest

from spdnet_datasets.core.fetch import download, md5sum, sha256sum
from tests.conftest import dataset_dir

pytestmark = [pytest.mark.slow, pytest.mark.network]


def test_birdset_xcm_metadata(tmp_path):
    import pyarrow.parquet as pq

    from spdnet_datasets.audio import birdset_xcm as bx

    url = bx._HF + bx.METADATA
    path = download(url, tmp_path / bx.METADATA, checksum=bx.FILES[url])
    assert pq.ParquetFile(path).metadata.num_rows == 89795


def test_ci4r_bucket_listing_and_one_image(tmp_path):
    from spdnet_datasets.radar_microdoppler import ci4r

    items = ci4r.list_bucket("xethru")
    assert len(items) == ci4r.N_IMAGES
    name, md5 = next((n, m) for n, m in items if "copy" in n)  # name with spaces
    url = f"https://storage.googleapis.com/{ci4r.BUCKET}/{urllib.parse.quote(name)}"
    path = download(url, tmp_path / "x.png", checksum=md5, algo="md5")
    local = dataset_dir("RADAR_DOPPLER", "CI4R-MULTI3") / name[len(ci4r.PREFIX):]
    if local.exists():
        assert md5sum(local) == md5sum(path)


def test_mvdoppler_labels(tmp_path):
    from spdnet_datasets.radar_microdoppler import mvdoppler as mv

    for url, sha256 in mv.FILES.items():
        path = download(url, tmp_path / url.rsplit("=", 1)[-1], checksum=sha256)
        assert sha256sum(path) == sha256


@pytest.mark.parametrize("module, url", [("radar_microdoppler.dopnet", "ARCHIVE_URL"),
                                         ("skeleton.hdm05", "ETH_URL")])
def test_large_archive_is_served(module, url):
    """First 4 bytes of the zip through ``urllib`` (never the whole body)."""
    import importlib

    target = getattr(importlib.import_module(f"spdnet_datasets.{module}"), url)
    req = urllib.request.Request(target, headers={"Range": "bytes=0-3"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        head, status = resp.read(4), resp.status
    assert head == b"PK\x03\x04" and status == 206
