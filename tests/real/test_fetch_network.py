"""Real download of one small file (needs Internet; ``pytest -m "slow and network"``)."""

import hashlib
import importlib
import urllib.request

import pytest

from spdnet_datasets.core.fetch import download, sha256sum
from spdnet_datasets.sar.cpazmal import HF_REVISION

URL = f"https://huggingface.co/datasets/musmb/CPAZMaL/resolve/{HF_REVISION}/dataset/CPAZMaL.txt"
# sha256 of the local copy SAR/PAZ_S1_proc_v2/CPAZMaL.txt (BERTHA), not of the download
SHA256 = "1abd2c9ea7a332f75cbe91a6bd74ccf7b0ca687fa3e0775cad79ebeb3a96e99d"
GIT_OID = "49ae02f090f1915325e70ff6dca8438de1cf6d79"  # Hugging Face tree API (git blob sha1)


@pytest.mark.slow
@pytest.mark.network
def test_real_download_of_a_small_file(tmp_path):
    path = download(URL, tmp_path / "CPAZMaL.txt", checksum=SHA256)
    data = path.read_bytes()
    assert len(data) == 1006 and sha256sum(path) == SHA256
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == GIT_OID
    assert download(URL, path, checksum=SHA256) == path  # idempotent, no second request


MAGIC = {".parquet": b"PAR1", ".hdf5": b"\x89HDF", ".zip": b"PK\x03\x04"}
REGISTERED = [(url, sha) for name in ("audio.esc50", "audio.gtzan", "sar.cpazmal", "sits.pastis_r")
              for url, sha in importlib.import_module(f"spdnet_datasets.{name}").FILES.items()]


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("url", [u for u, _ in REGISTERED])
def test_registered_url_serves_the_file(url):
    """First 4 bytes of every registered file through ``urllib`` (as ``download`` does)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-3"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        head = resp.read(4)  # never read() the body: a server ignoring Range sends it all
        status = resp.status
    suffix = next(s for s in MAGIC if url.split("?")[0].endswith(s))
    assert head == MAGIC[suffix]
    assert status == 206  # HTTP Range honoured: interrupted downloads resume
