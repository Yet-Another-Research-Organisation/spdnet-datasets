"""Fixtures shared by the unit tests."""

import hashlib
import io

import pytest

from spdnet_datasets.core import fetch as F


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_net(monkeypatch):
    """``serve(url -> bytes)``; returns the list of requested URLs."""
    served, calls = {}, []

    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Response(served[req.full_url])

    monkeypatch.setattr(F.urllib.request, "urlopen", urlopen)

    def serve(url, data):
        served[url] = data
        return hashlib.sha256(data).hexdigest()

    return serve, calls
