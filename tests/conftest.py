"""Shared test helpers: a MockTransport-backed ThrottledClient and a fixture loader."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from pmlab.http import ThrottledClient

FIXTURES = Path(__file__).parent / "fixtures"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def fixture() -> Callable[[str], Any]:
    def _load(name: str) -> Any:
        return json.loads((FIXTURES / name).read_text("utf-8"))

    return _load


@pytest.fixture
def mock_client(tmp_path: Path) -> Callable[..., ThrottledClient]:
    """Build a ThrottledClient whose network is served by ``handler`` (no real sockets)."""

    def _make(handler: Handler, **kwargs: Any) -> ThrottledClient:
        hclient = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
        kwargs.setdefault("rps", 1000.0)
        return ThrottledClient(base_url="http://test", cache_dir=tmp_path, client=hclient, **kwargs)

    return _make
