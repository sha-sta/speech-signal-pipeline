"""Unit tests for the throttle, on-disk cache, and retry paths of the shared HTTP client."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from pmlab.http import CACHE_FOREVER, RetryableHTTPError, ThrottledClient, TokenBucket

ClientFactory = Callable[..., ThrottledClient]


class FakeClock:
    """A clock whose time only moves when something sleeps — deterministic for throttle tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def test_token_bucket_allows_burst_then_throttles() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=4.0, capacity=4.0, clock=clock, sleep=clock.sleep)

    assert [bucket.acquire() for _ in range(4)] == [0.0, 0.0, 0.0, 0.0]  # full bucket, no wait

    waited = bucket.acquire()  # 5th token: bucket empty, must wait 1/4 s
    assert waited == pytest.approx(0.25)
    assert clock.t == pytest.approx(0.25)  # the sleep advanced the clock


def test_token_bucket_refills_over_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=2.0, capacity=2.0, clock=clock, sleep=clock.sleep)
    bucket.acquire()
    bucket.acquire()  # bucket now empty
    clock.t += 1.0  # 1 s elapses -> 2 tokens refill
    assert bucket.acquire() == 0.0
    assert bucket.acquire() == 0.0


def test_token_bucket_rejects_bad_rate() -> None:
    with pytest.raises(ValueError, match="rate"):
        TokenBucket(rate=0.0)


def test_cache_hit_avoids_second_request(mock_client: ClientFactory) -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"call": calls["n"]})

    client = mock_client(handler)

    first = client.get_json("/x", {"a": 1}, cache_ttl=CACHE_FOREVER)
    second = client.get_json("/x", {"a": 1}, cache_ttl=CACHE_FOREVER)
    assert first == second == {"call": 1}
    assert calls["n"] == 1  # second call served from disk cache

    client.get_json("/x", {"a": 2}, cache_ttl=CACHE_FOREVER)  # different params -> new request
    assert calls["n"] == 2

    client.get_json("/x", {"a": 1}, cache_ttl=None)  # cache_ttl=None -> always live
    assert calls["n"] == 3


def test_cache_respects_ttl(mock_client: ClientFactory) -> None:
    calls = {"n": 0}
    now = {"t": 1000.0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"call": calls["n"]})

    client = mock_client(handler, clock=lambda: now["t"])

    client.get_json("/x", cache_ttl=900)  # stored at t=1000
    now["t"] = 1000 + 899
    client.get_json("/x", cache_ttl=900)  # still fresh
    assert calls["n"] == 1
    now["t"] = 1000 + 901
    client.get_json("/x", cache_ttl=900)  # expired -> refetch
    assert calls["n"] == 2


def test_retry_on_5xx_then_success(mock_client: ClientFactory) -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = mock_client(handler, sleep=lambda _: None)
    assert client.get_json("/x", cache_ttl=None) == {"ok": True}
    assert calls["n"] == 3


def test_429_is_retryable(mock_client: ClientFactory) -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429 if calls["n"] == 1 else 200, json={"ok": True})

    client = mock_client(handler, sleep=lambda _: None)
    assert client.get_json("/x", cache_ttl=None) == {"ok": True}
    assert calls["n"] == 2


def test_4xx_is_not_retried(mock_client: ClientFactory) -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    client = mock_client(handler, sleep=lambda _: None)
    with pytest.raises(httpx.HTTPStatusError):
        client.get_json("/x", cache_ttl=None)
    assert calls["n"] == 1  # no retry on a client error


def test_retryable_error_carries_status() -> None:
    err = RetryableHTTPError(503, "http://x")
    assert err.status_code == 503
