"""Shared throttled HTTP client: token-bucket rate limit + retry + on-disk JSON cache.

All venue reads go through :class:`ThrottledClient`. The client is deliberately synchronous
(the study is analytical, not latency-sensitive) and injectable (``client``/``clock``/``sleep``)
so the throttle, cache, and retry paths are unit-testable without real network or wall-clock.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

# cache_ttl sentinels for get_json:
#   None          -> do not read or write the cache (always fetch live)
#   CACHE_FOREVER -> cache never expires (settled/immutable data)
#   <float secs>  -> cache entry expires after that many seconds (live listings)
CACHE_FOREVER = math.inf

ParamValue = str | int | float | bool | None
Params = Mapping[str, ParamValue]


class RetryableHTTPError(Exception):
    """Raised for 429/5xx responses so tenacity retries with backoff."""

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"retryable HTTP {status_code} for {url}")
        self.status_code = status_code


class TokenBucket:
    """A leaky token bucket. ``clock``/``sleep`` are injectable for deterministic tests."""

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def acquire(self, tokens: float = 1.0) -> float:
        """Consume ``tokens``, sleeping if the bucket is short. Returns seconds slept."""
        self._refill()
        waited = 0.0
        if self._tokens < tokens:
            waited = (tokens - self._tokens) / self.rate
            self._sleep(waited)
            self._refill()
        self._tokens -= tokens
        return waited


class ThrottledClient:
    """Throttled, retrying, cache-backed JSON GET client for one API host."""

    def __init__(
        self,
        *,
        base_url: str,
        rps: float,
        cache_dir: Path,
        user_agent: str = "pmlab/0.1",
        timeout: float = 30.0,
        max_attempts: int = 6,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )
        self._bucket = TokenBucket(rps, sleep=sleep)
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._retrying = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=0.5, max=60.0),
            retry=retry_if_exception_type(RetryableHTTPError),
            sleep=sleep,
            reraise=True,
        )

    def get_json(
        self, path: str, params: Params | None = None, *, cache_ttl: float | None = None
    ) -> Any:
        clean: dict[str, ParamValue] = {k: v for k, v in (params or {}).items() if v is not None}
        key = self._cache_key(path, clean)
        if cache_ttl is not None:
            hit = self._cache_read(key, cache_ttl)
            if hit is not None:
                return hit
        self._bucket.acquire()
        body = self._retrying(self._do_get, path, clean)
        if cache_ttl is not None:
            self._cache_write(key, body)
        return body

    def _do_get(self, path: str, params: dict[str, ParamValue]) -> Any:
        resp = self._client.get(path, params=params)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableHTTPError(resp.status_code, str(resp.request.url))
        resp.raise_for_status()
        return resp.json()

    def get_text(
        self, path: str, params: Params | None = None, *, cache_ttl: float | None = None
    ) -> str:
        """Throttled/retrying/cached GET returning the raw response body as text (HTML scrapers,
        M2). The cache key is namespaced (``TEXT:``) so it never collides with a JSON GET of the
        same URL, and the cached ``body`` is the response string."""
        clean: dict[str, ParamValue] = {k: v for k, v in (params or {}).items() if v is not None}
        key = self._cache_key(f"TEXT:{path}", clean)
        if cache_ttl is not None:
            hit = self._cache_read(key, cache_ttl)
            if hit is not None:
                return str(hit)
        self._bucket.acquire()
        body: str = self._retrying(self._do_get_text, path, clean)
        if cache_ttl is not None:
            self._cache_write(key, body)
        return body

    def _do_get_text(self, path: str, params: dict[str, ParamValue]) -> str:
        # Override the client-default ``Accept: application/json`` — HTML hosts (federalreserve.gov)
        # return 406 Not Acceptable for a JSON Accept on a web page.
        resp = self._client.get(
            path, params=params, headers={"Accept": "text/html,application/xhtml+xml,*/*"}
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableHTTPError(resp.status_code, str(resp.request.url))
        resp.raise_for_status()
        return resp.text

    # --- cache: data/cache/<sha1(path+sorted params)>.json ---

    def _cache_key(self, path: str, params: dict[str, ParamValue]) -> str:
        payload = json.dumps({"path": path, "params": params}, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_read(self, key: str, ttl: float) -> Any | None:
        try:
            rec = json.loads(self._cache_path(key).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if ttl != CACHE_FOREVER and (self._clock() - float(rec.get("fetched_at", 0.0))) > ttl:
            return None
        return rec.get("body")

    def _cache_write(self, key: str, body: Any) -> None:
        rec = {"fetched_at": self._clock(), "body": body}
        path = self._cache_path(key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec), "utf-8")
        tmp.replace(path)  # atomic on POSIX

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ThrottledClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
