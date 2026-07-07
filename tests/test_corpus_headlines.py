"""GDELT headline source (M3): date parsing, month enumeration, fetch, dedup — no live network."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from pmlab.corpus.headlines import (
    GdeltHeadlineSource,
    _months,
    _seendate_to_ts,
    fetch_headlines,
)
from pmlab.data.schemas import Headlines
from pmlab.http import ThrottledClient

ClientFactory = Callable[..., ThrottledClient]


def test_seendate_parsing() -> None:
    ts = _seendate_to_ts("20240612T191500Z")
    assert ts is not None
    assert _seendate_to_ts("garbage") is None
    assert _seendate_to_ts("") is None


def test_months_inclusive_range_wraps_year() -> None:
    assert _months("2025-11", "2026-02") == [(2025, 11), (2025, 12), (2026, 1), (2026, 2)]
    assert _months("2026-03", "2026-03") == [(2026, 3)]


def _gdelt(articles: list[dict[str, str]]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v2/doc/doc"
        assert request.url.params.get("mode") == "artlist"
        return httpx.Response(200, json={"articles": articles})

    return handler


def test_fetch_month_parses_and_skips_bad_rows(mock_client: ClientFactory) -> None:
    handler = _gdelt(
        [
            {"seendate": "20240612T191500Z", "domain": "cnn.com", "title": "Inflation cools"},
            {"seendate": "bad", "domain": "x.com", "title": "dropped — bad date"},
            {"seendate": "20240613T101500Z", "domain": "reuters.com", "title": ""},  # empty title
        ]
    )
    src = GdeltHeadlineSource(client=mock_client(handler))
    rows = src.fetch_month(2024, 6)
    assert len(rows) == 1
    assert rows[0].text == "Inflation cools" and rows[0].source == "cnn.com"


def test_fetch_headlines_dedupes_and_validates(mock_client: ClientFactory) -> None:
    # same article returned in overlapping months → one row after dedup on headline_id
    handler = _gdelt(
        [
            {"seendate": "20240612T191500Z", "domain": "cnn.com", "title": "Fed holds rates"},
            {"seendate": "20240612T191500Z", "domain": "cnn.com", "title": "Fed holds rates"},
        ]
    )
    src = GdeltHeadlineSource(client=mock_client(handler))
    frame = fetch_headlines(src, "2024-06", "2024-07")  # 2 months, same payload each
    Headlines.validate(frame)
    assert len(frame) == 1  # unique headline_id
    assert list(frame.columns) == ["headline_id", "ts", "source", "text"]


def test_fetch_survives_a_failing_month(mock_client: ClientFactory) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")  # first month fails
        return httpx.Response(200, json={"articles": [
            {"seendate": "20240712T191500Z", "domain": "ap.org", "title": "Jobs report"}]})

    src = GdeltHeadlineSource(client=mock_client(handler, max_attempts=1))
    frame = fetch_headlines(src, "2024-06", "2024-07")
    assert len(frame) == 1 and frame.iloc[0]["text"] == "Jobs report"
