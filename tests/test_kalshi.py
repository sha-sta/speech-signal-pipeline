"""Parse-against-fixture tests for KalshiPublic (recorded live 2026-07-02)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from pmlab.http import ThrottledClient
from pmlab.venues.kalshi import CANDLE_COLUMNS, KalshiPublic

ClientFactory = Callable[..., ThrottledClient]
Fixture = Callable[[str], Any]


def test_iter_events_paginates_and_parses(mock_client: ClientFactory, fixture: Fixture) -> None:
    events_page = fixture("kalshi_events.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/events"
        if request.url.params.get("cursor"):  # second page terminates the loop
            return httpx.Response(200, json={"events": [], "cursor": ""})
        return httpx.Response(200, json=events_page)

    kalshi = KalshiPublic(client=mock_client(handler))
    events = list(kalshi.iter_events(status="open"))

    assert len(events) == len(events_page["events"])
    assert events[0]["event_ticker"] == events_page["events"][0]["event_ticker"]
    assert events[0]["series_ticker"]


def test_get_series_unwraps_envelope(mock_client: ClientFactory, fixture: Fixture) -> None:
    series_doc = fixture("kalshi_series.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith("/series/")
        return httpx.Response(200, json=series_doc)

    kalshi = KalshiPublic(client=mock_client(handler))
    series = kalshi.get_series("KXELONMARS")

    expected = series_doc.get("series", series_doc)
    assert series["fee_type"] == expected["fee_type"]
    assert "fee_multiplier" in series


def test_iter_markets_parses(mock_client: ClientFactory, fixture: Fixture) -> None:
    markets_page = fixture("kalshi_markets.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets"
        return httpx.Response(200, json=markets_page)

    kalshi = KalshiPublic(client=mock_client(handler))
    markets = list(kalshi.iter_markets(status="open"))

    assert len(markets) == len(markets_page["markets"])
    # Post-migration fields: quotes are *_dollars strings, volume is *_fp (§2.1).
    assert "yes_bid_dollars" in markets[0]
    assert "volume_fp" in markets[0]


def test_candles_batch_parses_nested_ohlc(mock_client: ClientFactory, fixture: Fixture) -> None:
    batch = fixture("kalshi_candles_batch.json")
    entry = batch["markets"][0]
    ticker = entry["market_ticker"]

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.url.path == "/markets/candlesticks"
        return httpx.Response(200, json=batch)

    kalshi = KalshiPublic(client=mock_client(handler))
    df = kalshi.candles_batch([ticker], 0, 600)  # short window -> single request

    assert calls["n"] == 1
    assert list(df.columns) == CANDLE_COLUMNS
    assert len(df) == len(entry["candlesticks"])

    last_row = df.iloc[-1]
    last_candle = entry["candlesticks"][-1]
    assert int(last_row.ts) == last_candle["end_period_ts"]  # period-END timestamp
    assert last_row.ticker == ticker
    assert last_row.yes_bid_close == float(last_candle["yes_bid"]["close_dollars"])
    assert last_row.yes_ask_close == float(last_candle["yes_ask"]["close_dollars"])
    assert last_row.price_close == float(last_candle["price"]["close_dollars"])
    assert last_row.volume == float(last_candle["volume_fp"])


def test_candles_batch_empty_returns_typed_frame(mock_client: ClientFactory) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"markets": [{"market_ticker": "X", "candlesticks": []}]})

    kalshi = KalshiPublic(client=mock_client(handler))
    df = kalshi.candles_batch(["X"], 0, 600)
    assert len(df) == 0
    assert list(df.columns) == CANDLE_COLUMNS


def test_historical_candles_parses_plain_keys(mock_client: ClientFactory) -> None:
    # Historical shape verified live 2026-07-03 (S2 scan): plain nested keys, dollar-string
    # values, top-level volume/open_interest without the _fp suffix.
    doc = {
        "candlesticks": [
            {
                "end_period_ts": 1746633600,
                "price": {"open": "0.10", "high": "0.14", "low": "0.09", "close": "0.12",
                          "mean": "0.11"},
                "yes_bid": {"open": "0.09", "high": "0.13", "low": "0.08", "close": "0.11"},
                "yes_ask": {"open": "0.11", "high": "0.15", "low": "0.10", "close": "0.13"},
                "volume": 42,
                "open_interest": 7,
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/historical/markets/KXOLD-25JAN-T/candlesticks"
        return httpx.Response(200, json=doc)

    kalshi = KalshiPublic(client=mock_client(handler))
    df = kalshi.historical_candles("KXOLD-25JAN-T", 0, 600)

    assert list(df.columns) == CANDLE_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert row.yes_bid_close == 0.11  # would be NaN under a *_dollars-only parser
    assert row.yes_ask_open == 0.11
    assert row.price_mean == 0.11
    assert row.volume == 42.0
    assert row.open_interest == 7.0
