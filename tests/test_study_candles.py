"""Candle store (M3): target selection, window derivation, pull/merge/coverage.

``pull_candles`` is exercised end-to-end against a MockTransport candlestick endpoint (no sockets),
so the ticker filter, period tag and validation are all covered."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from pmlab.data.schemas import Candles
from pmlab.http import ThrottledClient
from pmlab.study.candles import (
    coverage_stats,
    load_candles,
    merge_candles,
    persist_candles,
    pull_candles,
    target_markets,
)
from pmlab.venues.kalshi import KalshiPublic

ClientFactory = Callable[..., ThrottledClient]

_DAY = 86_400


def _registry(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = {
        "market_ticker": None, "eligible": True, "result": "yes",
        "open_ts": pd.NA, "close_ts": pd.NA, "occurrence_ts": pd.NA, "expiration_ts": pd.NA,
    }
    int_cols = ["open_ts", "close_ts", "occurrence_ts", "expiration_ts"]
    return pd.DataFrame([{**cols, **r} for r in rows]).astype(dict.fromkeys(int_cols, "Int64"))


def _bar(ts: int, mid: float = 0.5) -> dict[str, Any]:
    def ohlc(v: float) -> dict[str, str]:
        keys = ("open_dollars", "high_dollars", "low_dollars", "close_dollars")
        return dict.fromkeys(keys, f"{v:.2f}")

    return {
        "end_period_ts": ts,
        "price": {**ohlc(mid), "mean_dollars": f"{mid:.2f}"},
        "yes_bid": ohlc(mid - 0.02),
        "yes_ask": ohlc(mid + 0.02),
        "volume_fp": "100.0",
        "open_interest_fp": "50.0",
    }


def _candle_handler(bars_per_ticker: int = 1) -> Callable[[httpx.Request], httpx.Response]:
    """One daily bar per requested ticker inside the requested window (+ a NOISE ticker to prove
    the caller filters to the markets it asked for)."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets/candlesticks"
        tickers = request.url.params["market_tickers"].split(",")
        start = int(request.url.params["start_ts"])
        markets = []
        for t in tickers:
            bars = [_bar(start + (i + 1) * _DAY) for i in range(bars_per_ticker)]
            markets.append({"market_ticker": t, "candlesticks": bars})
        markets.append({"market_ticker": "NOISE", "candlesticks": [_bar(start + _DAY)]})
        return httpx.Response(200, json={"markets": markets})

    return handler


def test_target_markets_keeps_eligible_binary() -> None:
    reg = _registry(
        [
            {"market_ticker": "A", "eligible": True, "result": "yes"},
            {"market_ticker": "B", "eligible": True, "result": "no"},
            {"market_ticker": "C", "eligible": False, "result": "yes"},  # ineligible
            {"market_ticker": "D", "eligible": True, "result": ""},  # open
            {"market_ticker": "E", "eligible": True, "result": "scalar"},  # odd settlement
        ]
    )
    assert set(target_markets(reg)["market_ticker"]) == {"A", "B"}


def test_pull_candles_end_to_end(mock_client: ClientFactory) -> None:
    reg = _registry(
        [
            {"market_ticker": "M1", "open_ts": 1_700_000_000,
             "occurrence_ts": 1_700_000_000 + 5 * _DAY, "close_ts": 1_700_000_000 + 5 * _DAY},
            {"market_ticker": "M2", "open_ts": 1_700_100_000,
             "occurrence_ts": 1_700_100_000 + 5 * _DAY, "close_ts": 1_700_100_000 + 5 * _DAY},
        ]
    )
    kalshi = KalshiPublic(client=mock_client(_candle_handler()))
    candles = pull_candles(kalshi, reg, period_minutes=1440)

    Candles.validate(candles)
    assert set(candles["ticker"]) == {"M1", "M2"}  # NOISE filtered out
    assert (candles["period_minutes"] == 1440).all()
    assert not candles.duplicated(subset=["ticker", "ts", "period_minutes"]).any()


def test_pull_candles_limit_caps_markets(mock_client: ClientFactory) -> None:
    reg = _registry(
        [
            {"market_ticker": f"M{i}", "open_ts": 1_700_000_000 + i * _DAY,
             "close_ts": 1_700_000_000 + (i + 3) * _DAY}
            for i in range(10)
        ]
    )
    kalshi = KalshiPublic(client=mock_client(_candle_handler()))
    candles = pull_candles(kalshi, reg, limit=3)
    assert candles["ticker"].nunique() == 3


def test_pull_candles_drops_markets_without_timestamps(mock_client: ClientFactory) -> None:
    reg = _registry([{"market_ticker": "NOTS"}])  # all timing columns NA
    kalshi = KalshiPublic(client=mock_client(_candle_handler()))
    candles = pull_candles(kalshi, reg)
    assert candles.empty


def test_merge_candles_dedupes_keeping_latest() -> None:
    base = pull_frame([("A", 100, 0.4)])
    dup = pull_frame([("A", 100, 0.9), ("A", 200, 0.5)])
    merged = merge_candles(base, dup)
    assert len(merged) == 2
    row = merged[(merged["ticker"] == "A") & (merged["ts"] == 100)].iloc[0]
    assert row["price_close"] == pytest.approx(0.9)  # incoming wins


def test_coverage_stats_counts_covered(mock_client: ClientFactory) -> None:
    reg = _registry(
        [
            {"market_ticker": "M1", "open_ts": 1_700_000_000, "close_ts": 1_700_000_000 + 3 * _DAY},
            {"market_ticker": "M2", "open_ts": 1_700_000_000, "close_ts": 1_700_000_000 + 3 * _DAY},
        ]
    )
    kalshi = KalshiPublic(client=mock_client(_candle_handler()))
    candles = pull_candles(kalshi, reg)
    stats = coverage_stats(candles, reg)
    assert stats["n_target"] == 2 and stats["n_covered"] == 2 and stats["n_missing"] == 0


def test_persist_load_roundtrip(tmp_path: Path) -> None:
    df = pull_frame([("A", 100, 0.4), ("B", 200, 0.6)])
    persist_candles(df, tmp_path / "c.parquet")
    back = load_candles(tmp_path / "c.parquet")
    assert len(back) == 2 and set(back["ticker"]) == {"A", "B"}
    assert load_candles(tmp_path / "missing.parquet").empty


def pull_frame(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    """A minimal valid Candles frame from ``(ticker, ts, mid)`` tuples (test helper)."""
    recs = []
    for ticker, ts, mid in rows:
        recs.append(
            {
                "ticker": ticker, "ts": ts, "period_minutes": 1440,
                "yes_bid_open": mid - 0.02, "yes_bid_high": mid - 0.02,
                "yes_bid_low": mid - 0.02, "yes_bid_close": mid - 0.02,
                "yes_ask_open": mid + 0.02, "yes_ask_high": mid + 0.02,
                "yes_ask_low": mid + 0.02, "yes_ask_close": mid + 0.02,
                "price_open": mid, "price_high": mid, "price_low": mid,
                "price_close": mid, "price_mean": mid,
                "volume": 100.0, "open_interest": 50.0,
            }
        )
    return Candles.validate(pd.DataFrame(recs))
