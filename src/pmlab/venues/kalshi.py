"""Kalshi public read client (unauthenticated). Endpoints verified live 2026-07-02.

Post-Mar-2026 migration facts baked in (see IMPLEMENTATION_PLAN §2.1):
  * market quote/size/volume fields are ``*_dollars`` strings and ``*_fp`` fixed-point strings;
    the legacy integer ``yes_bid``/``volume``/``open_interest`` fields are gone.
  * ``/markets`` objects carry ``event_ticker`` but NOT ``series_ticker`` — derive the series
    from the ticker prefix (``ticker.split('-')[0]``) or via the event.
  * candlestick timestamp is ``end_period_ts`` = END of the bucket (period-END semantics);
    OHLC live under nested ``price`` / ``yes_bid`` / ``yes_ask`` objects as ``*_dollars`` strings.
  * the ``/historical/...`` candlestick path nests the same objects but with PLAIN keys
    (``open``/``high``/``low``/``close``, dollar-string values; top-level ``volume`` /
    ``open_interest``, no ``_fp``) — verified live 2026-07-03 (S2 scan). ``_candle_rows``
    accepts both shapes.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd

from pmlab.config import Settings, get_settings
from pmlab.http import CACHE_FOREVER, ThrottledClient

_LIVE_TTL = 900.0  # 15 min for live listings (events/series/markets)

CANDLE_COLUMNS = [
    "ticker", "ts",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "price_open", "price_high", "price_low", "price_close", "price_mean",
    "volume", "open_interest",
]


def _f(x: Any) -> float:
    """Parse a Kalshi ``*_dollars``/``*_fp`` string (or number) to float; NaN if absent."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _g(d: dict[str, Any], key: str) -> Any:
    """Nested OHLC field under both venue shapes: ``<key>_dollars`` (live) or ``<key>``
    (historical path; plain dollar-string values)."""
    return d.get(f"{key}_dollars", d.get(key))


def _candle_rows(ticker: str | None, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in candles:
        price = c.get("price") or {}
        bid = c.get("yes_bid") or {}
        ask = c.get("yes_ask") or {}
        rows.append(
            {
                "ticker": ticker,
                "ts": int(c["end_period_ts"]),
                "yes_bid_open": _f(_g(bid, "open")),
                "yes_bid_high": _f(_g(bid, "high")),
                "yes_bid_low": _f(_g(bid, "low")),
                "yes_bid_close": _f(_g(bid, "close")),
                "yes_ask_open": _f(_g(ask, "open")),
                "yes_ask_high": _f(_g(ask, "high")),
                "yes_ask_low": _f(_g(ask, "low")),
                "yes_ask_close": _f(_g(ask, "close")),
                "price_open": _f(_g(price, "open")),
                "price_high": _f(_g(price, "high")),
                "price_low": _f(_g(price, "low")),
                "price_close": _f(_g(price, "close")),
                "price_mean": _f(_g(price, "mean")),
                "volume": _f(c.get("volume_fp", c.get("volume"))),
                "open_interest": _f(c.get("open_interest_fp", c.get("open_interest"))),
            }
        )
    return rows


class KalshiPublic:
    def __init__(
        self, client: ThrottledClient | None = None, settings: Settings | None = None
    ) -> None:
        s = settings or get_settings()
        self._c = client or ThrottledClient(
            base_url=s.kalshi_base,
            rps=s.kalshi_rps,
            cache_dir=s.cache_dir,
            user_agent=s.user_agent,
            timeout=s.http_timeout_s,
        )

    def list_series(self, category: str | None = None) -> list[dict[str, Any]]:
        """All series (optionally filtered by ``category``) in one call.

        Unlike ``/events``, the ``category`` filter *is* honored here and the response is a
        single unpaginated ``{"series": [...]}`` list — so this is the cheap way to enumerate a
        whole category (e.g. all ``Mentions`` series) without sweeping the platform's event feed
        (``/events?category=`` is silently ignored). Verified live 2026-07-03 (M1)."""
        data = self._c.get_json("/series", {"category": category}, cache_ttl=_LIVE_TTL)
        series: list[dict[str, Any]] = data.get("series", [])
        return series

    def iter_events(
        self,
        status: str | None = None,
        series_ticker: str | None = None,
        *,
        with_nested_markets: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Paginate ``/events``. With ``with_nested_markets`` each event carries its ``markets``
        list inline, avoiding a second ``/markets`` round-trip per event (verified live M1)."""
        cursor: str | None = None
        while True:
            params = {
                "limit": 200,
                "status": status,
                "series_ticker": series_ticker,
                "with_nested_markets": "true" if with_nested_markets else None,
                "cursor": cursor,
            }
            data = self._c.get_json("/events", params, cache_ttl=_LIVE_TTL)
            yield from data.get("events", [])
            cursor = data.get("cursor") or None
            if cursor is None:
                return

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        data = self._c.get_json(f"/series/{series_ticker}", cache_ttl=_LIVE_TTL)
        series: dict[str, Any] = data.get("series", data)
        return series

    def iter_markets(
        self,
        *,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = None,
        min_close_ts: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            params = {
                "limit": 1000,
                "event_ticker": event_ticker,
                "series_ticker": series_ticker,
                "status": status,
                "min_close_ts": min_close_ts,
                "cursor": cursor,
            }
            data = self._c.get_json("/markets", params, cache_ttl=_LIVE_TTL)
            yield from data.get("markets", [])
            cursor = data.get("cursor") or None
            if cursor is None:
                return

    def candles_batch(
        self,
        tickers: Sequence[str],
        start_ts: int,
        end_ts: int,
        period_minutes: int = 1,
    ) -> pd.DataFrame:
        """Batch candlesticks in long format, respecting the venue's two request caps (§2.1):
        ≤100 tickers/request AND ``tickers × buckets ≤ 10,000`` (both enforced here by chunking
        tickers into groups of 100 and the time axis so a request never exceeds the span cap)."""
        rows: list[dict[str, Any]] = []
        for group in _chunks(list(tickers), 100):
            budget_buckets = max(1, 9000 // len(group))  # margin under the 10k span cap
            span = budget_buckets * period_minutes * 60
            win_start = start_ts
            while win_start < end_ts:
                win_end = min(end_ts, win_start + span)
                data = self._c.get_json(
                    "/markets/candlesticks",
                    {
                        "market_tickers": ",".join(group),
                        "start_ts": win_start,
                        "end_ts": win_end,
                        "period_interval": period_minutes,
                    },
                    cache_ttl=CACHE_FOREVER,
                )
                for entry in data.get("markets", []):
                    rows.extend(
                        _candle_rows(entry.get("market_ticker"), entry.get("candlesticks", []))
                    )
                win_start = win_end
        return pd.DataFrame(rows, columns=CANDLE_COLUMNS)

    def historical_cutoff(self) -> dict[str, Any]:
        """Timestamps before which settled data has left the live window (then use /historical/)."""
        cutoff: dict[str, Any] = self._c.get_json("/historical/cutoff", cache_ttl=_LIVE_TTL)
        return cutoff

    def historical_candles(self, ticker: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        """Candlesticks for a market settled before the live cutoff.

        Response shape verified live 2026-07-03 (S2 scan): nested OHLC uses PLAIN keys with
        dollar-string values and top-level ``volume``/``open_interest`` — handled by
        ``_candle_rows``'s dual-shape getter. Known limitation (S2): any registry sweep built
        from the live ``/markets`` listing undercounts settled series that left the live window
        before a venue-side historical cutoff; a wider sweep must also cover
        ``/historical/markets``."""
        data = self._c.get_json(
            f"/historical/markets/{ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1},
            cache_ttl=CACHE_FOREVER,
        )
        candles = data.get("candlesticks", [])
        return pd.DataFrame(_candle_rows(ticker, candles), columns=CANDLE_COLUMNS)


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
