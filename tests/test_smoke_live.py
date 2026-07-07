"""Live connectivity tests — deselected by default; run with `uv run pytest -m live`."""

from __future__ import annotations

import itertools
import time

import pytest

from pmlab.venues.kalshi import KalshiPublic
from pmlab.venues.polymarket import ClobPublic, GammaClient, yes_token_id

pytestmark = pytest.mark.live


def test_kalshi_live_reads() -> None:
    kalshi = KalshiPublic()
    events = list(itertools.islice(kalshi.iter_events(status="open"), 3))
    assert events and events[0]["event_ticker"]
    series = kalshi.get_series(events[0]["series_ticker"])
    assert series.get("fee_type")


def test_polymarket_live_reads() -> None:
    gamma = GammaClient()
    clob = ClobPublic()
    markets = list(itertools.islice(gamma.iter_markets(closed=False), 5))
    market = next(m for m in markets if yes_token_id(m))
    token = yes_token_id(market)
    assert token is not None

    now = int(time.time())
    hist = clob.prices_history(token, now - 7 * 86_400, now)
    assert len(hist) > 0
    # prices-history last point should track the live midpoint (D4 semantics).
    assert abs(hist.iloc[-1].p - clob.midpoint(token)) < 0.05
