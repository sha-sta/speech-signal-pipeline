"""Parse-against-fixture tests for the Polymarket gamma + clob clients (fixtures 2026-07-02)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from pmlab.http import ThrottledClient
from pmlab.venues.polymarket import ClobPublic, GammaClient, parse_token_ids, yes_token_id

ClientFactory = Callable[..., ThrottledClient]
Fixture = Callable[[str], Any]


def test_iter_markets_offset_paginates(mock_client: ClientFactory, fixture: Fixture) -> None:
    page = fixture("poly_markets.json")  # a bare JSON list

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets"
        offset = int(request.url.params.get("offset", "0"))
        return httpx.Response(200, json=page if offset == 0 else [])

    gamma = GammaClient(client=mock_client(handler))
    markets = list(gamma.iter_markets(closed=False))

    assert len(markets) == len(page)
    assert markets[0]["conditionId"] == page[0]["conditionId"]


def test_token_helpers_align_with_outcomes(fixture: Fixture) -> None:
    market = fixture("poly_markets.json")[0]
    tokens = parse_token_ids(market)
    raw_outcomes = market["outcomes"]
    outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes

    assert len(tokens) == len(outcomes) == 2
    assert outcomes[0] == "Yes"
    assert yes_token_id(market) == tokens[0]  # index 0 == YES


def test_yes_token_id_handles_missing() -> None:
    assert yes_token_id({}) is None


def test_prices_history_parses(mock_client: ClientFactory, fixture: Fixture) -> None:
    hist = fixture("poly_prices_history.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/prices-history"
        assert "startTs" in request.url.params  # camelCase window params (§2.2)
        return httpx.Response(200, json=hist)

    clob = ClobPublic(client=mock_client(handler))
    df = clob.prices_history("tok", 0, 1000)

    assert list(df.columns) == ["ts", "p"]
    assert len(df) == len(hist["history"])
    assert int(df.iloc[-1].ts) == hist["history"][-1]["t"]
    assert df.iloc[-1].p == float(hist["history"][-1]["p"])


def test_midpoint_parses_string(mock_client: ClientFactory, fixture: Fixture) -> None:
    mid_doc = fixture("poly_midpoint.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/midpoint"
        return httpx.Response(200, json=mid_doc)

    clob = ClobPublic(client=mock_client(handler))
    assert clob.midpoint("tok") == float(mid_doc["mid"])
