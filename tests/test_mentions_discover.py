"""Sweep/catalog tests for mentions.discover against recorded fixtures (no network)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import httpx

from pmlab.data.schemas import MentionCatalog
from pmlab.mentions.discover import (
    attach_candle_coverage,
    build_catalog,
    infer_format,
    infer_speaker,
    sweep,
)
from pmlab.venues.kalshi import KalshiPublic

ClientFactory = Callable[..., Any]
Fixture = Callable[[str], Any]


def _sweep_handler(
    series_doc: dict[str, Any], ftn: dict[str, Any], earnings: dict[str, Any]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/series":
            assert request.url.params.get("category") == "Mentions"
            return httpx.Response(200, json=series_doc)
        if path == "/events":
            if request.url.params.get("cursor"):
                return httpx.Response(200, json={"events": [], "cursor": ""})
            st = request.url.params.get("series_ticker")
            if st == "KXFTNMENTION":
                return httpx.Response(200, json=ftn)
            if st == "KXEARNINGSMENTIONAVGO":
                return httpx.Response(200, json=earnings)
            return httpx.Response(200, json={"events": [], "cursor": ""})
        raise AssertionError(f"unexpected path {path}")

    return handler


def test_infer_format() -> None:
    assert infer_format("KXEARNINGSMENTIONAVGO", "Broadcom Earnings") == "earnings"
    assert infer_format("KXFTNMENTION", "Face The Nation") == "broadcast"
    assert infer_format("KXNFLMENTION", "NFL Mention") == "sports"
    assert infer_format("KXWCUPMENTION", "World Cup Mention") == "sports"  # "WC" = World Cup
    assert infer_format("KXNHLMENTION", "NHL Mention") == "sports"
    assert infer_format("KXATHLETEMENTION", "Athlete Mention") == "sports"
    assert infer_format("KXDWTSMENTION", "DWTS Mention") == "entertainment"
    assert infer_format("KXSURVIVORMENTION", "Survivor Mention") == "entertainment"
    assert infer_format("KXPOWELLMENTION", "Powell Mention") == "speech"


def test_infer_speaker_strips_boilerplate() -> None:
    assert infer_speaker("KXEARNINGSMENTIONAVGO", "Broadcom Earnings Call") == "Broadcom"
    assert infer_speaker("KXCUOMOMENTION", "Cuomo Mention") == "Cuomo"


def test_sweep_builds_schema_valid_catalog(mock_client: ClientFactory, fixture: Fixture) -> None:
    handler = _sweep_handler(
        fixture("kalshi_mentions_series.json"),
        fixture("kalshi_mentions_event_ftn.json"),
        fixture("kalshi_mentions_event_earnings.json"),
    )
    kalshi = KalshiPublic(client=mock_client(handler))
    catalog, raw, stats = sweep(kalshi, statuses=("settled",))

    MentionCatalog.validate(catalog)
    assert len(catalog) == 26  # 13 FTN + 13 earnings
    assert len(raw) == 26
    assert stats.n_parlay_excluded == 0
    assert set(catalog["format"]) == {"broadcast", "earnings"}
    # NQE + phrase legs both present; settlement sources carried and counted
    ftn_rows = catalog[catalog["series_ticker"] == "KXFTNMENTION"]
    assert (ftn_rows["n_settlement_sources"] == 14).all()
    assert ftn_rows["yes_sub_title"].str.contains("Event does not qualify").any()


def test_sweep_dedups_across_statuses(mock_client: ClientFactory, fixture: Fixture) -> None:
    handler = _sweep_handler(
        fixture("kalshi_mentions_series.json"),
        fixture("kalshi_mentions_event_ftn.json"),
        fixture("kalshi_mentions_event_earnings.json"),
    )
    kalshi = KalshiPublic(client=mock_client(handler))
    # same events returned for all three statuses → must not double-count
    catalog, _raw, _stats = sweep(kalshi, statuses=("settled", "closed", "open"))
    assert catalog["market_ticker"].is_unique
    assert len(catalog) == 26


def test_parlay_legs_excluded_and_counted(mock_client: ClientFactory, fixture: Fixture) -> None:
    ftn = copy.deepcopy(fixture("kalshi_mentions_event_ftn.json"))
    ftn["events"][0]["markets"][0]["mve_collection_ticker"] = "KXMVE-R"  # mark one leg a parlay
    handler = _sweep_handler(
        fixture("kalshi_mentions_series.json"), ftn, fixture("kalshi_mentions_event_earnings.json")
    )
    kalshi = KalshiPublic(client=mock_client(handler))
    catalog, _raw, stats = sweep(kalshi, statuses=("settled",))
    assert stats.n_parlay_excluded == 1
    assert len(catalog) == 25


def test_build_catalog_thin_wrapper(mock_client: ClientFactory, fixture: Fixture) -> None:
    handler = _sweep_handler(
        fixture("kalshi_mentions_series.json"),
        fixture("kalshi_mentions_event_ftn.json"),
        fixture("kalshi_mentions_event_earnings.json"),
    )
    kalshi = KalshiPublic(client=mock_client(handler))
    catalog, stats = build_catalog(kalshi, statuses=("settled",))
    assert len(catalog) == stats.n_catalog == 26


def test_attach_candle_coverage_flags_presence(
    mock_client: ClientFactory, fixture: Fixture
) -> None:
    ftn = fixture("kalshi_mentions_event_ftn.json")
    candles_doc = fixture("kalshi_candles_batch.json")
    covered = candles_doc["markets"][0]["market_ticker"]
    series_doc = {"series": [{"ticker": "KXFTNMENTION", "title": "FTN"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/series":
            return httpx.Response(200, json=series_doc)
        if path == "/events":
            if request.url.params.get("cursor"):
                return httpx.Response(200, json={"events": [], "cursor": ""})
            return httpx.Response(200, json=ftn)
        if path == "/markets/candlesticks":
            return httpx.Response(200, json=candles_doc)
        raise AssertionError(path)

    kalshi = KalshiPublic(client=mock_client(handler))
    catalog, _raw, _stats = sweep(kalshi, statuses=("settled",))
    # Rewrite one catalog ticker to the one the candle fixture returns, so coverage flags it True.
    catalog.loc[catalog.index[0], "market_ticker"] = covered
    out = attach_candle_coverage(kalshi, catalog)
    assert bool(out.loc[out["market_ticker"] == covered, "has_candles"].iloc[0]) is True
    assert int(out["has_candles"].sum()) == 1  # only the seeded ticker has bars
