"""Discover the Kalshi mention-market universe → ``data/catalog/mentions.parquet`` (§4.1).

**Enumeration strategy (supersedes §4.1's event-sweep; §0.4 verify-at-build 2026-07-03):** the
``/events?category=`` filter is silently ignored by the API, so the specced "sweep all events and
keep ``category=='Mentions'``" would page the *entire platform* feed. Instead we list the Mentions
**series** directly (``/series?category=Mentions`` — honored, unpaginated, ~376 series) and pull
each series' events with markets nested (``with_nested_markets=true``). Same coverage, far cheaper.
Parlay legs (``mve_collection_ticker`` set) are excluded and counted (trap #6 — log every drop).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pmlab.data.schemas import CATALOG_COLUMNS, MentionCatalog, empty_catalog
from pmlab.venues.kalshi import KalshiPublic

log = logging.getLogger("pmlab.mentions.discover")

MENTIONS_CATEGORY = "Mentions"
DEFAULT_STATUSES = ("settled", "closed", "open")

# Matched as substrings of the (uppercased) series ticker, e.g. "NHL" in "KXNHLMENTION". "WC" =
# World Cup, one of the largest sports mention series; safe as a token in this ticker namespace.
_SPORTS = (
    "NFL", "NBA", "NCAA", "MLB", "NHL", "UFC", "WWE", "MMA", "BOXING", "FIGHT", "DRAFT",
    "GOLF", "TENNIS", "SOCCER", "FIFA", "WC", "ATHLETE",
)
_ENTERTAINMENT_TICKER = (
    "DWTS", "LOVEISLAND", "REALITY", "SURVIVOR", "BACHELOR", "OSCARS", "GRAMMY", "SNL",
)
_BROADCAST_TICKER = ("FTN", "MTP", "FACENATION", "MEETPRESS", "THEVIEW", "FIRSTTAKE")
_BROADCAST_TITLE = (
    "face the nation", "meet the press", "the view", "first take", "squawk", "cnbc",
    "espn", "state of play", "nintendo direct", "podcast", "late night",
)


@dataclass
class DiscoveryStats:
    n_series: int = 0
    n_events: int = 0
    n_markets_seen: int = 0
    n_parlay_excluded: int = 0
    n_catalog: int = 0
    by_format: dict[str, int] = field(default_factory=dict)
    by_status: dict[str, int] = field(default_factory=dict)


def infer_format(series_ticker: str, series_title: str) -> str:
    t = series_ticker.upper()
    title = (series_title or "").lower()
    if "EARNINGS" in t or "MENTIONEARN" in t or "earnings" in title:
        return "earnings"
    if any(k in t for k in _SPORTS):
        return "sports"
    if any(k in t for k in _ENTERTAINMENT_TICKER) or any(
        k in title for k in ("dwts", "love island")
    ):
        return "entertainment"
    if any(k in t for k in _BROADCAST_TICKER) or any(k in title for k in _BROADCAST_TITLE):
        return "broadcast"
    return "speech"


def infer_speaker(series_ticker: str, series_title: str) -> str:
    """Coarse series-level speaker label (the per-market parsed speaker in ``resolution.py`` is
    authoritative; this is a fallback grouping key for the catalog)."""
    s = (series_title or "").strip()
    s = re.sub(
        r"\b(earnings call|earnings mention|earnings|mention|call)\b", "", s, flags=re.IGNORECASE
    )
    s = re.sub(r"\s+", " ", s).strip(" -–—:")
    return s or series_ticker


def _iso_to_ts(value: Any) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def iter_mention_markets(
    kalshi: KalshiPublic,
    statuses: Sequence[str] = DEFAULT_STATUSES,
    *,
    limit_series: int | None = None,
    stats: DiscoveryStats | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(market, event)`` pairs across all Mentions series (parlays excluded)."""
    series = kalshi.list_series(category=MENTIONS_CATEGORY)
    if limit_series is not None:
        series = series[:limit_series]
    if stats is not None:
        stats.n_series = len(series)
    seen_markets: set[str] = set()
    seen_events: set[str] = set()
    for s in series:
        series_ticker = s["ticker"]
        series_title = s.get("title", "")
        for status in statuses:
            for event in kalshi.iter_events(
                status=status, series_ticker=series_ticker, with_nested_markets=True
            ):
                event.setdefault("series_ticker", series_ticker)
                event["series_title"] = series_title  # series-level title (not on the event object)
                et = event.get("event_ticker", "")
                if et not in seen_events:
                    seen_events.add(et)
                    if stats is not None:
                        stats.n_events += 1
                for market in event.get("markets", []) or []:
                    ticker = market.get("ticker")
                    if not ticker or ticker in seen_markets:
                        continue
                    seen_markets.add(ticker)
                    if stats is not None:
                        stats.n_markets_seen += 1
                    if market.get("mve_collection_ticker"):  # parlay leg (§2) — exclude + count
                        if stats is not None:
                            stats.n_parlay_excluded += 1
                        continue
                    yield market, event


def raw_record(market: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    """A lean, self-contained record for one kept market — everything the offline parser needs
    (full market object + the event's series + governing ``settlement_sources``). Persisted to
    ``raw_events.jsonl`` so the ``parse`` stage re-runs without re-hitting the network."""
    return {
        "market": dict(market),
        "event": {
            "event_ticker": event.get("event_ticker", ""),
            "series_ticker": event.get("series_ticker", ""),
            "series_title": event.get("series_title", ""),
            "title": event.get("title", ""),
            "settlement_sources": event.get("settlement_sources", []),
        },
    }


def _catalog_row(market: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    series_ticker = str(event.get("series_ticker") or str(market["ticker"]).split("-")[0])
    series_title = str(event.get("series_title") or "")
    sources = [str(s.get("name", "")).strip() for s in (event.get("settlement_sources") or [])]
    sources = [s for s in sources if s]
    return {
        "market_ticker": str(market["ticker"]),
        "event_ticker": str(event.get("event_ticker") or ""),
        "series_ticker": series_ticker,
        "series_title": series_title,
        "speaker": infer_speaker(series_ticker, series_title or event.get("title", "")),
        "format": infer_format(series_ticker, series_title or event.get("title", "")),
        "yes_sub_title": str(market.get("yes_sub_title") or ""),
        "rules_primary": str(market.get("rules_primary") or ""),
        "rules_secondary": str(market.get("rules_secondary") or ""),
        "open_ts": _iso_to_ts(market.get("open_time")),
        "close_ts": _iso_to_ts(market.get("close_time")),
        "occurrence_ts": _iso_to_ts(market.get("occurrence_datetime")),
        "expiration_ts": _iso_to_ts(market.get("expiration_time")),
        "result": str(market.get("result") or ""),
        "volume": _to_float(market.get("volume_fp")),
        "open_interest": _to_float(market.get("open_interest_fp")),
        "status": str(market.get("status") or ""),
        "settlement_sources": "|".join(sources),
        "n_settlement_sources": len(sources),
        "has_candles": False,  # defaults; attach_candle_coverage fills these
        "n_candles": 0,
    }


def sweep(
    kalshi: KalshiPublic,
    statuses: Sequence[str] = DEFAULT_STATUSES,
    *,
    limit_series: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], DiscoveryStats]:
    """One live pass over the mention universe → (schema-valid catalog, raw records, stats)."""
    stats = DiscoveryStats()
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for m, e in iter_mention_markets(kalshi, statuses, limit_series=limit_series, stats=stats):
        rows.append(_catalog_row(m, e))
        raw.append(raw_record(m, e))
    df = pd.DataFrame(rows, columns=CATALOG_COLUMNS) if rows else empty_catalog()
    df = MentionCatalog.validate(df)
    stats.n_catalog = len(df)
    stats.by_format = df["format"].value_counts().to_dict()
    stats.by_status = df["status"].value_counts().to_dict()
    return df, raw, stats


def build_catalog(
    kalshi: KalshiPublic,
    statuses: Sequence[str] = DEFAULT_STATUSES,
    *,
    limit_series: int | None = None,
) -> tuple[pd.DataFrame, DiscoveryStats]:
    """Sweep the mention universe into a schema-valid catalog frame + discovery stats."""
    df, _raw, stats = sweep(kalshi, statuses, limit_series=limit_series)
    return df, stats


def attach_candle_coverage(
    kalshi: KalshiPublic,
    catalog: pd.DataFrame,
    *,
    period_minutes: int = 1440,
    pad_days: int = 2,
) -> pd.DataFrame:
    """Add ``has_candles``/``n_candles`` by batching candlesticks over each market's life window.

    Daily granularity is an *existence* probe (M3 pulls 1-min bars for the fill sim); markets that
    settled before the live ``/historical`` cutoff won't return here and surface as ``has_candles``
    ``False`` — the caller reports how many are pre-cutoff. Returns a copy with the two columns."""
    out = catalog.copy()
    out["n_candles"] = 0
    out["has_candles"] = False
    valid = out.dropna(subset=["open_ts", "close_ts"])
    if valid.empty:
        return out
    tickers = valid["market_ticker"].tolist()
    start = int(valid["open_ts"].min()) - pad_days * 86_400
    end = int(valid["close_ts"].max()) + pad_days * 86_400
    candles = kalshi.candles_batch(tickers, start, end, period_minutes)
    if len(candles):
        counts = candles.groupby("ticker").size()
        n = out["market_ticker"].map(counts).fillna(0).astype(int)
        out["n_candles"] = n
        out["has_candles"] = n > 0
    return out


def persist_catalog(catalog: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    MentionCatalog.validate(catalog)
    catalog.to_parquet(path, index=False)
    return path


def load_catalog(path: Path) -> pd.DataFrame:
    return MentionCatalog.validate(pd.read_parquet(path))
