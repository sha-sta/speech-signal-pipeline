"""Candle store (§6 step a, D4): pull + persist bid/ask OHLC for the backtestable markets.

The backtest needs, per market, the pre-event **market price** (a quote at the decision time) and
the **trade prints** between then and resolution (to test strict maker fills). This module fetches
those bars via :meth:`~pmlab.venues.kalshi.KalshiPublic.candles_batch` and stores them as a
:class:`~pmlab.data.schemas.Candles` frame.

**Granularity (a flagged §0.2 substitution).** The plan's aspirational note said "1-min candles";
this pulls **daily** bars universe-wide by default (``period_minutes`` is configurable). Rationale:
(1) the D9 primary metric — calibrated Brier vs market-implied Brier — depends only on the mid at
the decision time and the outcome, so it is *independent* of bar granularity; (2) 1-min bars for
~15k thin markets over their lifetimes is mostly empty and thousands of requests. The cost is that
daily bars *overestimate* maker fills (a day's high/low spans intraday moves a resting order might
not have caught) — the backtest and report flag this, and a finer re-run on the traded subset is
the documented refinement.

Requests are self-throttled and cached forever (historical bars are immutable), so re-pulls are
free and the store append-merges by ``(ticker, ts, period_minutes)``."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from pmlab.data.schemas import (
    BINARY_RESULTS,
    CANDLE_STORE_COLUMNS,
    Candles,
    empty_candles,
)
from pmlab.venues.kalshi import KalshiPublic

log = logging.getLogger("pmlab.study.candles")

_DAY_S = 86_400
PERIOD_ALIASES = {"daily": 1440, "hourly": 60, "minute": 1}
_DEFAULT_PRE_BUFFER_DAYS = 30  # window start fallback when a market has no open_ts
_DEFAULT_POST_BUFFER_DAYS = 2  # extend past close so the resolution-day bar is captured


def target_markets(registry: pd.DataFrame) -> pd.DataFrame:
    """Eligible, binary-scorable markets — the set worth pulling candles for (the backtest unit)."""
    mask = registry["eligible"].astype(bool) & registry["result"].isin(list(BINARY_RESULTS))
    return registry[mask].reset_index(drop=True)


def _windows(registry: pd.DataFrame) -> pd.DataFrame:
    """Per-market ``[start, end]`` fetch window (unix s) from the registry timing columns.

    Start = open, else event − buffer, else close − buffer. End = last of {close, expiration,
    event} + buffer. Markets with no usable timestamp at all are dropped (nothing to fetch)."""
    open_ts = pd.to_numeric(registry["open_ts"], errors="coerce")
    close_ts = pd.to_numeric(registry["close_ts"], errors="coerce")
    occ_ts = pd.to_numeric(registry["occurrence_ts"], errors="coerce")
    exp_ts = pd.to_numeric(registry["expiration_ts"], errors="coerce")

    last = pd.concat([close_ts, exp_ts, occ_ts], axis=1).max(axis=1)
    end = last + _DEFAULT_POST_BUFFER_DAYS * _DAY_S
    start = open_ts.copy()
    start = start.fillna(occ_ts - _DEFAULT_PRE_BUFFER_DAYS * _DAY_S)
    start = start.fillna(close_ts - _DEFAULT_PRE_BUFFER_DAYS * _DAY_S)

    out = pd.DataFrame(
        {"ticker": registry["market_ticker"].to_numpy(), "start": start, "end": end}
    )
    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] > out["start"]]
    out["start"] = out["start"].astype("int64")
    out["end"] = out["end"].astype("int64")
    return out.reset_index(drop=True)


def _time_local_chunks(windows: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    """Yield ``size``-row chunks after sorting by end time, so each chunk is time-local (its
    ``[min start, max end]`` span stays small → few candle buckets per request)."""
    ordered = windows.sort_values("end").reset_index(drop=True)
    for i in range(0, len(ordered), size):
        yield ordered.iloc[i : i + size]


def pull_candles(
    kalshi: KalshiPublic,
    registry: pd.DataFrame,
    *,
    period_minutes: int = 1440,
    limit: int | None = None,
    group_size: int = 100,
) -> pd.DataFrame:
    """Fetch candles for the registry's backtestable markets → a validated :class:`Candles` frame.

    ``limit`` caps the market count (dev/smoke). Bars are tagged with ``period_minutes`` and
    deduped on ``(ticker, ts, period_minutes)``."""
    windows = _windows(target_markets(registry))
    if limit is not None:
        windows = windows.head(limit)
    if windows.empty:
        return empty_candles()

    wanted = set(windows["ticker"])
    frames: list[pd.DataFrame] = []
    n_groups = (len(windows) + group_size - 1) // group_size
    for gi, chunk in enumerate(_time_local_chunks(windows, group_size), start=1):
        tickers = list(chunk["ticker"])
        start = int(chunk["start"].min())
        end = int(chunk["end"].max())
        df = kalshi.candles_batch(tickers, start, end, period_minutes)
        if len(df):
            frames.append(df)
        log.info("candles: group %d/%d (%d tickers) → %d bars so far",
                 gi, n_groups, len(tickers), sum(len(f) for f in frames))

    if not frames:
        return empty_candles()
    allbars = pd.concat(frames, ignore_index=True)
    allbars = allbars[allbars["ticker"].isin(wanted)]
    allbars["period_minutes"] = period_minutes
    allbars = allbars[CANDLE_STORE_COLUMNS]
    allbars = allbars.drop_duplicates(subset=["ticker", "ts", "period_minutes"], keep="last")
    allbars = allbars.reset_index(drop=True)
    return Candles.validate(allbars)


def persist_candles(candles: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Candles.validate(candles)
    candles.to_parquet(path, index=False)
    return path


def load_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_candles()
    return Candles.validate(pd.read_parquet(path))


def merge_candles(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Append bars, keeping the latest per ``(ticker, ts, period_minutes)`` (idempotent pull)."""
    if not len(existing):
        return Candles.validate(incoming)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["ticker", "ts", "period_minutes"], keep="last"
    ).reset_index(drop=True)
    return Candles.validate(combined)


def coverage_stats(candles: pd.DataFrame, registry: pd.DataFrame) -> dict[str, int]:
    """How many backtestable markets actually got bars (the honest denominator for the study)."""
    tgt = target_markets(registry)
    n_target = len(tgt)
    with_bars = set(candles["ticker"].unique()) if len(candles) else set()
    n_covered = int(tgt["market_ticker"].isin(with_bars).sum())
    return {
        "n_target": n_target,
        "n_covered": n_covered,
        "n_missing": n_target - n_covered,
        "n_bars": int(len(candles)),
    }
