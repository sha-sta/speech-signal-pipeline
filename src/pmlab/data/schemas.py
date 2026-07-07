"""Pandera schemas at the mention-pipeline boundaries (owner standard: validate at I/O edges).

Frames flowing through the study:

* :class:`MentionCatalog` — one row per discovered mention *market* (``mentions/discover.py``).
* :class:`MentionRegistry` — the catalog joined with the agent-parsed :class:`Resolution` fields
  and candle-coverage flags (``mentions/registry.py``); the study's unit of analysis.
* :class:`CorpusTranscripts` — one row per scraped speaking *occasion* (``corpus/store.py``).
* :class:`Headlines` — dated news headlines (``corpus/headlines.py``).

Timestamps are unix **seconds** as nullable ``Int64`` (Kalshi's ISO strings are parsed at the
boundary); missing → ``pd.NA``. List-valued fields (settlement sources) are persisted pipe-joined
so the frame stays flat for parquet/CSV round-trips.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# ``result`` is Kalshi's free-form settlement code. Observed live 2026-07-03: yes / no / "" (open)
# and "scalar" (a handful of oddly-settled binary markets). It is NOT enum-validated — it is an
# external field that has already drifted once (post-migration) and odd/cancelled markets must
# stay in the universe, not crash it. Only BINARY_RESULTS are scorable outcomes (M3).
BINARY_RESULTS = ("yes", "no")
FORMAT_VALUES = ("earnings", "speech", "broadcast", "sports", "entertainment", "other")
TEMPLATE_VALUES = ("speech", "earnings", "duration", "length", "superlative", "nqe", "other")
VARIANT_VALUES = ("exact", "plural_possessive", "aliases_listed", "count_threshold", "fuzzy")
RISK_VALUES = ("low", "medium", "high")

# Column order for building frames deterministically (incl. the empty case). Candle-coverage
# columns live on the catalog (defaulted in the sweep, filled by attach_candle_coverage).
CATALOG_COLUMNS = [
    "market_ticker", "event_ticker", "series_ticker", "series_title", "speaker", "format",
    "yes_sub_title", "rules_primary", "rules_secondary",
    "open_ts", "close_ts", "occurrence_ts", "expiration_ts",
    "result", "volume", "open_interest", "status",
    "settlement_sources", "n_settlement_sources",
    "has_candles", "n_candles",
]

REGISTRY_EXTRA_COLUMNS = [
    "template", "phrase", "variant_rule", "occasion", "cutoff_ts", "governing_sources",
    "early_resolution", "cancellation_clause", "is_nqe",
    "resolution_risk", "confidence", "notes", "eligible",
]
REGISTRY_COLUMNS = CATALOG_COLUMNS + REGISTRY_EXTRA_COLUMNS


class MentionCatalog(pa.DataFrameModel):
    """Discovered mention markets (parlay legs already excluded)."""

    market_ticker: Series[str] = pa.Field(unique=True, nullable=False)
    event_ticker: Series[str] = pa.Field(nullable=False)
    series_ticker: Series[str] = pa.Field(nullable=False)
    series_title: Series[str] = pa.Field(nullable=True)
    speaker: Series[str] = pa.Field(nullable=True)
    format: Series[str] = pa.Field(isin=FORMAT_VALUES)
    yes_sub_title: Series[str] = pa.Field(nullable=True)
    rules_primary: Series[str] = pa.Field(nullable=True)
    rules_secondary: Series[str] = pa.Field(nullable=True)
    open_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    close_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    occurrence_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    expiration_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    result: Series[str] = pa.Field(nullable=True)  # free-form settlement code; see BINARY_RESULTS
    volume: Series[float] = pa.Field(ge=0, nullable=True)
    open_interest: Series[float] = pa.Field(ge=0, nullable=True)
    status: Series[str] = pa.Field(nullable=True)
    settlement_sources: Series[str] = pa.Field(nullable=True)
    n_settlement_sources: Series[int] = pa.Field(ge=0)
    has_candles: Series[bool] = pa.Field()
    n_candles: Series[int] = pa.Field(ge=0)

    class Config:
        strict = True
        coerce = True


class MentionRegistry(pa.DataFrameModel):
    """Catalog + agent-parsed resolution fields + candle coverage; the study unit of analysis."""

    # --- carried from the catalog ---
    market_ticker: Series[str] = pa.Field(unique=True, nullable=False)
    event_ticker: Series[str] = pa.Field(nullable=False)
    series_ticker: Series[str] = pa.Field(nullable=False)
    series_title: Series[str] = pa.Field(nullable=True)
    speaker: Series[str] = pa.Field(nullable=True)
    format: Series[str] = pa.Field(isin=FORMAT_VALUES)
    yes_sub_title: Series[str] = pa.Field(nullable=True)
    rules_primary: Series[str] = pa.Field(nullable=True)
    rules_secondary: Series[str] = pa.Field(nullable=True)
    open_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    close_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    occurrence_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    expiration_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    result: Series[str] = pa.Field(nullable=True)  # free-form settlement code; see BINARY_RESULTS
    volume: Series[float] = pa.Field(ge=0, nullable=True)
    open_interest: Series[float] = pa.Field(ge=0, nullable=True)
    status: Series[str] = pa.Field(nullable=True)
    settlement_sources: Series[str] = pa.Field(nullable=True)
    n_settlement_sources: Series[int] = pa.Field(ge=0)
    # --- parsed resolution (mentions/resolution.py) ---
    template: Series[str] = pa.Field(isin=TEMPLATE_VALUES)
    phrase: Series[str] = pa.Field(nullable=True)
    variant_rule: Series[str] = pa.Field(isin=VARIANT_VALUES)
    occasion: Series[str] = pa.Field(nullable=True)
    cutoff_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    governing_sources: Series[str] = pa.Field(nullable=True)
    early_resolution: Series[bool] = pa.Field()
    cancellation_clause: Series[bool] = pa.Field()
    is_nqe: Series[bool] = pa.Field()
    resolution_risk: Series[str] = pa.Field(isin=RISK_VALUES)
    confidence: Series[float] = pa.Field(ge=0.0, le=1.0)
    notes: Series[str] = pa.Field(nullable=True)
    # --- coverage (mentions/discover.py) ---
    has_candles: Series[bool] = pa.Field()
    n_candles: Series[int] = pa.Field(ge=0)
    eligible: Series[bool] = pa.Field()

    class Config:
        strict = True
        coerce = True


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in columns})


def empty_catalog() -> pd.DataFrame:
    """A zero-row catalog frame with the declared columns (so downstream code never special-cases
    the empty sweep)."""
    return _empty(CATALOG_COLUMNS)


def empty_registry() -> pd.DataFrame:
    return _empty(REGISTRY_COLUMNS)


# --- corpus ----------------------------------------------------------------------------------

# One scraped speaking occasion. ``text`` is the full transcript body (kept inline — parquet
# compresses it well and the corpus is gitignored). ``date_ts`` is the occasion's unix-second
# timestamp.
CORPUS_COLUMNS = [
    "doc_id", "speaker", "format", "occasion", "date_ts", "source", "url",
    "n_words", "text",
]
HEADLINE_COLUMNS = ["headline_id", "ts", "source", "text"]


class CorpusTranscripts(pa.DataFrameModel):
    """Scraped speaking occasions (one row per transcript)."""

    doc_id: Series[str] = pa.Field(unique=True, nullable=False)
    speaker: Series[str] = pa.Field(nullable=False)
    format: Series[str] = pa.Field(isin=FORMAT_VALUES)
    occasion: Series[str] = pa.Field(nullable=True)
    date_ts: Series[int] = pa.Field(ge=0)  # leakage boundary — never nullable in the store
    source: Series[str] = pa.Field(nullable=False)
    url: Series[str] = pa.Field(nullable=True)
    n_words: Series[int] = pa.Field(ge=0)
    text: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


class Headlines(pa.DataFrameModel):
    """Dated news headlines (corpus/headlines.py)."""

    headline_id: Series[str] = pa.Field(unique=True, nullable=False)
    ts: Series[int] = pa.Field(ge=0)
    source: Series[str] = pa.Field(nullable=True)
    text: Series[str] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True


def empty_corpus() -> pd.DataFrame:
    return _empty(CORPUS_COLUMNS)


def empty_headlines() -> pd.DataFrame:
    return _empty(HEADLINE_COLUMNS)


# --- candles + backtest -----------------------------------------------------------------------

# One candlestick bar for a market (``venues/kalshi.py`` long format + the store's
# ``period_minutes`` tag so one file may hold mixed granularity). ``ts`` is the period-END s (Kalshi
# semantics). Prices are yes-side *dollars* ∈ [0, 1]; ``yes_bid``/``yes_ask`` are the resting
# book quotes and ``price`` is the trade (last) print — the strict maker fill test keys off the
# trade prints, never the quotes. Missing quotes/prints → NaN (a thin/absent bar), which the
# backtest treats as untradable.
CANDLE_STORE_COLUMNS = [
    "ticker", "ts", "period_minutes",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "price_open", "price_high", "price_low", "price_close", "price_mean",
    "volume", "open_interest",
]

# One backtested market: the study's analysis unit. Every eligible, binary-scorable market that had
# a tradable market price at its pre-event decision time gets a row — carrying the model
# probabilities, the calibrated probability, the market-implied probability, the realised outcome,
# and (when a position was actually taken and filled) the maker trade + fee + P&L. Metrics slice
# this frame: the Brier comparison uses all rows; P&L/hit-rate use ``filled`` rows. ``occasion`` is
# the block-bootstrap cluster key (§6). ``split`` is the time-ordered train/test tag — the D9
# primary metric is measured on ``test`` only.
BACKTEST_COLUMNS = [
    "market_ticker", "occasion", "speaker", "format", "resolution_risk",
    "decision_ts", "split", "y",
    "market_prob", "p_base", "p_topic", "p_cal",
    "traded", "side", "quote_price", "filled", "fill_ts", "fill_price",
    "fee", "contracts", "stake", "payout", "pnl", "decided_by_technicality",
]
SPLIT_VALUES = ("train", "test")
SIDE_VALUES = ("yes", "no")


class Candles(pa.DataFrameModel):
    """Candlestick bars for the scorable markets (the backtest's market-price substrate)."""

    ticker: Series[str] = pa.Field(nullable=False)
    ts: Series[int] = pa.Field(ge=0)
    period_minutes: Series[int] = pa.Field(isin=(1, 60, 1440))
    yes_bid_open: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    yes_bid_high: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    yes_bid_low: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    yes_bid_close: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    yes_ask_open: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    yes_ask_high: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    yes_ask_low: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    yes_ask_close: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    price_open: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    price_high: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    price_low: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    price_close: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    price_mean: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    volume: Series[float] = pa.Field(ge=0.0, nullable=True)
    open_interest: Series[float] = pa.Field(ge=0.0, nullable=True)

    class Config:
        strict = True
        coerce = True
        unique = ["ticker", "ts", "period_minutes"]


class Backtest(pa.DataFrameModel):
    """One row per backtested market (§6): model probs, market price, outcome, maker trade + P&L."""

    market_ticker: Series[str] = pa.Field(unique=True, nullable=False)
    occasion: Series[str] = pa.Field(nullable=True)
    speaker: Series[str] = pa.Field(nullable=True)
    format: Series[str] = pa.Field(isin=FORMAT_VALUES)
    resolution_risk: Series[str] = pa.Field(isin=RISK_VALUES)
    decision_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    split: Series[str] = pa.Field(isin=SPLIT_VALUES)
    y: Series[int] = pa.Field(isin=(0, 1))
    market_prob: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    p_base: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    p_topic: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    p_cal: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    traded: Series[bool] = pa.Field()
    side: Series[str] = pa.Field(isin=SIDE_VALUES, nullable=True)
    quote_price: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    filled: Series[bool] = pa.Field()
    fill_ts: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    fill_price: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    fee: Series[float] = pa.Field(ge=0.0)
    contracts: Series[float] = pa.Field(ge=0.0)
    stake: Series[float] = pa.Field(ge=0.0)
    payout: Series[float] = pa.Field()
    pnl: Series[float] = pa.Field()
    decided_by_technicality: Series[bool] = pa.Field()

    class Config:
        strict = True
        coerce = True


def empty_candles() -> pd.DataFrame:
    return _empty(CANDLE_STORE_COLUMNS)


def empty_backtest() -> pd.DataFrame:
    return _empty(BACKTEST_COLUMNS)
