"""Pre-event, maker-side, fee-aware, strict-fill backtest simulator (§6, D3–D7).

For each binary-scorable market the simulator does exactly one honest thing at the pre-event
decision time ``T``:

1. **Price.** Read the candle bar at/just-before ``T`` (never after). The market price is that
   bar's mid ``(yes_bid.close + yes_ask.close)/2`` (D4). A bar older than ``max_staleness_days`` or
   with a broken/absent quote is *not* a tradable price (trap #9) — the market is priced ``NaN`` and
   never traded.
2. **Signal.** Trade only when the price sits in the 15–60% band (D3) *and* the model disagrees by
   more than ``theta``. Buy yes if the model is higher, sell yes (buy no) if lower.
3. **Maker quote.** Rest one tick better than the relevant best quote (§6), never crossing.
4. **Strict fill.** Fill *only* if a later bar's **trade prints** cross the resting level — a buy at
   ``q`` fills when a later ``price_low ≤ q``; a sell fills when ``price_high ≥ q`` (D4, trap #4).
   Fills are tested against later bars only (``ts`` strictly after the decision bar), so a same-day
   print that may have preceded the order can never fill it — conservative by construction.
5. **Settle.** Hold to resolution; ``pnl = payout − stake − fee`` per contract, with the Kalshi
   maker fee (D7, ``ceil`` to the cent) and fee-free settlement.

Everything here is a pure function of the (predictions, candles, registry) frames — no network — so
the fill/fee/leakage logic is exhaustively unit-tested. The market metadata (occasion, speaker,
resolution risk) is joined from the registry; ``occasion`` is the report's bootstrap cluster.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab.data.schemas import BACKTEST_COLUMNS, Backtest, empty_backtest

log = logging.getLogger("pmlab.study.backtest")

_DAY_S = 86_400.0
TAKER_FEE_RATE = 0.07
MAKER_FEE_RATE = 0.0175  # only on series with fee_type == 'quadratic_with_maker_fees' (D7)
MAKER_FEE_TYPE = "quadratic_with_maker_fees"


@dataclass(frozen=True)
class BacktestParams:
    """Trading rule knobs (D3/D5 defaults). ``model_col`` selects which probability trades —
    ``p_cal`` for the primary model, ``p_base``/``p_topic`` for the §6 control backtests."""

    band_lo: float = 0.15
    band_hi: float = 0.60
    theta: float = 0.05
    tick: float = 0.01
    max_staleness_days: float = 7.0
    model_col: str = "p_cal"
    trade_split: str | None = "test"  # trade only out-of-sample by default; None = all splits


# --- pure trade mechanics ------------------------------------------------------------------------


def maker_fee(
    price: float, contracts: float, *, has_maker_fee: bool, rate: float = MAKER_FEE_RATE
) -> float:
    """Kalshi maker fee ``ceil_to_cent(rate · C · p · (1−p))`` (D7); 0 where the series has none.

    ``price`` is the fill price (yes-equivalent p). Applied per fill; settlement is fee-free."""
    if not has_maker_fee or contracts <= 0:
        return 0.0
    raw = rate * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


def maker_quote(side: str, yes_bid: float, yes_ask: float, tick: float) -> float | None:
    """Resting maker price one tick better than the best quote, never crossing the book (§6).

    Buy yes → improve the bid; sell yes → improve the ask. Returns ``None`` for a locked/crossed
    book where a one-tick improvement would cross (no passive fill possible)."""
    if side == "yes":
        q = round(yes_bid + tick, 2)
        return q if q < yes_ask else None
    q = round(yes_ask - tick, 2)
    return q if q > yes_bid else None


def settle(side: str, fill_price: float, y: int, fee: float) -> tuple[float, float, float]:
    """``(payout, stake, pnl)`` per contract held to resolution. ``pnl = payout − stake − fee``.

    Yes: stake = fill, payout = 1 if Yes. No (sell yes): stake = 1 − fill, payout = 1 if No."""
    if side == "yes":
        stake = fill_price
        payout = 1.0 if y == 1 else 0.0
    else:
        stake = 1.0 - fill_price
        payout = 1.0 if y == 0 else 0.0
    return payout, stake, payout - stake - fee


# --- per-market candle lookups -------------------------------------------------------------------


@dataclass
class _Bars:
    """Column arrays for one market's bars, sorted ascending by ``ts`` (fast decision/fill)."""

    ts: np.ndarray
    bid_close: np.ndarray
    ask_close: np.ndarray
    price_low: np.ndarray
    price_high: np.ndarray


def _bars_by_ticker(candles: pd.DataFrame) -> dict[str, _Bars]:
    out: dict[str, _Bars] = {}
    if candles.empty:
        return out
    for tkr, g in candles.sort_values("ts").groupby("ticker", sort=False):
        out[str(tkr)] = _Bars(
            ts=g["ts"].to_numpy(dtype="float64"),
            bid_close=g["yes_bid_close"].to_numpy(dtype="float64"),
            ask_close=g["yes_ask_close"].to_numpy(dtype="float64"),
            price_low=g["price_low"].to_numpy(dtype="float64"),
            price_high=g["price_high"].to_numpy(dtype="float64"),
        )
    return out


def _decision_bar(bars: _Bars, decision_ts: float, max_staleness_s: float) -> int | None:
    """Index of the last bar at/before ``decision_ts`` with a fresh, valid two-sided quote, else
    ``None`` (no pre-decision price, stale, or a broken/crossed quote — untradable, trap #9)."""
    if bars.ts.size == 0 or not np.isfinite(decision_ts):
        return None
    idx = int(np.searchsorted(bars.ts, decision_ts, side="right")) - 1
    if idx < 0 or (decision_ts - bars.ts[idx]) > max_staleness_s:
        return None
    bid, ask = bars.bid_close[idx], bars.ask_close[idx]
    if not (np.isfinite(bid) and np.isfinite(ask)) or ask < bid:
        return None
    if not (0.0 < (bid + ask) / 2.0 < 1.0):
        return None
    return idx


def _fills(bars: _Bars, side: str, quote: float, after_idx: int) -> float | None:
    """First later-bar ``ts`` whose trade print crosses ``quote`` (buy: ``price_low ≤ q``; sell:
    ``price_high ≥ q``), else ``None``. Only bars strictly after ``after_idx`` count (trap #4)."""
    lo = after_idx + 1
    if lo >= bars.ts.size:
        return None
    if side == "yes":
        hit = np.where(bars.price_low[lo:] <= quote)[0]
    else:
        hit = np.where(bars.price_high[lo:] >= quote)[0]
    if hit.size == 0:
        return None
    return float(bars.ts[lo + int(hit[0])])


# --- simulator -----------------------------------------------------------------------------------


def _has_maker_fee(market_ticker: str, fee_types: dict[str, str] | None) -> bool:
    """Whether a fill on this market pays the maker fee. Unknown series default to *charging* it —
    the conservative choice (overstates cost, understates edge; §0.3)."""
    if fee_types is None:
        return True
    series = market_ticker.split("-")[0]
    return fee_types.get(series, MAKER_FEE_TYPE) == MAKER_FEE_TYPE


def _meta(registry: pd.DataFrame) -> pd.DataFrame:
    cols = ["market_ticker", "occasion", "speaker", "format", "resolution_risk"]
    return registry[cols].drop_duplicates("market_ticker").reset_index(drop=True)


def run_backtest(
    predictions: pd.DataFrame,
    candles: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    params: BacktestParams | None = None,
    fee_types: dict[str, str] | None = None,
    contracts: float = 1.0,
) -> pd.DataFrame:
    """Backtest every market in ``predictions`` → a validated :class:`Backtest` frame (one row
    per market).

    ``predictions`` carries ``PREDICTION_COLUMNS`` (see :mod:`pmlab.model.calibrate`) plus whatever
    ``params.model_col`` names — the frame is produced by whatever model the caller researches;
    this harness only supplies the fill/fee/settle mechanics. ``fee_types`` maps
    ``series_ticker → fee_type`` (from ``list_series``); ``None`` conservatively charges the maker
    fee everywhere."""
    p = params or BacktestParams()
    preds = predictions
    if preds.empty:
        return empty_backtest()

    df = preds.merge(_meta(registry), on="market_ticker", how="left")
    bars_by = _bars_by_ticker(candles)
    stale_s = p.max_staleness_days * _DAY_S

    rows: list[dict[str, object]] = []
    for r in df.itertuples(index=False):
        model_p = float(getattr(r, p.model_col))
        y = int(r.y)
        dts = float(r.decision_ts) if pd.notna(r.decision_ts) else float("nan")
        row: dict[str, object] = {
            "market_ticker": r.market_ticker,
            "occasion": r.occasion,
            "speaker": r.speaker,
            "format": r.format if pd.notna(r.format) else "other",
            "resolution_risk": r.resolution_risk if pd.notna(r.resolution_risk) else "low",
            "decision_ts": r.decision_ts,
            "split": r.split,
            "y": y,
            "market_prob": np.nan,
            "p_base": float(r.p_base),
            "p_topic": float(r.p_topic),
            "p_cal": float(r.p_cal),
            "traded": False, "side": None, "quote_price": np.nan,
            "filled": False, "fill_ts": pd.NA, "fill_price": np.nan,
            "fee": 0.0, "contracts": 0.0, "stake": 0.0, "payout": 0.0, "pnl": 0.0,
            "decided_by_technicality": False,
        }

        bars = bars_by.get(str(r.market_ticker))
        idx = _decision_bar(bars, dts, stale_s) if bars is not None else None
        if bars is None or idx is None:
            rows.append(row)
            continue

        bid, ask = float(bars.bid_close[idx]), float(bars.ask_close[idx])
        market_prob = (bid + ask) / 2.0
        row["market_prob"] = market_prob

        trade_ok = p.trade_split is None or r.split == p.trade_split
        in_band = p.band_lo <= market_prob <= p.band_hi
        edge = model_p - market_prob
        if not (trade_ok and in_band and abs(edge) > p.theta):
            rows.append(row)
            continue

        side = "yes" if edge > 0 else "no"
        quote = maker_quote(side, bid, ask, p.tick)
        if quote is None:
            rows.append(row)
            continue

        row["traded"] = True
        row["side"] = side
        row["quote_price"] = quote
        fill_ts = _fills(bars, side, quote, idx)
        if fill_ts is not None:
            has_fee = _has_maker_fee(str(r.market_ticker), fee_types)
            fee = maker_fee(quote, contracts, has_maker_fee=has_fee)
            payout, stake, pnl = settle(side, quote, y, fee)
            row.update(
                filled=True, fill_ts=int(fill_ts), fill_price=quote, fee=fee,
                contracts=contracts, stake=stake * contracts, payout=payout * contracts,
                pnl=pnl * contracts,
                decided_by_technicality=str(row["resolution_risk"]) in {"medium", "high"},
            )
        rows.append(row)

    out = pd.DataFrame(rows, columns=BACKTEST_COLUMNS)
    out["fill_ts"] = out["fill_ts"].astype("Int64")
    out["decision_ts"] = out["decision_ts"].astype("Int64")
    return Backtest.validate(out)


def persist_backtest(bt: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Backtest.validate(bt)
    bt.to_parquet(path, index=False)
    return path


def load_backtest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_backtest()
    return Backtest.validate(pd.read_parquet(path))
