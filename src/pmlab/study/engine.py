"""Native-engine drop-in for :func:`pmlab.study.backtest.run_backtest`.

``run_backtest_native`` produces a frame **bit-identical** to the pure-Python simulator (the
parity gate in ``tests/test_engine_parity.py`` enforces it) by splitting responsibilities:

* This wrapper does everything pandas-shaped, reusing the reference implementation's own
  helpers so the two paths cannot drift: the registry meta join (``_meta``), the per-ticker bar
  grouping (``_bars_by_ticker``), and the fee-type lookup (``_has_maker_fee``). Strings become
  integer group ids and precomputed boolean gates.
* The C++ extension ``pmlab._engine`` (built from ``cpp/``, optional) runs the per-market hot
  loop over flat float64/int columns and returns output columns, which this wrapper reassembles
  into the validated :class:`Backtest` frame.

The extension is NOT part of the default install; without it this module imports fine and
:func:`engine_available` is False. Build it with::

    uv sync   # pybind11 lives in the dev group
    cmake --preset python -Dpybind11_DIR=$(uv run python -m pybind11 --cmakedir)   # in cpp/
    cmake --build cpp/build/python --parallel
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pandas as pd

from pmlab.data.schemas import BACKTEST_COLUMNS, Backtest, empty_backtest
from pmlab.study.backtest import (
    _DAY_S,
    BacktestParams,
    _bars_by_ticker,
    _has_maker_fee,
    _meta,
)

_engine: ModuleType | None
try:
    import pmlab._engine as _engine_module

    _engine = _engine_module
except ImportError:  # extension not built; the pure-Python path keeps working
    _engine = None


def engine_available() -> bool:
    return _engine is not None


def engine_version() -> str:
    if _engine is None:
        raise ImportError("pmlab._engine is not built; see src/pmlab/study/engine.py docstring")
    return str(_engine.version())


def run_backtest_native(
    predictions: pd.DataFrame,
    candles: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    params: BacktestParams | None = None,
    fee_types: dict[str, str] | None = None,
    contracts: float = 1.0,
) -> pd.DataFrame:
    """Same signature, same output, same bits as ``run_backtest`` — just a C++ hot loop."""
    if _engine is None:
        raise ImportError("pmlab._engine is not built; see src/pmlab/study/engine.py docstring")
    p = params or BacktestParams()
    if predictions.empty:
        return empty_backtest()

    df = predictions.merge(_meta(registry), on="market_ticker", how="left")
    n = len(df)
    bars_by = _bars_by_ticker(candles)

    # Concatenate the per-ticker bar arrays (dict insertion order) into one SoA block.
    group_of = {tkr: i for i, tkr in enumerate(bars_by)}
    if bars_by:
        groups = list(bars_by.values())
        bar_ts = np.concatenate([b.ts for b in groups])
        bar_bid = np.concatenate([b.bid_close for b in groups])
        bar_ask = np.concatenate([b.ask_close for b in groups])
        bar_plow = np.concatenate([b.price_low for b in groups])
        bar_phigh = np.concatenate([b.price_high for b in groups])
        counts = np.array([b.ts.size for b in groups], dtype=np.int64)
        offsets = np.zeros(len(groups), dtype=np.int64)
        np.cumsum(counts[:-1], out=offsets[1:])
    else:
        bar_ts = bar_bid = bar_ask = bar_plow = bar_phigh = np.empty(0, dtype=np.float64)
        offsets = counts = np.empty(0, dtype=np.int64)

    tickers = [str(t) for t in df["market_ticker"]]
    bar_group = np.fromiter((group_of.get(t, -1) for t in tickers), dtype=np.int64, count=n)
    decision_ts = pd.to_numeric(df["decision_ts"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    model_p = pd.to_numeric(df[p.model_col], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    y = df["y"].to_numpy(dtype=np.int32)
    if p.trade_split is None:
        trade_ok = np.ones(n, dtype=np.uint8)
    else:
        trade_ok = (df["split"] == p.trade_split).to_numpy(dtype=np.uint8)
    has_fee = np.fromiter(
        (_has_maker_fee(t, fee_types) for t in tickers), dtype=np.uint8, count=n
    )
    risk = [r if pd.notna(r) else "low" for r in df["resolution_risk"]]
    risk_technical = np.fromiter(
        (str(r) in {"medium", "high"} for r in risk), dtype=np.uint8, count=n
    )

    res = _engine.run_backtest(
        bar_ts, bar_bid, bar_ask, bar_plow, bar_phigh, offsets, counts,
        decision_ts, model_p, y, bar_group, trade_ok, has_fee, risk_technical,
        band_lo=p.band_lo, band_hi=p.band_hi, theta=p.theta, tick=p.tick,
        max_staleness_s=p.max_staleness_days * _DAY_S, contracts=contracts,
    )

    side_codes: np.ndarray = res["side"]
    fill_ts_f: np.ndarray = res["fill_ts"]
    out = pd.DataFrame(
        {
            "market_ticker": df["market_ticker"].to_numpy(),
            "occasion": df["occasion"].to_numpy(),
            "speaker": df["speaker"].to_numpy(),
            "format": [f if pd.notna(f) else "other" for f in df["format"]],
            "resolution_risk": risk,
            "decision_ts": df["decision_ts"].to_numpy(),
            "split": df["split"].to_numpy(),
            "y": [int(v) for v in df["y"]],
            "market_prob": res["market_prob"],
            "p_base": [float(v) for v in df["p_base"]],
            "p_topic": [float(v) for v in df["p_topic"]],
            "p_cal": [float(v) for v in df["p_cal"]],
            "traded": res["traded"].astype(bool),
            "side": [None if s < 0 else ("yes" if s == 0 else "no") for s in side_codes],
            "quote_price": res["quote_price"],
            "filled": res["filled"].astype(bool),
            "fill_ts": pd.array(
                [pd.NA if np.isnan(v) else int(v) for v in fill_ts_f], dtype="Int64"
            ),
            "fill_price": res["fill_price"],
            "fee": res["fee"],
            "contracts": res["contracts"],
            "stake": res["stake"],
            "payout": res["payout"],
            "pnl": res["pnl"],
            "decided_by_technicality": res["decided_by_technicality"].astype(bool),
        },
        columns=BACKTEST_COLUMNS,
    )
    out["decision_ts"] = out["decision_ts"].astype("Int64")
    return Backtest.validate(out)
