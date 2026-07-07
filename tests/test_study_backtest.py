"""Backtest simulator (M3): fee/quote/settle math, strict fills, staleness, end-to-end trades.

The pure mechanics (the trap-prone parts — fills at trade-through only, ceil-to-cent fees, no
same-bar fills) are tested directly; ``run_backtest`` is then exercised on a small synthetic
(predictions, candles, registry) triple with the model probability driven through ``p_base``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytest import approx as pytest_approx

from pmlab.data.schemas import CANDLE_STORE_COLUMNS, Backtest
from pmlab.study.backtest import (
    BacktestParams,
    _Bars,
    _decision_bar,
    _fills,
    maker_fee,
    maker_quote,
    run_backtest,
    settle,
)

# --- pure mechanics ------------------------------------------------------------------------------


def test_maker_fee_ceils_to_cent() -> None:
    # 0.0175 * 1 * 0.4 * 0.6 = 0.0042 -> ceil to $0.01
    assert maker_fee(0.4, 1.0, has_maker_fee=True) == 0.01
    # larger size scales then ceils: 0.0175 * 100 * 0.5 * 0.5 = 0.4375 -> 0.44
    assert maker_fee(0.5, 100.0, has_maker_fee=True) == 0.44


def test_maker_fee_zero_when_no_maker_fee_or_no_contracts() -> None:
    assert maker_fee(0.4, 1.0, has_maker_fee=False) == 0.0
    assert maker_fee(0.4, 0.0, has_maker_fee=True) == 0.0


def test_maker_quote_improves_best_side() -> None:
    assert maker_quote("yes", 0.38, 0.42, 0.01) == 0.39  # improve the bid
    assert maker_quote("no", 0.38, 0.42, 0.01) == 0.41  # improve the ask


def test_maker_quote_none_on_locked_book() -> None:
    # one-tick improvement would cross a 1-tick-wide book
    assert maker_quote("yes", 0.40, 0.41, 0.01) is None
    assert maker_quote("no", 0.40, 0.41, 0.01) is None


def test_settle_yes_side() -> None:
    payout, stake, pnl = settle("yes", 0.39, y=1, fee=0.01)
    assert (payout, stake) == (1.0, 0.39)
    assert pnl == pytest_approx(0.60)
    payout, stake, pnl = settle("yes", 0.39, y=0, fee=0.01)
    assert (payout, stake, round(pnl, 2)) == (0.0, 0.39, -0.40)


def test_settle_no_side() -> None:
    # sold yes at 0.41 (= bought no at 0.59); settles No -> win
    payout, stake, pnl = settle("no", 0.41, y=0, fee=0.01)
    assert (payout, round(stake, 2)) == (1.0, 0.59)
    assert pnl == pytest_approx(0.40)
    # settles Yes -> lose the stake
    _, _, pnl_loss = settle("no", 0.41, y=1, fee=0.01)
    assert pnl_loss == pytest_approx(-0.60)


def _bars(rows: list[tuple[float, float, float, float, float]]) -> _Bars:
    """rows = (ts, bid_close, ask_close, price_low, price_high)."""
    a = np.array(rows, dtype="float64")
    return _Bars(ts=a[:, 0], bid_close=a[:, 1], ask_close=a[:, 2],
                 price_low=a[:, 3], price_high=a[:, 4])


def test_decision_bar_picks_last_before_T() -> None:
    bars = _bars([(100, 0.38, 0.42, 0.3, 0.5), (200, 0.40, 0.44, 0.3, 0.5),
                  (300, 0.5, 0.55, 0.3, 0.5)])
    assert _decision_bar(bars, 250, max_staleness_s=1e9) == 1  # last bar <= 250 is ts=200


def test_decision_bar_none_when_no_prior_bar() -> None:
    bars = _bars([(300, 0.4, 0.44, 0.3, 0.5)])
    assert _decision_bar(bars, 250, max_staleness_s=1e9) is None


def test_decision_bar_none_when_stale() -> None:
    bars = _bars([(100, 0.4, 0.44, 0.3, 0.5)])
    assert _decision_bar(bars, 100 + 500, max_staleness_s=100) is None


def test_decision_bar_none_on_crossed_or_degenerate_quote() -> None:
    crossed = _bars([(100, 0.60, 0.40, 0.3, 0.5)])  # ask < bid
    assert _decision_bar(crossed, 200, max_staleness_s=1e9) is None
    nan_quote = _bars([(100, np.nan, 0.40, 0.3, 0.5)])
    assert _decision_bar(nan_quote, 200, max_staleness_s=1e9) is None


def test_fills_buy_only_on_later_trade_through() -> None:
    # decision bar idx 0; a later bar prints down through the resting bid 0.39
    bars = _bars([(100, 0.38, 0.42, 0.30, 0.42), (200, 0.35, 0.40, 0.34, 0.40)])
    assert _fills(bars, "yes", 0.39, after_idx=0) == 200.0


def test_fills_ignores_same_and_earlier_bars() -> None:
    # the decision bar itself printed at 0.30 (< 0.39) but that must NOT fill (trap #4);
    # the only later bar never trades down to 0.39
    bars = _bars([(100, 0.38, 0.42, 0.30, 0.42), (200, 0.41, 0.45, 0.41, 0.50)])
    assert _fills(bars, "yes", 0.39, after_idx=0) is None


def test_fills_sell_on_later_high() -> None:
    bars = _bars([(100, 0.38, 0.42, 0.34, 0.42), (200, 0.45, 0.50, 0.45, 0.52)])
    assert _fills(bars, "no", 0.41, after_idx=0) == 200.0


# --- end-to-end ----------------------------------------------------------------------------------


def _features(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a PREDICTION_COLUMNS frame directly (run_backtest consumes predictions, not raw
    features — the model producing them is the caller's research, not the harness's)."""
    n = len(rows)
    recs = [
        {
            "market_ticker": r["ticker"],
            "decision_ts": r["decision_ts"],
            # chronological split mirroring time_split(train_frac=0.6)
            "split": "train" if i < max(1, int(round(n * 0.6))) else "test",
            "y": int(r["y"]),
            "p_base": r["p"],
            "p_topic": r["p"],
            "p_cal": r["p"],
        }
        for i, r in enumerate(rows)
    ]
    return pd.DataFrame(recs)


def _registry(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_ticker": r["ticker"],
                "occasion": r.get("occasion", "occ"),
                "speaker": "spk",
                "format": "speech",
                "resolution_risk": r.get("risk", "low"),
            }
            for r in rows
        ]
    )


def _candles(bars: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    recs = []
    for ticker, ts, bid_c, ask_c, plow, phigh in bars:
        mid = (plow + phigh) / 2.0
        recs.append(
            {
                "ticker": ticker, "ts": int(ts), "period_minutes": 1440,
                "yes_bid_open": bid_c, "yes_bid_high": bid_c, "yes_bid_low": bid_c,
                "yes_bid_close": bid_c,
                "yes_ask_open": ask_c, "yes_ask_high": ask_c, "yes_ask_low": ask_c,
                "yes_ask_close": ask_c,
                "price_open": mid, "price_high": phigh, "price_low": plow,
                "price_close": mid, "price_mean": mid,
                "volume": 100.0, "open_interest": 50.0,
            }
        )
    return pd.DataFrame(recs, columns=CANDLE_STORE_COLUMNS)


_PARAMS = BacktestParams(model_col="p_base", trade_split=None, max_staleness_days=30.0)


def _run(features: pd.DataFrame, candles: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    out = run_backtest(features, candles, registry, params=_PARAMS)
    Backtest.validate(out)
    return out


def test_buy_yes_fills_and_wins() -> None:
    feats = _features([{"ticker": "A", "y": 1, "decision_ts": 1000, "p": 0.9}])
    # decision bar at ts=900 (mid 0.40, in band); later bar trades down through 0.39
    cands = _candles([("A", 900, 0.38, 0.42, 0.40, 0.42), ("A", 2000, 0.35, 0.40, 0.34, 0.41)])
    row = _run(feats, cands, _registry([{"ticker": "A"}])).iloc[0]
    assert row["traded"] and row["filled"] and row["side"] == "yes"
    assert row["quote_price"] == pytest_approx(0.39)
    assert row["fee"] == 0.01 and row["pnl"] == pytest_approx(0.60)  # 1 - 0.39 - 0.01


def test_edge_but_no_trade_through_is_unfilled() -> None:
    feats = _features([{"ticker": "A", "y": 1, "decision_ts": 1000, "p": 0.9}])
    # later bar never trades down to the 0.39 resting bid
    cands = _candles([("A", 900, 0.38, 0.42, 0.40, 0.42), ("A", 2000, 0.41, 0.45, 0.41, 0.50)])
    row = _run(feats, cands, _registry([{"ticker": "A"}])).iloc[0]
    assert row["traded"] and not row["filled"]
    assert row["pnl"] == 0.0 and row["stake"] == 0.0


def test_out_of_band_not_traded() -> None:
    feats = _features([{"ticker": "A", "y": 1, "decision_ts": 1000, "p": 0.95}])
    cands = _candles([("A", 900, 0.68, 0.72, 0.70, 0.72), ("A", 2000, 0.60, 0.66, 0.55, 0.66)])
    row = _run(feats, cands, _registry([{"ticker": "A"}])).iloc[0]
    assert row["market_prob"] == pytest_approx(0.70) and not row["traded"]


def test_below_theta_not_traded() -> None:
    feats = _features([{"ticker": "A", "y": 1, "decision_ts": 1000, "p": 0.43}])
    cands = _candles([("A", 900, 0.38, 0.42, 0.40, 0.42), ("A", 2000, 0.30, 0.35, 0.30, 0.42)])
    row = _run(feats, cands, _registry([{"ticker": "A"}])).iloc[0]
    assert not row["traded"]  # |0.43 - 0.40| = 0.03 <= theta 0.05


def test_stale_price_not_traded() -> None:
    feats = _features([{"ticker": "A", "y": 1, "decision_ts": 1_000_000, "p": 0.9}])
    # only bar is ~40 days before the decision (params staleness = 30 days) -> untradable
    cands = _candles([("A", 1_000_000 - 40 * 86_400, 0.38, 0.42, 0.40, 0.42)])
    row = _run(feats, cands, _registry([{"ticker": "A"}])).iloc[0]
    assert np.isnan(row["market_prob"]) and not row["traded"]


def test_sell_yes_fills_and_wins() -> None:
    feats = _features([{"ticker": "A", "y": 0, "decision_ts": 1000, "p": 0.1}])
    cands = _candles([("A", 900, 0.38, 0.42, 0.38, 0.42), ("A", 2000, 0.45, 0.55, 0.45, 0.55)])
    row = _run(feats, cands, _registry([{"ticker": "A"}])).iloc[0]
    assert row["side"] == "no" and row["filled"]
    assert row["quote_price"] == pytest_approx(0.41) and row["pnl"] == pytest_approx(0.40)


def test_decided_by_technicality_flag() -> None:
    feats = _features([{"ticker": "A", "y": 1, "decision_ts": 1000, "p": 0.9}])
    cands = _candles([("A", 900, 0.38, 0.42, 0.40, 0.42), ("A", 2000, 0.35, 0.40, 0.34, 0.41)])
    row = _run(feats, cands, _registry([{"ticker": "A", "risk": "medium"}])).iloc[0]
    assert row["filled"] and row["decided_by_technicality"]


def test_trade_split_gating() -> None:
    # many markets across time; trade only the out-of-sample (test) split
    rows = [{"ticker": f"M{i}", "y": i % 2, "decision_ts": 1000 + i * 1000, "p": 0.9}
            for i in range(20)]
    feats = _features(rows)
    cands = pd.concat(
        [_candles([(f"M{i}", 1000 + i * 1000 - 100, 0.38, 0.42, 0.40, 0.42),
                   (f"M{i}", 1000 + i * 1000 + 5000, 0.30, 0.36, 0.30, 0.42)])
         for i in range(20)],
        ignore_index=True,
    )
    reg = _registry(rows)
    out = run_backtest(feats, cands, reg, params=BacktestParams(model_col="p_base",
                                                                trade_split="test"))
    Backtest.validate(out)
    assert out["traded"].any()
    assert (out.loc[out["traded"], "split"] == "test").all()
