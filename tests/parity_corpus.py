"""Seeded synthetic (predictions, candles, registry) corpora for the C++ engine parity gate.

Deliberately adversarial: every rejection branch and boundary in run_backtest appears with real
frequency — markets with no bars, NaN/crossed/locked quotes, stale and boundary-exact decision
times, NaN decision_ts, NaN trade prints, duplicate registry rows (dedup keep-first), markets
missing from the registry entirely (NaN meta -> "other"/"low" defaults), fee series known, unknown,
and fee-free. The frames are what the real pipeline would hand run_backtest; only the values are
synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmlab.data.schemas import CANDLE_STORE_COLUMNS, FORMAT_VALUES

# Series universe: two charge the maker fee, one is explicitly fee-free, one has a fee type the
# simulator doesn't recognize, and UNKNOWN* are absent from the mapping (charged by default).
FEE_TYPES = {
    "KXMENTION": "quadratic_with_maker_fees",
    "KXSPEECH": "quadratic_with_maker_fees",
    "NOFEE": "quadratic",
    "WEIRD": "",
}
_SERIES = ["KXMENTION", "KXSPEECH", "NOFEE", "WEIRD", "UNKNOWN1", "UNKNOWN2"]

_BASE_TS = 1_700_000_000
_DAY = 86_400


def gen_corpus(
    seed: int, n_markets: int = 400, max_bars: int = 40
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (predictions, candles, registry) for one seeded corpus."""
    rng = np.random.default_rng(seed)

    pred_rows: list[dict[str, object]] = []
    candle_rows: list[dict[str, object]] = []
    reg_rows: list[dict[str, object]] = []

    for i in range(n_markets):
        series = _SERIES[int(rng.integers(len(_SERIES)))]
        ticker = f"{series}-EV{i % 97}-M{i}"
        t0 = _BASE_TS + int(rng.integers(0, 365)) * _DAY

        n_bars = 0 if rng.random() < 0.06 else int(rng.integers(1, max_bars + 1))
        bar_ts = t0 + np.arange(n_bars) * _DAY
        if n_bars > 2 and rng.random() < 0.15:  # duplicate-ts bars (mixed granularity stores)
            bar_ts[int(rng.integers(1, n_bars))] = bar_ts[int(rng.integers(1, n_bars))]
        for ts in bar_ts:
            bid_c = int(rng.integers(0, 100))
            spread = int(rng.integers(1, 7))
            r = rng.random()
            bid = bid_c / 100.0
            ask = min(bid_c + spread, 100) / 100.0
            if r < 0.05:
                ask = max(bid_c - spread, 0) / 100.0  # crossed quote -> untradable
            elif r < 0.10:
                bid = np.nan
            elif r < 0.14:
                ask = np.nan
            elif r < 0.20:
                ask = min(bid_c + 1, 100) / 100.0  # locked-ish one-tick book
            lo_c = max(bid_c - int(rng.integers(0, 8)), 0)
            hi_c = min(bid_c + spread + int(rng.integers(0, 8)), 100)
            plow: float = lo_c / 100.0
            phigh: float = hi_c / 100.0
            if rng.random() < 0.08:
                plow = np.nan  # thin bar with no prints
            if rng.random() < 0.08:
                phigh = np.nan
            mid_print = min(max((lo_c + hi_c) / 200.0, 0.0), 1.0)
            candle_rows.append({
                "ticker": ticker, "ts": int(ts), "period_minutes": 1440,
                "yes_bid_open": bid, "yes_bid_high": bid, "yes_bid_low": bid,
                "yes_bid_close": bid,
                "yes_ask_open": ask, "yes_ask_high": ask, "yes_ask_low": ask,
                "yes_ask_close": ask,
                "price_open": mid_print, "price_high": phigh, "price_low": plow,
                "price_close": mid_print, "price_mean": mid_print,
                "volume": float(rng.integers(0, 500)), "open_interest": float(rng.integers(0, 200)),
            })

        # Decision time: mostly a sane mid-history point, with every boundary represented.
        r = rng.random()
        decision_ts: object
        if r < 0.05 or n_bars == 0:
            decision_ts = pd.NA if r < 0.025 else t0 + int(rng.integers(0, 40)) * _DAY
        elif r < 0.10:
            decision_ts = int(bar_ts[0]) - _DAY  # before any bar
        elif r < 0.15:
            decision_ts = int(bar_ts[-1]) + 400 * _DAY  # far past the last bar (stale)
        elif r < 0.22:
            decision_ts = int(bar_ts[int(rng.integers(n_bars))])  # exactly on a bar ts
        elif r < 0.27:
            # exactly at the staleness limit for the default 7-day params (> rejects, == passes)
            decision_ts = int(bar_ts[int(rng.integers(n_bars))]) + 7 * _DAY
        else:
            decision_ts = int(bar_ts[int(rng.integers(n_bars))]) + int(rng.integers(0, 3 * _DAY))

        probs = rng.random(3)
        pred_rows.append({
            "market_ticker": ticker,
            "decision_ts": decision_ts,
            "split": "test" if rng.random() < 0.5 else "train",
            "y": int(rng.integers(0, 2)),
            "p_base": float(probs[0]), "p_topic": float(probs[1]), "p_cal": float(probs[2]),
        })

        if rng.random() < 0.08:
            continue  # market missing from the registry: NaN meta after the left join
        n_reg = 2 if rng.random() < 0.10 else 1  # duplicates exercise dedup keep-first
        for dup in range(n_reg):
            reg_rows.append({
                "market_ticker": ticker,
                "occasion": None if rng.random() < 0.1 else f"occ{i % 37}-{dup}",
                "speaker": None if rng.random() < 0.1 else f"spk{i % 11}",
                "format": None if rng.random() < 0.1
                else FORMAT_VALUES[int(rng.integers(len(FORMAT_VALUES)))],
                "resolution_risk": None if rng.random() < 0.15
                else ["low", "medium", "high"][int(rng.integers(3))],
            })

    predictions = pd.DataFrame(pred_rows)
    predictions["decision_ts"] = predictions["decision_ts"].astype("Int64")
    candles = pd.DataFrame(candle_rows, columns=CANDLE_STORE_COLUMNS)
    registry = pd.DataFrame(reg_rows)
    return predictions, candles, registry
