#!/usr/bin/env python3
"""Honest Python-vs-C++ benchmark for the backtest engine, on identical seeded data.

    uv run python scripts/bench_engine.py [--markets 15000] [--bars 400] [--repeat 5]
                                          [--corpus-out corpus.bin]

Measures, on the same synthetic corpus (tests/parity_corpus.py, seed 42):

* ``run_backtest``          — the pure-Python reference, end to end.
* ``run_backtest_native``   — pandas prep + C++ hot loop, end to end (what a caller sees).
* ``prepare + hot loop``    — the prep and the C++ call timed separately, because the honest
  headline is that pandas preparation dominates the native path; the C++ loop itself is the
  microseconds part, which is why sweeps (prep once, run many) are the real win.
* tick replay              — assemble_bars, Python vs C++, ticks/sec.

``--corpus-out`` additionally writes the binary corpus for ``engine_cli`` so the pure-C++ number
can be reproduced (and profiled: Instruments/perf on ``engine_cli corpus.bin /dev/null``) with
no Python in the process at all.
"""

from __future__ import annotations

import argparse
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from parity_corpus import FEE_TYPES, gen_corpus  # noqa: E402
from pmlab.probe.bars import BookTick, assemble_bars  # noqa: E402
from pmlab.study.backtest import BacktestParams, _meta, run_backtest  # noqa: E402
from pmlab.study.engine import (  # noqa: E402
    assemble_bars_native,
    engine_available,
    prepare_native_inputs,
    run_backtest_native,
)

_DAY_S = 86_400.0


def timed(fn, repeat: int) -> tuple[float, object]:
    """Median wall seconds over `repeat` runs (after one warmup), plus the last result."""
    fn()
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def write_corpus_bin(path: Path, arrays: dict[str, np.ndarray], p: BacktestParams,
                     contracts: float) -> None:
    """Write the PMCORP01 binary corpus consumed by cpp engine_cli (see corpus_io.hpp)."""
    with open(path, "wb") as f:
        f.write(b"PMCORP01")
        f.write(struct.pack("<QQQ", len(arrays["bar_ts"]), len(arrays["offsets"]),
                            len(arrays["decision_ts"])))
        f.write(struct.pack("<6d", p.band_lo, p.band_hi, p.theta, p.tick,
                            p.max_staleness_days * _DAY_S, contracts))
        for key in ("bar_ts", "bar_bid_close", "bar_ask_close", "bar_price_low",
                    "bar_price_high", "offsets", "counts", "decision_ts", "model_p", "y",
                    "bar_group", "trade_ok", "has_maker_fee", "risk_technical"):
            arrays[key].tofile(f)


def bench_backtest(n_markets: int, bars: int, repeat: int, corpus_out: Path | None) -> None:
    preds, candles, registry = gen_corpus(42, n_markets=n_markets, max_bars=bars)
    params = BacktestParams(trade_split=None, max_staleness_days=30.0)

    t_py, out_py = timed(
        lambda: run_backtest(preds, candles, registry, params=params, fee_types=FEE_TYPES),
        repeat,
    )
    t_native, out_native = timed(
        lambda: run_backtest_native(preds, candles, registry, params=params,
                                    fee_types=FEE_TYPES),
        repeat,
    )
    df = preds.merge(_meta(registry), on="market_ticker", how="left")
    t_prep, prepped = timed(lambda: prepare_native_inputs(df, candles, params, FEE_TYPES),
                            repeat)
    arrays, _ = prepped  # type: ignore[misc]

    import pmlab._engine as _engine

    def hot_loop() -> object:
        return _engine.run_backtest(
            **arrays, band_lo=params.band_lo, band_hi=params.band_hi, theta=params.theta,
            tick=params.tick, max_staleness_s=params.max_staleness_days * _DAY_S, contracts=1.0,
        )

    t_hot, _ = timed(hot_loop, max(repeat, 20))

    n = len(preds)
    n_bars = len(candles)
    print(f"\nbacktest: {n:,} markets, {n_bars:,} bars (median of {repeat})")
    print(f"  python run_backtest        {t_py * 1e3:10.1f} ms   {n / t_py:12,.0f} markets/s")
    print(f"  native end-to-end          {t_native * 1e3:10.1f} ms   "
          f"{n / t_native:12,.0f} markets/s   ({t_py / t_native:6.1f}x)")
    print(f"    pandas prep only         {t_prep * 1e3:10.1f} ms")
    print(f"    C++ hot loop only        {t_hot * 1e3:10.1f} ms   "
          f"{n / t_hot:12,.0f} markets/s   ({t_py / t_hot:6.1f}x)")
    print(f"  filled (both, must match): {int(out_py['filled'].sum())} / "  # type: ignore[index]
          f"{int(out_native['filled'].sum())}")  # type: ignore[index]

    if corpus_out is not None:
        write_corpus_bin(corpus_out, arrays, params, 1.0)
        print(f"  corpus for engine_cli -> {corpus_out}")

    # Sweep story: parse/prep once, evaluate a parameter grid against the SAME arrays.
    grid = [(theta, lo) for theta in np.linspace(0.01, 0.20, 25)
            for lo in np.linspace(0.05, 0.45, 8)]
    t0 = time.perf_counter()
    for theta, lo in grid:
        _engine.run_backtest(**arrays, band_lo=float(lo), band_hi=0.60, theta=float(theta),
                             tick=params.tick,
                             max_staleness_s=params.max_staleness_days * _DAY_S, contracts=1.0)
    t_sweep = time.perf_counter() - t0
    est_py = t_py * len(grid)
    print(f"  {len(grid)}-point sweep:  C++ {t_sweep:.2f}s vs Python est {est_py:,.0f}s "
          f"({est_py / t_sweep:,.0f}x)")


def bench_ticks(repeat: int) -> None:
    rng = np.random.default_rng(7)
    n = 1_000_000
    ts = 1_700_000_000 + np.sort(rng.integers(0, 6 * 3600, size=n))
    bid = np.round(rng.uniform(0.0, 0.99, n), 2)
    ask = np.minimum(bid + 0.01 + np.round(rng.uniform(0.0, 0.05, n), 2), 1.0)
    ticks = [BookTick(ts=int(t), yes_bid=float(b), yes_ask=float(a))
             for t, b, a in zip(ts, bid, ask, strict=True)]
    start, end = int(ts.min()), int(ts.max())

    t_py, bars_py = timed(lambda: assemble_bars(ticks, "T", start, end), max(1, repeat // 2))
    t_cpp, bars_cpp = timed(lambda: assemble_bars_native(ticks, "T", start, end), repeat)

    print(f"\ntick replay: {n:,} ticks -> {len(bars_py)} bars (median)")  # type: ignore[arg-type]
    print(f"  python assemble_bars       {t_py * 1e3:10.1f} ms   {n / t_py:12,.0f} ticks/s")
    print(f"  native (incl. conversion)  {t_cpp * 1e3:10.1f} ms   "
          f"{n / t_cpp:12,.0f} ticks/s   ({t_py / t_cpp:6.1f}x)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=15_000)
    ap.add_argument("--bars", type=int, default=400)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--corpus-out", type=Path, default=None)
    args = ap.parse_args()

    if not engine_available():
        raise SystemExit("pmlab._engine not built; see src/pmlab/study/engine.py docstring")
    bench_backtest(args.markets, args.bars, args.repeat, args.corpus_out)
    bench_ticks(args.repeat)


if __name__ == "__main__":
    main()
