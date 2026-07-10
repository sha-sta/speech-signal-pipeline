#!/usr/bin/env python3
"""Golden-file generator for the C++ engine's trade-mechanics parity suite (stdlib only).

Emits, into the directory given as argv[1]:

* ``pyround2.txt`` — ``<input_hex> <expected_hex>`` per line, where expected is Python's
  ``round(x, 2)``. Cases cover the realistic price surface (cent grid, cent +/- tick sums),
  the adversarial one (exact dyadic ties like 0.125 where round-half-even bites), and broad
  random fuzz including huge/tiny magnitudes and signed zeros.
* ``mechanics.txt`` — end-to-end cases for ``maker_quote`` / ``maker_fee`` / ``settle`` computed
  by the same formulas as ``src/pmlab/study/backtest.py`` (kept dependency-free here on purpose:
  the formulas are three lines each and the REAL cross-check against the installed package
  happens in the pytest parity gate; this file pins the C++ functions during pure-C++ CI runs).

Deterministic: fixed seed, no timestamps. The C++ test (tests/test_golden.cpp) replays every
line and compares bit patterns, so any libc/printf divergence from CPython rounding fails loud.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

SEED = 20260710


def hx(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x.hex()


def pyround_cases() -> list[float]:
    rng = random.Random(SEED)
    xs: list[float] = []
    # Cent grid and one-tick arithmetic on it: the entire realistic input surface of
    # maker_quote (Kalshi prices are integer cents / 100).
    for i in range(-500, 501):
        c = i / 100.0
        xs += [c, c + 0.01, c - 0.01]
    # Exact dyadic rationals: the only doubles whose 2-decimal rounding is a TRUE halfway tie,
    # so they exercise ties-to-even (0.125 -> 0.12, 0.375 -> 0.38, ...).
    for n in range(1, 13):
        d = 2**n
        xs += [k / d for k in range(-2 * d, 2 * d + 1)]
    # Near-tie decimals (not exact ties in binary, must round by the true binary value).
    for i in range(-2000, 2001):
        xs.append(i / 1000.0 + 0.0005)
    # Random fuzz: realistic band, then wide magnitudes.
    xs += [rng.uniform(-1.5, 1.5) for _ in range(100_000)]
    xs += [rng.uniform(-1e6, 1e6) for _ in range(10_000)]
    xs += [math.ldexp(rng.uniform(1.0, 2.0), rng.randint(-60, 60)) for _ in range(10_000)]
    # Specials.
    xs += [0.0, -0.0, 5e-324, -5e-324, 1e308, -1e308, float("nan"), float("inf"), float("-inf")]
    return xs


# The reference formulas, transcribed from src/pmlab/study/backtest.py (kept in lockstep by the
# pytest parity gate, which compares the full engine against the real package).
def maker_fee(price: float, contracts: float, has_maker_fee: bool, rate: float = 0.0175) -> float:
    if not has_maker_fee or contracts <= 0:
        return 0.0
    raw = rate * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


def maker_quote(side: str, yes_bid: float, yes_ask: float, tick: float) -> float | None:
    if side == "yes":
        q = round(yes_bid + tick, 2)
        return q if q < yes_ask else None
    q = round(yes_ask - tick, 2)
    return q if q > yes_bid else None


def settle(side: str, fill_price: float, y: int, fee: float) -> tuple[float, float, float]:
    if side == "yes":
        stake = fill_price
        payout = 1.0 if y == 1 else 0.0
    else:
        stake = 1.0 - fill_price
        payout = 1.0 if y == 0 else 0.0
    return payout, stake, payout - stake - fee


def write_pyround(path: Path) -> int:
    lines = [f"{hx(x)} {hx(round(x, 2))}" for x in pyround_cases()]
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def write_mechanics(path: Path) -> int:
    rng = random.Random(SEED + 1)
    out: list[str] = []
    tick = 0.01
    nan = float("nan")

    # Quotes: full cent-pair grid, both sides, plus NaN quotes.
    bids = [i / 100.0 for i in range(0, 100)]
    for bid in bids:
        for ask in bids:
            for side in ("yes", "no"):
                q = maker_quote(side, bid, ask, tick)
                out.append(f"quote {side} {hx(bid)} {hx(ask)} {hx(tick)} "
                           f"{'none' if q is None else hx(q)}")
    for side in ("yes", "no"):
        out.append(f"quote {side} {hx(nan)} {hx(0.5)} {hx(tick)} "
                   f"{'none' if maker_quote(side, nan, 0.5, tick) is None else 'X'}")
        out.append(f"quote {side} {hx(0.5)} {hx(nan)} {hx(tick)} "
                   f"{'none' if maker_quote(side, 0.5, nan, tick) is None else 'X'}")

    # Fees: cent-grid prices x contract sizes x fee flag, plus random contracts.
    sizes = [0.0, 1.0, 2.0, 10.0, 100.0, 250.5, -1.0]
    for i in range(0, 101):
        p = i / 100.0
        for c in sizes:
            for has in (True, False):
                out.append(f"fee {hx(p)} {hx(c)} {int(has)} {hx(maker_fee(p, c, has))}")
    for _ in range(5_000):
        p = rng.uniform(0.0, 1.0)
        c = rng.uniform(0.0, 1000.0)
        out.append(f"fee {hx(p)} {hx(c)} 1 {hx(maker_fee(p, c, True))}")

    # Settlement: cent-grid fills x side x outcome x a few fees.
    for i in range(0, 101):
        f = i / 100.0
        for side in ("yes", "no"):
            for y in (0, 1):
                for fee in (0.0, 0.01, 0.44):
                    payout, stake, pnl = settle(side, f, y, fee)
                    out.append(f"settle {side} {hx(f)} {y} {hx(fee)} "
                               f"{hx(payout)} {hx(stake)} {hx(pnl)}")

    path.write_text("\n".join(out) + "\n")
    return len(out)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: gen_golden_mechanics.py <output_dir>")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    n1 = write_pyround(out_dir / "pyround2.txt")
    n2 = write_mechanics(out_dir / "mechanics.txt")
    print(f"golden: {n1} pyround2 cases, {n2} mechanics cases -> {out_dir}")


if __name__ == "__main__":
    main()
