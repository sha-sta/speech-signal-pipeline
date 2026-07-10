"""Bars parity gate: C++ assemble_bars must be byte-identical to pmlab.probe.bars.

Randomized tick streams cover the semantics that bite: NaN sides, out-of-order arrival, equal-ts
ticks (stable order decides the close), boundary timestamps exactly on a minute edge, silent-gap
carry-forward, leading silence before any real bar, and start_ts inside the log (orphan buckets
that never emit). When recorded tapes exist under data/, they are replayed too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.probe.bars import BookTick, assemble_bars

pytest.importorskip("pmlab._engine", reason="native engine not built (see cpp/)")

from pmlab.study.engine import assemble_bars_native  # noqa: E402

pytestmark = pytest.mark.engine


def gen_ticks(seed: int, n: int) -> tuple[list[BookTick], int, int]:
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000 + int(rng.integers(0, 1000)) * 60
    ts = t0 + np.sort(rng.integers(0, 4 * 3600, size=n))
    # equal-ts runs and boundary hits
    ts[rng.random(n) < 0.10] = t0 + 60 * rng.integers(0, 240)
    bid = np.round(rng.uniform(0.0, 1.0, n), 2)
    ask = np.minimum(bid + np.round(rng.uniform(0.01, 0.06, n), 2), 1.0)
    bid[rng.random(n) < 0.07] = np.nan
    ask[rng.random(n) < 0.07] = np.nan
    order = np.arange(n)
    if rng.random() < 0.5:  # out-of-order arrival; assemble_bars sorts (stably) itself
        swap = rng.integers(0, n, size=n // 10)
        order[swap], order[swap[::-1]] = order[swap[::-1]].copy(), order[swap].copy()
    ticks = [BookTick(ts=int(ts[i]), yes_bid=float(bid[i]), yes_ask=float(ask[i])) for i in order]
    start = int(ts.min()) if rng.random() < 0.7 else int(ts.min()) + 300  # sometimes orphan start
    end = int(ts.max()) + (0 if rng.random() < 0.5 else 600)  # sometimes trailing silence
    return ticks, start, end


def assert_frames_bit_identical(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for col in a.columns:
        sa, sb = a[col], b[col]
        assert str(sa.dtype) == str(sb.dtype), f"{col}: dtype {sa.dtype} vs {sb.dtype}"
        if sa.dtype == np.float64:
            va, vb = sa.to_numpy(), sb.to_numpy()
            same = (va.view(np.uint64) == vb.view(np.uint64)) | (np.isnan(va) & np.isnan(vb))
            assert same.all(), f"{col}: first mismatch at {np.flatnonzero(~same)[0]}"
        else:
            pd.testing.assert_series_equal(sa, sb, check_exact=True, obj=col)


@pytest.mark.parametrize("seed", range(12))
def test_bars_parity_synthetic(seed: int) -> None:
    ticks, start, end = gen_ticks(seed, n=3000)
    py_bars = assemble_bars(ticks, "TKR", start, end)
    native_bars = assemble_bars_native(ticks, "TKR", start, end)
    assert_frames_bit_identical(py_bars, native_bars)
    assert len(py_bars) > 10


def test_bars_parity_empty() -> None:
    py_bars = assemble_bars([], "TKR", 0, 600)
    native_bars = assemble_bars_native([], "TKR", 0, 600)
    assert_frames_bit_identical(py_bars, native_bars)


def test_bars_parity_real_tapes_if_present() -> None:
    from pmlab.config import get_settings

    tapes = sorted(get_settings().data_dir.glob("**/book_ticks.parquet"))
    if not tapes:
        pytest.skip("no recorded tapes under data/; synthetic bars parity still enforced")
    for tape in tapes[:5]:
        df = pd.read_parquet(tape)
        ticker = str(df["ticker"].value_counts().idxmax())
        one = df[df["ticker"] == ticker].sort_values("recv_utc")
        ticks = [
            BookTick(ts=int(r.recv_utc), yes_bid=float(r.yes_bid), yes_ask=float(r.yes_ask))
            for r in one.itertuples()
        ]
        start, end = int(one["recv_utc"].min()), int(one["recv_utc"].max())
        assert_frames_bit_identical(
            assemble_bars(ticks, ticker, start, end),
            assemble_bars_native(ticks, ticker, start, end),
        )
