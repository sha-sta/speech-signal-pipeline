"""Book-tick → 1-min OHLC assembly: bucket boundaries, per-side OHLC, carry-forward."""

from __future__ import annotations

import numpy as np

from pmlab.probe.bars import BAR_COLUMNS, BarBuilder, BookTick, assemble_bars, bucket_end

T0 = 1_800_000_000  # on a 60-boundary


def test_bucket_end_half_open_interval():
    assert bucket_end(T0) == T0                 # boundary tick belongs to its own bar
    assert bucket_end(T0 + 1) == T0 + 60        # (T0, T0+60]
    assert bucket_end(T0 + 59) == T0 + 60
    assert bucket_end(T0 + 60) == T0 + 60
    assert bucket_end(T0 + 61) == T0 + 120


def test_builder_ohlc_within_minute():
    b = BarBuilder("T")
    for ts, bid, ask in [(T0 + 5, 0.20, 0.24), (T0 + 30, 0.22, 0.26), (T0 + 55, 0.21, 0.25)]:
        b.add(BookTick(ts, bid, ask))
    bar = b.close_minute(T0 + 60)
    assert bar is not None
    assert bar["ts"] == T0 + 60
    assert bar["yes_bid_open"] == 0.20 and bar["yes_bid_high"] == 0.22
    assert bar["yes_bid_low"] == 0.20 and bar["yes_bid_close"] == 0.21
    assert bar["yes_ask_open"] == 0.24 and bar["yes_ask_high"] == 0.26
    assert bar["yes_ask_close"] == 0.25
    assert np.isnan(bar["volume"])  # quote channel is volume-blind (report-only field)


def test_builder_carries_forward_silent_minute():
    b = BarBuilder("T")
    b.add(BookTick(T0 + 10, 0.30, 0.34))
    b.close_minute(T0 + 60)
    flat = b.close_minute(T0 + 120)  # no ticks this minute → flat bar at standing quote
    assert flat is not None
    assert flat["yes_bid_open"] == flat["yes_bid_close"] == 0.30
    assert flat["yes_ask_open"] == flat["yes_ask_close"] == 0.34
    assert b.last_tob == (0.30, 0.34)


def test_builder_no_quote_yet_returns_none():
    assert BarBuilder("T").close_minute(T0 + 60) is None


def test_assemble_bars_batch_shape_and_carry():
    ticks = [
        BookTick(T0 + 5, 0.20, 0.24),
        BookTick(T0 + 65, 0.25, 0.29),   # minute 2
        # minute 3 silent → carried
        BookTick(T0 + 190, 0.10, 0.14),  # minute 4 (T0+240)
    ]
    df = assemble_bars(ticks, "T", T0 + 1, T0 + 200)
    assert list(df.columns) == BAR_COLUMNS
    assert df["ts"].tolist() == [T0 + 60, T0 + 120, T0 + 180, T0 + 240]
    assert df.iloc[1]["yes_bid_close"] == 0.25
    assert df.iloc[2]["yes_bid_close"] == 0.25  # silent minute carried forward
    assert df.iloc[3]["yes_bid_open"] == 0.10


def test_depth_tracking():
    b = BarBuilder("T")
    b.add(BookTick(T0 + 5, 0.20, 0.24, depth_bid=300.0, depth_ask=150.0))
    assert b.last_depth == (300.0, 150.0)


def test_next_minute_tick_does_not_destroy_open_bar():
    """Live race: the decision loop closes minute S at wall-clock S+1, and a WS delta can land in
    that gap. The late tick must open bucket S+60 without wiping bar S's accumulator."""
    b = BarBuilder("T")
    b.add(BookTick(T0 + 5, 0.20, 0.24))
    b.add(BookTick(T0 + 30, 0.26, 0.30))
    b.add(BookTick(T0 + 61, 0.50, 0.54))  # next minute's tick arrives BEFORE close(T0+60)
    bar = b.close_minute(T0 + 60)
    assert bar is not None
    assert bar["yes_bid_open"] == 0.20 and bar["yes_bid_high"] == 0.26
    assert bar["yes_bid_close"] == 0.26            # bar S intact — not a flat carry-forward
    assert b.last_tob == (0.50, 0.54)              # standing quote IS the newest tick (t_d book)
    bar2 = b.close_minute(T0 + 120)
    assert bar2 is not None and bar2["yes_bid_open"] == 0.50  # bucket S+60 also intact


def test_silent_minute_uses_previous_close_not_newer_tick():
    """A silent minute is flat at the previous bar's CLOSE; a tick from the next minute must not
    leak into it."""
    b = BarBuilder("T")
    b.add(BookTick(T0 + 10, 0.30, 0.34))
    b.close_minute(T0 + 60)
    b.add(BookTick(T0 + 121, 0.40, 0.44))          # belongs to minute 3 (bucket T0+180)
    flat = b.close_minute(T0 + 120)                # minute 2: silent
    assert flat is not None
    assert flat["yes_bid_close"] == 0.30 and flat["yes_ask_close"] == 0.34


def test_flat_requires_a_prior_emitted_bar():
    """Before any real bar completes, silent minutes yield None (no quote history to carry)."""
    b = BarBuilder("T")
    b.add(BookTick(T0 + 61, 0.50, 0.54))           # only a future-bucket tick exists
    assert b.close_minute(T0) is None
    assert b.close_minute(T0 + 60) is None
