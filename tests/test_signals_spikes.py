"""Spike-detection mechanics: close-over-close magnitude crossings over 1-min bars."""

from __future__ import annotations

import pandas as pd
import pytest

from pmlab.signals.spikes import SpikeConfig, detect_spikes


def _bars(mids: list[float], start_ts: int = 1_800_000_000) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(mids):
        ts = start_ts + 60 * i
        rows.append({"ts": ts, "yes_bid_close": m - 0.01, "yes_ask_close": m + 0.01})
    return pd.DataFrame(rows)


def test_detect_spikes_flags_an_up_move_past_the_magnitude_threshold():
    # 12c up-move at minute 5, no further bars — nothing left to re-trigger the k=2 lookback.
    mids = [0.20] * 5 + [0.32]
    bars = _bars(mids)
    cfg = SpikeConfig(magnitude=0.08)
    events = detect_spikes("OCC", "T-A", bars, config=cfg)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["direction"] == "up"
    assert row["dmid"] == pytest.approx(0.12, abs=1e-6)
    assert row["S"] == bars["ts"].iloc[5]


def test_detect_spikes_flags_a_down_move():
    mids = [0.80] * 5 + [0.68]  # 12c down-move at minute 5
    bars = _bars(mids)
    events = detect_spikes("OCC", "T-B", bars, config=SpikeConfig(magnitude=0.08))
    assert len(events) == 1
    assert events.iloc[0]["direction"] == "down"


def test_detect_spikes_lookback_can_flag_more_than_one_bar():
    """Bidirectional, no one-trade-per-market filter (that was a trade-layer/menu concern, not a
    detection concern): a bar can still be flagged relative to a k=2 lookback even after the
    immediate k=1 move reverses, since it remains elevated vs. two bars back."""
    mids = [0.20] * 5 + [0.32] + [0.30]  # spike at minute 5, minute 6 still up vs minute 4
    bars = _bars(mids)
    events = detect_spikes("OCC", "T-F", bars, config=SpikeConfig(magnitude=0.08))
    assert len(events) == 2
    assert list(events["S"]) == [bars["ts"].iloc[5], bars["ts"].iloc[6]]


def test_detect_spikes_respects_the_price_band():
    # the move starts at mid=0.97, above the default band_hi (0.95) — must not fire
    mids = [0.97] * 5 + [0.85] + [0.85] * 3
    bars = _bars(mids)
    events = detect_spikes("OCC", "T-C", bars, config=SpikeConfig(magnitude=0.08))
    assert events.empty


def test_detect_spikes_below_threshold_is_silent():
    mids = [0.20] * 5 + [0.23] + [0.23] * 3  # 3c move, below an 8c threshold
    bars = _bars(mids)
    events = detect_spikes("OCC", "T-D", bars, config=SpikeConfig(magnitude=0.08))
    assert events.empty


def test_detect_spikes_window_bounds_the_scan():
    mids = [0.20] * 5 + [0.32] + [0.30] * 3
    bars = _bars(mids)
    # window excludes the spike bar entirely
    events = detect_spikes("OCC", "T-E", bars, config=SpikeConfig(magnitude=0.08),
                           window=(bars["ts"].iloc[0], bars["ts"].iloc[3]))
    assert events.empty
