"""Price-spike detection over 1-minute market bars.

Pure close-over-close magnitude crossings: at bar-close ``S``, the largest move
``mid_S − mid_{S−k}`` over a small lookback (default k ∈ {1, 2}) that starts inside a configured
price band. Detects both up- and down-spikes and returns one row per event — no execution, no
trade filters, no P&L. Thresholds are parameters (:class:`SpikeConfig`), not frozen constants, so
a caller can sweep or tune them for their own signal.

Pure function of a bars frame — no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EVENT_COLUMNS = ["occasion_id", "market_ticker", "S", "direction", "dmid", "mid_S"]


@dataclass(frozen=True)
class SpikeConfig:
    """Detection parameters. All are caller-supplied — there is no pinned default menu."""

    magnitude: float = 0.05          # min |mid_S − mid_{S−k}| to count as a spike
    band_lo: float = 0.05            # the pre-move mid (mid_{S−k}) must sit >= this …
    band_hi: float = 0.95            # … and <= this
    level_cap: float = 0.98          # the post-move mid (mid_S) must sit <= this
    lookback_bars: tuple[int, ...] = field(default=(1, 2))


def detect_spikes(
    occasion_id: str,
    market_ticker: str,
    bars: pd.DataFrame,
    *,
    config: SpikeConfig | None = None,
    window: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Detect close-over-close price spikes in one market's 1-min bars (``EVENT_COLUMNS``).

    ``bars`` needs ``ts``, ``yes_bid_close``, ``yes_ask_close`` (any span; windowed here).
    ``window`` (unix-s ``[lo, hi]``, inclusive) optionally bounds the scan to an in-play span. At
    each bar the largest-magnitude move over ``config.lookback_bars`` is kept (ties favor the
    smaller ``k``); it is emitted as an event iff its magnitude clears ``config.magnitude``, its
    starting mid sits in ``[band_lo, band_hi]``, and its ending mid sits at or below
    ``level_cap``."""
    cfg = config or SpikeConfig()
    b = bars.sort_values("ts").reset_index(drop=True)
    if window is not None:
        b = b[(b["ts"] >= window[0]) & (b["ts"] <= window[1])].reset_index(drop=True)
    min_history = max(cfg.lookback_bars) if cfg.lookback_bars else 0
    if len(b) < min_history + 1:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    mid = ((b["yes_bid_close"] + b["yes_ask_close"]) / 2.0).to_numpy("float64")
    ts = b["ts"].to_numpy("int64")

    rows: list[dict[str, object]] = []
    for i in range(min_history, len(b)):
        best = 0.0
        for k in cfg.lookback_bars:
            d = float(mid[i] - mid[i - k])
            if abs(d) > abs(best) and cfg.band_lo <= mid[i - k] <= cfg.band_hi:
                best = d
        if abs(best) < cfg.magnitude or mid[i] > cfg.level_cap:
            continue
        rows.append({
            "occasion_id": occasion_id, "market_ticker": market_ticker, "S": int(ts[i]),
            "direction": "up" if best > 0 else "down", "dmid": best, "mid_S": float(mid[i]),
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
