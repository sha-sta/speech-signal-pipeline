"""Book-tick → 1-min OHLC bar assembly.

Downstream spike detection (``signals/spikes.py``) consumes 1-min candles with **period-END**
timestamps: a bar with ``ts = S`` covers the half-open minute ``(S−60, S]`` and its ``*_open`` is
the first quote in that minute, ``*_close`` the last (matching Kalshi's own candlestick semantics,
`venues/kalshi.py`). This module turns a stream of top-of-book quotes into exactly that frame.

Empty minutes carry forward the previous emitted bar's CLOSE (a quote-based candle with no update
is flat at the resting book — and using the newest standing tick instead would leak a post-bar
quote into the bar; see :class:`BarBuilder`). ``volume`` is quote-channel-blind and left NaN.

Pure functions of the tick log — no network, no clock. Unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIOD_S = 60

# Bar frame a downstream consumer reads (`ticker`/`ts` + yes OHLC + volume). Kept
# column-compatible with `venues.kalshi.CANDLE_COLUMNS`' yes-side subset.
BAR_COLUMNS = [
    "ticker", "ts",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "volume",
]


@dataclass(frozen=True)
class BookTick:
    """One top-of-book observation. Prices are YES dollars in [0, 1]; a side is NaN if that side of
    the book is empty. ``depth_*`` are contracts available at the touch; ``*2`` fields are the next
    level past the touch (protocol §2.3: touch + one level deeper; report-only capacity)."""

    ts: int              # unix seconds the quote was observed (book-log time)
    yes_bid: float
    yes_ask: float
    depth_bid: float = float("nan")
    depth_ask: float = float("nan")
    yes_bid2: float = float("nan")
    yes_ask2: float = float("nan")
    depth_bid2: float = float("nan")
    depth_ask2: float = float("nan")


def bucket_end(ts: int, *, period_s: int = PERIOD_S) -> int:
    """Period-END of the bucket a tick at ``ts`` belongs to: maps ``(S−period, S]`` → ``S``.

    A tick exactly on a boundary belongs to that boundary's bar (``ts=S`` → ``S``)."""
    return ((int(ts) - 1) // period_s + 1) * period_s


@dataclass
class _Agg:
    """Streaming OHLC accumulator for one minute of one side (bid or ask)."""

    open: float
    high: float
    low: float
    close: float

    @classmethod
    def start(cls, x: float) -> _Agg:
        return cls(open=x, high=x, low=x, close=x)

    def push(self, x: float) -> None:
        if not np.isfinite(x):
            return
        if not np.isfinite(self.open):
            self.open = self.high = self.low = self.close = x
            return
        self.high = max(self.high, x)
        self.low = min(self.low, x)
        self.close = x


class BarBuilder:
    """Accumulates one market's top-of-book ticks and emits completed 1-min bars.

    Usage (live loop): ``add(tick)`` as quotes arrive, then ``close_minute(S)`` just after the clock
    passes ``S`` to get the bar for ``(S−60, S]``.

    Two live realities are handled explicitly:

    * Ticks for minute ``S+60`` can arrive *before* ``close_minute(S)`` runs — the decision loop
      closes ``S`` at wall-clock ``S+1``, and WS deltas land in that gap (most likely mid-spike).
      Accumulators are therefore kept **per bucket**: closing ``S`` cannot disturb ``S+60``.
    * A minute with no ticks carries forward the **previous emitted bar's close** — never the newest
      standing quote, which may already belong to a later minute (that would put a post-``S`` quote
      inside bar ``S``). Before any real bar has been emitted, silent minutes yield ``None``.

    ``last_tob``/``last_depth`` are the *standing* quote (latest tick seen) — the decision-instant
    book the synthetic S+1 open is built from; they deliberately DO include post-``S`` ticks."""

    def __init__(self, ticker: str, *, period_s: int = PERIOD_S) -> None:
        self.ticker = ticker
        self.period_s = period_s
        self._aggs: dict[int, tuple[_Agg, _Agg]] = {}         # bucket period-END → (bid, ask)
        self._last_close: tuple[float, float] | None = None   # last emitted REAL bar's closes
        self._last_bid = float("nan")
        self._last_ask = float("nan")
        self._last_depth_bid = float("nan")
        self._last_depth_ask = float("nan")

    @property
    def last_tob(self) -> tuple[float, float]:
        """Most recent (yes_bid, yes_ask) seen — the standing quote."""
        return self._last_bid, self._last_ask

    @property
    def last_depth(self) -> tuple[float, float]:
        return self._last_depth_bid, self._last_depth_ask

    def add(self, tick: BookTick) -> None:
        """Fold a tick into its minute's accumulator (per-bucket; out-of-order-safe across the
        minute boundary)."""
        end = bucket_end(tick.ts, period_s=self.period_s)
        pair = self._aggs.get(end)
        if pair is None:
            self._aggs[end] = (_Agg.start(tick.yes_bid), _Agg.start(tick.yes_ask))
        else:
            pair[0].push(tick.yes_bid)
            pair[1].push(tick.yes_ask)
        if np.isfinite(tick.yes_bid):
            self._last_bid = tick.yes_bid
        if np.isfinite(tick.yes_ask):
            self._last_ask = tick.yes_ask
        if np.isfinite(tick.depth_bid):
            self._last_depth_bid = tick.depth_bid
        if np.isfinite(tick.depth_ask):
            self._last_depth_ask = tick.depth_ask

    def close_minute(self, s: int) -> dict[str, object] | None:
        """Return the completed bar for the minute ending at ``s`` (``ts=s``), or None if no real
        bar has ever completed. A silent minute (no ticks in ``(s−60, s]``) is flat at the previous
        emitted bar's close."""
        pair = self._aggs.pop(s, None)
        if pair is not None:
            bid, ask = pair
            self._last_close = (bid.close, ask.close)
            return self._row(s, bid, ask)
        if self._last_close is not None:
            return self._row(s, _Agg.start(self._last_close[0]), _Agg.start(self._last_close[1]))
        return None

    def _row(self, s: int, bid: _Agg, ask: _Agg) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "ts": int(s),
            "yes_bid_open": bid.open, "yes_bid_high": bid.high,
            "yes_bid_low": bid.low, "yes_bid_close": bid.close,
            "yes_ask_open": ask.open, "yes_ask_high": ask.high,
            "yes_ask_low": ask.low, "yes_ask_close": ask.close,
            "volume": float("nan"),
        }


def assemble_bars(
    ticks: list[BookTick], ticker: str, start_ts: int, end_ts: int, *, period_s: int = PERIOD_S
) -> pd.DataFrame:
    """Batch: assemble every 1-min bar in ``[start_ts, end_ts]`` from a full tick log (offline /
    replay convenience; the live loop uses :class:`BarBuilder` incrementally). Bars are emitted for
    each period-END boundary in range once a real bar exists, carrying the last close through
    silent minutes."""
    builder = BarBuilder(ticker, period_s=period_s)
    ordered = sorted(ticks, key=lambda t: t.ts)
    first_end = bucket_end(start_ts, period_s=period_s)
    last_end = bucket_end(end_ts, period_s=period_s)
    rows: list[dict[str, object]] = []
    i = 0
    for s in range(first_end, last_end + 1, period_s):
        while i < len(ordered) and bucket_end(ordered[i].ts, period_s=period_s) <= s:
            builder.add(ordered[i])
            i += 1
        bar = builder.close_minute(s)
        if bar is not None:
            rows.append(bar)
    return pd.DataFrame(rows, columns=BAR_COLUMNS)
