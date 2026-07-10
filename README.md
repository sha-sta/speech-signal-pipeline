# speech-signal-pipeline

A research pipeline for real-time broadcast-speech signal detection: live stream/device audio
capture → streaming Whisper ASR → time-aligned transcripts + order-book recording → event/spike
detection → backtest harness.

## Provenance

This repository is a public extraction of the infrastructure layer from a larger private research
codebase. Strategy-specific components — feature engineering, model parameters, trading thresholds,
and the original git history — have been removed. What's left is the plumbing that ran real
captures in production research: political speeches and live sports broadcasts, both recorded
end-to-end (synchronized transcript + order-book tape) for later offline analysis. Nothing here
places an order, computes a trading edge, or encodes a specific strategy; it is capture,
detection, and backtesting *infrastructure* only.

## What's here

- `pmlab.probe` — live speech/broadcast capture: stream resolution (pinned video, a linked
  broadcast, or search), audio capture (`yt-dlp` or a local loopback device), streaming Whisper ASR
  with a LocalAgreement-2 commit policy, and Kalshi order-book recording. Record-only: it captures
  a synchronized transcript + book tape and makes no trading decisions.
- `pmlab.signals` — offline analysis over a recorded tape: transcript acquisition
  (`transcripts.py`), UTC alignment/anchoring between a transcript and a paired price series
  (`align.py`), and generic close-over-close price-spike detection (`spikes.py`).
- `pmlab.venues` — read-only Kalshi + Polymarket REST clients.
- `pmlab.http` — a shared throttled/cached HTTP client used by every venue and corpus source.
- `pmlab.mentions` — Kalshi mention-market universe discovery and a resolution registry, with a
  human-review round trip for low-confidence parses.
- `pmlab.corpus` — transcript and dated-news-headline scrapers, stored as append-safe parquet.
- `pmlab.model` — probability calibration primitives (isotonic/Platt scaling, Brier/log-loss
  scoring, a chronological train/test split).
- `pmlab.study` — a candle-history store and a maker-side, strict-fill backtest simulator that
  replays *your own* predictions frame against recorded market prices.
- `scripts/` — standalone microstructure/ASR-QA tooling for a recorded tape (book-shape survey,
  ASR-vs-book reprice-lag measurement, transcript QA).
- `cpp/` — an optional C++20 engine for the backtest hot path and tick-to-bar replay,
  bit-identical to the Python reference (enforced by a parity gate in CI) and built like a
  low-latency system: allocation-free hot loop, mmap-able binary tapes, sanitizers in CI,
  honest benchmarks. Python stays the reference and works without it; see
  `docs/BACKTEST_ENGINE.md`.

See `docs/ARCHITECTURE.md` for the data-flow diagram and design decisions, `docs/CAPTURE_GUIDE.md`
for the capture-ops runbook, `docs/BACKTEST_HARNESS.md` for the simulator's mechanics, and
`docs/BACKTEST_ENGINE.md` for the native engine and its benchmark methodology.

## Quickstart

Install with `uv` (recommended) or plain `pip`:

```bash
uv venv && uv sync            # base install
uv sync --extra probe         # + faster-whisper, for local ASR (heavy; only needed on a capture host)
```

```bash
# plain pip, equivalent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[probe]"     # drop [probe] to skip faster-whisper
```

Copy the environment template and fill in what you need:

```bash
cp .env.example .env          # Kalshi read-only API credentials, ntfy topic, data dir, etc.
```

Run the test suite (network-marked tests are deselected by default):

```bash
uv run pytest
```

### Capture example — local device audio

Tape a browser/device audio source (e.g. a loopback device such as BlackHole) plus the Kalshi
order book for a given event, no stream resolution involved:

```bash
uv run pmlab probe record-game <EVENT_TICKER> --audio-device "BlackHole 2ch"
```

Artifacts land under `data/sports/<EVENT_TICKER>/`: `book_ticks.parquet`, `transcript.parquet`,
`run.json`. See `docs/CAPTURE_GUIDE.md` for device setup and pre-flight checks.

### Capture example — a YouTube/live stream

Resolve and capture a live or scheduled stream by event, with join-time audio verification:

```bash
uv run pmlab probe run-today <EVENT_TICKER>
```

### Analysis example — spike detection on recorded bars

```python
import pandas as pd
from pmlab.probe.bars import BookTick, assemble_bars
from pmlab.signals.spikes import SpikeConfig, detect_spikes

ticks_df = pd.read_parquet("data/sports/<EVENT_TICKER>/book_ticks.parquet")
ticker = ticks_df["ticker"].iloc[0]
one_market = ticks_df[ticks_df["ticker"] == ticker].sort_values("recv_utc")
ticks = [
    BookTick(ts=int(r.recv_utc), yes_bid=r.yes_bid, yes_ask=r.yes_ask)
    for r in one_market.itertuples()
]
bars = assemble_bars(ticks, ticker, int(one_market["recv_utc"].min()), int(one_market["recv_utc"].max()))

events = detect_spikes("<EVENT_TICKER>", ticker, bars, config=SpikeConfig(magnitude=0.05))
print(events)
```

`SpikeConfig`'s fields are caller-supplied parameters, not tuned recommendations — see
`docs/ARCHITECTURE.md` for what each one means.

## Requirements

- Python 3.12.
- `ffmpeg` on `PATH` for any audio capture path.
- Device-audio capture (`probe record-game`) is macOS-only: it opens the named input through
  ffmpeg's `avfoundation` backend. System-audio loopback (capturing a browser tab or another app's
  output) additionally needs a virtual loopback device — [BlackHole](https://existential.audio/blackhole/)
  is a free, commonly used one — routed through a macOS Multi-Output Device.
- `faster-whisper` (the `probe` extra) runs local ASR on CPU; no GPU or paid API required, but
  budget real CPU headroom on the capture host — see `docs/CAPTURE_GUIDE.md` for a throughput
  self-test (`pmlab probe doctor`).
- Kalshi and Polymarket clients are unauthenticated/public-data reads by default; Kalshi's
  authenticated order-book WebSocket needs a read-only API key (`.env.example`) — this pipeline
  never imports or calls an order-placement endpoint.

## License

MIT — see `LICENSE`.
