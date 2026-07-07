# Architecture

## Data flow

```
                         ┌─────────────────────────┐
   audio source ───────► │  streaming Whisper ASR   │
   (yt-dlp stream /      │  (LocalAgreement-2,      │───► transcript.parquet
   avfoundation device)  │   bounded uncommitted     │      (word, t_start_utc,
                         │   window)                 │       t_end_utc, conf)
                         └─────────────────────────┘
                                                              │
   Kalshi WS orderbook ─► book-tick recorder ─► book_ticks.parquet
   (top-of-book + depth)                                │    │
                                                          │    │
                                                          ▼    ▼
                                                 1-min OHLC bar assembly
                                                          │
                                                          ▼
                                              close-over-close spike detection
                                                          │
                                                          ▼
                                              candle store + maker-fill backtest
```

Two independent recorders run concurrently against the same wall clock: an audio→ASR pipeline
that emits a time-stamped transcript, and a WebSocket orderbook reader that emits time-stamped
top-of-book ticks. Both write append-only parquet tapes under `data/<subdir>/<occasion_id>/`, so
every downstream step — bar assembly, spike detection, alignment QA, the backtest — is a pure
offline read over durable files, not a dependency on the live process. `pmlab.probe.bars` folds
ticks into 1-minute OHLC bars; `pmlab.signals.spikes` scans those bars for close-over-close price
moves; `pmlab.study.candles` + `pmlab.study.backtest` replay a predictions frame against a longer
history of the same kind of bars pulled from the venue's own candlestick API.

## Package tour

- **`pmlab.http`** — one shared `ThrottledClient` (token-bucket rate limit + retry + on-disk JSON
  cache) that every venue and corpus source is built on, so throttling and caching behavior is
  consistent and injectable (a mock transport swaps in for tests without touching a socket).
- **`pmlab.venues`** — unauthenticated REST clients for Kalshi and Polymarket market data
  (events, markets, series, candlesticks, order books, price history). Read-only by construction:
  neither client imports or exposes an order-placement endpoint.
- **`pmlab.mentions`** — discovers a venue's mention-market universe, parses each market's
  settlement rules into a structured resolution record with a confidence/risk tier, and builds a
  registry (catalog ⋈ parsed resolution ⋈ candle coverage) with a small human-review round trip
  for the lowest-confidence parses.
- **`pmlab.corpus`** — scrapers for a transcript corpus and a dated news-headline corpus, both
  stored as schema-validated, append-safe parquet. Neither computes a feature; they are I/O and
  storage layers a downstream analysis reads.
- **`pmlab.probe`** — the live capture orchestrator: resolves a stream or opens a named device,
  runs it through streaming ASR, tails the venue's WebSocket order book, and records both to
  parquet on a fixed cadence, with watchdogs that self-heal a wedged audio pipeline and an
  auto-stop rail that ends a capture once the program has gone quiet. Record-only — it makes no
  trading decision at any point.
- **`pmlab.signals`** — offline analysis over a recorded tape: transcript acquisition from
  alternate sources (e.g. platform auto-captions), UTC alignment between a transcript's
  relative-time words and an absolute-time price series, and generic close-over-close spike
  detection over 1-minute bars.
- **`pmlab.study`** — a candle-history store (pulls and persists bid/ask OHLC for a set of
  markets) and a maker-side, strict-fill backtest simulator that replays an external predictions
  frame against those bars.
- **`pmlab.model`** — small, dependency-light calibration primitives (isotonic and Platt scaling,
  a chronological train/test split, Brier and log-loss scoring) usable by any probability model
  you bring to the backtest.

## Design decisions worth stating

- **Append-only parquet tapes.** Every recorder (`ProbeStore.append_ticks`, the transcript writer,
  the corpus stores) does a read-modify-write append rather than holding state only in memory. A
  crash-restart mid-capture picks the tape back up instead of losing it, and every downstream
  consumer — including a live dashboard tailing the same file — reads the same durable artifact
  the capture process itself is writing.
- **Wall-clock stamping at socket read.** Order-book ticks are stamped the instant they're read off
  the socket (`recv_utc`), not when they're later processed or batched. This keeps the recorded
  edge latency honest — a busy writer queue shows up as writer backlog, never as a falsely fast
  tick timestamp.
- **Corruption-tolerant tail reads.** Recorders favor small, frequent flushes (e.g. a throttled
  transcript flush every ~10s) over one large write at the end, so a crash loses at most the last
  few seconds of a tape rather than the whole capture, and a live reader can tail a still-growing
  file.
- **LocalAgreement-2 streaming ASR with a bounded uncommitted window.** The streaming transcriber
  only commits a word once two consecutive decode passes agree on it, so the running transcript
  prefix a downstream consumer reads never gets silently rewritten. Because a decode pass can, in
  principle, never agree (silence, noise, a persistently revising tail), the uncommitted window is
  hard-bounded: past a configured number of seconds, the pipeline force-commits the current best
  hypothesis (flagged with a distinct confidence value) rather than let the buffer — and the
  transcript's staleness — grow without limit.
- **Watchdog bounces (audio-only restart).** A capture cross-checks its two independent channels
  against each other: if the order book is visibly moving but the transcript has been silent for
  longer than ASR's own commit-lag bound should allow, that's the signature of a wedged decode
  pipeline, not a quiet broadcast. The watchdog restarts only the audio/ASR task in that case —
  the order-book recorder and the capture's overall timing are untouched.
- **Quiet-rail auto-stop.** A capture ends itself once the transcript has been silent for a
  configured duration past a minimum capture length, on the assumption that a long enough silence
  means the program ended. This is a capture-lifetime heuristic only; it has no relationship to
  any trading or detection threshold.
- **Detached process discipline.** A live capture is a long-running, stateful process (a
  persistent WebSocket + audio subprocess pair); it is meant to run as a supervised background
  service (see `deploy/`), not inside a short-lived task runner that might reap it mid-capture and
  corrupt an in-progress parquet append. Anything that drives a live capture should launch it
  detached from its own lifecycle.
