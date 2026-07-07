"""Per-occasion capture orchestrator: arm → capture (audio ∥ book) → record → notify → auto-stop
→ summary. Host-agnostic (a Mac launchd service or a VPS; NOT serverless). Record-only: this
module never fires a trading decision — it captures a synchronized audio transcript + market-book
tape and lets an offline consumer (``signals/spikes.py``, ``signals/align.py``, or the scripts
under ``scripts/tape_exploration/``) analyze it afterward.

The intricate part is async plumbing (a WS book task, a threaded audio→Whisper task, and a
minute-cadence recording task). The *decisions* (when to stop, when to bounce a wedged audio
pipeline) live in pure helpers — :func:`next_minute_boundary`, :func:`stop_capture`,
:func:`snapshot_tob`, :func:`close_minute_bars` — which are unit-tested; the async wrapper only
schedules them. The live socket/model/subprocess paths are not exercised in tests (no live audio
to summon); first-arm live validation + the host/Whisper-throughput choice is surfaced to the
operator via :func:`host_prompt`.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd

from pmlab.config import Settings, get_settings
from pmlab.probe.audio import PcmStream, StreamingWhisper, TranscriptBuffer
from pmlab.probe.bars import PERIOD_S, BarBuilder
from pmlab.probe.book import stream_orderbook
from pmlab.probe.notify import Event, Notifier
from pmlab.probe.occasions import LiveStream, UpcomingOccasion
from pmlab.probe.record import BOOK_TICK_COLUMNS, LiveState, ProbeStore, RunSummary

log = logging.getLogger("pmlab.probe.run")


@dataclass(frozen=True)
class ProbeConfig:
    lead_s: int = 1800            # arm before scheduled time
    min_capture_s: int = 600      # don't auto-stop on silence before this
    max_capture_s: int = 3 * 3600 # hard cap after scheduled start
    quiet_s: int = 300            # stop when transcript silent this long (speech/program ended)
    decision_period_s: int = PERIOD_S
    sample_s: float = 10.0        # join-time audio sample length before starting the recording loop
    sample_timeout_s: float = 90.0  # first-chunk timeout (stream not live yet)
    min_sample_words: int = 8     # below this the sample is "quiet", not "dead" (music/crowd)
    # The ASR-blind watchdog: an independent ops constant, unrelated to any downstream analysis
    # threshold — it only trips a self-heal bounce of the audio pipeline, never anything else.
    watch_move_threshold: float = 0.08
    watch_move_window_s: int = 300
    watch_blind_s: int = 180      # must exceed the ASR force-commit bound + the flush cadence


# --- pure scheduling / stop / tick helpers (unit-tested) -----------------------------------------


def next_minute_boundary(now_ts: int, *, period_s: int = PERIOD_S) -> int:
    """The next period-END boundary strictly after ``now_ts`` — when the minute just closed."""
    return (now_ts // period_s + 1) * period_s


class CaptureLimits(Protocol):
    """Structural subset of :class:`ProbeConfig` that :func:`stop_capture` needs — lets the
    record-only sports/broadcast capture (``pmlab.probe.record_game.RecordConfig``) reuse this
    helper verbatim without inheriting the speech-probe's other fields. Read-only properties (not
    plain attributes) so a frozen dataclass's read-only fields satisfy this Protocol under mypy
    strict."""

    @property
    def max_capture_s(self) -> int: ...
    @property
    def min_capture_s(self) -> int: ...
    @property
    def quiet_s(self) -> int: ...


def stop_capture(
    now_utc: int, *, capture_start_utc: int, last_word_utc: int, occurrence_ts: int,
    cfg: CaptureLimits,
) -> str | None:
    """Auto-stop reason (stop when the program has clearly ended, or the hard cap is hit), or
    None to keep capturing."""
    if now_utc >= occurrence_ts + cfg.max_capture_s:
        return "max_duration"
    elapsed = now_utc - capture_start_utc
    silent = now_utc - last_word_utc if last_word_utc else 0
    if elapsed >= cfg.min_capture_s and last_word_utc and silent >= cfg.quiet_s:
        return "speech_ended"
    return None


def snapshot_tob(
    builders: Mapping[str, BarBuilder],
) -> dict[str, tuple[float, float, float, float]]:
    """Live top-of-book per ticker from each builder's standing quote (the builder sees every
    tick, so ``last_tob`` is the current executable quote). Returns ``(yes_bid, yes_ask,
    depth_bid, depth_ask)``."""
    out: dict[str, tuple[float, float, float, float]] = {}
    for ticker, builder in builders.items():
        bid, ask = builder.last_tob
        d_bid, d_ask = builder.last_depth
        out[ticker] = (bid, ask, d_bid, d_ask)
    return out


def close_minute_bars(builders: Mapping[str, BarBuilder], s: int) -> None:
    """Close each market's minute-``s`` bar (record-only: this does not evaluate any signal, it
    just advances every builder's bar boundary so the recorded tape has one row per closed
    minute)."""
    for builder in builders.values():
        builder.close_minute(s)


def blind_watch(
    mid_hist: Mapping[str, list[tuple[int, float]]], now_utc: int, last_word_utc: int, *,
    capture_start_utc: int, threshold: float, move_window_s: int, blind_s: int,
) -> bool:
    """The cross-channel ASR-blind detector — books-say-something-happened + transcript-frozen is
    the signature of a wedged Whisper, and it's cheap to detect by comparing two independent
    channels. True iff ANY ticker's mid moved ``≥ threshold`` within the last ``move_window_s``
    seconds AND no word has committed for ``blind_s`` seconds (from the last committed word, or
    from capture start if none has committed yet — ``0`` is this module's "no word yet" sentinel,
    matching :func:`stop_capture`'s convention)."""
    moved = False
    for pts in mid_hist.values():
        window = [m for t, m in pts if now_utc - t <= move_window_s]
        if window and max(window) - min(window) >= threshold:
            moved = True
            break
    if not moved:
        return False
    silent_since = last_word_utc if last_word_utc else capture_start_utc
    return now_utc - silent_since >= blind_s


def resume_live_state(
    occasion: UpcomingOccasion, live_state: LiveState | None,
) -> tuple[UpcomingOccasion, int]:
    """Resolve the occasion/attempt a capture should use, given the persisted ``live_state.json``
    (``None`` on a fresh occasion — attempt 1).

    A relaunch or re-invocation (in-loop bounce, operator manual, ``run-today --now``) must load
    the ORIGINAL ``occurrence_ts``, never re-stamp it — a reset clock silently shrinks the
    recorded elapsed-time-since-start for every downstream consumer of the tape."""
    if live_state is None:
        return occasion, 1
    resumed = replace(occasion, occurrence_ts=live_state.occurrence_ts)
    return resumed, live_state.attempt + 1


def host_prompt(rtf: float) -> str:
    """The first-arm operator message: pick a host, given the measured Whisper real-time factor
    (RTF < 1 ⇒ CPU keeps up with the live stream)."""
    ok = rtf and rtf < 1.0
    verdict = "keeps up (RTF<1) ✓" if ok else "may lag (RTF≥1) — prefer a stronger box"
    return (
        f"First arm: choose a host. Whisper '{get_settings().whisper_model}' RTF={rtf:.2f} "
        f"→ CPU {verdict}. Options: (a) this Mac plugged in as a launchd service ($0), or "
        f"(b) a small VPS. Set PMLAB_PROBE_HOST=mac-launchd|vps. NOT serverless (persistent "
        f"audio+WS process)."
    )


# --- live orchestration (network / subprocess / model) -------------------------------------------


async def verify_stream(
    video: LiveStream, *, whisper: StreamingWhisper, cfg: ProbeConfig, note: Notifier,
) -> tuple[str, PcmStream | None, list[np.ndarray], int]:
    """Open ``video``'s audio, collect ~``cfg.sample_s``, transcribe with the already-loaded
    Whisper, and push an operator-veto ntfy — automates the manual check for "did we join the
    right stream."

    Returns ``(status, stream, sample_chunks, anchor_utc)``: ``status`` is ``"dead"`` (no chunk
    within ``cfg.sample_timeout_s``, or the stream iterator was already exhausted — ``stream``
    is ``None`` and the underlying (yt-dlp, ffmpeg) pair has already been killed, see FIX1 below),
    ``"quiet"`` (decoded but < ``cfg.min_sample_words`` words — early joins are music/crowd, NOT a
    failure, so the stream is kept), or ``"ok"``. ``anchor_utc`` is the stream clock's anchor
    (first-sample wall time minus samples-so-far); ``sample_chunks`` are the raw PCM chunks
    consumed, so the caller can feed them through :meth:`StreamingWhisper.feed` afterward and
    lose nothing."""
    from pmlab.probe.audio import SAMPLE_RATE, stream_pcm

    loop = asyncio.get_running_loop()
    url = f"https://www.youtube.com/watch?v={video.video_id}"
    stream = await loop.run_in_executor(None, partial(stream_pcm, url))
    chunks: list[np.ndarray] = []
    anchor_utc = 0
    total_s = 0.0
    got_first = False
    while total_s < cfg.sample_s:
        fut = loop.run_in_executor(None, partial(next, stream.chunks, None))
        try:
            chunk = (await asyncio.wait_for(fut, timeout=cfg.sample_timeout_s)
                     if not got_first else await fut)
        except TimeoutError:
            chunk = None
        if chunk is None:
            # Abandon (timed out) or natural exhaustion — either way kill the pair directly: on
            # timeout the executor thread is still blocked in ffmpeg.stdout.read(), and killing
            # ffmpeg makes that read return EOF so the thread unblocks and returns to the pool
            # (the exhaustion case is a no-op here — the generator's own finally already reaped
            # both procs, and .kill() tolerates that).
            stream.kill()
            note.notify(Event.HEALTH, f'audio verify [dead] {video.video_id}: ""')
            return "dead", None, chunks, 0
        if not got_first:
            anchor_utc = int(time.time()) - int(len(chunk) / SAMPLE_RATE)
            got_first = True
        chunks.append(chunk)
        total_s += len(chunk) / SAMPLE_RATE
    hyp = await loop.run_in_executor(None, whisper.transcribe_sample, np.concatenate(chunks))
    words = [w for w, _t0, _t1 in sorted(hyp, key=lambda h: h[1])]
    status = "ok" if len(words) >= cfg.min_sample_words else "quiet"
    text = " ".join(words)[:180]
    note.notify(Event.HEALTH, f'audio verify [{status}] {video.video_id}: "{text}"')
    return status, stream, chunks, anchor_utc


async def lock_stream(
    occasion: UpcomingOccasion, video_id: str | None, *,
    whisper: StreamingWhisper, cfg: ProbeConfig, note: Notifier,
) -> tuple[LiveStream, PcmStream, list[np.ndarray], int] | None:
    """Resolve + verify the audio stream for ``occasion`` (precedence: operator pin → Kalshi-linked
    milestone → search), or ``None`` if nothing usable was found.

    A pin retries the SAME stream up to 3 times — the operator already knows it's right, so a
    dead sample is a transient join issue, not a wrong-stream signal. An auto-selected candidate
    instead advances down the precedence chain on dead: joining a wedged or wrong stream is worse
    than trying the next one. Fires the pinned/no-stream CRASH ntfy itself, so :func:`capture`'s
    audio task only has to check for ``None``."""
    from pmlab.probe.occasions import kalshi_linked_stream, resolve_stream

    if video_id is not None:
        pinned = LiveStream(video_id, "operator-pinned", "operator",
                            is_live=True, was_live=False, release_ts=None)
        for attempt in range(3):
            status, stream, chunks, anchor = await verify_stream(
                pinned, whisper=whisper, cfg=cfg, note=note)
            if status != "dead":
                assert stream is not None
                return pinned, stream, chunks, anchor
            log.warning("pinned stream %s dead (attempt %d/3)", video_id, attempt + 1)
        note.notify(Event.CRASH, f"pinned stream produced no audio: {video_id}")
        return None

    candidates: list[LiveStream] = []
    linked = await asyncio.get_running_loop().run_in_executor(
        None, kalshi_linked_stream, occasion.event_ticker)
    if linked is not None:
        candidates.append(LiveStream(linked, "kalshi-linked", "kalshi-milestone",
                                     is_live=True, was_live=False, release_ts=None))
    searched = resolve_stream(occasion)
    if searched is not None:
        candidates.append(searched)
    for cand in candidates:
        status, stream, chunks, anchor = await verify_stream(
            cand, whisper=whisper, cfg=cfg, note=note)
        if status != "dead":
            assert stream is not None
            return cand, stream, chunks, anchor
        log.warning("candidate stream %s (%s) dead — advancing", cand.video_id, cand.channel)
    note.notify(Event.CRASH, f"no live/scheduled stream found for {occasion.speaker}")
    return None


class TickWriter:
    """Off-event-loop parquet appends: the tape append is a whole-file read-modify-write that
    grows with the capture — run inline it would block the WS/recording loops for the duration of
    each rewrite. The loop enqueues batches; this daemon thread drains them. ``close()`` flushes
    the queue (sentinel + join) so the final batch always lands. An append error is logged and
    counted, never fatal — a recorder hiccup must not end a capture."""

    def __init__(self, store: ProbeStore) -> None:
        self._store = store
        self._q: queue.Queue[pd.DataFrame | None] = queue.Queue()
        self.n_errors = 0
        self._t = threading.Thread(target=self._drain, name="tick-writer", daemon=True)
        self._t.start()

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

    def submit(self, df: pd.DataFrame) -> None:
        self._q.put(df)

    def _drain(self) -> None:
        while True:
            df = self._q.get()
            if df is None:
                return
            try:
                self._store.append_ticks(df)
            except Exception:
                self.n_errors += 1
                log.exception("tick writer: append failed (batch dropped)")

    def close(self, timeout_s: float = 60.0) -> None:
        self._q.put(None)
        self._t.join(timeout=timeout_s)
        if self._t.is_alive():
            log.error("tick writer: close timed out with %d batches queued", self.queue_depth)


async def _book_task(
    occasion: UpcomingOccasion, builders: dict[str, BarBuilder],
    writer: TickWriter, settings: Settings, state: dict[str, int],
) -> None:
    """Consume the WS orderbook (which reconstructs books internally): feed the reconstructed
    top-of-book into each bar builder and batch-record every tick.

    Reconnects forever with capped backoff — a handshake timeout or mid-capture drop must never
    end the capture. Cancellation is the only exit."""
    from pmlab.probe.bars import BookTick

    tickers = [m.market_ticker for m in occasion.markets]
    backoff = 2.0
    while True:
        pending: list[dict[str, object]] = []
        try:
            async for upd in stream_orderbook(tickers, settings=settings):
                backoff = 2.0  # healthy stream → reset
                t: BookTick = upd.tick
                builders[upd.market_ticker].add(t)
                pending.append({
                    "ticker": upd.market_ticker, "recv_utc": t.ts, "seq": upd.seq,
                    "yes_bid": t.yes_bid, "yes_ask": t.yes_ask,
                    "depth_bid": t.depth_bid, "depth_ask": t.depth_ask,
                    "yes_bid2": t.yes_bid2, "yes_ask2": t.yes_ask2,
                    "depth_bid2": t.depth_bid2, "depth_ask2": t.depth_ask2,
                })
                state["n_ticks"] += 1
                state["last_tick_recv"] = t.ts
                if len(pending) >= 200:
                    writer.submit(pd.DataFrame(pending, columns=BOOK_TICK_COLUMNS))
                    pending = []
            log.warning("book stream ended; reconnecting in %.0fs", backoff)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # reconnect on ANY stream error — never end the capture
            log.warning("book stream error (%s: %s); reconnecting in %.0fs",
                        type(exc).__name__, exc, backoff)
        finally:
            if pending:
                writer.submit(pd.DataFrame(pending, columns=BOOK_TICK_COLUMNS))
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


async def capture(  # noqa: C901 - a linear orchestration; complexity is inherent, not branchy
    occasion: UpcomingOccasion, *, settings: Settings | None = None,
    notifier: Notifier | None = None, cfg: ProbeConfig | None = None, host: str = "manual",
    video_id: str | None = None,
) -> RunSummary:
    """Capture one occasion end-to-end, record-only. Live: opens the WS book + audio stream,
    records both to parquet every minute, notifies, and auto-stops. Returns the run summary (also
    written to disk)."""
    s = settings or get_settings()
    note = notifier or Notifier(topic=s.ntfy_topic, base=s.ntfy_base, dry=not s.ntfy_topic)
    cfg = cfg or ProbeConfig()
    store = ProbeStore(occasion.event_ticker, s.data_dir)
    capture_start = int(time.time())

    # Attempt 1 writes the state fresh; a relaunch/re-invocation resumes it (see
    # resume_live_state's docstring for the hazard this closes). Residual window (accepted): a
    # crash after resolve but before this first write_live_state still resets the clock on a
    # wholly separate CLI re-invocation — there is nothing on disk yet to resume from.
    occasion, attempt = resume_live_state(occasion, store.load_live_state())
    store.write_live_state(LiveState(occurrence_ts=occasion.occurrence_ts, attempt=attempt))

    builders = {m.market_ticker: BarBuilder(m.market_ticker) for m in occasion.markets}
    transcript = TranscriptBuffer(occasion.event_ticker)
    whisper = StreamingWhisper(capture_start_utc=capture_start, model_name=s.whisper_model)
    writer = TickWriter(store)
    state = {"n_ticks": 0, "last_word_utc": 0, "last_tick_recv": 0}
    locked_video_id: str | None = None            # bounce watchdog: the pin a bounce re-opens
    active_stream: PcmStream | None = None        # FIX1: the handle _bounce_audio/teardown kill
    mid_hist: dict[str, list[tuple[int, float]]] = {m.market_ticker: [] for m in occasion.markets}
    n_bounces = 0
    last_bounce_utc = 0
    stop_reason = ""  # threaded into the final RunSummary so run_with_relaunch can decide

    note.notify(Event.CAPTURE_STARTED,
               f"{occasion.speaker}: {len(occasion.markets)} markets, host={host}")

    async def audio_loop() -> None:
        loop = asyncio.get_running_loop()
        # Precedence: pin → kalshi-linked milestone stream → search, each sampled and ntfy'd
        # before it is trusted (:func:`lock_stream`). A dead pin retries itself; a dead
        # auto-candidate advances.
        locked = await lock_stream(occasion, video_id, whisper=whisper, cfg=cfg, note=note)
        if locked is None:
            return
        video, stream, sample_chunks, anchor_utc = locked
        nonlocal locked_video_id, active_stream
        locked_video_id = video.video_id  # lets a bounce re-pin the SAME stream
        active_stream = stream  # FIX1: so a bounce/teardown can .kill() this handle on abandon
        note.notify(Event.CAPTURE_STARTED,
                    f"stream locked: {video.title[:90]} [{video.channel}] ({video.video_id})")
        # Audio t=0 == the instant the live-edge join delivered its first (sample-phase) samples —
        # anchored inside :func:`lock_stream`'s verification, not at capture start (a scheduled
        # stream can keep stream_pcm waiting for minutes, which would shift every word timestamp).
        whisper.capture_start_utc = anchor_utc

        last_flush = 0.0
        last_health = 0.0
        asr_alerted = False

        async def _feed(chunk: np.ndarray) -> None:
            nonlocal last_flush, last_health, asr_alerted
            # FIX5: a bounce cancelling this task mid-await drops that one chunk's in-flight
            # words — bounded, expected (blind_watch only trips when NOTHING is committing).
            words = await loop.run_in_executor(
                None, partial(whisper.feed, chunk, now_utc=int(time.time())))
            # Health: the commit-lag metric + a once-per-capture page when the bound fires (the
            # run self-healed — the phone should still know the transcript got noisier).
            if time.time() - last_health >= 60.0:
                log.info("asr health: uncommitted=%.1fs force_commits=%d head_drops=%d",
                         whisper.uncommitted_s, whisper.n_force_commits, whisper.n_head_drops)
                last_health = time.time()
            if (whisper.n_force_commits or whisper.n_head_drops) and not asr_alerted:
                asr_alerted = True
                note.notify(Event.HEALTH, (
                    f"ASR bound hit: {whisper.n_force_commits} force-commits / "
                    f"{whisper.n_head_drops} head-drops — self-healed, transcript may be noisier"))
            if words:
                transcript.add(words)
                state["last_word_utc"] = words[-1].t_end_utc
                # Throttled incremental tape flush: a crash now loses ≤10 s of transcript instead
                # of the whole tape, and a live reader can tail it. Overwrite-write of the full
                # frame — idempotent; the final write is unchanged.
                if time.time() - last_flush >= 10.0:
                    store.write_transcript(transcript.frame())
                    last_flush = time.time()

        # The verification sample is real audio — feed it through the normal commit pipeline
        # before continuing with live chunks, so the transcript loses nothing.
        for chunk in sample_chunks:
            await _feed(chunk)
        while True:
            chunk = await loop.run_in_executor(None, partial(next, stream.chunks, None))
            if chunk is None:
                return
            await _feed(chunk)

    # Created here (not down with book_t/mon_t) so `_bounce_audio`'s `nonlocal audio_t` below has
    # an enclosing-scope binding to reassign — mypy requires the assignment textually precede a
    # nested function's `nonlocal` of the same name.
    audio_t = asyncio.create_task(audio_loop())

    async def _bounce_audio() -> None:
        """The audio pipeline is the suspected wedge — cancel it, rebuild a fresh Whisper (the
        old object is the suspect), and restart pinned to the SAME locked stream. The book task
        and ``speech_start`` are untouched."""
        nonlocal audio_t, whisper, video_id, active_stream
        audio_t.cancel()
        # FIX1: cancelling audio_t alone does not stop the executor thread abandoned mid-
        # ffmpeg.stdout.read() — a run_in_executor future already running can't be cancelled.
        # Kill the (yt-dlp, ffmpeg) pair directly so that read() returns EOF and the thread
        # unblocks; only then can the gather below actually observe audio_t finish.
        if active_stream is not None:
            active_stream.kill()
        await asyncio.gather(audio_t, return_exceptions=True)
        video_id = locked_video_id or video_id
        # FIX6: capture_start_utc is carried here for continuity, but audio_loop re-anchors the
        # fresh Whisper at its own first-chunk arrival regardless (see the anchoring comment
        # above) — this value is effectively overwritten, never actually relied on.
        whisper = StreamingWhisper(capture_start_utc=whisper.capture_start_utc,
                                   model_name=whisper.model_name)
        audio_t = asyncio.create_task(audio_loop())
        audio_t.add_done_callback(_support_watch("audio"))

    async def monitor_loop() -> None:
        nonlocal n_bounces, last_bounce_utc, stop_reason
        while True:
            s_bound = next_minute_boundary(int(time.time()))
            await asyncio.sleep(max(0.0, s_bound + 1 - time.time()))  # +1s: let the book settle
            now = int(time.time())
            close_minute_bars(builders, s_bound)
            # Track each ticker's live mid so blind_watch can compare it against the transcript.
            for ticker, (bid, ask, _db, _da) in snapshot_tob(builders).items():
                if bid == bid and ask == ask:  # both finite (NaN != NaN)
                    pts = mid_hist.setdefault(ticker, [])
                    pts.append((now, (bid + ask) / 2.0))
                    cutoff = now - cfg.watch_move_window_s
                    mid_hist[ticker] = [p for p in pts if p[0] >= cutoff]
            if (n_bounces < 5 and now - last_bounce_utc >= 300
                    and blind_watch(mid_hist, now, state["last_word_utc"],
                                    capture_start_utc=capture_start,
                                    threshold=cfg.watch_move_threshold,
                                    move_window_s=cfg.watch_move_window_s,
                                    blind_s=cfg.watch_blind_s)):
                n_bounces += 1
                last_bounce_utc = now
                note.notify(Event.HEALTH, (
                    f"ASR-blind bounce: mid moved ≥{cfg.watch_move_threshold:.2f} in "
                    f"{cfg.watch_move_window_s}s, no words {cfg.watch_blind_s}s"))
                await _bounce_audio()
            # Health: honest edge lag (recv stamped at socket read) + writer backlog.
            edge_lag = now - state["last_tick_recv"] if state["last_tick_recv"] else 0
            log.info("book health: edge_lag=%ds write_queue=%d writer_errors=%d",
                     edge_lag, writer.queue_depth, writer.n_errors)
            if writer.queue_depth > 20:
                log.warning("tick writer backlog: %d batches queued", writer.queue_depth)
            reason = stop_capture(now, capture_start_utc=capture_start,
                                  last_word_utc=state["last_word_utc"],
                                  occurrence_ts=occasion.occurrence_ts, cfg=cfg)
            if reason:
                log.info("auto-stop: %s", reason)
                stop_reason = reason
                return

    # The MONITOR loop owns the capture lifetime. Book/audio are support tasks: their death is
    # notified but must not end the capture (book reconnects internally; audio ending early just
    # means a book-only tail). A naive FIRST_COMPLETED over all three tasks would let a transient
    # WS handshake failure end the whole capture silently, with crashed=False.
    # (audio_t itself was created above, right after audio_loop's definition — see the comment.)
    book_t = asyncio.create_task(_book_task(occasion, builders, writer, s, state))
    mon_t = asyncio.create_task(monitor_loop())

    def _support_watch(name: str) -> Callable[[asyncio.Task[None]], None]:
        def cb(t: asyncio.Task[None]) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                log.error("%s task died: %s: %s", name, type(exc).__name__, exc)
                note.notify(Event.CRASH, f"{name} task died: {type(exc).__name__}: {exc}")
        return cb

    book_t.add_done_callback(_support_watch("book"))
    audio_t.add_done_callback(_support_watch("audio"))
    crashed = False
    try:
        await mon_t
    except Exception as exc:  # the lifetime owner blowing up is the real crash → notify
        crashed = True
        note.notify(Event.CRASH, f"{occasion.event_ticker}: {type(exc).__name__}: {exc}")
        log.exception("capture crashed")
    finally:
        for t in (book_t, audio_t):
            t.cancel()
        # FIX1: normal teardown also abandons audio_t's executor thread mid-read — kill the
        # (yt-dlp, ffmpeg) pair so the gather below can actually observe it finish, same as
        # _bounce_audio's abandon path.
        if active_stream is not None:
            active_stream.kill()
        await asyncio.gather(book_t, audio_t, return_exceptions=True)
        # after the book task is done enqueueing: flush the tape (final batch must land on disk)
        await asyncio.get_running_loop().run_in_executor(None, writer.close)

    summary = RunSummary(
        occasion_id=occasion.event_ticker, speaker=occasion.speaker,
        event_type=occasion.event_type, host=host, whisper_model=s.whisper_model,
        armed_utc=occasion.arm_at(lead_s=cfg.lead_s), capture_start_utc=capture_start,
        capture_end_utc=int(time.time()), speech_start_utc=occasion.occurrence_ts,
        n_markets_armed=len(occasion.markets), n_ticks=state["n_ticks"],
        n_words=transcript.n_words, crashed=crashed, stop_reason=stop_reason,
    )
    store.write_transcript(transcript.frame())  # final tape
    store.write_summary(summary)
    note.notify(Event.COMPLETE, (
        f"{occasion.speaker}: {state['n_ticks']} ticks / "
        f"{transcript.n_words} words" + (" [CRASHED]" if crashed else "")))
    return summary


def run_occasion(
    occasion: UpcomingOccasion, *, host: str = "manual", cfg: ProbeConfig | None = None,
    video_id: str | None = None,
) -> RunSummary:
    """Sync entry point (launchd/CLI): capture one occasion to completion. ``cfg`` lets the
    operator widen the capture window for long-program events (a long pre-program before the
    actual speech/segment can otherwise trip the hard cap or the quiet rail early);
    ``video_id`` pins the audio stream.

    A death anywhere — including capture setup, before the in-loop notifier coverage starts —
    must page the phone, not just the console. Notify CRASH, re-raise."""
    try:
        return asyncio.run(capture(occasion, host=host, cfg=cfg, video_id=video_id))
    except BaseException as exc:  # noqa: BLE001 - process boundary: page, then re-raise
        s = get_settings()
        Notifier(topic=s.ntfy_topic, base=s.ntfy_base, dry=not s.ntfy_topic).notify(
            Event.CRASH, f"{occasion.event_ticker}: capture died: {type(exc).__name__}: {exc}")
        raise


_RELAUNCH_MAX_WORDS = 1000  # a "speech_ended" stop below this word count usually means late-start
_RELAUNCH_BACKOFF_S = 10.0


def run_with_relaunch(
    occasion: UpcomingOccasion, *, host: str = "manual", cfg: ProbeConfig | None = None,
    video_id: str | None = None, max_relaunches: int = 3,
) -> RunSummary:
    """Productize an ad-hoc watchdog loop into ``run-today``/``daemon``. A ``speech_ended``
    auto-stop with a small transcript (< ``_RELAUNCH_MAX_WORDS`` words) usually means the quiet
    rail fired on a late program (music/announcer), not the actual program ending — relaunch
    instead of exiting. ``max_duration`` never relaunches (the capture window is genuinely over);
    a crash (``stop_reason == ""``) is the caller's job, not this loop's.

    The carry-over (the ORIGINAL ``occurrence_ts`` survives a relaunch) is :func:`capture`'s own
    responsibility via ``live_state.json`` — this loop only decides WHETHER to relaunch, never
    touches the clock itself, so a re-invocation (``run-today --now``) gets the same protection
    for free."""
    cfg = cfg or ProbeConfig()
    summary = run_occasion(occasion, host=host, cfg=cfg, video_id=video_id)
    relaunches = 0
    while (
        relaunches < max_relaunches
        and summary.stop_reason == "speech_ended"
        and summary.n_words < _RELAUNCH_MAX_WORDS
        and int(time.time()) < occasion.occurrence_ts + cfg.max_capture_s
    ):
        relaunches += 1
        s = get_settings()
        Notifier(topic=s.ntfy_topic, base=s.ntfy_base, dry=not s.ntfy_topic).notify(
            Event.HEALTH, (f"quiet-rail stop with {summary.n_words} words — relaunching "
                          f"{relaunches}/{max_relaunches}"))
        time.sleep(_RELAUNCH_BACKOFF_S)
        summary = run_occasion(occasion, host=host, cfg=cfg, video_id=video_id)
    return summary
