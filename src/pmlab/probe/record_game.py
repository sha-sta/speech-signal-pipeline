"""Record-only broadcast capture — no fire decisions, no signals: tapes the Kalshi WS orderbook
+ a local avfoundation audio device (e.g. a loopback device fed by a browser/TV tab's audio) into
local Whisper. This module makes no decisions and touches no order endpoint; the tape is meant for
offline analysis (spike detection, alignment QA, microstructure).

Deliberately NOT a thin wrapper of :mod:`pmlab.probe.run`'s ``capture``: there is no stream to
resolve (``--audio-device`` names a local device directly — never yt-dlp/YouTube search, which
would risk taping the wrong broadcast) and there is no decision-making wiring at all. It reuses
that module's book/writer/health primitives (:func:`~pmlab.probe.run._book_task`,
:class:`~pmlab.probe.run.TickWriter`, :func:`~pmlab.probe.run.stop_capture`,
:func:`~pmlab.probe.run.blind_watch`, :func:`~pmlab.probe.run.snapshot_tob`,
:func:`~pmlab.probe.run.resume_live_state`) verbatim.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from pmlab.config import Settings, get_settings
from pmlab.probe.audio import SAMPLE_RATE, PcmStream, StreamingWhisper, TranscriptBuffer, device_pcm
from pmlab.probe.bars import BarBuilder
from pmlab.probe.notify import Event, Notifier
from pmlab.probe.occasions import UpcomingOccasion
from pmlab.probe.record import LiveState, ProbeStore, RunSummary
from pmlab.probe.run import (
    TickWriter,
    _book_task,
    blind_watch,
    next_minute_boundary,
    resume_live_state,
    snapshot_tob,
    stop_capture,
)

log = logging.getLogger("pmlab.probe.record_game")


@dataclass(frozen=True)
class RecordConfig:
    sample_s: float = 10.0
    # Local device: the first chunk is immediate (no join-a-live-stream wait) — a miss means
    # ffmpeg/device failure, not a slow join, hence a much tighter timeout than the live-stream
    # default.
    sample_timeout_s: float = 30.0
    min_sample_words: int = 8
    min_capture_s: int = 900
    # Studio/halftime audio keeps words flowing; sustained silence means the operator's stream
    # actually stopped (postgame, stream drop) — wider than the speech probe's default quiet rail.
    quiet_s: int = 600
    max_capture_s: int = int(3.5 * 3600)
    watch_move_threshold: float = 0.08
    watch_move_window_s: int = 300
    watch_blind_s: int = 180


async def verify_device(
    device: str, *, whisper: StreamingWhisper, cfg: RecordConfig, note: Notifier,
) -> tuple[str, PcmStream | None, list[np.ndarray], int]:
    """The join-verification analog for a local avfoundation device (no stream resolution to
    verify — only that the device is actually producing the intended audio, not silence/wrong
    channel). Returns ``(status, stream, sample_chunks, anchor_utc)`` exactly like
    :func:`pmlab.probe.run.verify_stream`. The ntfy text includes the sample's RMS — the tell for
    whether a Multi-Output Device is actually routing audio into the loopback device (silence →
    rms≈0); this ntfy IS the wrong-channel operator veto for record-only mode."""
    loop = asyncio.get_running_loop()
    stream = await loop.run_in_executor(None, partial(device_pcm, device))
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
            # Same FIX1 reasoning as verify_stream: kill the handle so an abandoned executor
            # thread blocked in ffmpeg.stdout.read() unblocks (EOF) and returns to the pool.
            stream.kill()
            note.notify(Event.HEALTH, f'audio verify [dead] device={device}: ""')
            return "dead", None, chunks, 0
        if not got_first:
            anchor_utc = int(time.time()) - int(len(chunk) / SAMPLE_RATE)
            got_first = True
        chunks.append(chunk)
        total_s += len(chunk) / SAMPLE_RATE
    sample = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
    hyp = await loop.run_in_executor(None, whisper.transcribe_sample, sample)
    words = [w for w, _t0, _t1 in sorted(hyp, key=lambda h: h[1])]
    status = "ok" if len(words) >= cfg.min_sample_words else "quiet"
    rms = float(np.sqrt(np.mean(np.square(sample)))) if len(sample) else 0.0
    text = " ".join(words)[:180]
    note.notify(Event.HEALTH, f'audio verify [{status}] device={device} rms={rms:.3f}: "{text}"')
    return status, stream, chunks, anchor_utc


async def record_capture(  # noqa: C901 - linear orchestration, mirrors run.capture's shape
    occasion: UpcomingOccasion, *, device: str, settings: Settings | None = None,
    notifier: Notifier | None = None, cfg: RecordConfig | None = None, host: str = "manual",
) -> RunSummary:
    """Capture one occasion's book + local-device transcript, record-only. No decision-making, no
    signals — mirrors :func:`pmlab.probe.run.capture`'s async shape minus everything
    decision-related and minus stream resolution/locking (the device is named by the operator,
    not searched)."""
    s = settings or get_settings()
    note = notifier or Notifier(topic=s.ntfy_topic, base=s.ntfy_base, dry=not s.ntfy_topic)
    cfg = cfg or RecordConfig()
    store = ProbeStore(occasion.event_ticker, s.data_dir, subdir="sports")
    capture_start = int(time.time())

    # A relaunch/re-invocation must resume the ORIGINAL kickoff stamp, never reset it.
    occasion, attempt = resume_live_state(occasion, store.load_live_state())
    store.write_live_state(LiveState(occurrence_ts=occasion.occurrence_ts, attempt=attempt))

    builders = {m.market_ticker: BarBuilder(m.market_ticker) for m in occasion.markets}
    transcript = TranscriptBuffer(occasion.event_ticker)  # source stays "whisper-live" (schema)
    whisper = StreamingWhisper(capture_start_utc=capture_start, model_name=s.whisper_model)
    writer = TickWriter(store)
    state = {"n_ticks": 0, "last_word_utc": 0, "last_tick_recv": 0}
    active_stream: PcmStream | None = None  # FIX1: the handle bounce/teardown .kill()s on abandon
    mid_hist: dict[str, list[tuple[int, float]]] = {m.market_ticker: [] for m in occasion.markets}
    n_bounces = 0
    last_bounce_utc = 0
    stop_reason = ""

    note.notify(Event.CAPTURE_STARTED,
               f"{occasion.speaker}: record-only, {len(occasion.markets)} markets, device={device}")

    async def audio_loop() -> None:
        nonlocal active_stream
        loop = asyncio.get_running_loop()
        status, stream, sample_chunks, anchor_utc = await verify_device(
            device, whisper=whisper, cfg=cfg, note=note)
        if status == "dead":
            note.notify(Event.CRASH, f"audio device produced no audio: {device}")
            return
        assert stream is not None
        active_stream = stream
        whisper.capture_start_utc = anchor_utc

        last_flush = 0.0
        last_health = 0.0
        asr_alerted = False

        async def _feed(chunk: np.ndarray) -> None:
            nonlocal last_flush, last_health, asr_alerted
            words = await loop.run_in_executor(
                None, partial(whisper.feed, chunk, now_utc=int(time.time())))
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
                if time.time() - last_flush >= 10.0:
                    store.write_transcript(transcript.frame())
                    last_flush = time.time()

        for chunk in sample_chunks:
            await _feed(chunk)
        while True:
            chunk = await loop.run_in_executor(None, partial(next, stream.chunks, None))
            if chunk is None:
                return
            await _feed(chunk)

    audio_t = asyncio.create_task(audio_loop())

    async def _bounce_audio() -> None:
        """Same device re-opened fresh (no stream pinning needed — there is only one named
        device)."""
        nonlocal audio_t, whisper, active_stream
        audio_t.cancel()
        if active_stream is not None:
            active_stream.kill()
        await asyncio.gather(audio_t, return_exceptions=True)
        whisper = StreamingWhisper(capture_start_utc=whisper.capture_start_utc,
                                   model_name=whisper.model_name)
        audio_t = asyncio.create_task(audio_loop())
        audio_t.add_done_callback(_support_watch("audio"))

    async def health_loop() -> None:
        nonlocal n_bounces, last_bounce_utc, stop_reason
        while True:
            s_bound = next_minute_boundary(int(time.time()))
            await asyncio.sleep(max(0.0, s_bound + 1 - time.time()))
            now = int(time.time())
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

    book_t = asyncio.create_task(_book_task(occasion, builders, writer, s, state))
    health_t = asyncio.create_task(health_loop())

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
        await health_t
    except Exception as exc:  # the lifetime owner blowing up is the real crash → notify
        crashed = True
        note.notify(Event.CRASH, f"{occasion.event_ticker}: {type(exc).__name__}: {exc}")
        log.exception("record capture crashed")
    finally:
        for t in (book_t, audio_t):
            t.cancel()
        if active_stream is not None:
            active_stream.kill()
        await asyncio.gather(book_t, audio_t, return_exceptions=True)
        await asyncio.get_running_loop().run_in_executor(None, writer.close)

    summary = RunSummary(
        occasion_id=occasion.event_ticker, speaker=occasion.speaker, event_type="sports",
        host=host, whisper_model=s.whisper_model,
        # No arming/lead-time step exists in record-only mode (the CLI stamps kickoff and starts
        # capturing immediately) — armed_utc is the capture start, not a lead-time anchor.
        armed_utc=capture_start, capture_start_utc=capture_start,
        capture_end_utc=int(time.time()), speech_start_utc=occasion.occurrence_ts,
        n_markets_armed=len(occasion.markets), n_ticks=state["n_ticks"],
        n_words=transcript.n_words, crashed=crashed, stop_reason=stop_reason,
        note=f"record-only capture, device={device}",
    )
    store.write_transcript(transcript.frame())
    store.write_summary(summary)
    note.notify(Event.COMPLETE, (
        f"{occasion.speaker}: record-only, {state['n_ticks']} ticks / "
        f"{transcript.n_words} words" + (" [CRASHED]" if crashed else "")))
    return summary


def run_record(
    occasion: UpcomingOccasion, *, device: str, cfg: RecordConfig | None = None,
    host: str = "manual",
) -> RunSummary:
    """Sync entry point. A death anywhere — including setup — must page the phone, not just the
    console. Notify CRASH, re-raise."""
    try:
        return asyncio.run(record_capture(occasion, device=device, host=host, cfg=cfg))
    except BaseException as exc:  # noqa: BLE001 - process boundary: page, then re-raise
        s = get_settings()
        Notifier(topic=s.ntfy_topic, base=s.ntfy_base, dry=not s.ntfy_topic).notify(
            Event.CRASH,
            f"{occasion.event_ticker}: record capture died: {type(exc).__name__}: {exc}")
        raise


_RELAUNCH_MAX_WORDS = 1000  # a "speech_ended" (quiet-rail) stop below this usually means late-start
_RELAUNCH_BACKOFF_S = 10.0


def run_record_with_relaunch(
    occasion: UpcomingOccasion, *, device: str, cfg: RecordConfig | None = None,
    host: str = "manual", max_relaunches: int = 3,
) -> RunSummary:
    """Mirrors :func:`pmlab.probe.run.run_with_relaunch`: a quiet-rail stop with a small transcript
    usually means the program hadn't actually started yet, not that it ended — relaunch instead of
    exiting. ``max_duration`` never relaunches; the kickoff/live-state carry-over is
    :func:`record_capture`'s own responsibility via ``live_state.json``."""
    cfg = cfg or RecordConfig()
    summary = run_record(occasion, device=device, host=host, cfg=cfg)
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
        summary = run_record(occasion, device=device, host=host, cfg=cfg)
    return summary
