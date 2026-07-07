"""Capture orchestration pure helpers: minute scheduling, auto-stop, TOB snapshot, and bar-close
recording wired through real BarBuilders (book-sourced quotes, no sockets)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import numpy as np
import pytest

from pmlab.probe.audio import SAMPLE_RATE
from pmlab.probe.bars import BarBuilder, BookTick
from pmlab.probe.notify import Event, Notifier
from pmlab.probe.occasions import LiveStream, UpcomingOccasion
from pmlab.probe.record import LiveState, RunSummary
from pmlab.probe.run import (
    ProbeConfig,
    blind_watch,
    close_minute_bars,
    lock_stream,
    next_minute_boundary,
    resume_live_state,
    run_with_relaunch,
    snapshot_tob,
    stop_capture,
    verify_stream,
)

T0 = 1_800_000_000
END = T0 + 3600


def test_next_minute_boundary():
    assert next_minute_boundary(T0) == T0 + 60
    assert next_minute_boundary(T0 + 1) == T0 + 60
    assert next_minute_boundary(T0 + 59) == T0 + 60
    assert next_minute_boundary(T0 + 60) == T0 + 120


def test_stop_capture_reasons():
    cfg = ProbeConfig(min_capture_s=600, max_capture_s=3600, quiet_s=300)
    # hard cap after scheduled start
    assert stop_capture(T0 + 3600, capture_start_utc=T0, last_word_utc=T0 + 3599,
                        occurrence_ts=T0, cfg=cfg) == "max_duration"
    # silent past the quiet window (and past min capture) → speech ended
    assert stop_capture(T0 + 1200, capture_start_utc=T0, last_word_utc=T0 + 800,
                        occurrence_ts=T0, cfg=cfg) == "speech_ended"
    # still talking → keep going
    assert stop_capture(T0 + 1200, capture_start_utc=T0, last_word_utc=T0 + 1180,
                        occurrence_ts=T0, cfg=cfg) is None
    # within min capture, even if quiet → keep going
    assert stop_capture(T0 + 400, capture_start_utc=T0, last_word_utc=T0 + 10,
                        occurrence_ts=T0, cfg=cfg) is None


def test_snapshot_tob_from_builders():
    b = BarBuilder("A")
    b.add(BookTick(T0 + 5, 0.30, 0.34, depth_bid=200.0, depth_ask=90.0))
    tob = snapshot_tob({"A": b})
    assert tob["A"] == (0.30, 0.34, 200.0, 90.0)


def test_close_minute_bars_closes_every_builder_at_the_boundary():
    """Record-only: closing bars just advances each builder's boundary — no decision output."""
    a = BarBuilder("A")
    b = BarBuilder("B")
    a.add(BookTick(T0 - 30, 0.20, 0.24, depth_bid=100.0, depth_ask=50.0))
    b.add(BookTick(T0 - 30, 0.60, 0.64, depth_bid=100.0, depth_ask=50.0))
    close_minute_bars({"A": a, "B": b}, T0)
    tob = snapshot_tob({"A": a, "B": b})
    assert tob["A"][0] == pytest.approx(0.20) and tob["B"][0] == pytest.approx(0.60)


# --- TickWriter: off-loop tape appends, flush-on-close, error containment -------------------------


def test_tick_writer_drains_and_flushes_on_close(tmp_path):
    import pandas as pd

    from pmlab.probe.record import BOOK_TICK_COLUMNS, ProbeStore
    from pmlab.probe.run import TickWriter

    store = ProbeStore("OCC-W", tmp_path)
    w = TickWriter(store)
    row = {"ticker": "T", "recv_utc": 1, "seq": 1, "yes_bid": 0.2, "yes_ask": 0.3,
           "depth_bid": 1.0, "depth_ask": 2.0, "yes_bid2": float("nan"),
           "yes_ask2": float("nan"), "depth_bid2": float("nan"), "depth_ask2": float("nan")}
    w.submit(pd.DataFrame([row], columns=BOOK_TICK_COLUMNS))
    w.submit(pd.DataFrame([row | {"seq": 2}], columns=BOOK_TICK_COLUMNS))
    w.close()
    tape = pd.read_parquet(store.ticks_path)
    assert len(tape) == 2 and tape["seq"].tolist() == [1, 2]
    assert w.n_errors == 0


def test_tick_writer_survives_a_bad_batch(tmp_path):
    import pandas as pd

    from pmlab.probe.record import BOOK_TICK_COLUMNS, ProbeStore
    from pmlab.probe.run import TickWriter

    store = ProbeStore("OCC-E", tmp_path)
    w = TickWriter(store)
    w.submit(pd.DataFrame([{"wrong": 1}]))  # schema mismatch → logged, counted, not fatal
    row = {"ticker": "T", "recv_utc": 1, "seq": 9, "yes_bid": 0.2, "yes_ask": 0.3,
           "depth_bid": 1.0, "depth_ask": 2.0, "yes_bid2": float("nan"),
           "yes_ask2": float("nan"), "depth_bid2": float("nan"), "depth_ask2": float("nan")}
    w.submit(pd.DataFrame([row], columns=BOOK_TICK_COLUMNS))
    w.close()
    assert w.n_errors == 1
    tape = pd.read_parquet(store.ticks_path)
    assert tape["seq"].tolist() == [9]  # the good batch after the bad one still landed


# --- cross-channel ASR-blind watchdog (blind_watch) -----------------------------------------------


def test_blind_watch_trips_on_book_move_plus_transcript_silence():
    mid_hist = {"A": [(T0, 0.20), (T0 + 60, 0.20), (T0 + 120, 0.30)]}  # 10¢ move
    assert blind_watch(mid_hist, T0 + 300, T0, capture_start_utc=T0,
                       threshold=0.08, move_window_s=300, blind_s=180) is True


def test_blind_watch_false_when_book_flat():
    mid_hist = {"A": [(T0, 0.20), (T0 + 60, 0.21), (T0 + 120, 0.20)]}  # flat
    assert blind_watch(mid_hist, T0 + 300, T0, capture_start_utc=T0,
                       threshold=0.08, move_window_s=300, blind_s=180) is False


def test_blind_watch_false_when_transcript_still_fresh():
    mid_hist = {"A": [(T0, 0.20), (T0 + 120, 0.30)]}  # 10¢ move
    assert blind_watch(mid_hist, T0 + 130, T0 + 100, capture_start_utc=T0,
                       threshold=0.08, move_window_s=300, blind_s=180) is False  # only 30s silent


def test_blind_watch_prunes_moves_outside_the_window():
    # the 10¢ move happened, but it's now outside the move_window_s lookback
    mid_hist = {"A": [(T0, 0.20), (T0 + 60, 0.30)]}
    assert blind_watch(mid_hist, T0 + 1000, T0, capture_start_utc=T0,
                       threshold=0.08, move_window_s=300, blind_s=180) is False


def test_blind_watch_no_word_yet_uses_capture_start():
    """last_word_utc==0 (this module's "no word committed yet" sentinel) falls back to elapsed
    capture time, not a spurious "always silent since epoch"."""
    mid_hist = {"A": [(T0, 0.20), (T0 + 60, 0.30)]}
    # capture started 200s ago, well past blind_s=180 → trips
    assert blind_watch(mid_hist, T0 + 200, 0, capture_start_utc=T0,
                       threshold=0.08, move_window_s=300, blind_s=180) is True
    # capture started only 60s ago → not yet past the blind bound
    assert blind_watch(mid_hist, T0 + 60, 0, capture_start_utc=T0,
                       threshold=0.08, move_window_s=300, blind_s=180) is False


# --- join-time audio verification (verify_stream / lock_stream) -----------------------------------


class _FakeWhisper:
    """Stand-in for the already-loaded StreamingWhisper — only transcribe_sample is exercised."""

    def __init__(self, words: list[str]) -> None:
        self._words = words

    def transcribe_sample(self, pcm: np.ndarray) -> list[tuple[str, float, float]]:
        return [(w, float(i), float(i) + 0.5) for i, w in enumerate(self._words)]


def _pcm_chunk() -> np.ndarray:
    return np.zeros(SAMPLE_RATE, dtype="float32")


class _FakePcmStream:
    """Stand-in for :class:`pmlab.probe.audio.PcmStream` (FIX1) — a chunk iterator plus a
    ``.kill()`` the abandon paths (verify-timeout, exhausted/dead) are expected to call."""

    def __init__(self, chunks: object) -> None:
        self.chunks = iter(chunks)
        self.killed = False

    def kill(self) -> None:
        self.killed = True


def _mute_notifier() -> tuple[Notifier, list[tuple[Event, str]]]:
    sent: list[tuple[Event, str]] = []
    note = Notifier(dry=True)
    note.notify = lambda ev, msg: sent.append((ev, msg)) or None  # type: ignore[method-assign,assignment]
    return note, sent


def _occasion() -> UpcomingOccasion:
    return UpcomingOccasion(event_ticker="OCC", speaker="Spk", event_type="rally",
                            occurrence_ts=1_800_000_000, markets=[])


def test_verify_stream_ok_feeds_no_gap(monkeypatch) -> None:
    import pmlab.probe.audio as audio_mod

    monkeypatch.setattr(audio_mod, "stream_pcm",
                        lambda url, **kw: _FakePcmStream([_pcm_chunk()] * 10))
    cfg = ProbeConfig(sample_s=5.0, min_sample_words=3)
    note, sent = _mute_notifier()
    whisper = _FakeWhisper(["hello", "world", "today"])
    video = LiveStream("V1", "t", "c", is_live=True, was_live=False, release_ts=None)

    status, stream, chunks, anchor = asyncio.run(
        verify_stream(video, whisper=whisper, cfg=cfg, note=note))
    assert status == "ok" and stream is not None
    assert len(chunks) == 5  # 5 one-second chunks == cfg.sample_s
    assert anchor > 0
    assert len(sent) == 1 and sent[0][0] == Event.HEALTH
    assert "[ok]" in sent[0][1] and "hello world today" in sent[0][1]


def test_verify_stream_quiet_keeps_the_stream(monkeypatch) -> None:
    import pmlab.probe.audio as audio_mod

    monkeypatch.setattr(audio_mod, "stream_pcm",
                        lambda url, **kw: _FakePcmStream([_pcm_chunk()] * 3))
    cfg = ProbeConfig(sample_s=2.0, min_sample_words=8)
    note, sent = _mute_notifier()
    whisper = _FakeWhisper(["um"])  # early-join music/crowd: below min_sample_words
    video = LiveStream("V2", "t", "c", is_live=True, was_live=False, release_ts=None)

    status, stream, _chunks, _anchor = asyncio.run(
        verify_stream(video, whisper=whisper, cfg=cfg, note=note))
    assert status == "quiet" and stream is not None  # quiet is NOT a failure — stream is kept
    assert sent[0][0] == Event.HEALTH and "[quiet]" in sent[0][1]


def test_verify_stream_dead_on_exhausted_iterator(monkeypatch) -> None:
    import pmlab.probe.audio as audio_mod

    fake = _FakePcmStream([])
    monkeypatch.setattr(audio_mod, "stream_pcm", lambda url, **kw: fake)
    cfg = ProbeConfig(sample_s=2.0)
    note, sent = _mute_notifier()
    video = LiveStream("V3", "t", "c", is_live=True, was_live=False, release_ts=None)

    status, stream, chunks, anchor = asyncio.run(
        verify_stream(video, whisper=_FakeWhisper([]), cfg=cfg, note=note))
    assert status == "dead" and stream is None and chunks == [] and anchor == 0
    assert sent[0][0] == Event.HEALTH and "[dead]" in sent[0][1]
    assert fake.killed is True  # FIX1: the exhausted-iterator abandon path kills the handle


def test_verify_stream_timeout_kills_the_stream(monkeypatch) -> None:
    """FIX1: the first-chunk ``asyncio.wait_for`` timeout abandons the executor future — the
    handle must be killed so the (would-be) blocked ffmpeg read unblocks and the thread returns
    to the pool, instead of leaking the (yt-dlp, ffmpeg, thread) triple."""
    import pmlab.probe.audio as audio_mod

    class _HangingChunks:
        def __iter__(self) -> _HangingChunks:
            return self

        def __next__(self) -> np.ndarray:
            time.sleep(0.2)  # longer than cfg.sample_timeout_s below → forces the timeout branch
            return _pcm_chunk()

    fake = _FakePcmStream(_HangingChunks())
    monkeypatch.setattr(audio_mod, "stream_pcm", lambda url, **kw: fake)
    cfg = ProbeConfig(sample_s=2.0, sample_timeout_s=0.02)
    note, sent = _mute_notifier()
    video = LiveStream("V4", "t", "c", is_live=True, was_live=False, release_ts=None)

    status, stream, chunks, anchor = asyncio.run(
        verify_stream(video, whisper=_FakeWhisper([]), cfg=cfg, note=note))
    assert status == "dead" and stream is None and chunks == [] and anchor == 0
    assert fake.killed is True


def test_lock_stream_pinned_retries_three_times_then_pages(monkeypatch) -> None:
    import pmlab.probe.audio as audio_mod

    calls = {"n": 0}

    def fake_stream_pcm(url: str, **kw: object) -> object:
        calls["n"] += 1
        return _FakePcmStream([])  # always dead

    monkeypatch.setattr(audio_mod, "stream_pcm", fake_stream_pcm)
    note, sent = _mute_notifier()
    cfg = ProbeConfig()

    result = asyncio.run(
        lock_stream(_occasion(), "PINNED1", whisper=_FakeWhisper([]), cfg=cfg, note=note))
    assert result is None
    assert calls["n"] == 3
    assert sent[-1] == (Event.CRASH, "pinned stream produced no audio: PINNED1")


def test_lock_stream_auto_advances_past_dead_candidate(monkeypatch) -> None:
    import pmlab.probe.audio as audio_mod
    import pmlab.probe.occasions as occ_mod

    monkeypatch.setattr(occ_mod, "kalshi_linked_stream", lambda ticker, **kw: "DEAD1")
    good = LiveStream("GOOD1", "good stream", "chan", is_live=True, was_live=False, release_ts=1)
    monkeypatch.setattr(occ_mod, "resolve_stream", lambda occasion, **kw: good)

    def fake_stream_pcm(url: str, **kw: object) -> object:
        return _FakePcmStream([]) if "DEAD1" in url else _FakePcmStream([_pcm_chunk()] * 3)

    monkeypatch.setattr(audio_mod, "stream_pcm", fake_stream_pcm)
    note, sent = _mute_notifier()
    cfg = ProbeConfig(sample_s=2.0, min_sample_words=1)

    result = asyncio.run(
        lock_stream(_occasion(), None, whisper=_FakeWhisper(["hi"]), cfg=cfg, note=note))
    assert result is not None
    video, stream, _chunks, _anchor = result
    assert video.video_id == "GOOD1" and stream is not None
    statuses = [msg.split("]")[0].split("[")[1] for ev, msg in sent if ev == Event.HEALTH]
    assert statuses == ["dead", "ok"]  # DEAD1 tried and rejected before GOOD1 locked


def test_lock_stream_no_candidates_crashes(monkeypatch) -> None:
    import pmlab.probe.occasions as occ_mod

    monkeypatch.setattr(occ_mod, "kalshi_linked_stream", lambda ticker, **kw: None)
    monkeypatch.setattr(occ_mod, "resolve_stream", lambda occasion, **kw: None)
    note, sent = _mute_notifier()

    result = asyncio.run(
        lock_stream(_occasion(), None, whisper=_FakeWhisper([]), cfg=ProbeConfig(), note=note))
    assert result is None
    assert sent == [(Event.CRASH, "no live/scheduled stream found for Spk")]


# --- bounded relaunch + occurrence_ts carry-over across a relaunch --------------------------------


def test_resume_live_state_fresh_occasion_is_attempt_one() -> None:
    occ = _occasion()
    resumed, attempt = resume_live_state(occ, None)
    assert resumed == occ and attempt == 1


def test_resume_live_state_overrides_occurrence_ts() -> None:
    """A relaunch must load the ORIGINAL occurrence_ts, not keep whatever the fresh occasion
    object carries (a re-invocation may have stamped a new "now")."""
    occ = _occasion()  # occurrence_ts = T0 in this fixture's occasion
    stale_now = replace(occ, occurrence_ts=occ.occurrence_ts + 9_999)
    live_state = LiveState(occurrence_ts=occ.occurrence_ts, attempt=1)
    resumed, attempt = resume_live_state(stale_now, live_state)
    assert resumed.occurrence_ts == occ.occurrence_ts  # NOT stale_now's
    assert attempt == 2


def _summary(**kw: object) -> RunSummary:
    base: dict[str, object] = {
        "occasion_id": "OCC", "speaker": "s", "event_type": "rally",
        "host": "manual", "whisper_model": "base",
    }
    base.update(kw)
    return RunSummary(**base)  # type: ignore[arg-type]


def test_run_with_relaunch_bounded_on_quiet_stop_with_small_transcript(monkeypatch) -> None:
    from pmlab.probe import run as run_mod

    monkeypatch.setattr(run_mod.time, "sleep", lambda _s: None)
    note, sent = _mute_notifier()
    monkeypatch.setattr(run_mod, "Notifier", lambda **kw: note)
    calls = {"n": 0}

    def fake_run_occasion(occasion: UpcomingOccasion, **kw: object) -> RunSummary:
        calls["n"] += 1
        return _summary(n_words=10, stop_reason="speech_ended")

    monkeypatch.setattr(run_mod, "run_occasion", fake_run_occasion)
    occ = replace(_occasion(), occurrence_ts=int(time.time()) + 10_000)
    summary = run_with_relaunch(occ, max_relaunches=3)
    assert calls["n"] == 4  # original attempt + 3 bounded relaunches
    assert summary.stop_reason == "speech_ended"
    assert sum(1 for ev, msg in sent if ev == Event.HEALTH and "relaunching" in msg) == 3


def test_run_with_relaunch_no_relaunch_on_large_transcript(monkeypatch) -> None:
    from pmlab.probe import run as run_mod

    monkeypatch.setattr(run_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(run_mod, "Notifier", lambda **kw: _mute_notifier()[0])
    calls = {"n": 0}

    def fake_run_occasion(occasion: UpcomingOccasion, **kw: object) -> RunSummary:
        calls["n"] += 1
        return _summary(n_words=5000, stop_reason="speech_ended")

    monkeypatch.setattr(run_mod, "run_occasion", fake_run_occasion)
    occ = replace(_occasion(), occurrence_ts=int(time.time()) + 10_000)
    summary = run_with_relaunch(occ)
    assert calls["n"] == 1 and summary.n_words == 5000


def test_run_with_relaunch_no_relaunch_on_max_duration(monkeypatch) -> None:
    from pmlab.probe import run as run_mod

    monkeypatch.setattr(run_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(run_mod, "Notifier", lambda **kw: _mute_notifier()[0])
    calls = {"n": 0}

    def fake_run_occasion(occasion: UpcomingOccasion, **kw: object) -> RunSummary:
        calls["n"] += 1
        return _summary(n_words=1, stop_reason="max_duration")  # capture window genuinely over

    monkeypatch.setattr(run_mod, "run_occasion", fake_run_occasion)
    occ = replace(_occasion(), occurrence_ts=int(time.time()) + 10_000)
    summary = run_with_relaunch(occ)
    assert calls["n"] == 1 and summary.stop_reason == "max_duration"


def test_run_occasion_pages_crash_and_reraises(monkeypatch) -> None:
    """A capture that dies — even in setup — must notify CRASH, then re-raise."""
    from pmlab.probe import run as run_mod
    from pmlab.probe.notify import Event, Notification, Notifier
    from pmlab.probe.occasions import UpcomingOccasion

    sent: list[tuple[Event, str]] = []

    def fake_notify(self: Notifier, event: Event, message: str) -> Notification:
        sent.append((event, message))
        return Notification(event, "t", message, 1, "", sent=False)

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("setup died")

    monkeypatch.setattr(run_mod, "capture", boom)
    monkeypatch.setattr(Notifier, "notify", fake_notify)
    occ = UpcomingOccasion(event_ticker="OCC-X", speaker="s", event_type="remarks",
                           occurrence_ts=1, markets=[])
    with pytest.raises(RuntimeError, match="setup died"):
        run_mod.run_occasion(occ)
    assert sent and sent[0][0] == Event.CRASH and "OCC-X" in sent[0][1]
