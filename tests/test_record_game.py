"""Record-only broadcast capture: pure config + the async orchestration helpers exercised with
fakes, same style as test_probe_run.py's verify_stream/lock_stream/relaunch tests. No live
ffmpeg/whisper/WS paths are exercised."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from pmlab.probe.audio import SAMPLE_RATE
from pmlab.probe.notify import Event, Notification, Notifier
from pmlab.probe.occasions import UpcomingOccasion
from pmlab.probe.record import RunSummary
from pmlab.probe.record_game import (
    RecordConfig,
    run_record_with_relaunch,
    verify_device,
)

T0 = 1_800_000_000


def test_record_config_defaults() -> None:
    cfg = RecordConfig()
    assert cfg.sample_s == 10.0
    assert cfg.sample_timeout_s == 30.0
    assert cfg.min_sample_words == 8
    assert cfg.min_capture_s == 900
    assert cfg.quiet_s == 600
    assert cfg.max_capture_s == int(3.5 * 3600)
    assert cfg.watch_move_threshold == 0.08
    assert cfg.watch_move_window_s == 300
    assert cfg.watch_blind_s == 180


# --- fakes (device_pcm/PcmStream/whisper stand-ins; mirrors test_probe_run.py's fakes) ------------


class _FakeWhisper:
    def __init__(self, words: list[str]) -> None:
        self._words = words

    def transcribe_sample(self, pcm: np.ndarray) -> list[tuple[str, float, float]]:
        return [(w, float(i), float(i) + 0.5) for i, w in enumerate(self._words)]


def _pcm_chunk(value: float = 0.0) -> np.ndarray:
    return np.full(SAMPLE_RATE, value, dtype="float32")


class _FakePcmStream:
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
    return UpcomingOccasion(event_ticker="EVENT-EXAMPLE", speaker="Broadcast",
                            event_type="sports", occurrence_ts=T0, markets=[])


# --- verify_device: the wrong-channel operator veto for a local device ----------------------------


def test_verify_device_ok_reports_rms_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import pmlab.probe.record_game as rg

    monkeypatch.setattr(rg, "device_pcm", lambda device, **kw: _FakePcmStream([_pcm_chunk()] * 10))
    cfg = RecordConfig(sample_s=5.0, min_sample_words=3)
    note, sent = _mute_notifier()
    whisper = _FakeWhisper(["hello", "world", "today"])

    status, stream, chunks, anchor = asyncio.run(
        verify_device("BlackHole 2ch", whisper=whisper, cfg=cfg, note=note))
    assert status == "ok" and stream is not None
    assert len(chunks) == 5  # 5 one-second chunks == cfg.sample_s
    assert anchor > 0
    assert len(sent) == 1 and sent[0][0] == Event.HEALTH
    msg = sent[0][1]
    assert "[ok]" in msg and "device=BlackHole 2ch" in msg and "rms=" in msg
    assert "hello world today" in msg


def test_verify_device_rms_reflects_signal_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rms figure is the operator's wrong-channel tell — silence reads rms≈0, a real signal
    reads nonzero."""
    import pmlab.probe.record_game as rg

    loud = _pcm_chunk(0.5)
    monkeypatch.setattr(rg, "device_pcm", lambda device, **kw: _FakePcmStream([loud] * 3))
    cfg = RecordConfig(sample_s=2.0, min_sample_words=1)
    note, sent = _mute_notifier()

    status, *_ = asyncio.run(
        verify_device("BlackHole 2ch", whisper=_FakeWhisper(["hi"]), cfg=cfg, note=note))
    assert status == "ok"
    assert "rms=0.500" in sent[0][1]


def test_verify_device_quiet_keeps_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    import pmlab.probe.record_game as rg

    monkeypatch.setattr(rg, "device_pcm", lambda device, **kw: _FakePcmStream([_pcm_chunk()] * 3))
    cfg = RecordConfig(sample_s=2.0, min_sample_words=8)
    note, sent = _mute_notifier()

    status, stream, _chunks, _anchor = asyncio.run(
        verify_device("BlackHole 2ch", whisper=_FakeWhisper(["um"]), cfg=cfg, note=note))
    assert status == "quiet" and stream is not None  # quiet is NOT a failure — device is kept
    assert sent[0][0] == Event.HEALTH and "[quiet]" in sent[0][1]


def test_verify_device_dead_on_exhausted_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    import pmlab.probe.record_game as rg

    fake = _FakePcmStream([])
    monkeypatch.setattr(rg, "device_pcm", lambda device, **kw: fake)
    cfg = RecordConfig(sample_s=2.0)
    note, sent = _mute_notifier()

    status, stream, chunks, anchor = asyncio.run(
        verify_device("BlackHole 2ch", whisper=_FakeWhisper([]), cfg=cfg, note=note))
    assert status == "dead" and stream is None and chunks == [] and anchor == 0
    assert sent[0][0] == Event.HEALTH
    assert "[dead]" in sent[0][1] and "device=BlackHole 2ch" in sent[0][1]
    assert fake.killed is True


def test_verify_device_timeout_kills_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """FIX1 (shared with verify_stream): the first-chunk timeout abandons the executor future —
    the handle must be killed so the (would-be) blocked ffmpeg read unblocks."""
    import pmlab.probe.record_game as rg

    class _HangingChunks:
        def __iter__(self) -> _HangingChunks:
            return self

        def __next__(self) -> np.ndarray:
            time.sleep(0.2)  # longer than cfg.sample_timeout_s below → forces the timeout branch
            return _pcm_chunk()

    fake = _FakePcmStream(_HangingChunks())
    monkeypatch.setattr(rg, "device_pcm", lambda device, **kw: fake)
    cfg = RecordConfig(sample_s=2.0, sample_timeout_s=0.02)
    note, sent = _mute_notifier()

    status, stream, chunks, anchor = asyncio.run(
        verify_device("BlackHole 2ch", whisper=_FakeWhisper([]), cfg=cfg, note=note))
    assert status == "dead" and stream is None and chunks == [] and anchor == 0
    assert fake.killed is True


# --- run_record pages CRASH and re-raises ---------------------------------------------------------


def test_run_record_pages_crash_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    from pmlab.probe import record_game as rg_mod

    sent: list[tuple[Event, str]] = []

    def fake_notify(self: Notifier, event: Event, message: str) -> Notification:
        sent.append((event, message))
        return Notification(event, "t", message, 1, "", sent=False)

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("setup died")

    monkeypatch.setattr(rg_mod, "record_capture", boom)
    monkeypatch.setattr(Notifier, "notify", fake_notify)
    occ = _occasion()
    with pytest.raises(RuntimeError, match="setup died"):
        rg_mod.run_record(occ, device="BlackHole 2ch")
    assert sent and sent[0][0] == Event.CRASH and occ.event_ticker in sent[0][1]


# --- run_record_with_relaunch: bounded relaunch, quiet-rail stop with a small transcript ----------


def _summary(**kw: object) -> RunSummary:
    base: dict[str, object] = {
        "occasion_id": "OCC", "speaker": "s", "event_type": "sports",
        "host": "manual", "whisper_model": "base",
    }
    base.update(kw)
    return RunSummary(**base)  # type: ignore[arg-type]


def test_run_record_with_relaunch_bounded_on_quiet_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from pmlab.probe import record_game as rg_mod

    monkeypatch.setattr(rg_mod.time, "sleep", lambda _s: None)
    note, sent = _mute_notifier()
    monkeypatch.setattr(rg_mod, "Notifier", lambda **kw: note)
    calls = {"n": 0}

    def fake_run_record(occasion: UpcomingOccasion, **kw: object) -> RunSummary:
        calls["n"] += 1
        return _summary(n_words=10, stop_reason="speech_ended")

    monkeypatch.setattr(rg_mod, "run_record", fake_run_record)
    occ = UpcomingOccasion(event_ticker="OCC", speaker="s", event_type="sports",
                          occurrence_ts=int(time.time()) + 10_000, markets=[])
    summary = run_record_with_relaunch(occ, device="BlackHole 2ch", max_relaunches=3)
    assert calls["n"] == 4  # original attempt + 3 bounded relaunches
    assert summary.stop_reason == "speech_ended"
    assert sum(1 for ev, msg in sent if ev == Event.HEALTH and "relaunching" in msg) == 3


def test_run_record_with_relaunch_no_relaunch_on_large_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pmlab.probe import record_game as rg_mod

    monkeypatch.setattr(rg_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(rg_mod, "Notifier", lambda **kw: _mute_notifier()[0])
    calls = {"n": 0}

    def fake_run_record(occasion: UpcomingOccasion, **kw: object) -> RunSummary:
        calls["n"] += 1
        return _summary(n_words=5000, stop_reason="speech_ended")

    monkeypatch.setattr(rg_mod, "run_record", fake_run_record)
    occ = UpcomingOccasion(event_ticker="OCC", speaker="s", event_type="sports",
                          occurrence_ts=int(time.time()) + 10_000, markets=[])
    summary = run_record_with_relaunch(occ, device="BlackHole 2ch")
    assert calls["n"] == 1 and summary.n_words == 5000


def test_run_record_with_relaunch_no_relaunch_on_max_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pmlab.probe import record_game as rg_mod

    monkeypatch.setattr(rg_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(rg_mod, "Notifier", lambda **kw: _mute_notifier()[0])
    calls = {"n": 0}

    def fake_run_record(occasion: UpcomingOccasion, **kw: object) -> RunSummary:
        calls["n"] += 1
        return _summary(n_words=1, stop_reason="max_duration")

    monkeypatch.setattr(rg_mod, "run_record", fake_run_record)
    occ = UpcomingOccasion(event_ticker="OCC", speaker="s", event_type="sports",
                          occurrence_ts=int(time.time()) + 10_000, markets=[])
    summary = run_record_with_relaunch(occ, device="BlackHole 2ch")
    assert calls["n"] == 1 and summary.stop_reason == "max_duration"
