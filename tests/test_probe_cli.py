"""run-today's stream-resolution poll loop (pulled out of the typer command into
``_resolve_occasion_time_loop`` specifically so it's unit-testable without a CliRunner / typer
Option-default gymnastics). Covers: --video-id resolves time from the pinned video, never the
general search; and failed-resolution paging (so a stalled resolution loop can't fail silently
for hours with zero notification)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest
import typer

import pmlab.probe.cli as cli_mod
from pmlab.probe.notify import Event, Notifier
from pmlab.probe.occasions import UpcomingOccasion, ticker_day_bounds

NOW = 1_800_000_000


def _mute_notifier() -> tuple[Notifier, list[tuple[Event, str]]]:
    sent: list[tuple[Event, str]] = []
    note = Notifier(dry=True)
    note.notify = lambda ev, msg: sent.append((ev, msg)) or None  # type: ignore[method-assign,assignment]
    return note, sent


def _day_only_occ(ticker: str, occurrence_ts: int = NOW) -> UpcomingOccasion:
    return UpcomingOccasion(event_ticker=ticker, speaker="Jane Doe", event_type="meeting",
                            occurrence_ts=occurrence_ts, markets=[], time_known=False)


# --- --video-id resolves from the pin, never the general search -----------------------------------


def test_loop_video_id_resolves_immediately_without_search_or_sleep(monkeypatch):
    occ = _day_only_occ("E-30JAN01")
    resolved = replace(occ, occurrence_ts=NOW - 500, time_known=True)

    monkeypatch.setattr(cli_mod, "resolve_occasion_time_from_video", lambda o, vid, **kw: resolved)

    def _search_boom(*a: object, **kw: object) -> None:
        raise AssertionError("resolve_occasion_time (search path) must not run with --video-id")

    def _sleep_boom(*a: object) -> None:
        raise AssertionError("must not poll/sleep when the pin resolves on the first attempt")

    monkeypatch.setattr(cli_mod, "resolve_occasion_time", _search_boom)
    monkeypatch.setattr(cli_mod.time, "sleep", _sleep_boom)
    note, sent = _mute_notifier()

    out = cli_mod._resolve_occasion_time_loop(
        occ, "E-30JAN01", video_id="PINNED1", poll_min=15.0, not_before_ts=None, note=note)

    assert out is resolved
    assert sent == []


def test_loop_video_id_retries_the_same_pin_not_search(monkeypatch):
    occ = _day_only_occ("E-30JAN01")
    resolved = replace(occ, occurrence_ts=NOW - 100, time_known=True)
    attempts = {"n": 0}

    def fake_resolve_video(o: UpcomingOccasion, vid: str, **kw: object) -> UpcomingOccasion | None:
        attempts["n"] += 1
        assert vid == "PINNED1"
        return resolved if attempts["n"] == 3 else None

    def _search_boom(*a: object, **kw: object) -> None:
        raise AssertionError("resolve_occasion_time (search path) must not run with --video-id")

    sleeps: list[float] = []
    monkeypatch.setattr(cli_mod, "resolve_occasion_time_from_video", fake_resolve_video)
    monkeypatch.setattr(cli_mod, "resolve_occasion_time", _search_boom)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda s: sleeps.append(s))
    note, sent = _mute_notifier()

    out = cli_mod._resolve_occasion_time_loop(
        occ, "E-30JAN01", video_id="PINNED1", poll_min=1.0, not_before_ts=None, note=note)

    assert out is resolved
    assert attempts["n"] == 3
    assert sleeps == [60.0, 60.0]  # two failed attempts, polled at poll_min cadence


def test_loop_forwards_not_before_to_the_pinned_video_resolver(monkeypatch):
    occ = _day_only_occ("E-30JAN01")
    resolved = replace(occ, occurrence_ts=NOW + 500, time_known=True)
    seen: dict[str, object] = {}

    def fake_resolve_video(o: UpcomingOccasion, vid: str, *,
                           not_before_ts: int | None = None) -> UpcomingOccasion:
        seen["not_before_ts"] = not_before_ts
        return resolved

    monkeypatch.setattr(cli_mod, "resolve_occasion_time_from_video", fake_resolve_video)
    note, _sent = _mute_notifier()

    cli_mod._resolve_occasion_time_loop(
        occ, "E-30JAN01", video_id="PINNED1", poll_min=15.0, not_before_ts=12345, note=note)

    assert seen["not_before_ts"] == 12345


def test_loop_without_video_id_uses_general_search(monkeypatch):
    occ = _day_only_occ("E-30JAN01")
    resolved = replace(occ, occurrence_ts=NOW - 100, time_known=True)
    seen: dict[str, object] = {}

    def fake_resolve(o: UpcomingOccasion, *, not_before_ts: int | None = None) -> UpcomingOccasion:
        seen["called"] = True
        return resolved

    monkeypatch.setattr(cli_mod, "resolve_occasion_time", fake_resolve)

    def _pin_boom(*a: object, **kw: object) -> None:
        raise AssertionError("resolve_occasion_time_from_video must not run without --video-id")

    monkeypatch.setattr(cli_mod, "resolve_occasion_time_from_video", _pin_boom)
    note, _sent = _mute_notifier()

    out = cli_mod._resolve_occasion_time_loop(
        occ, "E-30JAN01", video_id=None, poll_min=15.0, not_before_ts=None, note=note)

    assert out is resolved
    assert seen.get("called") is True


# --- day-boundary exit (unchanged behavior; guards the refactor) ----------------------------------


def test_loop_exits_when_ticker_day_has_ended(monkeypatch):
    occ = _day_only_occ("E-20JAN01", occurrence_ts=1_577_836_800)  # 2020-01-01: long past

    monkeypatch.setattr(cli_mod, "resolve_occasion_time", lambda *a, **kw: None)

    def _sleep_boom(*a: object) -> None:
        raise AssertionError("must not sleep once the ticker's day has already ended")

    monkeypatch.setattr(cli_mod.time, "sleep", _sleep_boom)
    note, _sent = _mute_notifier()

    with pytest.raises(typer.Exit):
        cli_mod._resolve_occasion_time_loop(
            occ, "E-20JAN01", video_id=None, poll_min=15.0, not_before_ts=None, note=note)


# --- failed-resolution paging ---------------------------------------------------------------------


def test_loop_pages_once_after_three_consecutive_failures(monkeypatch):
    occ = _day_only_occ("E-30JAN01")
    resolved = replace(occ, occurrence_ts=NOW, time_known=True)
    attempts = {"n": 0}

    def fake_resolve(o: UpcomingOccasion, **kw: object) -> UpcomingOccasion | None:
        attempts["n"] += 1
        return resolved if attempts["n"] == 5 else None

    monkeypatch.setattr(cli_mod, "resolve_occasion_time", fake_resolve)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda *a: None)
    note, sent = _mute_notifier()

    out = cli_mod._resolve_occasion_time_loop(
        occ, "E-30JAN01", video_id=None, poll_min=15.0, not_before_ts=None, note=note)

    assert out is resolved
    assert attempts["n"] == 5
    health = [(ev, msg) for ev, msg in sent if ev == Event.HEALTH]
    assert len(health) == 1  # pages exactly once (at fail_streak==3), not again at 4
    assert "E-30JAN01" in health[0][1]
    assert "3 consecutive" in health[0][1]


def test_loop_never_pages_before_three_failures(monkeypatch):
    occ = _day_only_occ("E-30JAN01")
    resolved = replace(occ, occurrence_ts=NOW, time_known=True)
    attempts = {"n": 0}

    def fake_resolve(o: UpcomingOccasion, **kw: object) -> UpcomingOccasion | None:
        attempts["n"] += 1
        return resolved if attempts["n"] == 2 else None

    monkeypatch.setattr(cli_mod, "resolve_occasion_time", fake_resolve)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda *a: None)
    note, sent = _mute_notifier()

    cli_mod._resolve_occasion_time_loop(
        occ, "E-30JAN01", video_id=None, poll_min=15.0, not_before_ts=None, note=note)

    assert sent == []  # only 1 failed attempt before resolving — below the page threshold


def test_loop_repeats_the_page_at_the_30_min_cadence(monkeypatch):
    """A fake clock lets the loop run past the (real) 30-min page interval deterministically: the
    ticker's day-end guard is what eventually terminates it, so start close enough to that bound
    to observe >1 page within the run."""
    day_end = ticker_day_bounds(date(2030, 1, 1))[1]
    clock = {"t": float(day_end - 7200)}  # 2h before the ticker's day ends

    def fake_time() -> float:
        return clock["t"]

    def fake_resolve(o: UpcomingOccasion, **kw: object) -> None:
        clock["t"] += 600  # 10 simulated minutes per failed attempt
        return None

    occ = _day_only_occ("E-30JAN01", occurrence_ts=int(clock["t"]))
    monkeypatch.setattr(cli_mod, "resolve_occasion_time", fake_resolve)
    monkeypatch.setattr(cli_mod.time, "time", fake_time)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda *a: None)
    note, sent = _mute_notifier()

    with pytest.raises(typer.Exit):
        cli_mod._resolve_occasion_time_loop(
            occ, "E-30JAN01", video_id=None, poll_min=15.0, not_before_ts=None, note=note)

    health = [msg for ev, msg in sent if ev == Event.HEALTH]
    assert len(health) == 3  # fail_streak 3, 6, 9 (30 simulated min apart) before day-end cuts it
