"""`pmlab probe ...` — the live speech/broadcast capture CLI.

Read-only market data + local ASR; $0 API; ZERO order endpoints; record-only (no trading
decisions are made anywhere in this package). Subcommands:
* ``sweep``   — list upcoming armable speech occasions (live registry).
* ``next``    — the next occasion + when it auto-arms.
* ``run``     — capture one occasion now (blocks through the speech; auto-stops).
* ``run-today`` — capture a named event today, resolving its stream if needed, then blocking.
* ``record-game`` — record-only broadcast capture (WS book + a named audio device).
* ``daemon``  — the fully-automatic loop: sweep → wait to T−30 → capture → repeat.
* ``doctor``  — measure local Whisper throughput + print the first-arm host choice.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace

import typer
from rich.console import Console
from rich.table import Table

from pmlab.config import get_settings
from pmlab.probe.notify import Event, Notifier
from pmlab.probe.occasions import (
    DEFAULT_WITHIN_S,
    UpcomingOccasion,
    parse_not_before,
    parse_ticker_date,
    resolve_occasion_time,
    resolve_occasion_time_from_video,
    sweep_upcoming,
    ticker_day_bounds,
)
from pmlab.probe.run import ProbeConfig, host_prompt, run_occasion, run_with_relaunch

# Page once the stream-resolution poll loop has failed this many consecutive attempts, then at
# most this often after — a long run of silent failed polls can otherwise burn an entire capture
# window with zero notification.
RESOLVE_FAIL_PAGE_THRESHOLD = 3
RESOLVE_FAIL_PAGE_INTERVAL_S = 30 * 60

log = logging.getLogger("pmlab.probe.cli")
console = Console()
probe_app = typer.Typer(help="Live speech/broadcast capture probe — record-only, $0, no orders")


def _fmt_ts(ts: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def _print_occasions(occasions: list[UpcomingOccasion]) -> None:
    table = Table(title="Upcoming armable speech occasions")
    for col in ("event", "speaker", "type", "scheduled", "arms at", "eligible mkts"):
        table.add_column(col)
    for o in occasions:
        table.add_row(o.event_ticker, o.speaker, o.event_type, _fmt_ts(o.occurrence_ts),
                      _fmt_ts(o.arm_at()), str(len(o.markets)))
    console.print(table)


@probe_app.command()
def sweep(within_hours: float = typer.Option(6.0, help="horizon to look ahead")) -> None:
    """List upcoming speech occasions with eligible markets (fresh live registry)."""
    occ = sweep_upcoming(now_ts=int(time.time()), within_s=int(within_hours * 3600))
    if not occ:
        console.print("[yellow]no upcoming armable speech occasions in the horizon[/yellow]")
        return
    _print_occasions(occ)


@probe_app.command()
def next(within_hours: float = typer.Option(72.0)) -> None:  # noqa: A001 - typer command name
    """Show the next armable occasion and when it auto-arms."""
    occ = sweep_upcoming(now_ts=int(time.time()), within_s=int(within_hours * 3600))
    if not occ:
        console.print("[yellow]nothing upcoming[/yellow]")
        raise typer.Exit(0)
    o = occ[0]
    console.print(f"[bold]{o.event_ticker}[/bold] {o.speaker} ({o.event_type}) — "
                  f"scheduled {_fmt_ts(o.occurrence_ts)}, arms {_fmt_ts(o.arm_at())}, "
                  f"{len(o.markets)} eligible markets")


def _resolve_host() -> str:
    host = get_settings().probe_host
    if not host:
        console.print("[yellow]PMLAB_PROBE_HOST unset — run `pmlab probe doctor` to choose a host "
                      "(Mac launchd vs VPS); defaulting to 'manual' for this run.[/yellow]")
        return "manual"
    return host


@probe_app.command()
def run(event_ticker: str = typer.Argument(..., help="occasion to capture now")) -> None:
    """Capture one occasion end-to-end now (blocks until the speech ends / auto-stop)."""
    occ = [o for o in sweep_upcoming(now_ts=int(time.time()), within_s=DEFAULT_WITHIN_S * 4)
           if o.event_ticker == event_ticker]
    if not occ:
        console.print(f"[red]{event_ticker} not found in the upcoming sweep[/red]")
        raise typer.Exit(1)
    summary = run_occasion(occ[0], host=_resolve_host())
    console.print(json.loads(summary.to_json()))


def _resolve_occasion_time_loop(
    occ: UpcomingOccasion, event_ticker: str, *, video_id: str | None, poll_min: float,
    not_before_ts: int | None, note: Notifier,
) -> UpcomingOccasion:
    """The ``run-today`` stream-resolution poll loop, pulled out of the command for direct unit
    testing. An operator ``--video-id`` pin resolves the occurrence time FROM ITSELF (never the
    general speaker-name search — a pin blocked on the general search can stall for a long time
    while the pinned stream sits live the whole time). After 3 consecutive failed resolution
    attempts, page once; repeat at most every ``RESOLVE_FAIL_PAGE_INTERVAL_S`` after that —
    unattended, this loop would otherwise fail silently for the entire day."""
    fail_streak = 0
    last_page_ts = 0.0
    while not occ.time_known:
        resolved = (
            resolve_occasion_time_from_video(occ, video_id, not_before_ts=not_before_ts)
            if video_id is not None
            else resolve_occasion_time(occ, not_before_ts=not_before_ts)
        )
        if resolved is not None:
            return resolved
        fail_streak += 1
        tdate = parse_ticker_date(event_ticker)
        if tdate and int(time.time()) >= ticker_day_bounds(tdate)[1]:
            console.print(f"[red]{event_ticker}: its day ended with no findable stream[/red]")
            raise typer.Exit(1)
        if (fail_streak >= RESOLVE_FAIL_PAGE_THRESHOLD
                and time.time() - last_page_ts >= RESOLVE_FAIL_PAGE_INTERVAL_S):
            note.notify(Event.HEALTH, (
                f"{event_ticker}: stream resolution has failed {fail_streak} consecutive "
                f"attempts — still polling every {poll_min:.0f} min"))
            last_page_ts = time.time()
        console.print(f"no live/scheduled stream for {occ.speaker} yet — "
                      f"retrying in {poll_min:.0f} min")
        time.sleep(poll_min * 60)
    return occ


@probe_app.command("run-today")
def run_today(
    event_ticker: str = typer.Argument(..., help="event to capture today (ticker date = truth)"),
    poll_min: float = typer.Option(15.0, help="stream-schedule poll cadence while waiting"),
    now: bool = typer.Option(False, "--now", help="skip stream polling; capture immediately"),
    not_before: str | None = typer.Option(
        None, "--not-before",
        help="reject streams starting before this time (HH:MM local or ISO) — pins a "
             "multi-appearance day to the named event's slot"),
    quiet_min: float | None = typer.Option(
        None, "--quiet-min",
        help="override the transcript-silence auto-stop (minutes) — widen for long-program "
             "events where music/announcer gaps precede the speech"),
    max_capture_hours: float | None = typer.Option(
        None, "--max-capture-hours",
        help="override the hard capture cap (hours after occurrence start)"),
    video_id: str | None = typer.Option(
        None, "--video-id",
        help="pin the audio stream to this YouTube video id (skips stream resolution — for "
             "multi-stream nights where the nearest-start heuristic grabs the wrong feed)"),
    max_relaunches: int = typer.Option(
        3, "--max-relaunches",
        help="bounded relaunches on a quiet-rail stop with a small transcript (late-start "
             "programs) — carries the original occurrence_ts across attempts"),
) -> None:
    """Capture EVENT today even when Kalshi's occurrence_datetime is broken (the ticker date owns
    the DAY): occurrence_ts comes from the stream's scheduled/live start (fallback: now), then a
    watched-mode capture. Blocks until the speech ends — run under nohup/launchd for the day."""
    start_ts = int(time.time())
    not_before_ts = parse_not_before(not_before) if not_before else None
    cfg = ProbeConfig()
    if quiet_min is not None:
        cfg = replace(cfg, quiet_s=int(quiet_min * 60))
    if max_capture_hours is not None:
        cfg = replace(cfg, max_capture_s=int(max_capture_hours * 3600))
    cands = [o for o in sweep_upcoming(now_ts=start_ts, within_s=24 * 3600)
             if o.event_ticker == event_ticker]
    if not cands:
        console.print(f"[red]{event_ticker} not found among today's armable occasions[/red]")
        raise typer.Exit(1)
    occ = cands[0]
    if now:
        occ = replace(occ, occurrence_ts=int(time.time()), time_known=True)
    if not_before_ts is not None:
        console.print(f"stream floor: rejecting starts before {_fmt_ts(not_before_ts)}")
    s = get_settings()
    note = Notifier(topic=s.ntfy_topic, base=s.ntfy_base, dry=not s.ntfy_topic)
    occ = _resolve_occasion_time_loop(occ, event_ticker, video_id=video_id, poll_min=poll_min,
                                      not_before_ts=not_before_ts, note=note)
    console.print(f"[bold]{occ.event_ticker}[/bold] {occ.speaker} — occurrence "
                  f"{_fmt_ts(occ.occurrence_ts)}, arms {_fmt_ts(occ.arm_at())}, "
                  f"{len(occ.markets)} eligible markets")
    while True:
        remaining = occ.arm_at(lead_s=cfg.lead_s) - int(time.time())
        if remaining <= 0:
            break
        console.print(f"arms in {remaining / 60:.0f} min")
        time.sleep(min(remaining, 15 * 60))
    summary = run_with_relaunch(occ, host=_resolve_host(), cfg=cfg, video_id=video_id,
                                max_relaunches=max_relaunches)
    console.print(json.loads(summary.to_json()))


@probe_app.command("record-game")
def record_game(
    event_ticker: str = typer.Argument(..., help="event to capture (an event whose markets "
                                                  "reference a broadcast, e.g. a sports match)"),
    audio_device: str = typer.Option(
        ..., "--audio-device", help="avfoundation loopback device name, e.g. 'BlackHole 2ch' — "
                                    "a loopback of the operator's own broadcast stream, mixed via "
                                    "a Multi-Output Device"),
    kickoff: str | None = typer.Option(
        None, "--kickoff", help="start time (HH:MM local or ISO); default now. Some series carry "
                                "a series-window deadline as their registry occurrence_ts, NOT "
                                "the actual start time — this mode always stamps the real start "
                                "itself, from this flag or from 'now'"),
    max_capture_hours: float = typer.Option(3.5, "--max-capture-hours"),
    quiet_min: float = typer.Option(10.0, "--quiet-min",
                                    help="auto-stop after this many minutes of transcript silence"),
    max_relaunches: int = typer.Option(3, "--max-relaunches"),
) -> None:
    """Record-only broadcast capture: tapes the Kalshi WS orderbook + a named avfoundation audio
    device into local Whisper. NO decisions, NO signals, NO orders — and NO stream resolution
    (audio comes ONLY from ``--audio-device``; never yt-dlp/YouTube search, which would risk
    taping the wrong broadcast). Artifacts land under ``data/sports/<event_ticker>/``."""
    from pmlab.probe.occasions import lookup_event
    from pmlab.probe.record_game import RecordConfig, run_record_with_relaunch

    occ = lookup_event(event_ticker)
    if occ is None:
        console.print(f"[red]{event_ticker}: no eligible/unresolved markets found[/red]")
        raise typer.Exit(1)
    cfg = RecordConfig(max_capture_s=int(max_capture_hours * 3600), quiet_s=int(quiet_min * 60))
    kickoff_ts = parse_not_before(kickoff) if kickoff else int(time.time())
    occ = replace(occ, occurrence_ts=kickoff_ts, time_known=True)
    console.print(f"[bold]{occ.event_ticker}[/bold] {occ.speaker} — record-only, "
                  f"device={audio_device}, start {_fmt_ts(occ.occurrence_ts)}, "
                  f"{len(occ.markets)} eligible markets")
    # A future --kickoff waits here (like run-today's arm loop) so the join-verify sample fires
    # AT the actual start, when the operator's stream is actually playing — not against pregame
    # silence on the loopback device.
    while True:
        remaining = kickoff_ts - int(time.time())
        if remaining <= 0:
            break
        console.print(f"starts in {remaining / 60:.0f} min")
        time.sleep(min(remaining, 5 * 60))
    summary = run_record_with_relaunch(occ, device=audio_device, cfg=cfg, host=_resolve_host(),
                                       max_relaunches=max_relaunches)
    console.print(json.loads(summary.to_json()))


@probe_app.command()
def daemon(
    within_hours: float = typer.Option(6.0),
    poll_min: float = typer.Option(30.0, help="re-sweep cadence when idle"),
) -> None:
    """Fully-automatic loop: sweep → sleep until the next occasion's T−30 → capture → repeat."""
    host = _resolve_host()
    cfg = ProbeConfig()
    console.print("[green]probe daemon started[/green] (read-only, $0, no orders)")
    while True:
        occ = sweep_upcoming(now_ts=int(time.time()), within_s=int(within_hours * 3600))
        nexto = occ[0] if occ else None
        if nexto is None:
            time.sleep(poll_min * 60)
            continue
        if not nexto.time_known:
            # Day-known occasion (broken occurrence_ts, ticker-date override): from the noon
            # anchor's arm time onward, poll the stream search for the true start; before that
            # (or while YouTube shows nothing schedulable) just re-sweep later.
            resolved = (resolve_occasion_time(nexto)
                        if int(time.time()) >= nexto.arm_at(lead_s=cfg.lead_s) else None)
            if resolved is None:
                time.sleep(poll_min * 60)
                continue
            nexto = resolved
        wait = nexto.arm_at(lead_s=cfg.lead_s) - int(time.time())
        if wait > 0:
            log.info("next: %s arms in %.0f min", nexto.event_ticker, wait / 60)
            time.sleep(min(wait, poll_min * 60))
            continue
        run_with_relaunch(nexto, host=host)  # blocks through the occasion, then re-sweeps


@probe_app.command()
def doctor() -> None:
    """Measure local Whisper throughput (RTF) and print the first-arm host choice."""
    import numpy as np

    s = get_settings()
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        console.print("[yellow]faster-whisper not installed — run `uv sync --extra probe`. "
                      "It is required for $0 live ASR.[/yellow]")
        console.print(host_prompt(float("nan")))
        return
    dur_s = 15.0
    audio = (0.01 * np.random.randn(int(dur_s * 16_000))).astype("float32")
    model = WhisperModel(s.whisper_model, device="cpu", compute_type="int8")
    t0 = time.time()
    list(model.transcribe(audio, language="en", word_timestamps=True)[0])
    rtf = (time.time() - t0) / dur_s
    console.print(f"whisper '{s.whisper_model}' RTF={rtf:.2f} (on {dur_s:.0f}s CPU audio)")
    console.print(host_prompt(rtf))
