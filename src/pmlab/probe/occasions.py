"""Occasion sweep + auto-arm.

Finds UPCOMING speech occasions with eligible markets and resolves a live-stream URL, so a host
can auto-arm ahead of the scheduled time with zero manual steps. The eligibility predicate and
registry assembly reuse the mention-market pipeline (``mentions/``); this module only *selects*
upcoming rows and builds the event→YouTube-query glue.

Pure selection/query/arming helpers are unit-tested; the live registry build + video resolution wrap
existing modules and hit the network.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

from pmlab.config import Settings, get_settings

log = logging.getLogger("pmlab.probe.occasions")

SPEECH_FORMAT = "speech"
DEFAULT_LEAD_S = 1800          # arm 30 min before the scheduled occasion (§3)
DEFAULT_WITHIN_S = 6 * 3600    # sweep horizon: occasions in the next 6 h
ET = ZoneInfo("America/New_York")   # Kalshi's civil day, for ticker-encoded event dates
TICKER_DISAGREE_S = 48 * 3600  # occurrence_ts vs ticker-day gap beyond which the ticker wins
DAY_ONLY_MAX_GAP_S = 16 * 3600  # stream relevance from a noon anchor: covers the civil day
TIMED_MAX_GAP_S = 6 * 3600     # stream relevance when the occasion time is trusted

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
_DATE_SEG = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})[A-Z]*$")


def parse_ticker_date(ticker: str) -> date | None:
    """Event date encoded in a ticker's dash-suffix (e.g. ``…-26JUL04`` → 2026-07-04; market
    forms like ``…-26APR29C`` carry a trailing variant letter; sports events append team codes,
    e.g. ``…-26JUN14CARVGK``). Kalshi's ``occurrence_datetime`` ships a broken far-future default
    on some mention events, so the ticker date is the more reliable truth. Rightmost parseable
    segment wins."""
    for seg in reversed(str(ticker).upper().split("-")):
        m = _DATE_SEG.match(seg.strip())
        if not m:
            continue
        yy, mon, dd = m.groups()
        month = _MONTHS.get(mon)
        if month is None:
            continue
        try:
            return date(2000 + int(yy), month, int(dd))
        except ValueError:  # a date-shaped segment with an impossible day IS the (typo'd) date
            return None
    return None


def ticker_day_bounds(d: date) -> tuple[int, int]:
    """``[start, end)`` unix bounds of the ticker's civil day in ET."""
    start = datetime.combine(d, dtime(0, 0), tzinfo=ET)
    return int(start.timestamp()), int((start + timedelta(days=1)).timestamp())


def ticker_day_anchor_ts(d: date) -> int:
    """Noon-ET placeholder for a day-known/time-unknown occasion (≤12 h from any same-day true
    start; :func:`resolve_occasion_time` swaps in the stream's scheduled start)."""
    return int(datetime.combine(d, dtime(12, 0), tzinfo=ET).timestamp())


@dataclass(frozen=True)
class EligibleMarket:
    market_ticker: str
    yes_sub_title: str


@dataclass(frozen=True)
class UpcomingOccasion:
    """An armable speech occasion: its event, speaker, scheduled time, and eligible markets."""

    event_ticker: str
    speaker: str
    event_type: str
    occurrence_ts: int
    markets: list[EligibleMarket]
    time_known: bool = True  # False → occurrence_ts is a noon-ET day anchor from the ticker date

    def arm_at(self, *, lead_s: int = DEFAULT_LEAD_S) -> int:
        return self.occurrence_ts - lead_s


def _event_type_hint(series_title: str) -> str:
    """Coarse event_type hint for a length-prior model. A downstream model can back off to a
    global length pool for an unknown type, so this only needs to be stable, not exhaustive."""
    t = (series_title or "").lower()
    for kind in ("rally", "presser", "press conference", "debate", "inaugural", "dinner",
                 "ceremony", "meeting", "address"):
        if kind in t:
            return "presser" if "press" in kind else kind
    return "remarks"


def _unresolved(result: pd.Series) -> pd.Series:
    """True where the market has not settled. Open Kalshi markets carry ``result == ""`` (an empty
    string), NOT NaN — so a bare ``.isna()`` wrongly drops every live market. Treat blank-or-null
    as unsettled; a settled market has ``result`` in the free-form code set (``yes``/``no``/…)."""
    r = result.astype("string")
    return r.isna() | (r.str.strip() == "")


def select_occasions(
    registry: pd.DataFrame, *, now_ts: int, within_s: int = DEFAULT_WITHIN_S,
) -> list[UpcomingOccasion]:
    """Upcoming speech occasions with ≥1 eligible, unresolved market whose scheduled time is in
    ``(now_ts, now_ts + within_s]``. Pure over a registry-shaped frame."""
    df = registry
    when = df["occurrence_ts"]
    if "close_ts" in df:
        when = when.fillna(df["close_ts"])
    when = when.astype("float64")
    # Kalshi's occurrence_datetime is untrustworthy on mention events (a broken far-future default
    # has been observed, which would put an occasion outside every sweep window). When the ticker
    # encodes the event day and disagrees with occurrence_ts by >48 h (or the ts is missing),
    # trust the ticker DAY: anchor at noon ET, mark time-unknown, and let the stream search
    # resolve the actual start (:func:`resolve_occasion_time`).
    tdate = df["event_ticker"].map(parse_ticker_date)
    anchor = tdate.map(lambda d: float(ticker_day_anchor_ts(d)) if d else float("nan"))
    day_only = anchor.notna() & (when.isna() | ((when - anchor).abs() > TICKER_DISAGREE_S))
    when_eff = when.where(~day_only, anchor)
    day_start = tdate.map(lambda d: float(ticker_day_bounds(d)[0]) if d else float("nan"))
    day_end = tdate.map(lambda d: float(ticker_day_bounds(d)[1]) if d else float("nan"))
    # Only genuine single-stream speeches: template=="speech". The `duration` template
    # ("say X before <date>") is also format=="speech"+eligible and carries an occurrence_ts, but
    # it is an open-window bet with no canonical video to capture — exclude it (see the Explore of
    # mentions/resolution.py). Guarded so a frame without the column (minimal tests) still works.
    is_speech = (
        df["template"] == "speech" if "template" in df.columns
        else pd.Series(True, index=df.index)
    )
    base = (
        (df["format"] == SPEECH_FORMAT)
        & is_speech
        & df["eligible"].astype(bool)
        & _unresolved(df["result"])                 # not yet settled (blank/NaN, not a code)
    )
    # Timed rows keep the strict (now, now+within] window; day-only rows are armable whenever
    # their ET civil day overlaps it (the true start could be anywhere in the day).
    in_window = (
        (~day_only & when_eff.notna() & (when_eff > now_ts) & (when_eff <= now_ts + within_s))
        | (day_only & (day_end > now_ts) & (day_start <= now_ts + within_s))
    )
    sel = df[base & in_window]
    out: list[UpcomingOccasion] = []
    for event_ticker, g in sel.groupby("event_ticker", sort=True):
        markets = [
            EligibleMarket(str(r.market_ticker), str(r.yes_sub_title))
            for r in g.itertuples(index=False)
        ]
        title = str(g["series_title"].iloc[0]) if "series_title" in g else ""
        timed = when_eff.loc[g.index][~day_only.loc[g.index]]
        occ_when = timed if len(timed) else when_eff.loc[g.index]
        out.append(UpcomingOccasion(
            event_ticker=str(event_ticker),
            speaker=str(g["speaker"].iloc[0]),
            event_type=_event_type_hint(title),
            occurrence_ts=int(occ_when.min()),
            markets=markets,
            time_known=bool(len(timed)),
        ))
    out.sort(key=lambda o: o.occurrence_ts)
    return out


def select_event(registry: pd.DataFrame, event_ticker: str) -> UpcomingOccasion | None:
    """Generic event capture selection (record-only mode): every eligible, unresolved row of ONE
    named ``event_ticker`` — deliberately NO format/template/speech filter and NO time window.
    Some series carry a non-kickoff/non-occurrence deadline as ``occurrence_ts`` (e.g. a
    tournament-window close rather than a specific match time), so there is no scheduled-time
    window to check here; the caller (record-only CLI) stamps the real occurrence time itself."""
    df = registry
    sel = df[
        (df["event_ticker"] == event_ticker)
        & df["eligible"].astype(bool)
        & _unresolved(df["result"])
    ]
    if sel.empty:
        return None
    markets = [
        EligibleMarket(str(r.market_ticker), str(r.yes_sub_title))
        for r in sel.itertuples(index=False)
    ]
    speaker = event_ticker
    if "speaker" in sel.columns:
        spk = sel["speaker"].iloc[0]
        if pd.notna(spk) and str(spk).strip():
            speaker = str(spk)
    when = sel["occurrence_ts"]
    if "close_ts" in sel:
        when = when.fillna(sel["close_ts"])
    when = when.astype("float64")
    occurrence_ts = int(when.min()) if when.notna().any() else 0
    return UpcomingOccasion(
        event_ticker=event_ticker, speaker=speaker, event_type="sports",
        occurrence_ts=occurrence_ts, markets=markets, time_known=False,
    )


def speech_query(speaker: str, *, event_type: str = "") -> str:
    """YouTube search query for a speaker's live speech (the event→query glue). ``search_speech_
    videos`` ranks live-with-release-timestamp first, so 'live' biases to the streamable feed."""
    parts = [speaker.strip()]
    if event_type and event_type not in ("remarks",):
        parts.append(event_type)
    parts.append("speech live")
    return " ".join(p for p in parts if p)


# --- live wiring (network) -----------------------------------------------------------------------


def build_live_registry(
    settings: Settings | None = None, *, statuses: Sequence[str] = ("open",),
) -> pd.DataFrame:
    """Fresh registry over currently-open events (the M1 pipeline, live). Reused verbatim:
    ``sweep → parse_raw_records → build_registry`` — same eligibility the study froze."""
    from pmlab.mentions.discover import sweep
    from pmlab.mentions.registry import build_registry
    from pmlab.mentions.resolution import parse_raw_records
    from pmlab.venues.kalshi import KalshiPublic

    kalshi = KalshiPublic(settings=settings or get_settings())
    catalog, raw, _stats = sweep(kalshi, statuses)
    resolutions = parse_raw_records(raw)
    return build_registry(catalog, resolutions)


def sweep_upcoming(
    settings: Settings | None = None, *, now_ts: int, within_s: int = DEFAULT_WITHIN_S,
) -> list[UpcomingOccasion]:
    """Live sweep → armable occasions (build a fresh open-event registry, then select)."""
    return select_occasions(build_live_registry(settings), now_ts=now_ts, within_s=within_s)


def lookup_event(event_ticker: str, settings: Settings | None = None) -> UpcomingOccasion | None:
    """Live wrapper for :func:`select_event` (record-only sports capture): fresh open-event
    registry → the one named event's eligible/unresolved markets, or ``None``."""
    return select_event(build_live_registry(settings), event_ticker)


@dataclass(frozen=True)
class LiveStream:
    """One YouTube candidate for a live occasion. ``release_ts`` is the actual (is_live) or
    scheduled (upcoming) stream start — the audio clock's anchor and the relevance filter."""

    video_id: str
    title: str
    channel: str
    is_live: bool
    was_live: bool
    release_ts: int | None


def _parse_live_rows(stdout: str) -> list[LiveStream]:
    """Parse ``yt-dlp --print`` rows (id, is_live, was_live, release_timestamp, channel, title).
    yt-dlp prints ``NA`` for missing fields; live streams have NO duration, which is why the
    study's ``search_speech_videos`` (duration-filtered, archives-first) cannot be reused here."""
    out: list[LiveStream] = []
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        vid, is_l, was_l, rel, channel = (p.strip() for p in parts[:5])
        title = "\t".join(parts[5:]).strip()
        if not vid or vid.upper() == "NA":
            continue
        try:
            rel_ts: int | None = int(float(rel))
        except ValueError:
            rel_ts = None
        out.append(LiveStream(
            video_id=vid, title=title, channel=channel,
            is_live=is_l.lower() == "true", was_live=was_l.lower() == "true", release_ts=rel_ts,
        ))
    return out


def search_live_streams(query: str, *, n: int = 8) -> list[LiveStream]:
    """YouTube candidates with live-ness metadata (network; parsing is :func:`_parse_live_rows`)."""
    from pmlab.signals.transcripts import _yt_dlp

    fields = "%(id)s\t%(is_live)s\t%(was_live)s\t%(release_timestamp)s\t%(channel)s\t%(title)s"
    r = _yt_dlp([
        "--skip-download", "--no-warnings", "--socket-timeout", "30",
        "--print", fields, f"ytsearch{n}:{query}",
    ])
    return _parse_live_rows(r.stdout)


def video_metadata(video_id: str) -> LiveStream | None:
    """Live-ness metadata for ONE video, addressed directly by id (network; same fields/parsing as
    :func:`search_live_streams`) — the operator-pin path (:func:`resolve_occasion_time_from_video`)
    never searches, it just asks yt-dlp about the pinned video."""
    from pmlab.signals.transcripts import _yt_dlp

    fields = "%(id)s\t%(is_live)s\t%(was_live)s\t%(release_timestamp)s\t%(channel)s\t%(title)s"
    r = _yt_dlp([
        "--skip-download", "--no-warnings", "--socket-timeout", "30",
        "--print", fields, f"https://www.youtube.com/watch?v={video_id}",
    ])
    rows = _parse_live_rows(r.stdout)
    return rows[0] if rows else None


def pick_stream(
    cands: list[LiveStream], *, occurrence_ts: int, max_gap_s: int = 6 * 3600,
) -> LiveStream | None:
    """The capture-safe choice: a CURRENTLY-LIVE stream whose start is near the occasion, else an
    upcoming scheduled one in the window. Archived videos (``was_live``) and release-less
    candidates are NEVER picked — old tape or a 24/7 channel (release weeks ago) transcribed into
    the live decision loop is strictly worse than no capture (the runner crash-notifies instead)."""
    def gap(c: LiveStream) -> int:
        return abs((c.release_ts or 0) - occurrence_ts)

    ok = [c for c in cands if not c.was_live and c.release_ts is not None and gap(c) <= max_gap_s]
    live = sorted((c for c in ok if c.is_live), key=gap)
    if live:
        return live[0]
    upcoming = sorted((c for c in ok if not c.is_live), key=gap)
    return upcoming[0] if upcoming else None


def resolve_stream(occasion: UpcomingOccasion, *, n: int = 8) -> LiveStream | None:
    """Best live/scheduled stream for an occasion (None → the runner crash-notifies). Day-only
    occasions widen the relevance window from the noon anchor to span their civil day.

    The speaker-only query is ADDITIVELY enriched with the Kalshi milestone's own title when one
    is linked (:func:`kalshi_milestone_title`), non-fatal on failure: broadcaster titles often
    share far more words with the venue's own event description than with a bare "<speaker>
    speech live" query, and relying on the bare query alone can burn many minutes of failed polls
    against a stream that was live the whole time."""
    query = speech_query(occasion.speaker, event_type=occasion.event_type)
    cands = search_live_streams(query, n=n)
    title = kalshi_milestone_title(occasion.event_ticker)
    if title:
        cands = cands + search_live_streams(f"{occasion.speaker} {title}", n=n)
    max_gap = TIMED_MAX_GAP_S if occasion.time_known else DAY_ONLY_MAX_GAP_S
    return pick_stream(cands, occurrence_ts=occasion.occurrence_ts, max_gap_s=max_gap)


# --- Kalshi-linked stream — the authoritative feed when the venue links one -----------------------
#
# The kalshi.com event page hydrates its "milestone" with ``product_details.youtube_link`` via the
# unauthenticated v1 ``/bff/cards`` endpoint. When present this beats any search heuristic: it is
# the feed the market's own resolution graders watch, so a wrong-stream capture is far less likely
# than trusting a name-based search alone. Fallback chain in the capture: operator pin →
# kalshi-linked → scored search.

KALSHI_V1_BASE = "https://api.elections.kalshi.com/v1"
_YT_ID = re.compile(r"(?:v=|youtu\.be/|/live/|/embed/|/shorts/)([A-Za-z0-9_-]{11})")


def parse_youtube_id(url: str) -> str | None:
    """Video id from any common YouTube URL form (watch?v=, youtu.be/, /live/, /embed/)."""
    m = _YT_ID.search(url or "")
    return m.group(1) if m else None


def _matching_milestones(payload: dict[str, object], event_ticker: str) -> list[dict[str, object]]:
    """Milestone dicts in a v1 ``/bff/cards`` payload that reference ``event_ticker`` — via its
    card's ``milestone_id`` or the milestone's own related/primary ticker lists (pure; shared by
    :func:`parse_linked_stream` and :func:`parse_milestone_title`)."""
    hydrated = payload.get("hydrated_data")
    milestones = hydrated.get("milestones") if isinstance(hydrated, dict) else None
    if not isinstance(milestones, dict):
        return []
    cards = payload.get("cards")
    card_mids = {c.get("milestone_id") for c in cards
                 if isinstance(c, dict) and c.get("event_ticker") == event_ticker
                 } if isinstance(cards, list) else set()
    out: list[dict[str, object]] = []
    for mid, m in milestones.items():
        if not isinstance(m, dict):
            continue
        related = set(m.get("related_event_tickers") or []) | set(
            m.get("primary_event_tickers") or [])
        if mid not in card_mids and event_ticker not in related:
            continue
        out.append(m)
    return out


def parse_linked_stream(payload: dict[str, object], event_ticker: str) -> str | None:
    """Extract the milestone-linked YouTube id for ``event_ticker`` from a v1 ``/bff/cards``
    payload (pure). A milestone counts if the event's card points at it or it lists the event
    among its related/primary tickers."""
    for m in _matching_milestones(payload, event_ticker):
        details = m.get("product_details")
        link = str(details.get("youtube_link") or "") if isinstance(details, dict) else ""
        vid = parse_youtube_id(link)
        if vid:
            return vid
    return None


def parse_milestone_title(payload: dict[str, object], event_ticker: str) -> str | None:
    """Extract the milestone's own title for ``event_ticker`` from a v1 ``/bff/cards`` payload
    (pure; same matching as :func:`parse_linked_stream`) — the venue's event description, used to
    enrich the YouTube search query (:func:`resolve_stream`) beyond a bare speaker-name search."""
    for m in _matching_milestones(payload, event_ticker):
        title = str(m.get("title") or "").strip()
        if title:
            return title
    return None


_cards_cache: dict[str, dict[str, object] | None] = {}


def _fetch_cards_payload(event_ticker: str, *, timeout_s: float = 10.0) -> dict[str, object] | None:
    """Fetch the v1 ``/bff/cards`` payload for one event (network; never raises — a failure caches
    ``None``, same as a genuinely absent milestone). Cached per ``event_ticker`` for the process
    lifetime: ``run-today``'s poll loop and :func:`resolve_stream` both consult this on every
    retry, and the milestone doesn't change mid-poll."""
    if event_ticker in _cards_cache:
        return _cards_cache[event_ticker]
    import httpx

    payload: dict[str, object] | None = None
    try:
        r = httpx.get(
            f"{KALSHI_V1_BASE}/bff/cards", params={"event_tickers": event_ticker},
            headers={"User-Agent": get_settings().user_agent}, timeout=timeout_s,
        )
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict):
            payload = body
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("bff/cards lookup failed for %s (%s: %s)",
                    event_ticker, type(exc).__name__, exc)
    _cards_cache[event_ticker] = payload
    return payload


def kalshi_linked_stream(event_ticker: str, *, timeout_s: float = 10.0) -> str | None:
    """The Kalshi-linked YouTube video id for an event, or None (absent / network failure —
    never raises; the caller falls through to stream search)."""
    payload = _fetch_cards_payload(event_ticker, timeout_s=timeout_s)
    vid = parse_linked_stream(payload, event_ticker) if payload is not None else None
    if vid:
        log.info("kalshi-linked stream for %s: %s", event_ticker, vid)
    return vid


def kalshi_milestone_title(event_ticker: str, *, timeout_s: float = 10.0) -> str | None:
    """The Kalshi milestone's own title for an event, or None (absent / network failure — never
    raises; the caller keeps its existing speaker-only query)."""
    payload = _fetch_cards_payload(event_ticker, timeout_s=timeout_s)
    return parse_milestone_title(payload, event_ticker) if payload is not None else None


LIVE_JOIN_MAX_S = 45 * 60       # day-only: join a live stream only if it started this recently
SCHED_PAST_TOLERANCE_S = 300    # day-only: a "scheduled" stream must not have lapsed


def parse_not_before(spec: str, *, now: datetime | None = None) -> int:
    """An operator ``--not-before`` spec → epoch seconds. Accepts ``HH:MM`` (today, local clock)
    or a full ISO datetime (naive = local)."""
    now_dt = now if now is not None else datetime.now().astimezone()
    try:
        t = dtime.fromisoformat(spec)
    except ValueError:
        return int(datetime.fromisoformat(spec).astimezone().timestamp())
    return int(datetime.combine(now_dt.date(), t, tzinfo=now_dt.tzinfo).timestamp())


def resolve_occasion_time(
    occasion: UpcomingOccasion, *, n: int = 8, now_ts: int | None = None,
    not_before_ts: int | None = None,
) -> UpcomingOccasion | None:
    """Concrete start time for a day-only occasion: the ticker owns the DAY, the live or
    scheduled stream's ``release_ts`` owns the TIME. None → nothing schedulable on YouTube yet;
    poll again later. A timed occasion passes through unchanged.

    Day-only occasions have a FAKE (noon-anchor) occurrence, so the gap filter alone cannot
    reject stale streams: an old overnight live sits inside the day window and, joined, burns the
    capture hours before the real slot. Accept only a live stream that STARTED within the last
    ``LIVE_JOIN_MAX_S`` (the event is plausibly just beginning) or a scheduled stream whose
    start has not lapsed.

    ``not_before_ts`` is the operator's floor for **multi-appearance days**: any stream (live or
    scheduled) starting before it is rejected. This matters when the same speaker has multiple
    same-day appearances — freshness alone cannot tell sibling events apart, and title matching
    is brittle (streamers name events sloppily). When the operator knows the slot, the floor
    pins it."""
    if occasion.time_known:
        return occasion
    now = int(time.time()) if now_ts is None else now_ts
    video = resolve_stream(occasion, n=n)
    if video is None or video.release_ts is None:
        return None
    if not_before_ts is not None and video.release_ts < not_before_ts:
        return None
    if video.is_live and now - video.release_ts > LIVE_JOIN_MAX_S:
        return None
    if not video.is_live and video.release_ts < now - SCHED_PAST_TOLERANCE_S:
        return None
    return replace(occasion, occurrence_ts=int(video.release_ts), time_known=True)


def resolve_occasion_time_from_video(
    occasion: UpcomingOccasion, video_id: str, *, not_before_ts: int | None = None,
    now_ts: int | None = None,
) -> UpcomingOccasion | None:
    """Concrete start time for a day-only occasion FROM AN OPERATOR-PINNED VIDEO — the
    ``--video-id`` counterpart to :func:`resolve_occasion_time`. An operator pin IS the stream; it
    must never sit behind the general search-poll loop (a pin that's blocked on the speaker-name
    search instead of resolving from itself can stall indefinitely on a broadcaster title that
    never matches).

    Prefers the pinned video's own ``release_timestamp`` (its scheduled or live-broadcast start);
    a currently-live video with no timestamp falls back to ``now`` (it is plainly already
    happening). None → the video isn't resolvable yet (not found, or a scheduled video with no
    timestamp) — the caller polls the SAME pin again, never the general search."""
    if occasion.time_known:
        return occasion
    now = int(time.time()) if now_ts is None else now_ts
    meta = video_metadata(video_id)
    if meta is None:
        return None
    ts = meta.release_ts if meta.release_ts is not None else (now if meta.is_live else None)
    if ts is None:
        return None
    if not_before_ts is not None and ts < not_before_ts:
        return None
    return replace(occasion, occurrence_ts=int(ts), time_known=True)
