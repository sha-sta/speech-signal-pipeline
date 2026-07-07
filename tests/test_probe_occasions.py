"""Occasion selection + query/arming glue (pure; the live sweep/video paths hit the network)."""

from __future__ import annotations

from datetime import UTC

import pandas as pd

from pmlab.probe.occasions import (
    DEFAULT_LEAD_S,
    UpcomingOccasion,
    select_event,
    select_occasions,
    speech_query,
)

NOW = 1_800_000_000


def _row(event: str, mkt: str, **kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "event_ticker": event, "market_ticker": mkt, "format": "speech", "template": "speech",
        "eligible": True, "result": "", "occurrence_ts": NOW + 3600, "close_ts": NOW + 7200,
        "speaker": "X", "series_title": "X Mentions", "yes_sub_title": "Y",  # open → result=""
    }
    return {**base, **kw}


def _reg() -> pd.DataFrame:
    speaker_kw = {"speaker": "Jane Doe", "series_title": "Speaker Rally Mentions"}
    rows = [
        _row("E1", "M1", **speaker_kw, yes_sub_title="Iran"),               # result="" → unresolved
        _row("E1", "M2", **speaker_kw, yes_sub_title="China", result=None),  # None also unresolved
        _row("E2", "M3", occurrence_ts=NOW + 10 * 3600),   # beyond the 6h horizon
        _row("E3", "M4", format="sports"),                 # not a speech
        _row("E4", "M5", eligible=False),                  # ineligible
        _row("E5", "M6", result="yes"),                    # already resolved
        _row("E6", "M7", occurrence_ts=NOW - 3600, close_ts=NOW - 60),  # in the past
        _row("E7", "M8", template="duration"),             # open-window bet, no stream → excluded
    ]
    return pd.DataFrame(rows)


def test_select_upcoming_speech_occasions():
    occ = select_occasions(_reg(), now_ts=NOW, within_s=6 * 3600)
    assert [o.event_ticker for o in occ] == ["E1"]
    e1 = occ[0]
    assert e1.speaker == "Jane Doe"
    assert e1.event_type == "rally"                     # from "Speaker Rally Mentions"
    assert e1.occurrence_ts == NOW + 3600
    assert {m.market_ticker for m in e1.markets} == {"M1", "M2"}
    assert e1.arm_at() == NOW + 3600 - DEFAULT_LEAD_S


def test_select_empty_when_nothing_upcoming():
    assert select_occasions(_reg(), now_ts=NOW, within_s=600) == []  # E1 is 1h out, > 10min horizon


# --- select_event (sports record-only capture): NO format/template/speech filter, NO time window --


def test_select_event_picks_eligible_unresolved_rows_of_named_event():
    """Sports rows fail the speech filter (format="sports") and carry no scheduled occurrence_ts
    at all (WC events stamp the series deadline via close_ts) — select_event must not care."""
    rows = [
        _row("EVSPORTS-26JUL05MATCH", "M1", format="sports", template="", speaker=None,
             yes_sub_title="PlayerA", occurrence_ts=None, close_ts=NOW + 19 * 86400),
        _row("EVSPORTS-26JUL05MATCH", "M2", format="sports", template="", speaker=None,
             yes_sub_title="PlayerB", occurrence_ts=None, close_ts=NOW + 19 * 86400),
        _row("EVSPORTS-26JUL05MATCH", "M3", format="sports", template="", speaker=None,
             yes_sub_title="PlayerC", eligible=False),                    # ineligible → excluded
        _row("EVSPORTS-26JUL05MATCH", "M4", format="sports", template="", speaker=None,
             yes_sub_title="PlayerD", result="yes"),                      # settled → excluded
        _row("EVOTHERMATCH-26JUL05", "M5", format="sports", template="", speaker=None,
             yes_sub_title="Other"),                                     # different event
    ]
    occ = select_event(pd.DataFrame(rows), "EVSPORTS-26JUL05MATCH")
    assert occ is not None
    assert occ.event_ticker == "EVSPORTS-26JUL05MATCH"
    assert occ.event_type == "sports"
    assert {m.market_ticker for m in occ.markets} == {"M1", "M2"}
    assert occ.time_known is False
    assert occ.occurrence_ts == NOW + 19 * 86400  # occurrence_ts NaN → fillna(close_ts)
    assert occ.speaker == "EVSPORTS-26JUL05MATCH"  # speaker NaN on every row → ticker fallback


def test_select_event_speaker_from_registry_when_present():
    rows = [_row("EVT", "M1", format="sports", template="", speaker="Fox Sports",
                 yes_sub_title="PlayerA")]
    occ = select_event(pd.DataFrame(rows), "EVT")
    assert occ is not None and occ.speaker == "Fox Sports"


def test_select_event_occurrence_ts_zero_when_all_missing():
    rows = [_row("EVT", "M1", format="sports", template="", speaker=None,
                 yes_sub_title="PlayerA", occurrence_ts=None, close_ts=None)]
    occ = select_event(pd.DataFrame(rows), "EVT")
    assert occ is not None and occ.occurrence_ts == 0


def test_select_event_none_when_nothing_matches():
    assert select_event(_reg(), "NOPE") is None


def test_speech_query_biases_live():
    assert speech_query("Jane Doe", event_type="rally") == "Jane Doe rally speech live"
    assert speech_query("Jerome Powell", event_type="remarks") == "Jerome Powell speech live"


def test_upcoming_occasion_arm_offset():
    o = UpcomingOccasion("E", "spk", "rally", NOW + 5000, markets=[])
    assert o.arm_at(lead_s=1800) == NOW + 5000 - 1800


# --- ticker-encoded event dates (Kalshi occurrence_datetime is a broken far-future default) ------


def test_parse_ticker_date_formats():
    from datetime import date

    from pmlab.probe.occasions import parse_ticker_date

    assert parse_ticker_date("EVMENTION-26JUL04") == date(2026, 7, 4)
    assert parse_ticker_date("KXPERSONMENTION-26APR29C") == date(2026, 4, 29)   # variant letter
    assert parse_ticker_date("EVMENTION-26JUL04-OIL") == date(2026, 7, 4)  # market ticker
    assert parse_ticker_date("KXMADDOWMENTION-26MAY18") == date(2026, 5, 18)
    assert parse_ticker_date("KXNHLMENTION-26JUN14CARVGK") == date(2026, 6, 14)  # team codes
    assert parse_ticker_date("NODATEHERE") is None
    assert parse_ticker_date("KXX-26FEB30") is None                             # impossible day
    assert parse_ticker_date("KXX-26XXX04") is None                             # not a month


def test_ticker_day_overrides_broken_occurrence_ts():
    from datetime import date

    from pmlab.probe.occasions import ticker_day_anchor_ts

    anchor = ticker_day_anchor_ts(date(2026, 7, 4))
    now = anchor - 6 * 3600                                  # 06:00 ET on the event day
    far_future = anchor + 300 * 86400                        # Kalshi's broken default
    rows = [_row("EVMENTION-26JUL04", "M1", occurrence_ts=far_future, close_ts=None)]
    occ = select_occasions(pd.DataFrame(rows), now_ts=now, within_s=6 * 3600)
    assert [o.event_ticker for o in occ] == ["EVMENTION-26JUL04"]
    assert occ[0].occurrence_ts == anchor and occ[0].time_known is False


def test_ticker_day_trusts_agreeing_occurrence_ts():
    from datetime import date

    from pmlab.probe.occasions import ticker_day_anchor_ts

    real = ticker_day_anchor_ts(date(2026, 7, 4)) + 7 * 3600  # 19:00 ET, same civil day
    rows = [_row("EVMENTION-26JUL04", "M1", occurrence_ts=real)]
    occ = select_occasions(pd.DataFrame(rows), now_ts=real - 3600, within_s=6 * 3600)
    assert occ and occ[0].occurrence_ts == real and occ[0].time_known is True


def test_ticker_day_expired_is_excluded():
    from datetime import date

    from pmlab.probe.occasions import ticker_day_anchor_ts, ticker_day_bounds

    anchor = ticker_day_anchor_ts(date(2026, 7, 4))
    day_end = ticker_day_bounds(date(2026, 7, 4))[1]
    rows = [_row("EVMENTION-26JUL04", "M1",
                 occurrence_ts=anchor + 300 * 86400, close_ts=None)]
    assert select_occasions(pd.DataFrame(rows), now_ts=day_end + 3600, within_s=6 * 3600) == []


def test_missing_ts_with_ticker_date_is_day_only():
    from datetime import date

    from pmlab.probe.occasions import ticker_day_anchor_ts

    anchor = ticker_day_anchor_ts(date(2026, 7, 4))
    rows = [_row("EVMENTION-26JUL04", "M1", occurrence_ts=None, close_ts=None)]
    occ = select_occasions(pd.DataFrame(rows), now_ts=anchor - 3600, within_s=6 * 3600)
    assert occ and occ[0].occurrence_ts == anchor and occ[0].time_known is False


def test_resolve_occasion_time_passthrough_when_timed():
    from pmlab.probe.occasions import resolve_occasion_time

    o = UpcomingOccasion("E", "spk", "rally", NOW + 5000, markets=[], time_known=True)
    assert resolve_occasion_time(o) is o                     # no network touched


def test_resolve_occasion_time_rejects_stale_and_lapsed_streams(monkeypatch):
    """Day-only guard: a live stream that started hours ago is NOT the awaited event (live
    failure 2026-07-04: the Jul-4 arm joined a 3 h-old overnight stream), and a 'scheduled'
    stream whose start already lapsed never went live. A fresh live join or a genuinely
    upcoming schedule resolves the time."""
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import LiveStream, resolve_occasion_time

    day_only = UpcomingOccasion("E-26JUL04", "spk", "rally", NOW, markets=[], time_known=False)

    def fake(streams):
        monkeypatch.setattr(occ_mod, "resolve_stream", lambda o, n=8: streams)

    fake(LiveStream("a", "overnight", "Ch", True, False, NOW - 3 * 3600))   # live, 3h stale
    assert resolve_occasion_time(day_only, now_ts=NOW) is None
    fake(LiveStream("b", "never aired", "Ch", False, False, NOW - 3600))    # sched, lapsed
    assert resolve_occasion_time(day_only, now_ts=NOW) is None
    fake(LiveStream("c", "just started", "Ch", True, False, NOW - 600))     # live, 10 min in
    r = resolve_occasion_time(day_only, now_ts=NOW)
    assert r is not None and r.occurrence_ts == NOW - 600 and r.time_known
    fake(LiveStream("d", "tonight", "Ch", False, False, NOW + 4 * 3600))    # sched, upcoming
    r = resolve_occasion_time(day_only, now_ts=NOW)
    assert r is not None and r.occurrence_ts == NOW + 4 * 3600


def test_resolve_occasion_time_not_before_floor(monkeypatch):
    """Multi-appearance-day guard: a speaker with two same-day appearances can have an earlier
    unrelated appearance's stream wrongly attached to a later occasion's row — same speaker, same
    day, wrong event. The operator floor rejects any stream, live or scheduled, starting before
    the named slot; the freshness rails still apply above the floor."""
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import LiveStream, resolve_occasion_time

    day_only = UpcomingOccasion("E-26JUL04", "spk", "rally", NOW, markets=[], time_known=False)
    floor = NOW - 900

    def fake(stream):
        monkeypatch.setattr(occ_mod, "resolve_stream", lambda o, n=8: stream)

    # fresh live stream, but it started before the floor → wrong sibling event, rejected
    fake(LiveStream("a", "afternoon appearance", "Ch", True, False, NOW - 1200))
    assert resolve_occasion_time(day_only, now_ts=NOW, not_before_ts=floor) is None
    # same stream, no floor → accepted (proves the floor is what rejected it)
    r = resolve_occasion_time(day_only, now_ts=NOW)
    assert r is not None and r.occurrence_ts == NOW - 1200
    # scheduled stream before the floor → rejected even though not lapsed
    fake(LiveStream("b", "earlier slot", "Ch", False, False, floor - 60))
    assert resolve_occasion_time(day_only, now_ts=NOW - 7200, not_before_ts=floor) is None
    # fresh live stream at/after the floor → accepted
    fake(LiveStream("c", "the rally", "Ch", True, False, NOW - 600))
    r = resolve_occasion_time(day_only, now_ts=NOW, not_before_ts=floor)
    assert r is not None and r.occurrence_ts == NOW - 600 and r.time_known
    # upcoming scheduled stream after the floor → accepted
    fake(LiveStream("d", "tonight", "Ch", False, False, NOW + 4 * 3600))
    r = resolve_occasion_time(day_only, now_ts=NOW, not_before_ts=floor)
    assert r is not None and r.occurrence_ts == NOW + 4 * 3600


def test_parse_not_before_specs():
    from datetime import datetime

    from pmlab.probe.occasions import parse_not_before

    anchor = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
    # HH:MM = that day, anchor's clock
    assert parse_not_before("18:30", now=anchor) == int(
        datetime(2026, 7, 4, 18, 30, tzinfo=UTC).timestamp())
    # full ISO with offset is honored verbatim
    assert parse_not_before("2026-07-04T18:30:00-04:00") == int(
        datetime(2026, 7, 4, 22, 30, tzinfo=UTC).timestamp())


# --- live-stream discovery (pure pieces) ---------------------------------------------------------


def test_parse_live_rows_handles_na_and_tabs():
    from pmlab.probe.occasions import _parse_live_rows

    stdout = (
        f"vid1\tTrue\tFalse\t{NOW}\tRSBN\tLIVE: speaker rally\n"          # currently live
        f"vid2\tFalse\tFalse\t{NOW + 600}\tC-SPAN\tJane\tspeech\n"     # scheduled; tab in title
        "vid3\tFalse\tTrue\tNA\tNews\told archived rally\n"            # archive, no release
        "NA\tFalse\tFalse\tNA\tX\tbroken row\n"                        # missing id → dropped
        "short\trow\n"                                                  # malformed → dropped
    )
    rows = _parse_live_rows(stdout)
    assert [r.video_id for r in rows] == ["vid1", "vid2", "vid3"]
    assert rows[0].is_live and not rows[0].was_live and rows[0].release_ts == NOW
    assert rows[1].title == "Jane\tspeech" and rows[1].release_ts == NOW + 600
    assert rows[2].was_live and rows[2].release_ts is None


def test_pick_stream_prefers_live_excludes_archives_and_far_channels():
    from pmlab.probe.occasions import LiveStream, pick_stream

    occ = NOW
    live_near = LiveStream("a", "LIVE: rally", "RSBN", True, False, occ - 600)
    scheduled = LiveStream("b", "upcoming", "C-SPAN", False, False, occ + 900)
    channel_247 = LiveStream("c", "24/7 news", "Sky", True, False, occ - 21 * 86400)  # weeks old
    archive = LiveStream("d", "old rally", "News", False, True, occ - 3600)
    no_release = LiveStream("e", "???", "X", True, False, None)

    # live-near beats scheduled; 24/7 channel excluded by the release window
    assert pick_stream([scheduled, channel_247, live_near], occurrence_ts=occ) == live_near
    # no live → the scheduled stream
    assert pick_stream([scheduled, archive], occurrence_ts=occ) == scheduled
    # archives / release-less / far channels alone → None (crash-notify beats wrong tape)
    assert pick_stream([archive, no_release, channel_247], occurrence_ts=occ) is None


# --- S1: Kalshi-linked stream (pure extraction from the v1 /bff/cards payload) --------------------


def _cards_payload(link: str | None = "https://www.youtube.com/watch?v=IyEPha0Oafs") -> dict:
    details = {"youtube_link": link} if link is not None else {}
    return {
        "cards": [{"event_ticker": "EVMENTION-26JUL04", "milestone_id": "m-1"}],
        "hydrated_data": {"milestones": {
            "m-1": {"related_event_tickers": ["EVMENTION-26JUL04"],
                    "product_details": details},
        }},
    }


def test_parse_youtube_id_forms():
    from pmlab.probe.occasions import parse_youtube_id

    assert parse_youtube_id("https://www.youtube.com/watch?v=IyEPha0Oafs") == "IyEPha0Oafs"
    assert parse_youtube_id("https://youtu.be/IyEPha0Oafs?t=1") == "IyEPha0Oafs"
    assert parse_youtube_id("https://www.youtube.com/live/IyEPha0Oafs") == "IyEPha0Oafs"
    assert parse_youtube_id("https://example.com/nope") is None
    assert parse_youtube_id("") is None


def test_parse_linked_stream_via_card_milestone():
    from pmlab.probe.occasions import parse_linked_stream

    assert parse_linked_stream(_cards_payload(), "EVMENTION-26JUL04") == "IyEPha0Oafs"


def test_parse_linked_stream_via_related_tickers_only():
    from pmlab.probe.occasions import parse_linked_stream

    p = _cards_payload()
    p["cards"] = []  # no card pointer — the milestone's own ticker list must still match
    assert parse_linked_stream(p, "EVMENTION-26JUL04") == "IyEPha0Oafs"


def test_parse_linked_stream_absent_or_foreign():
    from pmlab.probe.occasions import parse_linked_stream

    assert parse_linked_stream(_cards_payload(link=None), "EVMENTION-26JUL04") is None
    assert parse_linked_stream(_cards_payload(), "KXOTHER-26JUL04") is None
    assert parse_linked_stream({}, "EVMENTION-26JUL04") is None


# --- milestone TITLE enriches the search --------------------------------------------------------


def test_parse_milestone_title_via_card_milestone():
    from pmlab.probe.occasions import parse_milestone_title

    payload = _cards_payload()
    payload["hydrated_data"]["milestones"]["m-1"]["title"] = (
        "Speaker holds a meeting with a foreign leader")
    assert parse_milestone_title(payload, "EVMENTION-26JUL04") == (
        "Speaker holds a meeting with a foreign leader")


def test_parse_milestone_title_absent_or_foreign():
    from pmlab.probe.occasions import parse_milestone_title

    assert parse_milestone_title(_cards_payload(), "EVMENTION-26JUL04") is None  # no title key
    payload = _cards_payload()
    payload["hydrated_data"]["milestones"]["m-1"]["title"] = "Speaker holds a meeting"
    assert parse_milestone_title(payload, "KXOTHER-26JUL04") is None  # different event
    assert parse_milestone_title({}, "EVMENTION-26JUL04") is None


def test_fetch_cards_payload_is_cached(monkeypatch):
    import httpx

    import pmlab.probe.occasions as occ_mod

    occ_mod._cards_cache.clear()
    calls = {"n": 0}

    class _FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            calls["n"] += 1
            return _cards_payload()

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResp())
    try:
        occ_mod._fetch_cards_payload("EVMENTION-26JUL04")
        occ_mod._fetch_cards_payload("EVMENTION-26JUL04")
        assert calls["n"] == 1  # second call served from the cache, no second HTTP round trip
    finally:
        occ_mod._cards_cache.clear()


def test_kalshi_linked_stream_and_milestone_title_share_one_fetch(monkeypatch):
    import httpx

    import pmlab.probe.occasions as occ_mod

    occ_mod._cards_cache.clear()
    calls = {"n": 0}

    class _FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            calls["n"] += 1
            p = _cards_payload()
            p["hydrated_data"]["milestones"]["m-1"]["title"] = (
                "Speaker holds a meeting with a foreign leader")
            return p

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResp())
    try:
        vid = occ_mod.kalshi_linked_stream("EVMENTION-26JUL04")
        title = occ_mod.kalshi_milestone_title("EVMENTION-26JUL04")
        assert vid == "IyEPha0Oafs"
        assert title == "Speaker holds a meeting with a foreign leader"
        assert calls["n"] == 1  # kalshi_linked_stream's fetch was reused, not re-fetched
    finally:
        occ_mod._cards_cache.clear()


def test_kalshi_milestone_title_none_on_network_failure(monkeypatch):
    import httpx

    import pmlab.probe.occasions as occ_mod

    occ_mod._cards_cache.clear()

    def _boom(*a: object, **kw: object) -> None:
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "get", _boom)
    try:
        assert occ_mod.kalshi_milestone_title("EVMENTION-26JUL04") is None  # never raises
    finally:
        occ_mod._cards_cache.clear()


def test_resolve_stream_additive_milestone_query(monkeypatch):
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import LiveStream, resolve_stream

    occ = UpcomingOccasion("E-26JUL07", "Jane Doe", "meeting", NOW, markets=[],
                           time_known=True)
    queries: list[str] = []
    base = LiveStream("BASE1", "search hit", "C-SPAN", False, False, NOW - 6000)
    enriched = LiveStream(
        "MATCH1", "LIVE: Speaker holds a meeting with a foreign leader | NEWS 5",
        "NEWS 5", True, False, NOW - 300)

    def fake_search(query: str, *, n: int = 8) -> list[LiveStream]:
        queries.append(query)
        return [enriched] if "a foreign leader" in query else [base]

    monkeypatch.setattr(occ_mod, "search_live_streams", fake_search)
    monkeypatch.setattr(occ_mod, "kalshi_milestone_title",
                        lambda ticker, **kw: "Speaker holds a meeting with a foreign leader")

    r = resolve_stream(occ)
    assert len(queries) == 2  # the base speaker query AND the milestone-enriched query both ran
    assert r == enriched  # live + fresher beats the stale base-query hit


def test_resolve_stream_falls_back_to_base_query_when_no_milestone_title(monkeypatch):
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import LiveStream, resolve_stream

    occ = UpcomingOccasion("E-26JUL07", "Jane Doe", "meeting", NOW, markets=[],
                           time_known=True)
    base = LiveStream("BASE1", "search hit", "C-SPAN", True, False, NOW - 300)

    monkeypatch.setattr(occ_mod, "search_live_streams", lambda query, *, n=8: [base])
    monkeypatch.setattr(occ_mod, "kalshi_milestone_title", lambda ticker, **kw: None)

    assert resolve_stream(occ) == base  # no second (milestone) search ran, existing query kept


# --- --video-id resolves occasion time from the pinned video --------------------------------------


def test_video_metadata_parses_pinned_video_row(monkeypatch):
    import pmlab.signals.transcripts as tr_mod
    from pmlab.probe.occasions import video_metadata

    class _Result:
        stdout = f"VID1\tTrue\tFalse\t{NOW}\tNEWS 5\tLIVE: Speaker holds a meeting\n"

    def fake_yt_dlp(args: list[str], **kw: object) -> _Result:
        assert args[-1] == "https://www.youtube.com/watch?v=VID1"
        return _Result()

    monkeypatch.setattr(tr_mod, "_yt_dlp", fake_yt_dlp)
    meta = video_metadata("VID1")
    assert meta is not None
    assert meta.video_id == "VID1" and meta.is_live and meta.release_ts == NOW


def test_video_metadata_none_when_yt_dlp_finds_nothing(monkeypatch):
    import pmlab.signals.transcripts as tr_mod
    from pmlab.probe.occasions import video_metadata

    class _Result:
        stdout = ""

    monkeypatch.setattr(tr_mod, "_yt_dlp", lambda args, **kw: _Result())
    assert video_metadata("NOPE") is None


def test_resolve_occasion_time_from_video_passthrough_when_timed():
    from pmlab.probe.occasions import resolve_occasion_time_from_video

    o = UpcomingOccasion("E", "spk", "rally", NOW + 5000, markets=[], time_known=True)
    assert resolve_occasion_time_from_video(o, "VID1") is o  # no network touched


def test_resolve_occasion_time_from_video_uses_release_ts(monkeypatch):
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import LiveStream, resolve_occasion_time_from_video

    day_only = UpcomingOccasion("E-26JUL07", "Jane Doe", "meeting", NOW, markets=[],
                                time_known=False)
    monkeypatch.setattr(occ_mod, "video_metadata",
                        lambda vid: LiveStream(vid, "t", "c", True, False, NOW - 600))
    r = resolve_occasion_time_from_video(day_only, "VID1", now_ts=NOW)
    assert r is not None and r.occurrence_ts == NOW - 600 and r.time_known


def test_resolve_occasion_time_from_video_live_no_ts_falls_back_to_now(monkeypatch):
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import LiveStream, resolve_occasion_time_from_video

    day_only = UpcomingOccasion("E-26JUL07", "Jane Doe", "meeting", NOW, markets=[],
                                time_known=False)
    monkeypatch.setattr(occ_mod, "video_metadata",
                        lambda vid: LiveStream(vid, "t", "c", True, False, None))
    r = resolve_occasion_time_from_video(day_only, "VID1", now_ts=NOW)
    assert r is not None and r.occurrence_ts == NOW and r.time_known


def test_resolve_occasion_time_from_video_scheduled_no_ts_is_none(monkeypatch):
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import LiveStream, resolve_occasion_time_from_video

    day_only = UpcomingOccasion("E-26JUL07", "Jane Doe", "meeting", NOW, markets=[],
                                time_known=False)
    monkeypatch.setattr(occ_mod, "video_metadata",
                        lambda vid: LiveStream(vid, "t", "c", False, False, None))
    assert resolve_occasion_time_from_video(day_only, "VID1", now_ts=NOW) is None


def test_resolve_occasion_time_from_video_none_when_not_found(monkeypatch):
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import resolve_occasion_time_from_video

    day_only = UpcomingOccasion("E-26JUL07", "Jane Doe", "meeting", NOW, markets=[],
                                time_known=False)
    monkeypatch.setattr(occ_mod, "video_metadata", lambda vid: None)
    assert resolve_occasion_time_from_video(day_only, "VID1", now_ts=NOW) is None


def test_resolve_occasion_time_from_video_respects_not_before(monkeypatch):
    import pmlab.probe.occasions as occ_mod
    from pmlab.probe.occasions import LiveStream, resolve_occasion_time_from_video

    day_only = UpcomingOccasion("E-26JUL07", "Jane Doe", "meeting", NOW, markets=[],
                                time_known=False)
    floor = NOW - 900
    monkeypatch.setattr(occ_mod, "video_metadata",
                        lambda vid: LiveStream(vid, "t", "c", True, False, NOW - 1200))
    assert resolve_occasion_time_from_video(
        day_only, "VID1", now_ts=NOW, not_before_ts=floor) is None
    r = resolve_occasion_time_from_video(day_only, "VID1", now_ts=NOW)
    assert r is not None and r.occurrence_ts == NOW - 1200  # no floor → accepted
