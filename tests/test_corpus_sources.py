"""FederalReserveSource tests — parsing helpers + the fetch loop over mock-served HTML."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import httpx

from pmlab.corpus.sources import (
    FederalReserveSource,
    _ts_from_long_date,
    _ts_from_yyyymmdd,
    extract_main_text,
)
from pmlab.http import ThrottledClient

_BODY = " ".join(["Inflation and the labor market remain the committee's focus this period."] * 8)

_CALENDAR = '<html><body><a href="/newsevents/pressreleases/monetary20260318a.htm">Statement</a>' \
            '<a href="/monetarypolicy/fomcpresconf20260318.htm">Presser</a></body></html>'
_STATEMENT = f'<html><body><h1>Federal Reserve issues FOMC statement</h1>' \
             f'<div id="article">{_BODY}</div></body></html>'
_SPEECH_INDEX = '<html><body><a href="/newsevents/speech/barr20260217a.htm">Speech</a>' \
                '<a href="/newsevents/speech/powell20260610a.htm">Speech2</a></body></html>'
_SPEECH = f'<html><body><p class="speaker">Governor Michael S. Barr</p>' \
          f'<p class="article__time">February 17, 2026</p>' \
          f'<h1>AI and the Economy</h1><div id="article">{_BODY}</div></body></html>'

_ROUTES = {
    "/monetarypolicy/fomccalendars.htm": _CALENDAR,
    "/newsevents/pressreleases/monetary20260318a.htm": _STATEMENT,
    "/newsevents/speech/2026-speeches.htm": _SPEECH_INDEX,
    "/newsevents/speech/barr20260217a.htm": _SPEECH,
    "/newsevents/speech/powell20260610a.htm": _SPEECH,
}


def _handler(request: httpx.Request) -> httpx.Response:
    body = _ROUTES.get(request.url.path)
    if body is None:
        return httpx.Response(404, text="not found")
    return httpx.Response(200, text=body)


# --- pure helpers -------------------------------------------------------------------------------


def test_extract_main_text_picks_content_and_strips_noise() -> None:
    html = '<html><body><nav>menu junk</nav><h1>Title</h1>' \
           '<div id="article">the real body text here</div><footer>foot</footer></body></html>'
    title, text = extract_main_text(html)
    assert title == "Title"
    assert "real body text" in text
    assert "menu junk" not in text and "foot" not in text


def test_date_parsers() -> None:
    assert _ts_from_yyyymmdd("20260318") == int(
        datetime(2026, 3, 18, tzinfo=UTC).timestamp()
    )
    assert _ts_from_long_date("Released February 17, 2026 at 2pm") == int(
        datetime(2026, 2, 17, tzinfo=UTC).timestamp()
    )
    assert _ts_from_long_date("no date here") is None


# --- source fetch over mocked HTTP --------------------------------------------------------------


def test_fed_source_fetches_statements_and_speeches(
    mock_client: Callable[..., ThrottledClient],
) -> None:
    src = FederalReserveSource(client=mock_client(_handler))
    recs = list(src.fetch(speech_years=[2026]))
    by_url = {r.url.split("/")[-1]: r for r in recs}

    # one statement (the Chair) + two speeches
    stmt = by_url["monetary20260318a.htm"]
    assert stmt.speaker == "the Chair of the Federal Reserve"
    assert stmt.format == "speech"
    assert stmt.date_ts == int(datetime(2026, 3, 18, tzinfo=UTC).timestamp())
    assert stmt.to_row()["n_words"] > 50

    speech = by_url["barr20260217a.htm"]
    assert speech.speaker == "Governor Michael S. Barr"
    assert speech.date_ts == int(datetime(2026, 2, 17, tzinfo=UTC).timestamp())
    assert speech.occasion == "AI and the Economy"


def test_fed_source_statements_only(mock_client: Callable[..., ThrottledClient]) -> None:
    src = FederalReserveSource(client=mock_client(_handler))
    recs = list(src.fetch(statements=True, speeches=False))
    assert len(recs) == 1
    assert recs[0].url.endswith("monetary20260318a.htm")


def test_fed_source_limit(mock_client: Callable[..., ThrottledClient]) -> None:
    src = FederalReserveSource(client=mock_client(_handler))
    assert len(list(src.fetch(limit=1, speech_years=[2026]))) == 1


def test_fed_source_missing_year_index_is_skipped(
    mock_client: Callable[..., ThrottledClient],
) -> None:
    # 2099 index 404s → no speeches, but statements still come through (no crash).
    src = FederalReserveSource(client=mock_client(_handler))
    recs = list(src.fetch(speech_years=[2099]))
    assert [r for r in recs if "monetary" in r.url]
    assert not [r for r in recs if "speech" in r.url]
