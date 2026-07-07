"""Transcript scrapers → ``data/corpus/`` (§5). Reuses the M0 throttled/cached HTTP client (D12).

The corpus is the base-rate substrate: dated, per-occasion speaking transcripts for the recurring
speakers whose Kalshi mention markets we model. This module ships the source that proved robust and
static under M2's [verify-at-build] probing (2026-07-03):

* :class:`FederalReserveSource` — federalreserve.gov, unauthenticated, static HTML, deep history.
  Two occasion streams: **FOMC statements** (attributed to the Fed Chair) enumerated from the FOMC
  calendar, and **Fed speeches** (Powell, governors) enumerated from the yearly speech indices.
  Together these give ≥2 real speakers with many occasions each — the M2 corpus acceptance target.

Deliberately *not* shipped: some official sources (e.g. whitehouse.gov) resist static scraping —
JS-hydrated listings, REST APIs that 403 unauthenticated clients, and custom post types absent
from the XML sitemaps are common obstacles — so covering those speakers needs a JS-capable fetch
or a licensed transcript feed and is left as a follow-up. :class:`TranscriptSource` is the
extension point for that.

All fetches go through :class:`~pmlab.http.ThrottledClient` (self-throttled, retried, on-disk
cached); historical docs are immutable so they cache forever.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, Field
from selectolax.parser import HTMLParser, Node

from pmlab.config import Settings, get_settings
from pmlab.corpus.store import make_doc_id, to_corpus_frame, word_count
from pmlab.http import CACHE_FOREVER, ThrottledClient

log = logging.getLogger("pmlab.corpus.sources")

# Content containers to try (largest wins); noise stripped first. Covers federalreserve.gov's
# speech pages (#article) and statement pages (.col-md-8 / #content) in one list.
_CONTENT_SELECTORS = (
    "#article",
    "div.col-xs-12.col-sm-8.col-md-8",
    "#content",
    "article",
    ".entry-content",
    "main",
    "div.field-docs-content",
)
_NOISE_SELECTORS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


class RawTranscript(BaseModel):
    """One scraped speaking occasion, pre-store."""

    speaker: str
    format: str = "speech"
    occasion: str = ""
    date_ts: int
    source: str
    url: str = ""
    text: str = Field(default="", repr=False)

    def to_row(self) -> dict[str, object]:
        return {
            "doc_id": make_doc_id(self.source, self.url, self.occasion),
            "speaker": self.speaker,
            "format": self.format,
            "occasion": self.occasion,
            "date_ts": int(self.date_ts),
            "source": self.source,
            "url": self.url,
            "n_words": word_count(self.text),
            "text": self.text,
        }


class TranscriptSource(Protocol):
    """A speaker/venue scraper. ``fetch`` yields dated transcripts; the store dedupes by doc_id."""

    def fetch(self, *, limit: int | None = None) -> Iterator[RawTranscript]: ...


# --- shared helpers ------------------------------------------------------------------------------


def extract_main_text(html: str) -> tuple[str, str]:
    """Return ``(h1_title, body_text)`` — the largest content container's text, noise removed."""
    tree = HTMLParser(html)
    h1 = tree.css_first("h1")
    title = h1.text(strip=True) if h1 else ""
    for sel in _NOISE_SELECTORS:
        for noise in tree.css(sel):
            noise.decompose()
    best: Node | None = None
    best_len = 0
    for sel in _CONTENT_SELECTORS:
        cand = tree.css_first(sel)
        if cand is not None:
            n = len(cand.text(strip=True))
            if n > best_len:
                best, best_len = cand, n
    container = best if best is not None else tree.body
    text = container.text(separator=" ", strip=True) if container is not None else ""
    return title, re.sub(r"\s+", " ", text).strip()


def _ts_from_yyyymmdd(digits: str) -> int:
    return int(datetime.strptime(digits, "%Y%m%d").replace(tzinfo=UTC).timestamp())


def _ts_from_long_date(text: str) -> int | None:
    """Parse ``"February 17, 2026"`` (federalreserve.gov ``.article__time``) → unix seconds."""
    m = re.search(r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),\s+(\d{4})", text)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y")
    except ValueError:
        return None
    return int(dt.replace(tzinfo=UTC).timestamp())


# --- federalreserve.gov --------------------------------------------------------------------------


class FederalReserveSource:
    """FOMC statements (the Chair) + Fed speeches (named officials) from federalreserve.gov."""

    HOST = "https://www.federalreserve.gov"
    NAME = "federalreserve.gov"
    CALENDAR = "/monetarypolicy/fomccalendars.htm"
    _STATEMENT_RE = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")
    _SPEECH_RE = re.compile(r"/newsevents/speech/([a-z]+)(\d{8})[a-z]?\.htm")

    def __init__(
        self, client: ThrottledClient | None = None, settings: Settings | None = None
    ) -> None:
        s = settings or get_settings()
        self._c = client or ThrottledClient(
            base_url=self.HOST, rps=2.0, cache_dir=s.cache_dir, user_agent=s.user_agent
        )

    def _get(self, path: str) -> str:
        return self._c.get_text(path, cache_ttl=CACHE_FOREVER)

    def iter_statement_paths(self) -> list[str]:
        """Distinct FOMC statement page paths from the calendar, most recent first."""
        html = self._c.get_text(self.CALENDAR, cache_ttl=900.0)  # calendar updates; short TTL
        seen: dict[str, None] = {}
        for m in self._STATEMENT_RE.finditer(html):
            seen.setdefault(f"/newsevents/pressreleases/monetary{m.group(1)}a.htm", None)
        return sorted(seen, reverse=True)

    def iter_speech_paths(self, years: Sequence[int]) -> list[tuple[str, str, str]]:
        """``(path, speaker_slug, yyyymmdd)`` for each speech in the given yearly indices."""
        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for year in years:
            try:
                html = self._c.get_text(f"/newsevents/speech/{year}-speeches.htm", cache_ttl=900.0)
            except Exception as exc:  # noqa: BLE001 - a missing year index shouldn't abort the run
                log.warning("fed speeches %s index unavailable: %s", year, exc)
                continue
            for m in self._SPEECH_RE.finditer(html):
                path = m.group(0)
                if path not in seen:
                    seen.add(path)
                    out.append((path, m.group(1), m.group(2)))
        return out

    def _statement(self, path: str) -> RawTranscript | None:
        digits = self._STATEMENT_RE.search(path)
        if not digits:
            return None
        _title, text = extract_main_text(self._get(path))
        if word_count(text) < 50:  # a real statement is a few hundred words; guard empties
            return None
        ts = _ts_from_yyyymmdd(digits.group(1))
        month = datetime.fromtimestamp(ts, tz=UTC).strftime("%B %Y")
        return RawTranscript(
            speaker="the Chair of the Federal Reserve",
            format="speech",
            occasion=f"FOMC statement — {month}",
            date_ts=ts,
            source=self.NAME,
            url=self.HOST + path,
            text=text,
        )

    def _speech(self, path: str, yyyymmdd: str) -> RawTranscript | None:
        html = self._get(path)
        tree = HTMLParser(html)
        spk = tree.css_first(".speaker") or tree.css_first("p.speaker")
        speaker = spk.text(strip=True) if spk else "Federal Reserve official"
        dt_el = tree.css_first(".article__time") or tree.css_first("p.article__time")
        ts = (_ts_from_long_date(dt_el.text(strip=True)) if dt_el else None) or _ts_from_yyyymmdd(
            yyyymmdd
        )
        title, text = extract_main_text(html)
        if word_count(text) < 50:
            return None
        return RawTranscript(
            speaker=speaker,
            format="speech",
            occasion=title,
            date_ts=ts,
            source=self.NAME,
            url=self.HOST + path,
            text=text,
        )

    def fetch(
        self,
        *,
        limit: int | None = None,
        statements: bool = True,
        speeches: bool = True,
        speech_years: Sequence[int] | None = None,
    ) -> Iterator[RawTranscript]:
        n = 0
        if statements:
            for path in self.iter_statement_paths():
                if limit is not None and n >= limit:
                    return
                rec = self._statement(path)
                if rec is not None:
                    n += 1
                    yield rec
        if speeches:
            years = list(speech_years or (datetime.now(tz=UTC).year, datetime.now(tz=UTC).year - 1))
            for path, _slug, yyyymmdd in self.iter_speech_paths(years):
                if limit is not None and n >= limit:
                    return
                try:
                    rec = self._speech(path, yyyymmdd)
                except Exception as exc:  # noqa: BLE001 - skip a bad page, keep the corpus building
                    log.warning("fed speech %s failed: %s", path, exc)
                    continue
                if rec is not None:
                    n += 1
                    yield rec


def fetch_corpus(
    sources: Sequence[TranscriptSource],
    *,
    limit_per_source: int | None = None,
) -> pd.DataFrame:
    """Run each source and collect a schema-valid :class:`CorpusTranscripts` frame."""
    rows: list[dict[str, object]] = []
    for src in sources:
        name = type(src).__name__
        count = 0
        for rec in src.fetch(limit=limit_per_source):
            rows.append(rec.to_row())
            count += 1
        log.info("%s: %d transcripts", name, count)
    return to_corpus_frame(rows)


def persist_raw_corpus(corpus: pd.DataFrame, path: Path) -> Path:
    from pmlab.corpus.store import persist_corpus

    return persist_corpus(corpus, path)
