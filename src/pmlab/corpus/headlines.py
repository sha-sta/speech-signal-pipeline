"""Dated news-headline source and store.

A dated, broad news-headline corpus is a generally useful complement to a transcript corpus — e.g.
for asking whether some text was already circulating in the news around a given time.
:class:`GdeltHeadlineSource` pulls one from the free GDELT 2.0 Doc API — article titles with a
``seendate``, queryable back years — in monthly batches, self-throttled to GDELT's
1-request-per-5-seconds limit and cached forever (historical windows are immutable).

The default query below samples a broad political/economic/market news agenda; swap ``query`` for
whatever agenda your own downstream analysis needs — this module only fetches and stores headlines,
it does not score or interpret them.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from pmlab.config import Settings, get_settings
from pmlab.data.schemas import HEADLINE_COLUMNS, Headlines, empty_headlines
from pmlab.http import CACHE_FOREVER, ThrottledClient

log = logging.getLogger("pmlab.corpus.headlines")

# Broad political/economic/market agenda (GDELT query syntax: OR is explicit, quotes = phrase).
DEFAULT_QUERY = (
    '("White House" OR "Federal Reserve" OR inflation OR recession OR economy OR election OR '
    'Congress OR tariff OR Ukraine OR Israel OR "interest rate" OR "stock market" OR "artificial '
    'intelligence" OR Bitcoin)'
)
GDELT_HOST = "https://api.gdeltproject.org"
GDELT_PATH = "/api/v2/doc/doc"
GDELT_RPS = 0.19  # ~1 request / 5.3s — under GDELT's published 1/5s limit


class RawHeadline(BaseModel):
    ts: int
    source: str
    text: str

    def to_row(self) -> dict[str, object]:
        hid = hashlib.sha1(f"{self.ts}|{self.source}|{self.text}".encode()).hexdigest()[:16]
        return {"headline_id": hid, "ts": int(self.ts), "source": self.source, "text": self.text}


def _seendate_to_ts(seendate: str) -> int | None:
    """GDELT ``seendate`` (``20240612T191500Z``) → unix seconds; ``None`` if unparseable."""
    try:
        dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None
    return int(dt.timestamp())


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=UTC)
    return start.strftime("%Y%m%d%H%M%S"), end.strftime("%Y%m%d%H%M%S")


def _months(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of ``(year, month)`` from ``YYYY-MM`` ``start`` to ``end``."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out: list[tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        y, m = y + (m == 12), (m % 12) + 1
    return out


class GdeltHeadlineSource:
    def __init__(
        self,
        client: ThrottledClient | None = None,
        settings: Settings | None = None,
        query: str = DEFAULT_QUERY,
    ) -> None:
        s = settings or get_settings()
        self.query = query
        self._c = client or ThrottledClient(
            base_url=GDELT_HOST, rps=GDELT_RPS, cache_dir=s.cache_dir, user_agent=s.user_agent
        )

    def fetch_month(self, year: int, month: int, *, maxrecords: int = 250) -> list[RawHeadline]:
        start, end = _month_bounds(year, month)
        data = self._c.get_json(
            GDELT_PATH,
            {
                "query": self.query, "mode": "artlist", "format": "json",
                "maxrecords": maxrecords, "startdatetime": start, "enddatetime": end,
                "sort": "hybridrel",
            },
            cache_ttl=CACHE_FOREVER,
        )
        rows: list[RawHeadline] = []
        for art in data.get("articles", []):
            ts = _seendate_to_ts(str(art.get("seendate", "")))
            title = str(art.get("title", "")).strip()
            if ts is None or not title:
                continue
            rows.append(RawHeadline(ts=ts, source=str(art.get("domain", "gdelt")), text=title))
        return rows

    def fetch(
        self, start_month: str, end_month: str, *, maxrecords: int = 250
    ) -> Iterator[RawHeadline]:
        for i, (y, m) in enumerate(_months(start_month, end_month), start=1):
            try:
                batch = self.fetch_month(y, m, maxrecords=maxrecords)
            except Exception as exc:  # noqa: BLE001 - a bad month shouldn't abort the whole pull
                log.warning("gdelt %04d-%02d failed: %s", y, m, exc)
                continue
            log.info("gdelt %04d-%02d: %d headlines (%d months requested)", y, m, len(batch), i)
            yield from batch


def fetch_headlines(
    source: GdeltHeadlineSource, start_month: str, end_month: str, *, maxrecords: int = 250
) -> pd.DataFrame:
    """Collect a deduped, schema-valid :class:`Headlines` frame over the month range.

    GDELT repeats articles across overlapping windows, so rows are deduped on ``headline_id``
    *before* validation (``to_headline_frame`` enforces uniqueness)."""
    seen: set[str] = set()
    uniq: list[dict[str, object]] = []
    for h in source.fetch(start_month, end_month, maxrecords=maxrecords):
        row = h.to_row()
        hid = str(row["headline_id"])
        if hid not in seen:
            seen.add(hid)
            uniq.append(row)
    return to_headline_frame(uniq)


# --- headline store ----------------------------------------------------------------------------


def to_headline_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=HEADLINE_COLUMNS) if records else empty_headlines()
    return Headlines.validate(df)


def persist_headlines(headlines: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Headlines.validate(headlines)
    headlines.to_parquet(path, index=False)
    return path


def load_headlines(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_headlines()
    return Headlines.validate(pd.read_parquet(path))
