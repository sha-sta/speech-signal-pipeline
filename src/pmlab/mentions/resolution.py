"""Resolution schema + deterministic rule parser + JSONL I/O for the mention registry.

Per IMPLEMENTATION_PLAN §4.2–4.3 and D6, resolution-clause parsing is done by the executing
agent (no paid LLM), $0. This module is the agent's *tool*: it turns a Kalshi mention market's
``rules_primary``/``rules_secondary`` into a structured :class:`Resolution` with a **baseline**
risk tier and extraction confidence. The agent then reviews the baseline (archetype judgment +
spot-checks) and the human gate signs off the low-confidence / high-risk tail.

Grammar observed live 2026-07-03 (three templates cover the universe; see the recorded fixtures
``tests/fixtures/kalshi_mentions_event_*.json``):

* **speech / broadcast** — ``If <SPEAKER> says <PHRASE> as part of <OCCASION>, then the market
  resolves to Yes.`` ``<SPEAKER>`` ∈ {a person, ``A or B or C…``, ``any <ORG> representative``,
  ``any person``}.
* **earnings** — ``If <PHRASE> is said by any <COMPANY> representative (including the operator of
  the call) during the next <COMPANY> earnings call (including the Q+A), then …``.
* **NQE** — ``If a qualifying event does not occur, then the market resolves to Yes.`` (the
  cancellation-clause leg; ``yes_sub_title == "Event does not qualify"``). Not a phrase prediction.

Resolution-risk driver (§0.3, §4.3): the dominant risk is that the governing source *misses* a
true occurrence (approved-outlet dependency → the Sanders–Greensboro $3.6M all-No event) or that
the speaker/an affiliate trivially triggers/suppresses the phrase. ``rules_secondary`` almost
always names **video of the event as the primary resolution source** with approved outlets only as
a transcript fallback — which makes the miss-risk *low* for televised/streamed events and
concentrates real risk in sparse-source events with no canonical video.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# Binary phrase-prediction templates enter the model set; the rest are carried but out-of-scope.
Template = Literal["speech", "earnings", "duration", "length", "superlative", "nqe", "other"]
BINARY_TEMPLATES = ("speech", "earnings", "duration")
VariantRule = Literal["exact", "plural_possessive", "aliases_listed", "count_threshold", "fuzzy"]
RiskTier = Literal["low", "medium", "high"]

_WS = re.compile(r"\s+")
# "<speaker> says <phrase> [as part of|at] <occasion>" — a single televised/streamed occasion.
_SPEECH_RE = re.compile(
    r"^If (?P<speaker>.+?) says (?P<phrase>.+?) (?:as part of|at) (?P<occasion>.+?)"
    r"\s*,?\s*then the market resolves to Yes\.?$"
)
_EARNINGS_RE = re.compile(
    r"^If (?P<phrase>.+?) is said by any (?P<company>.+?) representative.*?"
    r"during the next .+? earnings call.*?then the market resolves to Yes\.?$"
)
# "<phrase> … is stated by <speaker> before/after <date-window>" — an open window with NO single
# canonical video; resolution leans on approved outlets + the speaker's socials (elevated risk).
_DURATION_RE = re.compile(
    r"^If (?P<phrase>.+?)(?:, or a plural or possessive form of [^,]+?)?, is stated by "
    r"(?P<speaker>.+?) (?P<window>(?:before|after|between) .+?)"
    r"\s*,?\s*then the market resolves to Yes\.?$"
)
# Out-of-scope Kalshi mention variants (recognized so they don't masquerade as parse failures, but
# excluded from the binary phrase-frequency model): speech length, and "which topic dominates".
_LENGTH_RE = re.compile(
    r"^If .+? speaks for .+? minutes at .+?, then the market resolves to Yes\.?$"
)
_SUPERLATIVE_RE = re.compile(
    r"^If (?P<speaker>.+?) mentions the topic of (?P<phrase>.+?) most as part of "
    r"(?P<occasion>.+?)\s*,?\s*then the market resolves to Yes\.?$"
)
_COUNT_RE = re.compile(r"\(\s*\d+\s*\+?\s*times?\s*\)", re.IGNORECASE)


class Resolution(BaseModel):
    """Parsed resolution clause for one mention market (the risk gate, §4.3)."""

    market_ticker: str
    series: str
    template: Template
    speaker: str
    occasion: str
    phrase: str
    variant_rule: VariantRule
    cutoff_ts: int | None = None
    governing_sources: list[str] = Field(default_factory=list)
    early_resolution: bool = False
    cancellation_clause: bool = False
    is_nqe: bool = False
    resolution_risk: RiskTier = "medium"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    notes: str = ""

    @property
    def eligible(self) -> bool:
        """Enters the model set (§4.3): a **binary phrase** prediction (speech/earnings/duration),
        low/medium risk, confidently parsed. NQE cancellation legs, out-of-scope market types
        (length/superlative), ``high``-risk and low-confidence parses are carried but excluded."""
        return (
            self.template in BINARY_TEMPLATES
            and not self.is_nqe
            and self.resolution_risk in ("low", "medium")
            and self.confidence >= 0.8
        )


def normalize_ws(text: str | None) -> str:
    return _WS.sub(" ", (text or "").replace("\n", " ")).strip()


def classify_template(rules_primary: str, yes_sub_title: str) -> Template:
    r = normalize_ws(rules_primary)
    if (
        yes_sub_title.strip().lower() == "event does not qualify"
        or "qualifying event does not occur" in r.lower()
    ):
        return "nqe"
    if _EARNINGS_RE.match(r):
        return "earnings"
    if _LENGTH_RE.match(r):  # "speaks for N minutes" — a scalar length market, not a phrase
        return "length"
    if _SUPERLATIVE_RE.match(r):  # "mentions the topic of X most" — comparative, not binary
        return "superlative"
    if _DURATION_RE.match(r):  # check before speech: "is stated by … before/after <date>"
        return "duration"
    if _SPEECH_RE.match(r):
        return "speech"
    return "other"


def classify_variant(yes_sub_title: str) -> VariantRule:
    """Classify the phrase-matching rule from the outcome label.

    Kalshi enumerates accepted alternatives with `` / `` in the label; single labels match exactly
    (with plural/possessive per the universal payout criterion in ``rules_secondary``). A
    ``(N+ times)`` suffix is a counting threshold (higher ambiguity)."""
    s = yes_sub_title.strip()
    if not s:
        return "fuzzy"
    if _COUNT_RE.search(s):
        return "count_threshold"
    if "/" in s:
        parts = [p.strip() for p in s.split("/") if p.strip()]
        if len(parts) >= 2 and _share_stem(parts):
            return "plural_possessive"
        return "aliases_listed"
    return "exact"


def _share_stem(parts: list[str], stem_len: int = 4) -> bool:
    """True if every alternative shares a leading stem — a morphological family (Afford/Affordable/
    Affordability, Terrorist/Terrorism) rather than distinct aliases (Oil/Gas, China/Russia)."""
    heads = [re.sub(r"[^a-z]", "", p.lower())[:stem_len] for p in parts]
    return len(set(heads)) == 1 and bool(heads[0])


def _iso_to_ts(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _baseline_risk(
    template: Template,
    variant: VariantRule,
    n_sources: int,
    video_primary: bool,
) -> tuple[RiskTier, str]:
    """Conservative baseline the agent refines (§4.3). Risk is the approved-outlet **miss** risk,
    modulated by matching ambiguity."""
    base: RiskTier
    if template == "duration":
        # windowed "stated by X anytime before/after <date>": no single canonical video, resolves
        # off approved outlets + the speaker's socials → the dominant approved-outlet-miss surface.
        if n_sources >= 3:
            base, why = "medium", "windowed statement, no canonical video → outlet-miss risk"
        else:
            base, why = "high", "windowed statement, sparse approved sources → high miss risk"
    elif template == "earnings" or video_primary:
        base, why = "low", "video/transcript-primary resolution → low approved-outlet-miss risk"
    elif n_sources >= 5:
        base, why = "low", f"{n_sources} approved outlets → low miss risk"
    elif n_sources >= 1:
        base, why = "medium", f"only {n_sources} approved source(s) → outlet-dependency risk"
    else:
        base, why = "high", "no governing source declared"
    if variant in ("count_threshold", "fuzzy"):  # counting / ambiguous matching bumps one tier
        bumped: dict[RiskTier, RiskTier] = {"low": "medium", "medium": "high", "high": "high"}
        base = bumped[base]
        why += f"; +{variant} ambiguity"
    return base, why


def parse_market(market: Mapping[str, Any], event: Mapping[str, Any]) -> Resolution:
    """Deterministically parse one market into a :class:`Resolution` with baseline risk/confidence.

    ``event`` supplies the series and the approved ``settlement_sources`` (the governing-source set
    lives on the event, not the market — §2)."""
    ticker = str(market["ticker"])
    series = str(event.get("series_ticker") or ticker.split("-")[0])
    rules_primary = str(market.get("rules_primary") or "")
    rules_secondary = str(market.get("rules_secondary") or "")
    yes_sub_title = str(market.get("yes_sub_title") or "")
    sources = [str(s.get("name", "")).strip() for s in (event.get("settlement_sources") or [])]
    sources = [s for s in sources if s]
    rs_lower = rules_secondary.lower()
    video_primary = "video" in rs_lower and "primarily used" in rs_lower
    cutoff_ts = _iso_to_ts(market.get("close_time")) or _iso_to_ts(
        market.get("expected_expiration_time")
    )

    template = classify_template(rules_primary, yes_sub_title)
    norm = normalize_ws(rules_primary)

    if template == "nqe":
        return Resolution(
            market_ticker=ticker, series=series, template="nqe",
            speaker="", occasion="", phrase=yes_sub_title.strip(),
            variant_rule="exact", cutoff_ts=cutoff_ts, governing_sources=sources,
            early_resolution=bool(market.get("can_close_early")),
            cancellation_clause=True, is_nqe=True,
            resolution_risk="medium", confidence=0.95,
            notes="qualifying-event-does-not-occur leg (cancellation clause); not a phrase pred",
        )

    speaker = occasion = ""
    confidence = 0.4
    if template == "earnings":
        m = _EARNINGS_RE.match(norm)
        if m:
            speaker = f"any {m['company'].strip()} representative"
            occasion = f"{m['company'].strip()} earnings call"
            confidence = 0.9
    elif template == "speech":
        m = _SPEECH_RE.match(norm)
        if m:
            speaker = m["speaker"].strip()
            occasion = m["occasion"].strip().rstrip(".")
            confidence = 0.9
            if speaker.lower() in ("any person",) or " or " in speaker:
                confidence = 0.8  # who-counts is looser, but still a clean template match
    elif template == "duration":
        m = _DURATION_RE.match(norm)
        if m:
            speaker = m["speaker"].strip()
            occasion = "public statements " + m["window"].strip().rstrip(".")
            confidence = 0.85  # clean template, but the open-window occasion is inherently looser
    elif template == "superlative":
        m = _SUPERLATIVE_RE.match(norm)
        if m:
            speaker = m["speaker"].strip()
            occasion = m["occasion"].strip().rstrip(".")
            confidence = 0.85
    elif template == "length":
        m = _LENGTH_RE.match(norm)
        if m:
            confidence = 0.85  # recognized speech-length (scalar) market; carried, not modeled

    variant = classify_variant(yes_sub_title)
    risk, why = _baseline_risk(template, variant, len(sources), video_primary)
    if template == "other":
        confidence = min(confidence, 0.4)
        why = "unrecognized rule template — needs manual parse; " + why
    payout = "plural/possessive accepted" if "plural or possessive" in rs_lower else ""
    note = "; ".join(x for x in [f"template={template}", why, payout] if x)

    return Resolution(
        market_ticker=ticker, series=series, template=template,
        speaker=speaker, occasion=occasion, phrase=yes_sub_title.strip(),
        variant_rule=variant, cutoff_ts=cutoff_ts, governing_sources=sources,
        early_resolution=bool(market.get("can_close_early")),
        cancellation_clause=False, is_nqe=False,
        resolution_risk=risk, confidence=confidence, notes=note,
    )


def parse_raw_records(records: Iterable[Mapping[str, Any]]) -> list[Resolution]:
    """Parse ``{"market": ..., "event": ...}`` raw records (from discovery) into Resolutions.

    This is the deterministic baseline the agent reviews (§4.3/D6): it fixes speaker/occasion/
    phrase/variant/cutoff/sources mechanically and assigns a conservative risk tier; agent judgment
    then refines the tier for the low-confidence / high-risk tail via review.csv."""
    return [parse_market(rec["market"], rec["event"]) for rec in records]


def write_resolutions(resolutions: Iterable[Resolution], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in resolutions:
            fh.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_resolutions(path: Path) -> list[Resolution]:
    out: list[Resolution] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Resolution.model_validate_json(line))
    return out


def iter_resolutions(path: Path) -> Iterator[Resolution]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Resolution.model_validate_json(line)
