"""Parser tests against recorded ``rules_primary`` fixtures (M1 acceptance).

Fixtures ``kalshi_mentions_event_{ftn,earnings}.json`` are real events recorded live 2026-07-03;
the FTN event exercises all three templates and every variant style in one shot.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pmlab.mentions.resolution import (
    Resolution,
    classify_template,
    classify_variant,
    parse_market,
    parse_raw_records,
    read_resolutions,
    write_resolutions,
)

Fixture = Callable[[str], Any]


def _event(fixture: Fixture, name: str) -> dict[str, Any]:
    event: dict[str, Any] = fixture(name)["events"][0]
    return event


def _market(event: dict[str, Any], suffix: str) -> dict[str, Any]:
    return next(m for m in event["markets"] if m["ticker"].endswith(suffix))


# --- template + variant classifiers ------------------------------------------------------------


def test_classify_template_covers_three_shapes(fixture: Fixture) -> None:
    ftn = _event(fixture, "kalshi_mentions_event_ftn.json")
    earn = _event(fixture, "kalshi_mentions_event_earnings.json")
    speech = _market(ftn, "-TRUM")
    nqe = _market(ftn, "-NQE")
    earnings = _market(earn, "-HYPE")
    assert classify_template(speech["rules_primary"], speech["yes_sub_title"]) == "speech"
    assert classify_template(nqe["rules_primary"], nqe["yes_sub_title"]) == "nqe"
    assert classify_template(earnings["rules_primary"], earnings["yes_sub_title"]) == "earnings"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Nuclear", "exact"),
        ("Israel / Israeli", "plural_possessive"),
        ("Afford / Affordable / Affordability", "plural_possessive"),
        ("Oil / Gas / Gasoline", "aliases_listed"),
        ("China / Russia", "aliases_listed"),
        ("Inflation (3+ times)", "count_threshold"),
        ("", "fuzzy"),
    ],
)
def test_classify_variant(label: str, expected: str) -> None:
    assert classify_variant(label) == expected


# --- full parse of real markets ----------------------------------------------------------------


def test_parse_speech_extracts_speaker_and_occasion(fixture: Fixture) -> None:
    ftn = _event(fixture, "kalshi_mentions_event_ftn.json")
    r = parse_market(_market(ftn, "-TRUM"), ftn)
    assert r.template == "speech"
    assert r.speaker == "Lindsey Graham"
    assert r.occasion == "Face the Nation"
    assert r.variant_rule == "count_threshold"
    assert r.phrase == "Trump (3+ times)"
    assert r.cutoff_ts and r.cutoff_ts > 0
    assert len(r.governing_sources) == 14
    # count threshold bumps a video-primary broadcast from low → medium, but still eligible
    assert r.resolution_risk == "medium"
    assert r.eligible


def test_parse_speech_exact_phrase_is_low_risk(fixture: Fixture) -> None:
    ftn = _event(fixture, "kalshi_mentions_event_ftn.json")
    r = parse_market(_market(ftn, "-NUCL"), ftn)
    assert r.variant_rule == "exact"
    assert r.resolution_risk == "low"
    assert r.eligible


def test_parse_earnings_video_primary_is_low_risk_despite_single_source(fixture: Fixture) -> None:
    earn = _event(fixture, "kalshi_mentions_event_earnings.json")
    r = parse_market(_market(earn, "-HYPE"), earn)
    assert r.template == "earnings"
    assert r.speaker.startswith("any Broadcom")
    assert "earnings call" in r.occasion
    # only one settlement source (the company) — but the call video/transcript is canonical → low
    assert len(r.governing_sources) == 1
    assert r.resolution_risk == "low"


def test_parse_nqe_leg_is_ineligible(fixture: Fixture) -> None:
    ftn = _event(fixture, "kalshi_mentions_event_ftn.json")
    r = parse_market(_market(ftn, "-NQE"), ftn)
    assert r.is_nqe
    assert r.cancellation_clause
    assert not r.eligible  # cancellation leg is not a phrase prediction


def test_every_fixture_market_parses_to_a_known_template(fixture: Fixture) -> None:
    for name in ("kalshi_mentions_event_ftn.json", "kalshi_mentions_event_earnings.json"):
        ev = _event(fixture, name)
        for m in ev["markets"]:
            r = parse_market(m, ev)
            assert r.template in ("speech", "earnings", "nqe"), (m["ticker"], r.template)
            assert r.market_ticker == m["ticker"]
            assert r.series == ev["series_ticker"]


# --- eligibility gate --------------------------------------------------------------------------


def test_eligibility_requires_low_med_risk_and_confidence() -> None:
    base: dict[str, Any] = {
        "market_ticker": "X-Y", "series": "X", "template": "speech", "speaker": "A",
        "occasion": "O", "phrase": "p", "variant_rule": "exact", "governing_sources": ["S"],
    }
    assert Resolution(**base, resolution_risk="low", confidence=0.9).eligible
    assert Resolution(**base, resolution_risk="medium", confidence=0.8).eligible
    assert not Resolution(**base, resolution_risk="high", confidence=0.99).eligible
    assert not Resolution(**base, resolution_risk="low", confidence=0.79).eligible
    assert not Resolution(**base, resolution_risk="low", confidence=0.9, is_nqe=True).eligible


# --- JSONL round-trip + raw-record parse -------------------------------------------------------


def test_jsonl_roundtrip(fixture: Fixture, tmp_path: Path) -> None:
    ftn = _event(fixture, "kalshi_mentions_event_ftn.json")
    parsed = [parse_market(m, ftn) for m in ftn["markets"]]
    path = tmp_path / "resolutions.jsonl"
    n = write_resolutions(parsed, path)
    loaded = read_resolutions(path)
    assert n == len(parsed) == len(loaded)
    assert [r.model_dump() for r in loaded] == [r.model_dump() for r in parsed]


def _template_rec(fixture: Fixture, kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    recs = fixture("kalshi_mentions_templates.json")["records"]
    rec = next(r for r in recs if r["kind"] == kind)
    return rec["market"], rec["event"]


def test_duration_template_is_elevated_risk_and_eligible(fixture: Fixture) -> None:
    market, event = _template_rec(fixture, "duration")
    r = parse_market(market, event)
    assert r.template == "duration"
    assert r.speaker == "Donald Trump"
    assert r.occasion.startswith("public statements")
    # no canonical video + open window → elevated to medium, but still a binary phrase → eligible
    assert r.resolution_risk == "medium"
    assert r.eligible


def test_fed_says_at_occasion_parses_as_speech(fixture: Fixture) -> None:
    market, event = _template_rec(fixture, "fed_at")
    r = parse_market(market, event)
    assert r.template == "speech"  # "says X at his <occasion>" form (not "as part of")
    assert "Chair of the Federal Reserve" in r.speaker
    assert r.resolution_risk == "low"  # FOMC presser is video-primary


def test_length_and_superlative_are_recognized_but_ineligible(fixture: Fixture) -> None:
    length_m, length_e = _template_rec(fixture, "length")
    sup_m, sup_e = _template_rec(fixture, "superlative")
    length = parse_market(length_m, length_e)
    sup = parse_market(sup_m, sup_e)
    assert length.template == "length"
    assert sup.template == "superlative"
    # recognized (not 'other'), but out of scope for the binary phrase-frequency model
    assert not length.eligible
    assert not sup.eligible


def test_parse_raw_records_matches_parse_market(fixture: Fixture) -> None:
    ftn = _event(fixture, "kalshi_mentions_event_ftn.json")
    ev_ctx = {k: ftn[k] for k in ("event_ticker", "series_ticker", "settlement_sources")}
    records = [{"market": m, "event": ev_ctx} for m in ftn["markets"]]
    out = parse_raw_records(records)
    assert len(out) == len(ftn["markets"])
    assert {r.template for r in out} == {"speech", "nqe"}
