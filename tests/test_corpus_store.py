"""Phrase-matcher, speaker-canonicalisation, occurrence-index and store round-trip tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

from pmlab.corpus.store import (
    build_occurrences,
    canonical_speaker,
    count_occurrences,
    load_corpus,
    merge_corpus,
    parse_phrase_spec,
    persist_corpus,
    phrase_key,
    phrase_occurs,
    to_corpus_frame,
)


@pytest.mark.parametrize(
    ("label", "text", "count", "occurs"),
    [
        ("Nuclear", "We go nuclear. Nuclear energy, nuclear!", 3, True),
        ("Billionaire", "the billionaires and one billionaire", 2, True),  # plural tolerated
        ("Oil / Gas / Gasoline", "oil prices and gasoline", 2, True),      # any alias
        ("Afford / Affordable / Affordability", "affordability, affordable", 2, True),
        ("Senator (3+ times)", "Senator Senator", 2, False),               # below threshold
        ("Senator (3+ times)", "Senator, Senator, Senator", 3, True),      # meets threshold
        ("World Cup", "the World Cups are here", 1, True),                 # multiword + plural
        ("ICE", "nice ice cream", 1, True),                               # whole-word, not 'nice'
        ("China / Russia", "China, China and Russia", 3, True),
        ("", "anything at all", 0, False),                                # empty → unmatchable
    ],
)
def test_phrase_matcher(label: str, text: str, count: int, occurs: bool) -> None:
    spec = parse_phrase_spec(label)
    assert count_occurrences(text, spec) == count
    assert phrase_occurs(text, spec) is occurs


def test_phrase_spec_metadata() -> None:
    spec = parse_phrase_spec("Senator (3+ times)")
    assert spec.min_count == 3
    assert spec.alternates == ["Senator"]
    assert spec.variant == "count_threshold"
    assert spec.matchable
    assert not parse_phrase_spec("").matchable


def test_phrase_key_groups_recurrence_and_separates_thresholds() -> None:
    # order-independent alternate set → same key
    assert phrase_key("Oil / Gas") == phrase_key("Gas / Oil")
    # a counting threshold is a distinct prediction → distinct key
    assert phrase_key("Senator") != phrase_key("Senator (3+ times)")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Senator Jane Doe", "jane doe"),                 # generic title stripping, no alias
        ("the Chair of the Federal Reserve", "jerome powell"),
        ("Chair Jerome H. Powell", "jerome powell"),      # title + middle initial stripped
        ("Governor Michael S. Barr", "michael barr"),
        ("Kathy Hochul", "kathy hochul"),
    ],
)
def test_canonical_speaker(raw: str, expected: str) -> None:
    assert canonical_speaker(raw) == expected


def test_build_occurrences_only_matches_own_speaker() -> None:
    corpus = to_corpus_frame(
        [
            {"doc_id": "a", "speaker": "Jerome Powell", "format": "speech", "occasion": "o1",
             "date_ts": 100, "source": "t", "url": "", "n_words": 2, "text": "inflation risk"},
            {"doc_id": "b", "speaker": "Jane Doe", "format": "speech", "occasion": "o2",
             "date_ts": 200, "source": "t", "url": "", "n_words": 2, "text": "tariffs china"},
        ]
    )
    powell_spec = parse_phrase_spec("inflation")
    groups = {("jerome powell", "speech"): [powell_spec]}
    occ = build_occurrences(corpus, groups)
    # only Powell's transcript is scanned for the Powell spec (Doe's row is not in the group)
    assert set(occ["speaker_key"]) == {"jerome powell"}
    assert occ.iloc[0]["occurred"]
    assert int(occ.iloc[0]["count"]) == 1


def test_corpus_store_roundtrip_and_merge(tmp_path: Path) -> None:
    frame = to_corpus_frame(
        [
            {"doc_id": "x", "speaker": "Jerome Powell", "format": "speech", "occasion": "o",
             "date_ts": 1, "source": "fed", "url": "u", "n_words": 1, "text": "hi"},
        ]
    )
    path = tmp_path / "c.parquet"
    persist_corpus(frame, path)
    assert len(load_corpus(path)) == 1
    # merge is idempotent on doc_id (re-fetch keeps one row)
    merged = merge_corpus(load_corpus(path), frame)
    assert len(merged) == 1


def test_load_missing_corpus_is_empty(tmp_path: Path) -> None:
    assert load_corpus(tmp_path / "nope.parquet").empty


def test_to_corpus_frame_validates_schema() -> None:
    # date_ts must be present + non-negative (leakage boundary) — a bad row fails validation.
    with pytest.raises(SchemaError):
        to_corpus_frame([{"doc_id": "z", "speaker": "X", "format": "speech", "occasion": "",
                          "date_ts": -5, "source": "t", "url": "", "n_words": 0, "text": ""}])


def test_corpus_frame_columns() -> None:
    df = to_corpus_frame([])
    assert list(df.columns) == [
        "doc_id", "speaker", "format", "occasion", "date_ts", "source", "url", "n_words", "text",
    ]
    assert isinstance(df, pd.DataFrame)
