"""Registry-build and review.csv round-trip tests (no network)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from pmlab.data.schemas import CATALOG_COLUMNS, MentionCatalog, MentionRegistry
from pmlab.mentions.discover import _catalog_row
from pmlab.mentions.registry import (
    apply_review,
    build_registry,
    compute_eligible,
    read_review_csv,
    write_review_csv,
)
from pmlab.mentions.resolution import Resolution, parse_market

Fixture = Callable[[str], Any]


def _catalog_and_resolutions(fixture: Fixture) -> tuple[pd.DataFrame, list[Resolution]]:
    ev = fixture("kalshi_mentions_event_ftn.json")["events"][0]
    ev["series_title"] = "Face The Nation"
    rows = [_catalog_row(m, ev) for m in ev["markets"]]
    catalog = MentionCatalog.validate(pd.DataFrame(rows, columns=CATALOG_COLUMNS))
    resolutions = [parse_market(m, ev) for m in ev["markets"]]
    return catalog, resolutions


def test_build_registry_joins_refines_speaker_and_validates(fixture: Fixture) -> None:
    catalog, resolutions = _catalog_and_resolutions(fixture)
    reg = build_registry(catalog, resolutions)
    MentionRegistry.validate(reg)
    assert len(reg) == 13
    # per-market parsed speaker replaces the coarse series-level label
    trum = reg[reg["market_ticker"].str.endswith("-TRUM")].iloc[0]
    assert trum["speaker"] == "Lindsey Graham"
    # NQE leg carried but ineligible; exact-phrase leg eligible
    nqe = reg[reg["market_ticker"].str.endswith("-NQE")].iloc[0]
    assert bool(nqe["is_nqe"]) and not bool(nqe["eligible"])
    assert bool(reg[reg["market_ticker"].str.endswith("-NUCL")].iloc[0]["eligible"])


def test_build_registry_carries_default_coverage(fixture: Fixture) -> None:
    catalog, resolutions = _catalog_and_resolutions(fixture)
    # catalog carries coverage columns defaulted (False/0) until attach_candle_coverage runs
    reg = build_registry(catalog, resolutions)
    assert (~reg["has_candles"]).all()
    assert (reg["n_candles"] == 0).all()


def test_compute_eligible_matches_rule() -> None:
    df = pd.DataFrame(
        {
            "template": ["speech", "speech", "nqe", "speech", "length"],
            "is_nqe": [False, False, True, False, False],
            "resolution_risk": ["low", "high", "low", "medium", "low"],
            "confidence": [0.9, 0.9, 0.99, 0.75, 0.9],
        }
    )
    # eligible only when binary-template AND low/med risk AND conf≥0.8 AND not NQE
    assert list(compute_eligible(df)) == [True, False, False, False, False]


def test_review_csv_sorted_worst_first(fixture: Fixture, tmp_path: Path) -> None:
    catalog, resolutions = _catalog_and_resolutions(fixture)
    reg = build_registry(catalog, resolutions)
    path = tmp_path / "review.csv"
    write_review_csv(reg, path)
    review = read_review_csv(path)
    assert list(review["confidence"]) == sorted(review["confidence"])  # ascending confidence


def test_apply_review_overrides_risk_and_recomputes_eligible(
    fixture: Fixture, tmp_path: Path
) -> None:
    catalog, resolutions = _catalog_and_resolutions(fixture)
    reg = build_registry(catalog, resolutions)
    path = tmp_path / "review.csv"
    write_review_csv(reg, path)

    review = read_review_csv(path)
    target = reg[reg["market_ticker"].str.endswith("-NUCL")]["market_ticker"].iloc[0]
    assert bool(reg[reg["market_ticker"] == target]["eligible"].iloc[0]) is True
    # owner downgrades this parse to high risk
    review.loc[review["market_ticker"] == target, "resolution_risk"] = "high"

    updated = apply_review(reg, review)
    row = updated[updated["market_ticker"] == target].iloc[0]
    assert row["resolution_risk"] == "high"
    assert bool(row["eligible"]) is False  # eligibility re-derived after override


def test_apply_review_confidence_downgrade_makes_ineligible(
    fixture: Fixture, tmp_path: Path
) -> None:
    catalog, resolutions = _catalog_and_resolutions(fixture)
    reg = build_registry(catalog, resolutions)
    review = reg[["market_ticker", "confidence"]].copy()
    target = reg[reg["market_ticker"].str.endswith("-NUCL")]["market_ticker"].iloc[0]
    review.loc[review["market_ticker"] == target, "confidence"] = 0.5

    updated = apply_review(reg, review)
    assert bool(updated[updated["market_ticker"] == target]["eligible"].iloc[0]) is False
