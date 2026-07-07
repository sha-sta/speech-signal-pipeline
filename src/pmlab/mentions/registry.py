"""Build the mention registry (catalog ⋈ parsed resolutions ⋈ candle coverage) + the review.csv
human-gate round-trip (§4.4).

The registry is the study's unit of analysis. ``review.csv`` is the small committed artifact the
owner spot-checks: it is sorted confidence-ascending (worst parses first) and a handful of columns
are **re-ingestable** — the owner edits risk/confidence/variant/phrase/notes for flagged rows and
:func:`apply_review` overlays them back, after which ``eligible`` is recomputed deterministically.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from pmlab.data.schemas import (
    CATALOG_COLUMNS,
    REGISTRY_COLUMNS,
    MentionRegistry,
    empty_registry,
)
from pmlab.mentions.resolution import BINARY_TEMPLATES, Resolution

log = logging.getLogger("pmlab.mentions.registry")

# Columns the owner may override in review.csv; everything else is read-only context.
REVIEW_EDITABLE = ("resolution_risk", "confidence", "variant_rule", "phrase", "occasion", "notes")
# Human-facing column order for review.csv (worst-parse-first triage view).
REVIEW_COLUMNS = [
    "market_ticker", "series_ticker", "speaker", "format", "template",
    "phrase", "yes_sub_title", "variant_rule", "occasion",
    "resolution_risk", "confidence", "is_nqe", "eligible",
    "n_settlement_sources", "has_candles", "result",
    "notes", "rules_primary",  # full governing_sources list stays in the parquet, not the CSV
]


def _clean_speaker(parsed: str, fallback: str) -> str:
    """Prefer the per-market parsed speaker (the actual person), but fall back to the series-level
    label for generic speakers (``any <org> representative`` / ``any person`` / NQE)."""
    p = (parsed or "").strip()
    if p and not p.lower().startswith("any "):
        return p
    return fallback


def _resolution_frame(resolutions: Iterable[Resolution]) -> pd.DataFrame:
    rows = []
    for r in resolutions:
        rows.append(
            {
                "market_ticker": r.market_ticker,
                "_parsed_speaker": r.speaker,
                "template": r.template,
                "phrase": r.phrase,
                "variant_rule": r.variant_rule,
                "occasion": r.occasion,
                "cutoff_ts": r.cutoff_ts,
                "governing_sources": "|".join(r.governing_sources),
                "early_resolution": r.early_resolution,
                "cancellation_clause": r.cancellation_clause,
                "is_nqe": r.is_nqe,
                "resolution_risk": r.resolution_risk,
                "confidence": r.confidence,
                "notes": r.notes,
            }
        )
    cols = [
        "market_ticker", "_parsed_speaker", "template", "phrase", "variant_rule", "occasion",
        "cutoff_ts", "governing_sources", "early_resolution", "cancellation_clause", "is_nqe",
        "resolution_risk", "confidence", "notes",
    ]
    return pd.DataFrame(rows, columns=cols)


def compute_eligible(df: pd.DataFrame) -> pd.Series:
    """§4.3 model-set gate: a binary phrase prediction (speech/earnings/duration, not NQE),
    low/medium risk, confidently parsed. Derived — never hand-set — so the invariant survives
    review re-ingest."""
    return (
        df["template"].isin(list(BINARY_TEMPLATES))
        & (~df["is_nqe"].astype(bool))
        & df["resolution_risk"].isin(["low", "medium"])
        & (df["confidence"].astype(float) >= 0.8)
    )


def build_registry(
    catalog: pd.DataFrame,
    resolutions: Iterable[Resolution],
) -> pd.DataFrame:
    """Join the catalog with parsed resolutions; carry candle coverage if already attached."""
    cat = catalog.copy()
    for col, default in (("has_candles", False), ("n_candles", 0)):
        if col not in cat.columns:
            log.warning("catalog missing %s; defaulting (run attach_candle_coverage first)", col)
            cat[col] = default

    res = _resolution_frame(resolutions)
    missing = set(cat["market_ticker"]) - set(res["market_ticker"])
    if missing:
        log.warning("%d catalog markets lack a resolution; dropped from registry", len(missing))

    merged = cat.merge(res, on="market_ticker", how="inner", validate="one_to_one")
    if merged.empty:
        return empty_registry()

    merged["speaker"] = [
        _clean_speaker(p, s)
        for p, s in zip(merged["_parsed_speaker"], merged["speaker"], strict=True)
    ]
    merged = merged.drop(columns=["_parsed_speaker"])
    merged["eligible"] = compute_eligible(merged)

    registry = merged[REGISTRY_COLUMNS]
    return MentionRegistry.validate(registry)


def write_review_csv(
    registry: pd.DataFrame, path: Path, *, only_flagged: bool = False, rule_chars: int = 200
) -> Path:
    """Emit the human-gate triage CSV, worst parses (lowest confidence, then higher risk) first.

    ``rules_primary`` is truncated to ``rule_chars`` (templates are short) so the committed CSV
    stays small at universe scale. ``only_flagged=True`` restricts to rows that actually warrant a
    look — non-``low`` risk, low confidence, unrecognized template, or ineligible — which is the
    committable artifact when the full registry runs to thousands of clean rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    risk_rank = {"high": 0, "medium": 1, "low": 2}
    view = registry.copy()
    if only_flagged:  # rows that actually warrant a look — not the clean low-risk eligible bulk
        flag = (
            (view["resolution_risk"] != "low")
            | (~view["eligible"])
            | (view["variant_rule"] == "fuzzy")
            | (view["confidence"] < 0.8)
        )
        view = view[flag]
    view["_risk_rank"] = view["resolution_risk"].map(risk_rank).fillna(1)
    view = view.sort_values(["confidence", "_risk_rank", "market_ticker"])
    view = view.drop(columns="_risk_rank")
    view = view[REVIEW_COLUMNS].copy()
    # collapse embedded newlines so each market is one CSV line (clean git diffs), then truncate
    rules = view["rules_primary"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    view["rules_primary"] = rules.str.slice(0, rule_chars)
    view.to_csv(path, index=False)
    return path


def read_review_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_review(registry: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """Overlay owner edits (REVIEW_EDITABLE columns) from a review.csv back onto the registry,
    keyed by ``market_ticker``; recompute ``eligible``; re-validate."""
    out = registry.set_index("market_ticker").copy()
    edits = review.set_index("market_ticker")
    unknown = set(edits.index) - set(out.index)
    if unknown:
        log.warning("review.csv has %d unknown market_tickers; ignored", len(unknown))
    for col in REVIEW_EDITABLE:
        if col in edits.columns:
            incoming = edits[col].reindex(out.index)
            out[col] = incoming.where(incoming.notna(), out[col])
    out = out.reset_index()
    out["eligible"] = compute_eligible(out)
    return MentionRegistry.validate(out[REGISTRY_COLUMNS])


def persist_registry(registry: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    MentionRegistry.validate(registry)
    registry.to_parquet(path, index=False)
    return path


def load_registry(path: Path) -> pd.DataFrame:
    return MentionRegistry.validate(pd.read_parquet(path))


assert set(CATALOG_COLUMNS).issubset(REGISTRY_COLUMNS)  # registry is a catalog superset
