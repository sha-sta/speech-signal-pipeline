"""UTC alignment + anchor QA for streaming transcripts.

Caption/ASR timings usually arrive relative to stream or video start. This module places them on
absolute UTC by anchoring against independent reactions in a paired time series: for each anchor,
the transcript-relative time its associated event is judged to have occurred should coincide with
the time series crossing some threshold (default: a price series crossing :data:`CROSS_LEVEL`).
The offset ``video_start_utc = median(cross_utc − said_rel)`` over anchors; the spread of
per-anchor residuals is the QA (``|Δ| ≤ MAX_ABS_RESIDUAL_S``) and doubles as a reaction-lag
measurement. Occasions failing QA on more than a third of their anchors should be excluded by the
caller.

Anchor detection is pluggable (:data:`AnchorTimeFn`): the default (:func:`phrase_said_time`) looks
for a phrase in the transcript, but any function of a word frame → time works, e.g. a sentiment
threshold or a named-entity mention.

Pure given (raw words, anchors, a paired time series) — no network.
"""

from __future__ import annotations

import bisect
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from pmlab.corpus.store import PhraseSpec, parse_phrase_spec
from pmlab.signals.transcripts import RawWord

log = logging.getLogger("pmlab.signals.align")

CROSS_LEVEL = 0.90
MAX_ABS_RESIDUAL_S = 120.0  # anchor tolerance
QA_MIN_ANCHOR_FRAC = 2 / 3  # >= this fraction of anchors must be within tolerance

TRANSCRIPT_COLUMNS = [
    "occasion_id", "word_idx", "word", "t_start_utc", "t_end_utc", "source", "asr_conf",
]


class Transcript(pa.DataFrameModel):
    """One transcribed word on absolute UTC. ``t_end_utc`` is the field a downstream consumer
    should gate any leakage-safe prefix on."""

    occasion_id: Series[str] = pa.Field(nullable=False)
    word_idx: Series[int] = pa.Field(ge=0)
    word: Series[str] = pa.Field(nullable=False)
    t_start_utc: Series[int] = pa.Field(ge=0)
    t_end_utc: Series[int] = pa.Field(ge=0)
    source: Series[str] = pa.Field(nullable=False)  # e.g. "youtube-captions", "whisper:live"
    asr_conf: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)

    class Config:
        strict = True
        coerce = True
        unique = ["occasion_id", "word_idx"]


AnchorTimeFn = Callable[[pd.DataFrame], "int | None"]


def phrase_occurrence_times(words: pd.DataFrame, spec: PhraseSpec) -> list[int]:
    """The ``t_end_utc`` of the final word of each occurrence of ``spec`` in ``words``, in time
    order. Generic text/timing alignment: builds the space-joined transcript text with a
    char→word map, then maps each regex match's end back to the word that closes it."""
    if not spec.matchable or spec.pattern is None or words.empty:
        return []
    toks = words["word"].astype(str).tolist()
    ends = words["t_end_utc"].astype("int64").tolist()
    starts: list[int] = []
    pos = 0
    for w in toks:
        starts.append(pos)
        pos += len(w) + 1  # + single joining space
    text = " ".join(toks)
    times: list[int] = []
    for m in spec.pattern.finditer(text):
        last_char = max(m.start(), m.end() - 1)
        wi = bisect.bisect_right(starts, last_char) - 1
        wi = min(max(wi, 0), len(ends) - 1)
        times.append(ends[wi])
    return sorted(times)


def phrase_said_time(spec: PhraseSpec) -> AnchorTimeFn:
    """Default anchor-time extractor: the time ``spec``'s ``min_count``-th occurrence is satisfied
    in a word frame, else ``None``."""

    def _fn(words: pd.DataFrame) -> int | None:
        times = phrase_occurrence_times(words, spec)
        return times[spec.min_count - 1] if len(times) >= spec.min_count else None

    return _fn


@dataclass(frozen=True)
class Anchor:
    """One price-reaction anchor: pairs a time series (``series_key``, matched against a
    ``ticker`` column in the frame passed to :func:`estimate_alignment`) with a function that
    returns the transcript-relative time its associated event is judged to have occurred (or
    ``None`` if it never does). Use :meth:`from_phrase` for the common phrase-in-transcript case,
    or build ``said_time_fn`` however else suits the domain."""

    series_key: str
    said_time_fn: AnchorTimeFn

    @classmethod
    def from_phrase(cls, series_key: str, label: str) -> Anchor:
        return cls(series_key, phrase_said_time(parse_phrase_spec(label)))


@dataclass
class AlignResult:
    occasion_id: str
    video_start_utc: int | None
    n_anchors: int
    n_ok: int
    residuals_s: list[float] = field(default_factory=list)
    reaction_lags_s: list[float] = field(default_factory=list)
    aligned: bool = False
    # anchor-fusion provenance: 'price_only' | 'release+price_qa' | 'price_over_release' |
    # 'release_only' | ''.
    method: str = ""
    release_start_utc: int | None = None

    @property
    def max_abs_residual_s(self) -> float:
        return max((abs(r) for r in self.residuals_s), default=float("nan"))


def _raw_to_word_frame(occasion_id: str, words: list[RawWord]) -> pd.DataFrame:
    """A frame ``said_time_fn`` can consume, carrying *video-relative ms* in the t_end field."""
    return pd.DataFrame({
        "occasion_id": occasion_id,
        "word_idx": range(len(words)),
        "word": [w.word for w in words],
        "t_start_utc": [w.t_start_ms for w in words],
        "t_end_utc": [w.t_end_ms for w in words],
    })


def first_cross_utc(
    mbars: pd.DataFrame, level: float = CROSS_LEVEL, *, window: tuple[int, int] | None = None
) -> int | None:
    """First bar ts (UTC) where a price series crosses ``level`` on its "yes" side.

    Uses the trade print ``price_close`` where present, else the mid ``(bid+ask)/2``. ``window``
    (unix-s ``[lo, hi]``) bounds the search to the in-play span. ESSENTIAL: a near-certain series
    can sit at/above ``level`` for a long time pre-event, so an un-windowed 'first cross' anchors
    to whenever the series opened, not to the anchor event itself."""
    if mbars.empty:
        return None
    b = mbars.sort_values("ts")
    if window is not None:
        b = b[(b["ts"] >= window[0]) & (b["ts"] <= window[1])]
        if b.empty:
            return None
    price = b["price_close"].to_numpy("float64")
    mid = (b["yes_bid_close"].to_numpy("float64") + b["yes_ask_close"].to_numpy("float64")) / 2.0
    ts = b["ts"].to_numpy("int64")
    ref = np.where(np.isfinite(price), price, mid)
    hit = np.where(ref >= level)[0]
    return int(ts[hit[0]]) if hit.size else None


def estimate_alignment(
    occasion_id: str,
    words: list[RawWord],
    anchors: list[Anchor],
    candles: pd.DataFrame,
    *,
    window: tuple[int, int] | None = None,
    release_start_utc: int | None = None,
) -> AlignResult:
    """Estimate ``video_start_utc`` + QA.

    ``release_start_utc`` is an independent absolute-time anchor (e.g. a live-broadcast start
    timestamp), used with no reaction-lag bias. Fusion order: release anchor primary when it
    agrees with the price-anchor median within ``MAX_ABS_RESIDUAL_S`` (VOD-trim guard); price
    median wins on disagreement (flagged); release-only accepted when no price anchor exists
    (flagged, weaker QA). Without a release anchor: the original ≥3-concordant price rule."""
    wf = _raw_to_word_frame(occasion_id, words)
    bars_by = (
        {} if candles.empty else {str(t): g for t, g in candles.groupby("ticker", sort=False)}
    )
    offsets: list[float] = []
    per_anchor: list[tuple[str, float]] = []  # (series_key, offset)
    for a in anchors:
        said_rel_ms = a.said_time_fn(wf)
        mbars = bars_by.get(a.series_key)
        if said_rel_ms is None or mbars is None:
            continue
        cross = first_cross_utc(mbars, window=window)
        if cross is None:
            continue
        offset = cross - said_rel_ms / 1000.0  # candidate video_start_utc
        offsets.append(offset)
        per_anchor.append((a.series_key, offset))

    if not offsets and release_start_utc is None:
        return AlignResult(occasion_id, None, 0, 0)
    if not offsets:  # release-only occasion: accept, flag weaker QA
        return AlignResult(
            occasion_id=occasion_id, video_start_utc=int(release_start_utc or 0),
            n_anchors=0, n_ok=0, aligned=True,
            method="release_only", release_start_utc=release_start_utc,
        )

    # Robust price-anchor offset: median, then refine to the inlier cluster (rejects spurious
    # early/late crosses — a series can cross the level for reasons not tied to the exact anchor).
    price_start = statistics.median(offsets)
    inliers = [o for o in offsets if abs(o - price_start) <= MAX_ABS_RESIDUAL_S]
    if inliers:
        price_start = statistics.median(inliers)

    method = "price_only"
    start = price_start
    if release_start_utc is not None:
        if abs(release_start_utc - price_start) <= MAX_ABS_RESIDUAL_S:
            start, method = float(release_start_utc), "release+price_qa"
        else:
            method = "price_over_release"  # trimmed VOD / wrong release ts — trust the price series

    residuals = [off - start for _, off in per_anchor]  # vs release: true reaction lags
    lags = [off - min(offsets) for _, off in per_anchor]  # relative lag proxy
    n_ok = sum(1 for r in residuals if abs(r) <= MAX_ABS_RESIDUAL_S)
    if method == "release+price_qa":
        # Release gives the clock; concordant price anchors corroborate. 1–2 anchors suffice.
        aligned = n_ok >= 1
    else:
        # exclude if > 1/3 of anchors fail — but require ≥3 concordant anchors for a small set, so
        # sub-minute 3-anchor consensus counts rather than being vetoed by 1–2 noisy ones.
        aligned = n_ok >= 3 and (n_ok / len(offsets)) >= max(0.5, QA_MIN_ANCHOR_FRAC - 0.2)
    return AlignResult(
        occasion_id=occasion_id, video_start_utc=int(round(start)),
        n_anchors=len(offsets), n_ok=n_ok, residuals_s=residuals, reaction_lags_s=lags,
        aligned=aligned, method=method, release_start_utc=release_start_utc,
    )


def to_utc_transcript(
    occasion_id: str, words: list[RawWord], video_start_utc: int, source: str
) -> pd.DataFrame:
    """Emit the :class:`Transcript` frame on absolute UTC using the estimated offset."""
    rows = []
    for i, w in enumerate(words):
        rows.append({
            "occasion_id": occasion_id,
            "word_idx": i,
            "word": w.word,
            "t_start_utc": int(video_start_utc + round(w.t_start_ms / 1000.0)),
            "t_end_utc": int(video_start_utc + round(w.t_end_ms / 1000.0)),
            "source": source,
            "asr_conf": np.nan,
        })
    out = pd.DataFrame(rows, columns=TRANSCRIPT_COLUMNS)
    return Transcript.validate(out)
