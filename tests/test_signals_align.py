"""Anchor-fusion tests for align.estimate_alignment.

Synthetic tape: a transcript whose target word sits at a known video-relative time, and 1-min
candles whose 0.90-cross sits at a known UTC bar — so every fusion path's arithmetic is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmlab.signals.align import Anchor, estimate_alignment
from pmlab.signals.transcripts import RawWord

TRUE_START = 1_777_600_000  # true video_start_utc
SAID_REL_S = 600            # "iran" ends ~600 s into the video
CROSS_TS = TRUE_START + SAID_REL_S + 60  # market crosses 0.90 one bar after the utterance


def _words() -> list[RawWord]:
    toks = ["we", "will", "talk", "about", "iran", "today"]
    out = []
    for i, w in enumerate(toks):
        start_ms = (SAID_REL_S - 5 + i) * 1000
        out.append(RawWord(word=w, t_start_ms=start_ms, t_end_ms=start_ms + 900))
    return out


def _candles(ticker: str, cross_ts: int) -> pd.DataFrame:
    ts = np.arange(cross_ts - 1800, cross_ts + 1800, 60, dtype="int64")
    price = np.where(ts >= cross_ts, 0.95, 0.50)
    return pd.DataFrame({
        "ticker": ticker, "ts": ts,
        "yes_bid_close": price - 0.02, "yes_ask_close": price + 0.02,
        "price_close": price,
    })


def _anchor(ticker: str = "T-IRAN") -> Anchor:
    return Anchor.from_phrase(ticker, "Iran")


def test_price_only_needs_three_concordant_anchors():
    candles = pd.concat(
        [_candles(f"T-{i}", CROSS_TS) for i in range(3)], ignore_index=True
    )
    anchors = [Anchor.from_phrase(f"T-{i}", "Iran") for i in range(3)]
    r = estimate_alignment("OCC", _words(), anchors, candles)
    assert r.method == "price_only"
    assert r.aligned and r.n_ok == 3
    # offset = cross − said_rel; the word "iran" ends at ~595.9 s
    assert abs(r.video_start_utc - (TRUE_START + 60)) < 10  # reaction-lag bias, ≤ 1 bar

    r1 = estimate_alignment("OCC", _words(), anchors[:1], candles)
    assert r1.method == "price_only" and not r1.aligned  # 1 anchor alone: rejected


def test_release_plus_price_qa_accepts_single_anchor():
    r = estimate_alignment(
        "OCC", _words(), [_anchor()], _candles("T-IRAN", CROSS_TS),
        release_start_utc=TRUE_START,
    )
    assert r.method == "release+price_qa"
    assert r.aligned and r.n_ok == 1
    assert r.video_start_utc == TRUE_START  # release gives the unbiased clock
    # residual vs release = true market reaction lag ≥ 0 (cross one bar after utterance)
    assert 0 < r.residuals_s[0] <= 120


def test_price_overrides_disagreeing_release():
    bogus_release = TRUE_START - 600  # e.g. trimmed VOD / wrong release ts
    candles = pd.concat(
        [_candles(f"T-{i}", CROSS_TS) for i in range(3)], ignore_index=True
    )
    anchors = [Anchor.from_phrase(f"T-{i}", "Iran") for i in range(3)]
    r = estimate_alignment("OCC", _words(), anchors, candles, release_start_utc=bogus_release)
    assert r.method == "price_over_release"
    assert r.aligned  # 3 concordant price anchors still carry it
    assert abs(r.video_start_utc - (TRUE_START + 60)) < 10


def test_release_only_flagged_alignment():
    r = estimate_alignment("OCC", _words(), [], pd.DataFrame(), release_start_utc=TRUE_START)
    assert r.aligned and r.method == "release_only"
    assert r.video_start_utc == TRUE_START and r.n_anchors == 0


def test_nothing_to_anchor_fails():
    r = estimate_alignment("OCC", _words(), [], pd.DataFrame())
    assert not r.aligned and r.video_start_utc is None
