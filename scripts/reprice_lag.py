#!/usr/bin/env python3
"""Reprice-lag analysis: how fast does a market's order book react to a transcript mention?

Generic ASR-vs-book microstructure tool over one tape recorded by ``pmlab probe`` /
``pmlab probe record-game`` — a directory containing a transcript parquet (``occasion_id,
word_idx, word, t_start_utc, t_end_utc, source, asr_conf``) and a ``book_ticks.parquet``
(``ticker, recv_utc, seq, yes_bid, yes_ask, depth_bid, depth_ask, ...``). For each market's
tracked phrase, finds each transcript mention event and measures how long the book took to move
away from its pre-mention baseline (``lag_s``), plus a cross-market "fastest move" scan and an
optional notable-event-vocabulary calibration pass (useful when a market's own phrase rarely
fires but a proxy vocabulary, e.g. "goal"/"score" for a sports broadcast, does).

CAVEAT: if ``t_start_utc`` on the transcript is audio-time on a delayed capture (browser/device
loopback → ffmpeg → Whisper), not the wall-clock time the ASR actually committed the word, the
measured ``lag_s`` here is an OPTIMISTIC bound on live actionable latency — add the unrecorded ASR
commit delay on top (see the ``StreamingWhisper`` force-commit bound in ``pmlab.probe.audio``).

Read-only; writes only the ``--out`` parquet.

Usage:
    python scripts/reprice_lag.py --tape-dir data/sports/<event> --vocab vocab.json \\
        --event-ticker <event> --out reprice_lag_events.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEPTH_EPS = 1e-6

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# A generic "notable event" vocabulary for calibration when a market's own tracked phrase is too
# rare to measure lag reliably. The default is soccer-flavored (goal/score/card); override with
# --event-vocab-regex for other broadcast domains.
DEFAULT_EVENT_VOCAB_RE = (
    r"\bgoal\b|goals|score|scores|scored|equalizer|equaliser|\d-\d|penalty kick|red card|sent off"
)


def et(ts: float, utc_offset_h: float) -> str:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return "NaT"
    import datetime

    return (
        datetime.datetime.fromtimestamp(ts, datetime.UTC)
        .astimezone(datetime.timezone(datetime.timedelta(hours=utc_offset_h)))
        .strftime("%H:%M:%S")
    )


# ---------------------------------------------------------------------------
# Phrase tokenization / matching
# ---------------------------------------------------------------------------


def clean_token(raw: str) -> str:
    """Lowercase, strip accents, drop possessive 's, strip remaining punctuation."""
    s = unicodedata.normalize("NFKD", raw)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[’']s\b", "", s)  # possessive 's / 's
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def base_form(tok: str) -> str:
    """Strip common inflection suffixes."""
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) - len(suf) >= 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def phrase_to_alias_token_lists(phrase: str) -> list[list[str]]:
    """'Comeback / Come Back' -> [['comeback'], ['come', 'back']]"""
    aliases = [a.strip() for a in phrase.split("/")]
    out = []
    for alias in aliases:
        toks = [clean_token(w) for w in alias.split()]
        toks = [t for t in toks if t]
        if toks:
            out.append(toks)
    return out


@dataclass
class MarketPhrase:
    event_ticker: str
    market_ticker: str
    phrase: str
    alias_token_lists: list[list[str]] = field(default_factory=list)


def load_vocab(vocab_path: Path, event_ticker: str) -> list[MarketPhrase]:
    """``vocab.json`` shape: ``{market_ticker: {"phrase": str, "eligible": bool}}``."""
    v = json.loads(vocab_path.read_text())
    out = []
    for mkt, info in v.items():
        if not info.get("eligible", True):
            continue
        out.append(MarketPhrase(
            event_ticker=event_ticker, market_ticker=mkt, phrase=info["phrase"],
            alias_token_lists=phrase_to_alias_token_lists(info["phrase"]),
        ))
    return out


# ---------------------------------------------------------------------------
# Transcript scanning -> mention events
# ---------------------------------------------------------------------------


def find_matches(base_tokens: list[str], alias_token_lists: list[list[str]]) -> list[int]:
    """Return start indices in base_tokens where any alias matches."""
    n = len(base_tokens)
    starts = []
    for alias in alias_token_lists:
        length = len(alias)
        if length == 0 or length > n:
            continue
        for i in range(n - length + 1):
            if base_tokens[i:i + length] == alias:
                starts.append(i)
    return sorted(set(starts))


def collapse_to_events(times: list[float], gap_s: float) -> list[float]:
    """Collapse a sorted list of match timestamps into 'first occurrence after >= gap_s of
    absence' events. Returns t_word per event."""
    if not times:
        return []
    times = sorted(times)
    events = [times[0]]
    last = times[0]
    for t in times[1:]:
        if t - last >= gap_s:
            events.append(t)
        last = t
    return events


def scan_transcript_for_events(
    transcript: pd.DataFrame, phrases: list[MarketPhrase], event_gap_s: float,
) -> pd.DataFrame:
    words = transcript["word"].astype(str).tolist()
    t_starts = transcript["t_start_utc"].astype(float).tolist()
    base_tokens = [base_form(clean_token(w)) for w in words]

    rows = []
    for mp in phrases:
        starts = find_matches(base_tokens, mp.alias_token_lists)
        match_times = [t_starts[i] for i in starts]
        event_times = collapse_to_events(match_times, event_gap_s)
        n_total_matches = len(match_times)
        for t_word in event_times:
            rows.append({
                "event_ticker": mp.event_ticker, "market_ticker": mp.market_ticker,
                "phrase": mp.phrase, "t_word": t_word, "total_matches_in_tape": n_total_matches,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tick-level reprice measurement
# ---------------------------------------------------------------------------


def compute_mid(yes_bid: np.ndarray, yes_ask: np.ndarray) -> np.ndarray:
    bid_ok = ~np.isnan(yes_bid)
    ask_ok = ~np.isnan(yes_ask)
    mid = np.where(bid_ok & ask_ok, (yes_bid + yes_ask) / 2.0, np.nan)
    mid = np.where(bid_ok & ~ask_ok, yes_bid, mid)
    mid = np.where(~bid_ok & ask_ok, yes_ask, mid)
    return mid


def prep_ticker_series(ticks: pd.DataFrame) -> pd.DataFrame:
    """Dedupe on (ticker, recv_utc, seq), sort, compute mid, treat float-noise depths (<1e-6) as
    exact zero."""
    t = ticks.drop_duplicates(subset=["ticker", "recv_utc", "seq"]).copy()
    t = t.sort_values(["ticker", "recv_utc", "seq"]).reset_index(drop=True)
    for c in ("depth_bid", "depth_ask", "depth_bid2", "depth_ask2"):
        if c in t.columns:
            t.loc[t[c].abs() < DEPTH_EPS, c] = 0.0
    t["mid"] = compute_mid(t["yes_bid"].to_numpy(), t["yes_ask"].to_numpy())
    return t


@dataclass(frozen=True)
class RepriceParams:
    baseline_lookback_s: float = 600.0
    baseline_lookahead_cut_s: float = 120.0
    search_before_s: float = 300.0
    search_after_s: float = 300.0
    reprice_thresh: float = 0.03  # 3 cents


def measure_reprice(series: pd.DataFrame, t_word: float, p: RepriceParams) -> dict:
    """``series``: single-ticker tick df sorted by recv_utc, with a ``mid`` column."""
    recv = series["recv_utc"].to_numpy()
    empty = {"t_reprice": np.nan, "t_pull": np.nan, "lag_s": np.nan, "move_60s": np.nan,
                "move_300s": np.nan, "reprice_move": np.nan, "book_valid_at_word": None,
                "tick_coverage": False}
    if len(recv) == 0:
        return empty

    base_lo, base_hi = t_word - p.baseline_lookback_s, t_word - p.baseline_lookahead_cut_s
    win_lo, win_hi = t_word - p.search_before_s, t_word + p.search_after_s

    has_baseline = np.any((recv >= base_lo) & (recv <= base_hi))
    has_after = np.any(recv >= t_word + p.search_after_s - 10)
    # tick_coverage requires a usable baseline window + coverage through the search window end.
    # Ticks reaching all the way back to t_word - baseline_lookback_s are NOT required: near the
    # start of a capture the baseline window is legally truncated (fewer samples) but still valid.
    tick_coverage = bool(has_baseline and has_after)
    if not has_baseline:
        return {**empty, "tick_coverage": tick_coverage}

    base_mask = (recv >= base_lo) & (recv <= base_hi)
    baseline = np.nanmedian(series["mid"].to_numpy()[base_mask])
    if np.isnan(baseline):
        return {**empty, "tick_coverage": tick_coverage}

    win_mask = (recv >= win_lo) & (recv <= win_hi)
    w = series.loc[win_mask]
    w_recv = w["recv_utc"].to_numpy()
    w_mid = w["mid"].to_numpy()

    t_reprice = np.nan
    reprice_move = np.nan
    dev = np.abs(w_mid - baseline)
    hit = np.where(dev >= p.reprice_thresh)[0]
    if len(hit) > 0:
        t_reprice = float(w_recv[hit[0]])
        reprice_move = float(w_mid[hit[0]] - baseline)

    t_pull = np.nan
    ask_empty = (w["depth_ask"].to_numpy() < DEPTH_EPS) | np.isnan(w["yes_ask"].to_numpy())
    crossed = w["yes_bid"].to_numpy() >= w["yes_ask"].to_numpy()
    pull_hit = np.where(ask_empty | crossed)[0]
    if len(pull_hit) > 0:
        t_pull = float(w_recv[pull_hit[0]])

    def mid_at_or_before(ts: float) -> float:
        mask = recv <= ts
        if not np.any(mask):
            return np.nan
        idx = np.where(mask)[0][-1]
        return series["mid"].to_numpy()[idx]

    mid_60 = mid_at_or_before(t_word + 60)
    mid_300 = mid_at_or_before(t_word + 300)
    move_60s = mid_60 - baseline if not np.isnan(mid_60) else np.nan
    move_300s = mid_300 - baseline if not np.isnan(mid_300) else np.nan

    mask_before = recv <= (t_word - 1)
    book_valid_at_word = None
    if np.any(mask_before):
        idx = np.where(mask_before)[0][-1]
        row = series.iloc[idx]
        book_valid_at_word = bool(
            (not np.isnan(row.yes_bid)) and (not np.isnan(row.yes_ask))
            and (row.yes_bid < row.yes_ask)
            and (row.depth_bid >= 1) and (row.depth_ask >= 1)
        )

    lag_s = (t_reprice - t_word) if not np.isnan(t_reprice) else np.nan
    return {
        "t_reprice": t_reprice, "t_pull": t_pull, "lag_s": lag_s,
        "move_60s": move_60s, "move_300s": move_300s,
        "reprice_move": reprice_move, "book_valid_at_word": book_valid_at_word,
        "tick_coverage": tick_coverage,
    }


def run_tape(
    tape_name: str, event_ticker: str, transcript: pd.DataFrame, ticks: pd.DataFrame,
    phrases: list[MarketPhrase], p: RepriceParams, event_gap_s: float,
) -> pd.DataFrame:
    events = scan_transcript_for_events(transcript, phrases, event_gap_s)
    events["tape"] = tape_name

    ticks_p = prep_ticker_series(ticks)
    by_ticker = {tkr: g.reset_index(drop=True) for tkr, g in ticks_p.groupby("ticker")}

    out_rows = []
    for _, ev in events.iterrows():
        series = by_ticker.get(ev.market_ticker)
        if series is None:
            m = {"t_reprice": np.nan, "t_pull": np.nan, "lag_s": np.nan, "move_60s": np.nan,
                     "move_300s": np.nan, "book_valid_at_word": None, "tick_coverage": False}
        else:
            m = measure_reprice(series, ev.t_word, p)
        row = ev.to_dict()
        row.update(m)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Cross-market analyses (independent of any one market's own phrase)
# ---------------------------------------------------------------------------


def top_fastest_moves(ticks_p: pd.DataFrame, window_s: int = 30, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for tkr, g in ticks_p.groupby("ticker"):
        g = g.sort_values("recv_utc")
        recv = g["recv_utc"].to_numpy()
        mid = g["mid"].to_numpy()
        if len(recv) < 3:
            continue
        idx_prev = np.searchsorted(recv, recv - window_s, side="left")
        valid = idx_prev < np.arange(len(recv))
        moves = np.full(len(recv), np.nan)
        moves[valid] = np.abs(mid[valid] - mid[idx_prev[valid]])
        order = np.argsort(-np.nan_to_num(moves, nan=-1))
        taken_times: list[float] = []
        for i in order[: top_n * 3]:
            if np.isnan(moves[i]):
                continue
            t = recv[i]
            if any(abs(t - tt) < 300 for tt in taken_times):
                continue
            taken_times.append(t)
            rows.append({"ticker": tkr, "t_event": float(t), "move_abs": float(moves[i]),
                        "mid_before": float(mid[idx_prev[i]]), "mid_after": float(mid[i])})
    df = pd.DataFrame(rows).sort_values("move_abs", ascending=False) if rows else pd.DataFrame(rows)
    return df.head(top_n)


def cluster_vocab_words(
    transcript: pd.DataFrame, vocab_re: re.Pattern[str], gap_s: int = 20, min_hits: int = 2,
) -> list[dict]:
    """Cluster notable-event-vocabulary hits into candidate live-event calls: >=min_hits
    occurrences within gap_s of each other."""
    hits = transcript[transcript.word.str.contains(vocab_re, na=False)].copy()
    hits = hits.sort_values("t_start_utc")
    times = hits.t_start_utc.tolist()
    words = hits.word.tolist()
    clusters, cur = [], []
    for t, w in zip(times, words, strict=False):
        if cur and t - cur[-1][0] > gap_s:
            if len(cur) >= min_hits:
                clusters.append(cur)
            cur = []
        cur.append((t, w))
    if len(cur) >= min_hits:
        clusters.append(cur)
    return [{"t_call": c[0][0], "t_call_last": c[-1][0], "n_hits": len(c),
             "words": [w for _, w in c]}
            for c in clusters]


def earliest_cross_market_reaction(
    ticks_p: pd.DataFrame, t_call: float, before: int = 90, after: int = 180,
    step_window: int = 20, thresh: float = 0.04,
) -> dict | None:
    """Scan ALL tickers for a SHARP step move (>= thresh over a step_window-second lookback)
    landing inside [t_call-before, t_call+after]. Deliberately NOT the same test as
    :func:`measure_reprice`'s baseline-deviation test: a wide baseline/window also catches slow
    ambient multi-minute drift unrelated to the call. This step-change test isolates genuine
    sudden reprices."""
    best = None
    for tkr, g in ticks_p.groupby("ticker"):
        recv = g["recv_utc"].to_numpy()
        mid = g["mid"].to_numpy()
        if len(recv) < 3:
            continue
        idx_prev = np.searchsorted(recv, recv - step_window, side="left")
        valid = idx_prev < np.arange(len(recv))
        moves = np.full(len(recv), np.nan)
        moves[valid] = np.abs(mid[valid] - mid[idx_prev[valid]])
        win_mask = (recv >= t_call - before) & (recv <= t_call + after)
        cand = np.where(win_mask & (moves >= thresh))[0]
        if len(cand) == 0:
            continue
        i = cand[0]
        t_react = float(recv[i])
        if best is None or t_react < best["t_react"]:
            best = {"ticker": tkr, "t_react": t_react, "move": float(mid[i] - mid[idx_prev[i]])}
    return best


def nearest_phrase_within(
    transcript: pd.DataFrame, phrases: list[MarketPhrase], t_event: float, window: int = 120,
) -> list[tuple[str, str, float]]:
    words = transcript["word"].astype(str).tolist()
    t_starts = transcript["t_start_utc"].astype(float).tolist()
    base_tokens = [base_form(clean_token(w)) for w in words]
    lo, hi = t_event - window, t_event + window
    idx_lo = np.searchsorted(t_starts, lo)
    idx_hi = np.searchsorted(t_starts, hi)
    sub_tokens = base_tokens[max(0, idx_lo - 3):idx_hi + 3]
    sub_times = t_starts[max(0, idx_lo - 3):idx_hi + 3]
    hits = []
    for mp in phrases:
        for s in find_matches(sub_tokens, mp.alias_token_lists):
            hits.append((mp.phrase, mp.market_ticker, sub_times[s]))
    return hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tape-dir", required=True, help="directory holding the transcript + ticks")
    ap.add_argument("--transcript", default="transcript.parquet",
                    help="transcript filename within --tape-dir (or an absolute path)")
    ap.add_argument("--ticks", default="book_ticks.parquet",
                    help="book-ticks filename within --tape-dir (or an absolute path)")
    ap.add_argument("--vocab", required=True, help="path to a vocab.json: "
                    '{market_ticker: {"phrase": str, "eligible": bool}}')
    ap.add_argument("--event-ticker", required=True, help="label for this tape's event")
    ap.add_argument("--out", required=True, help="output parquet path for per-event lag rows")
    ap.add_argument("--utc-offset-h", type=float, default=-4.0,
                    help="local-time offset from UTC for display timestamps (default: UTC-4)")
    ap.add_argument("--transcript-floor-ts", type=int, default=None,
                    help="drop transcript words before this unix ts (e.g. a pregame video "
                         "artifact preceding the real capture)")
    ap.add_argument("--reprice-thresh", type=float, default=0.03)
    ap.add_argument("--event-gap-s", type=float, default=600.0)
    ap.add_argument("--event-vocab-regex", default=DEFAULT_EVENT_VOCAB_RE,
                    help="regex for the notable-event calibration pass")
    args = ap.parse_args()

    tape_dir = Path(args.tape_dir)
    transcript_path = Path(args.transcript)
    if not transcript_path.is_absolute():
        transcript_path = tape_dir / transcript_path
    ticks_path = Path(args.ticks)
    if not ticks_path.is_absolute():
        ticks_path = tape_dir / ticks_path

    def fmt(ts: float) -> str:
        return et(ts, args.utc_offset_h)

    print("=" * 90)
    print(f"LOADING TAPE ({args.event_ticker})")
    print("=" * 90)
    transcript = pd.read_parquet(transcript_path)
    if args.transcript_floor_ts is not None:
        n_before = len(transcript)
        transcript = transcript[transcript.t_start_utc >= args.transcript_floor_ts]
        transcript = transcript.sort_values("word_idx").reset_index(drop=True)
        print(f"transcript: {n_before} -> {len(transcript)} words after floor trim "
              f"(< {args.transcript_floor_ts} dropped)")
    ticks = pd.read_parquet(ticks_path)
    print(f"ticks: {len(ticks)} rows, {ticks.ticker.nunique()} tickers")
    print(f"tick coverage: {fmt(ticks.recv_utc.min())} - {fmt(ticks.recv_utc.max())}")

    print()
    print("=" * 90)
    print("MENTION EVENT SCAN")
    print("=" * 90)
    phrases = load_vocab(Path(args.vocab), args.event_ticker)
    p = RepriceParams(reprice_thresh=args.reprice_thresh)
    events = run_tape(args.event_ticker, args.event_ticker, transcript, ticks, phrases, p,
                      args.event_gap_s)
    print(f"{args.event_ticker}: {len(events)} mention events across "
          f"{events.market_ticker.nunique() if len(events) else 0} markets")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} ({len(events)} rows)")

    print()
    print("=" * 90)
    print("PER-TAPE EVENT TABLE (tick_coverage rows only, sorted by lag_s)")
    print("=" * 90)
    if len(events):
        cov = events[events.tick_coverage]
        print(f"\n--- {len(cov)}/{len(events)} events with tick coverage ---")
        disp = cov.copy()
        disp["t_word_et"] = disp.t_word.apply(fmt)
        disp = disp.sort_values("lag_s")
        print(disp[["market_ticker", "phrase", "t_word_et", "lag_s", "move_60s", "move_300s",
                    "book_valid_at_word", "total_matches_in_tape"]].to_string(index=False))

        print()
        print("=" * 90)
        print("LAG DISTRIBUTION (tick-covered events)")
        print("=" * 90)
        cov_all = events[events.tick_coverage & events.lag_s.notna()]
        if len(cov_all):
            lags = cov_all.lag_s.to_numpy()
            print(f"n={len(lags)} median={np.median(lags):.1f}s "
                  f"IQR=[{np.percentile(lags, 25):.1f},{np.percentile(lags, 75):.1f}] "
                  f"min={lags.min():.1f} max={lags.max():.1f} "
                  f"pct_negative={100 * np.mean(lags < 0):.1f}%")
        else:
            print("n=0")

    print()
    print("=" * 90)
    print("NOTABLE-EVENT CALIBRATION (auto-clustered vocabulary hits)")
    print("=" * 90)
    vocab_re = re.compile(args.event_vocab_regex, re.IGNORECASE)
    clusters = cluster_vocab_words(transcript, vocab_re)
    ticks_p = prep_ticker_series(ticks)
    print(f"\n{len(clusters)} candidate event clusters")
    cov_lo, cov_hi = ticks.recv_utc.min(), ticks.recv_utc.max()
    calib_lags = []
    for c in clusters:
        t_call = c["t_call"]
        in_cov = (t_call - 90 >= cov_lo) and (t_call + 180 <= cov_hi)
        reaction = earliest_cross_market_reaction(ticks_p, t_call) if in_cov else None
        lag = (reaction["t_react"] - t_call) if reaction else np.nan
        if not np.isnan(lag):
            calib_lags.append(lag)
        tail = (f"earliest reaction: {reaction['ticker'].split('-')[-1]} "
                f"at {fmt(reaction['t_react'])} "
                f"(lag={lag:+.0f}s, reprice_move={reaction['move']:+.3f})"
                if reaction else "no cross-market reaction found / no tick coverage")
        print(f"t_call={fmt(t_call)} n_hits={c['n_hits']:2d} words={c['words'][:6]} "
              f"tick_cov={in_cov} -> {tail}")
    if calib_lags:
        print(f"calibration lags: n={len(calib_lags)} median={np.median(calib_lags):.1f}s "
              f"min={min(calib_lags):.1f} max={max(calib_lags):.1f}")

    print()
    print("=" * 90)
    print("TOP-10 FASTEST MOVES (independent of transcript)")
    print("=" * 90)
    top = top_fastest_moves(ticks_p, window_s=30, top_n=10)
    for _, r in top.iterrows():
        hits = nearest_phrase_within(transcript, phrases, r.t_event, window=120)
        hit_str = ("; ".join(f"{ph}@{fmt(t)}(dt={t - r.t_event:+.0f}s)" for ph, _mk, t in hits)
                  or "NONE")
        print(f"{r.ticker:45s} t={fmt(r.t_event)} move={r.move_abs:.3f} "
              f"{r.mid_before:.2f}->{r.mid_after:.2f}  transcript_matches_within_120s: {hit_str}")

    print("\nDONE")


if __name__ == "__main__":
    main()
