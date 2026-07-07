#!/usr/bin/env python3
"""ASR transcript QA over a recorded capture tape.

Generic quality-assurance tooling for a ``transcript.parquet`` produced by ``pmlab probe`` /
``pmlab probe record-game`` (columns ``occasion_id, word_idx, word, t_start_utc, t_end_utc,
source, asr_conf``). Runs:
  1. gap scan — silences longer than a threshold in the committed transcript.
  2. throughput + quality — words/minute, sparse minutes, ``asr_conf`` distribution
     (NaN = agreed, -1.0 = force-committed under the uncommitted-window bound).
  3. phrase hit-rate — for a supplied vocabulary, how often (and when) each phrase's surface
     forms actually appear, with hand-curated morphological variants layered on top of
     alias/"/" lists.
  4. plausibility samples + hallucination flags — repeated n-gram loops, "thank you" filler
     blocks, identical-timestamp runs, non-ASCII runs, and backward-timestamp jitter — all
     well-known local-Whisper failure signatures.

No trade/edge claims — pure ASR/QA. Read-only; writes only the two ``--out-dir`` parquet files.

Usage:
    python scripts/transcript_qa.py --transcript data/sports/<event>/transcript.parquet \\
        --out-dir out/ --vocab vocab.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_STRIP_RE = re.compile(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$")

# Hand-curated morphological variants layered on top of a vocab entry's own "/" alias list.
# For single-token aliases: plural ('s'), possessive ("'s"), and, for verb-shaped aliases,
# -ed/-ing forms. Multi-word aliases get their trailing noun pluralized. This default set is
# soccer-flavored (an example domain); pass a different --variants-json for other domains.
_DEFAULT_SINGLE_VARIANTS: dict[str, list[str]] = {
    "champion": ["champions", "champion's"],
    "captain": ["captains", "captain's"],
    "comeback": ["comebacks"],
    "shutout": ["shutouts"],
    "record": ["records"],
    "equalizer": ["equalizers", "equalize", "equalized", "equalizes", "equalizing"],
    "suspend": ["suspended", "suspends", "suspension", "suspending"],
    "suspended": ["suspend", "suspends", "suspension"],
    "handball": ["handballs"],
    "nutmeg": ["nutmegs", "nutmegged"],
    "meg": ["megs", "megged"],
    "appeal": ["appealed", "appeals", "appealing"],
    "hattrick": ["hattricks"],
    "penalty": ["penalties"],
    "goal": ["goals"],
    "bicycle": ["bicycles"],
    "crossbar": ["crossbars"],
}
_DEFAULT_MULTIWORD_LAST_PLURAL: dict[str, str] = {
    "hat trick": "hat tricks",
    "red card": "red cards",
    "own goal": "own goals",
    "penalty kick": "penalty kicks",
    "golden boot": "golden boots",
    "come back": "come backs",
}


def fmt(ts: int | float, tz: timezone) -> str:
    return datetime.fromtimestamp(int(ts), tz=tz).strftime("%H:%M:%S")


# --------------------------------------------------------------------------------------
# 1. GAP SCAN
# --------------------------------------------------------------------------------------

def gap_scan(
    df: pd.DataFrame, tape: str, tz: timezone, floor_ts: int | None, gap_thresh_s: float,
) -> pd.DataFrame:
    d = df.sort_values("t_start_utc").reset_index(drop=True)
    if floor_ts is not None:
        d = d[d["t_start_utc"] >= floor_ts].reset_index(drop=True)
    starts = d["t_start_utc"].to_numpy()
    ends = d["t_end_utc"].to_numpy()
    rows = []
    for i in range(1, len(starts)):
        gap = starts[i] - ends[i - 1]
        if gap > gap_thresh_s:
            rows.append({
                "tape": tape,
                "gap_start_utc": int(ends[i - 1]), "gap_end_utc": int(starts[i]),
                "gap_start": fmt(ends[i - 1], tz), "gap_end": fmt(starts[i], tz),
                "duration_s": int(gap),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. THROUGHPUT + QUALITY
# --------------------------------------------------------------------------------------

def throughput_per_min(df: pd.DataFrame) -> pd.Series:
    minute = (df["t_start_utc"] // 60) * 60
    return df.groupby(minute).size()


# --------------------------------------------------------------------------------------
# 3. PHRASE HIT-RATE
# --------------------------------------------------------------------------------------

def normalize(word: str) -> str:
    return _STRIP_RE.sub("", word).lower()


def phrase_variants(
    phrase: str, single_variants: dict[str, list[str]], multiword_last_plural: dict[str, str],
) -> list[str]:
    """Expand a vocab `phrase` field into a list of lowercase surface forms to search for."""
    aliases = [a.strip() for a in phrase.split("/")]
    out: set[str] = set()
    for alias in aliases:
        base = alias.lower()
        out.add(base)
        if base in multiword_last_plural:
            out.add(multiword_last_plural[base])
        if " " not in base:
            for v in single_variants.get(base, []):
                out.add(v)
            if base not in single_variants and not base.endswith("s"):
                out.add(base + "s")  # generic plural fallback
    return sorted(out)


def find_ngram_hits(tokens: list[str], phrase_tokens: list[str]) -> list[int]:
    """Indices (into `tokens`) where the consecutive sequence `phrase_tokens` starts."""
    n = len(phrase_tokens)
    return [i for i in range(len(tokens) - n + 1) if tokens[i:i + n] == phrase_tokens]


def phrase_hit_rate(
    df: pd.DataFrame, vocab: dict, tape_label: str, tz: timezone,
    single_variants: dict[str, list[str]], multiword_last_plural: dict[str, str],
) -> pd.DataFrame:
    tokens = [normalize(w) for w in df["word"].tolist()]
    ts = df["t_start_utc"].tolist()
    rows = []
    for ticker, meta in vocab.items():
        if not meta.get("eligible", True):
            continue
        phrase = meta["phrase"]
        variants = phrase_variants(phrase, single_variants, multiword_last_plural)
        all_hit_idx: list[int] = []
        for v in variants:
            all_hit_idx.extend(find_ngram_hits(tokens, v.split(" ")))
        all_hit_idx = sorted(set(all_hit_idx))
        n = len(all_hit_idx)
        rows.append({
            "tape": tape_label, "market_ticker": ticker, "phrase": phrase,
            "variants_searched": "; ".join(variants), "n_matches": n,
            "t_first": fmt(ts[all_hit_idx[0]], tz) if n else None,
            "t_last": fmt(ts[all_hit_idx[-1]], tz) if n else None,
            "t_first_utc": int(ts[all_hit_idx[0]]) if n else None,
            "t_last_utc": int(ts[all_hit_idx[-1]]) if n else None,
        })
    return pd.DataFrame(rows).sort_values(["tape", "n_matches"], ascending=[True, False])


# --------------------------------------------------------------------------------------
# 4. ASR PLAUSIBILITY SAMPLES
# --------------------------------------------------------------------------------------

def sample_windows(
    df: pd.DataFrame, n_windows: int, window_len: int, seed: int, tz: timezone,
) -> list[str]:
    rng = random.Random(seed)
    words = df["word"].tolist()
    starts_utc = df["t_start_utc"].tolist()
    if len(words) <= window_len:
        return [" ".join(words)]
    max_start = len(words) - window_len
    picks = sorted(rng.sample(range(max_start), min(n_windows, max_start)))
    return [f"[{fmt(starts_utc[p], tz)}] " + " ".join(words[p:p + window_len]) for p in picks]


def hallucination_flags(df: pd.DataFrame, tape_label: str, tz: timezone) -> list[str]:
    flags = []
    words = df["word"].tolist()
    starts = df["t_start_utc"].tolist()

    # (a) exact repeated n-gram loops (same 3-gram repeated >=3x consecutively)
    norm = [normalize(w) for w in words]
    i = 0
    while i < len(norm) - 9:
        gram = tuple(norm[i:i + 3])
        if all(g for g in gram):
            reps = 1
            j = i + 3
            while j + 3 <= len(norm) and tuple(norm[j:j + 3]) == gram:
                reps += 1
                j += 3
            if reps >= 3:
                flags.append(f"{tape_label}: repeated 3-gram loop {gram} x{reps} starting "
                            f"{fmt(starts[i], tz)} (idx {i})")
                i = j
                continue
        i += 1

    # (b) 'Thank you.' filler blocks -- classic Whisper silence hallucination
    thankyou_idx = [k for k in range(len(words) - 1)
                    if normalize(words[k]) == "thank" and normalize(words[k + 1]) == "you"]
    if thankyou_idx:
        clusters, cur = [], [thankyou_idx[0]]
        for k in thankyou_idx[1:]:
            if k - cur[-1] <= 50:
                cur.append(k)
            else:
                clusters.append(cur)
                cur = [k]
        clusters.append(cur)
        for c in clusters:
            flags.append(f"{tape_label}: 'Thank you' occurrences x{len(c)} clustered near "
                        f"{fmt(starts[c[0]], tz)}-{fmt(starts[c[-1]], tz)}")

    # (c) long runs of words sharing identical t_start (batch-shared timestamps)
    same_ts_run, run_start = 1, 0
    for k in range(1, len(starts)):
        if starts[k] == starts[k - 1]:
            same_ts_run += 1
        else:
            if same_ts_run >= 5:
                flags.append(f"{tape_label}: {same_ts_run} consecutive words share identical "
                            f"t_start_utc at {fmt(starts[run_start], tz)} (idx {run_start})")
            same_ts_run, run_start = 1, k
    if same_ts_run >= 5:
        flags.append(f"{tape_label}: {same_ts_run} consecutive words share identical t_start_utc "
                    f"at {fmt(starts[run_start], tz)} (idx {run_start})")

    # (d) non-ascii / non-English runs
    non_ascii = [w for w in words if not all(ord(c) < 128 for c in w)]
    if non_ascii:
        flags.append(f"{tape_label}: {len(non_ascii)} non-ASCII tokens (sample: {non_ascii[:10]})")

    # (e) local emission-order jitter: t_start_utc should be non-decreasing in stored (word_idx)
    # order; backward jumps mean words landed out of true utterance order (breaks n-gram phrase
    # matching). NOT the same signal as (c) -- (c) fires on ties (normal batch-shared
    # timestamps); this fires only on true backward jumps.
    n_backward = sum(1 for k in range(1, len(starts)) if starts[k] < starts[k - 1])
    if n_backward:
        mags = [starts[k - 1] - starts[k] for k in range(1, len(starts))
                if starts[k] < starts[k - 1]]
        flags.append(f"{tape_label}: {n_backward}/{len(starts)} "
                    f"({100 * n_backward / len(starts):.1f}%) backward t_start_utc jumps "
                    f"(word-order jitter), median magnitude {sorted(mags)[len(mags) // 2]}s, "
                    f"max {max(mags)}s")

    return flags


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript", required=True, help="path to a transcript.parquet tape")
    ap.add_argument("--tape-name", default="tape", help="label used in printed output")
    ap.add_argument("--out-dir", required=True, help="directory for the two output parquet files")
    ap.add_argument("--vocab", default=None,
                    help='optional vocab.json: {market_ticker: {"phrase": str, "eligible": bool}}')
    ap.add_argument("--variants-json", default=None,
                    help="optional JSON with {\"single\": {...}, \"multiword_last_plural\": {...}} "
                         "to override the default (soccer-flavored) morphological variant table")
    ap.add_argument("--floor-ts", type=int, default=None,
                    help="drop words before this unix ts (e.g. a pregame/lead-in video artifact "
                         "preceding the real capture)")
    ap.add_argument("--gap-thresh-s", type=float, default=60.0,
                    help="report a gap when consecutive words are silent longer than this")
    ap.add_argument("--utc-offset-h", type=float, default=-4.0,
                    help="local-time offset from UTC for display timestamps (default: UTC-4)")
    ap.add_argument("--candles", default=None,
                    help="optional 1-min candle parquet for a candle-vs-word-hits cross-check")
    ap.add_argument("--candle-tickers", default="",
                    help="comma-separated market tickers to cross-check against --candles")
    ap.add_argument("--candle-move-thresh", type=float, default=0.05,
                    help="1-min price_mean delta (0-1 scale) to flag as a candle spike")
    args = ap.parse_args()

    tz = timezone(timedelta(hours=args.utc_offset_h))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    single_variants = _DEFAULT_SINGLE_VARIANTS
    multiword_last_plural = _DEFAULT_MULTIWORD_LAST_PLURAL
    if args.variants_json:
        v = json.loads(Path(args.variants_json).read_text())
        single_variants = v.get("single", single_variants)
        multiword_last_plural = v.get("multiword_last_plural", multiword_last_plural)

    df = pd.read_parquet(args.transcript)
    if args.floor_ts:
        match = df[df["t_start_utc"] >= args.floor_ts].reset_index(drop=True)
        head = df[df["t_start_utc"] < args.floor_ts].reset_index(drop=True)
    else:
        match, head = df, df.iloc[0:0]

    print("=" * 100)
    print(f"{args.tape_name}: total rows {len(df)} | pre-floor rows {len(head)} | "
          f"match rows {len(match)}")
    if len(head):
        print("  head span:", fmt(head["t_start_utc"].min(), tz), "->",
              fmt(head["t_start_utc"].max(), tz))
    print("  match span:", fmt(match["t_start_utc"].min(), tz), "->",
          fmt(match["t_start_utc"].max(), tz))
    if "attempt" in df.columns:
        print("  attempts:", sorted(df["attempt"].unique()))

    # ---- 1. gap scan ----
    gaps = gap_scan(match, args.tape_name, tz, args.floor_ts, args.gap_thresh_s)
    print("\n" + "=" * 100)
    print(f"GAP SCAN (>{args.gap_thresh_s:.0f}s)")
    with pd.option_context("display.width", 160, "display.max_rows", 100):
        print(gaps.to_string(index=False))

    match_span = match["t_end_utc"].max() - match["t_start_utc"].min()
    blind = gaps["duration_s"].sum() if len(gaps) else 0
    print(f"\nmatch span {match_span}s, blind {blind}s ({100 * blind / match_span:.2f}%)")

    # ---- 2. throughput + quality ----
    print("\n" + "=" * 100)
    print("THROUGHPUT (words/min), sparse minutes (<3 wpm)")
    tp = throughput_per_min(match)
    sparse = tp[tp < 3]
    print(f"  total minutes with data: {len(tp)}, sparse (<3 wpm): {len(sparse)}")
    print(f"  mean wpm {tp.mean():.1f}, median {tp.median():.1f}, max {tp.max()}")
    print("  sparse minute timestamps:", [fmt(t, tz) for t in sparse.index[:30]])

    n, n_nan = len(df), df["asr_conf"].isna().sum()
    n_neg1 = (df["asr_conf"] == -1.0).sum()
    print(f"\nasr_conf: n={n} nan={n_nan} ({100 * n_nan / n:.1f}%) neg1(force-commit)={n_neg1}")
    print(f"  source values: {df['source'].value_counts().to_dict()}")

    # ---- 3. phrase hit rate ----
    hits = pd.DataFrame()
    if args.vocab:
        print("\n" + "=" * 100)
        print("PHRASE HIT-RATE")
        vocab = json.loads(Path(args.vocab).read_text())
        hits = phrase_hit_rate(match, vocab, args.tape_name, tz, single_variants,
                               multiword_last_plural)
        with pd.option_context("display.width", 200, "display.max_rows", 100,
                               "display.max_colwidth", 40):
            print(hits[["tape", "market_ticker", "phrase", "n_matches", "t_first", "t_last"]]
                  .to_string(index=False))
        never = hits[hits["n_matches"] == 0]
        print("\nNEVER-APPEARING PHRASES:")
        print(never[["tape", "market_ticker", "phrase"]].to_string(index=False))

        if args.candles and args.candle_tickers:
            candles = pd.read_parquet(args.candles)
            print(f"\n{args.candle_move_thresh * 100:.0f}c+ 1-min candle moves vs word first/last:")
            for ticker in [t.strip() for t in args.candle_tickers.split(",") if t.strip()]:
                c = candles[candles["ticker"] == ticker].sort_values("ts").reset_index(drop=True)
                if len(c) == 0:
                    print(f"  {ticker}: no candle rows")
                    continue
                c["delta"] = c["price_mean"].diff()
                spikes = c[c["delta"].abs() > args.candle_move_thresh]
                spike_times = [fmt(t, tz) for t in spikes["ts"].tolist()]
                row = hits[hits["market_ticker"] == ticker]
                print(f"  {ticker}: candle spikes at {spike_times} | word hits n="
                      f"{row['n_matches'].iloc[0] if len(row) else 'NA'} "
                      f"first={row['t_first'].iloc[0] if len(row) else None} "
                      f"last={row['t_last'].iloc[0] if len(row) else None}")

    # ---- 4. plausibility samples + hallucination flags ----
    print("\n" + "=" * 100)
    print("SAMPLE WINDOWS")
    for w in sample_windows(match, 15, 15, seed=42, tz=tz):
        print(" ", w)

    print("\nHALLUCINATION FLAGS")
    for f in hallucination_flags(match, args.tape_name, tz):
        print(" ", f)

    # ---- persist ----
    gaps.to_parquet(out_dir / "qa_gaps.parquet", index=False)
    hits.to_parquet(out_dir / "qa_phrase_hits.parquet", index=False)
    print(f"\nWrote {out_dir / 'qa_gaps.parquet'} ({len(gaps)} rows)")
    print(f"Wrote {out_dir / 'qa_phrase_hits.parquet'} ({len(hits)} rows)")


if __name__ == "__main__":
    main()
