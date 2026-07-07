#!/usr/bin/env python3
"""Book-shape survey over a recorded order-book tape.

Generic microstructure tooling for a tape recorded by ``pmlab probe`` / ``pmlab probe
record-game`` (a ``book_ticks.parquet`` file: one row per top-of-book observation, columns
``ticker, recv_utc, seq, yes_bid, yes_ask, depth_bid, depth_ask, yes_bid2, yes_ask2, depth_bid2,
depth_ask2``). Classifies book state (CROSSED / EMPTY-ASK / EMPTY-BID / EMPTY-BOTH / VALID) at
tick, per-second-grid, and per-market-minute resolution, and writes:
  - <out-dir>/book_shape_minutes.parquet
  - <out-dir>/book_shape_gaps.parquet
plus a dense stdout report (percentages, spread/depth stats, rest-window stats, gap-dynamics
table).

NO trading/P&L claims are computed here — only descriptive book-shape statistics.

Usage:
    python scripts/book_shape.py --ticks data/sports/<event>/book_ticks.parquet --out-dir out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

DEPTH_EPS = 1e-6
STATES = ["VALID", "CROSSED", "EMPTY_ASK", "EMPTY_BID", "EMPTY_BOTH"]


# =========================================================================
# Core classification
# =========================================================================
def classify_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """Add effective (zero-thresholded) depths + tick-level state + tick mid."""
    df = df.copy()
    for c in ["depth_bid", "depth_ask", "depth_bid2", "depth_ask2"]:
        if c in df.columns:
            df[c + "_eff"] = np.where(df[c].to_numpy() > DEPTH_EPS, df[c].to_numpy(), 0.0)

    bid_present = df["depth_bid_eff"] > 0
    ask_present = df["depth_ask_eff"] > 0
    crossed = df["yes_bid"] >= df["yes_ask"]

    state = np.select(
        [
            ~bid_present & ~ask_present,
            ~ask_present,
            ~bid_present,
            crossed,
        ],
        ["EMPTY_BOTH", "EMPTY_ASK", "EMPTY_BID", "CROSSED"],
        default="VALID",
    )
    df["state"] = state
    df["bid_present"] = bid_present
    df["ask_present"] = ask_present

    mid = np.where(
        bid_present & ask_present,
        (df["yes_bid"] + df["yes_ask"]) / 2.0,
        np.where(bid_present, df["yes_bid"], np.where(ask_present, df["yes_ask"], np.nan)),
    )
    df["mid_tick"] = mid
    df["spread"] = df["yes_ask"] - df["yes_bid"]
    return df


def build_second_grid(ticks: pd.DataFrame, t_min: int, t_max: int) -> pd.DataFrame:
    """Per-ticker forward-filled per-second state grid."""
    full_index = np.arange(int(t_min), int(t_max) + 1, dtype=np.int64)
    frames = []
    for ticker, g in ticks.groupby("ticker", sort=False):
        g = g.sort_values("recv_utc").drop_duplicates("recv_utc", keep="last")
        g = g.set_index("recv_utc")
        cols = ["state", "mid_tick", "bid_present", "ask_present", "yes_bid", "yes_ask",
                "depth_bid_eff", "depth_ask_eff"]
        g = g[cols].reindex(full_index)
        first_valid = g["state"].first_valid_index()
        if first_valid is None:
            continue
        g = g.loc[first_valid:]
        g["state"] = g["state"].ffill()
        g["bid_present"] = g["bid_present"].ffill()
        g["ask_present"] = g["ask_present"].ffill()
        g["mid_lkm"] = g["mid_tick"].ffill().bfill()  # last-known-mid, back-filled only at head
        g["ticker"] = ticker
        g["t"] = g.index
        frames.append(g.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=["ticker", "t", "state", "mid_lkm"])
    return pd.concat(frames, ignore_index=True)


def minute_agg(grid: pd.DataFrame) -> pd.DataFrame:
    """Per ticker-minute majority state + mean mid, from a per-second grid."""
    grid = grid.copy()
    grid["minute_ts"] = (grid["t"] // 60) * 60
    recs = []
    for (ticker, minute_ts), g in grid.groupby(["ticker", "minute_ts"], sort=False):
        vc = g["state"].value_counts()
        majority = vc.idxmax()
        fracs = {f"frac_{s.lower()}": float(vc.get(s, 0)) / len(g) for s in STATES}
        recs.append({
            "ticker": ticker,
            "minute_ts": minute_ts,
            "state_majority": majority,
            "mid": g["mid_lkm"].mean(),
            "n_seconds": len(g),
            **fracs,
        })
    return pd.DataFrame(recs)


def band_of(mid: pd.Series, band_lo: float, band_hi: float) -> pd.Series:
    return pd.cut(mid, bins=[-0.01, band_lo, band_hi, 1.01],
                  labels=[f"<{band_lo}", f"{band_lo}-{band_hi}", f">{band_hi}"])


def phase_of(ts: pd.Series, anchor: int, breaks_min: list[float], labels: list[str]) -> pd.Series:
    """Bucket a timestamp series into named phases relative to ``anchor`` (unix s), given
    ``breaks_min`` minute offsets from the anchor (ascending) and one more label than breaks."""
    edges = [-np.inf, *(anchor + b * 60 for b in breaks_min), np.inf]
    return pd.cut(ts, bins=edges, labels=labels)


def run_length_encode(states: np.ndarray) -> list[tuple[object, int, int, int]]:
    """Return (state, length, start_idx, end_idx) for each run in a 1D array."""
    if len(states) == 0:
        return []
    change = np.where(states[1:] != states[:-1])[0] + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(states)]
    return [(states[s], e - s, s, e) for s, e in zip(starts, ends, strict=True)]


def q(s: pd.Series, ps: tuple[float, ...] = (0.25, 0.5, 0.75, 0.90)) -> pd.Series:
    return s.quantile(list(ps))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", required=True, help="path to a book_ticks.parquet tape")
    ap.add_argument("--out-dir", required=True, help="directory for the two output parquet files")
    ap.add_argument("--exclude-suffix", default="-NQE",
                    help="drop tickers ending with this suffix if present (e.g. a "
                         "non-qualifying-event market); '' to disable")
    ap.add_argument("--band-lo", type=float, default=0.05, help="lower price-band bound")
    ap.add_argument("--band-hi", type=float, default=0.95, help="upper price-band bound")
    ap.add_argument("--anchor-ts", type=int, default=None,
                    help="unix s anchor for phase labeling (e.g. an event's start time); "
                         "omit to skip phase breakdowns")
    ap.add_argument("--phase-breaks-min", default="50,65,113",
                    help="comma-separated minute offsets from --anchor-ts marking phase "
                         "boundaries")
    ap.add_argument("--phase-labels", default="pregame,P1,break,P2,post",
                    help="comma-separated phase labels (one more than --phase-breaks-min)")
    ap.add_argument("--window-start-min", type=float, default=None,
                    help="gated-window report: start, minutes after --anchor-ts")
    ap.add_argument("--window-end-min", type=float, default=None,
                    help="gated-window report: end, minutes after --anchor-ts")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    breaks_min = [float(x) for x in args.phase_breaks_min.split(",") if x.strip()]
    phase_labels = [x.strip() for x in args.phase_labels.split(",") if x.strip()]
    if len(phase_labels) != len(breaks_min) + 1:
        raise SystemExit("--phase-labels must have exactly one more entry than --phase-breaks-min")

    print("=" * 100)
    print("LOADING TAPE")
    print("=" * 100)
    raw = pd.read_parquet(args.ticks)
    if args.exclude_suffix:
        raw = raw[~raw["ticker"].str.endswith(args.exclude_suffix)]
    ticks = classify_ticks(raw)
    print(f"ticks: {len(ticks):,} rows, {ticks['ticker'].nunique()} tickers, "
          f"{ticks['recv_utc'].min()}..{ticks['recv_utc'].max()}")

    print("\nBuilding per-second grid...")
    grid = build_second_grid(ticks, ticks["recv_utc"].min(), ticks["recv_utc"].max())
    print(f"grid: {len(grid):,} ticker-seconds")

    minutes = minute_agg(grid)
    minutes["mid_band"] = band_of(minutes["mid"], args.band_lo, args.band_hi)
    if args.anchor_ts is not None:
        minutes["phase"] = phase_of(minutes["minute_ts"], args.anchor_ts, breaks_min, phase_labels)

    # =====================================================================
    # TASK 1 -- validity survey
    # =====================================================================
    print("\n" + "=" * 100)
    print("TASK 1 -- VALIDITY SURVEY (market-minute, majority-state)")
    print("=" * 100)

    n_minutes = len(minutes)
    overall = minutes["state_majority"].value_counts(normalize=True).reindex(STATES).fillna(0) * 100
    print(f"\nOverall ({n_minutes} market-minutes, majority-state basis):")
    print(overall.round(2).to_string())

    print("\nMean per-second fractions across ALL market-seconds (finer-grained, not majority):")
    frac_cols = [f"frac_{s.lower()}" for s in STATES]
    weighted = (minutes[frac_cols].multiply(minutes["n_seconds"], axis=0).sum()
                / minutes["n_seconds"].sum() * 100)
    weighted.index = STATES
    print(weighted.round(2).to_string())

    print("\nPer-market table (% of market-minutes, majority-state):")
    per_market = (minutes.groupby("ticker")["state_majority"]
                  .value_counts(normalize=True).unstack().reindex(columns=STATES).fillna(0) * 100)
    per_market["n_minutes"] = minutes.groupby("ticker").size()
    print(per_market.round(1).to_string())

    print("\nBy price band (mid of minute):")
    by_band = (minutes.groupby("mid_band", observed=True)["state_majority"]
               .value_counts(normalize=True).unstack().reindex(columns=STATES).fillna(0) * 100)
    by_band["n_minutes"] = minutes.groupby("mid_band", observed=True).size()
    print(by_band.round(2).to_string())

    if args.anchor_ts is not None:
        print("\nBy phase:")
        by_phase = (minutes.groupby("phase", observed=True)["state_majority"]
                    .value_counts(normalize=True).unstack().reindex(columns=STATES).fillna(0) * 100)
        by_phase["n_minutes"] = minutes.groupby("phase", observed=True).size()
        by_phase = by_phase.reindex(phase_labels)
        print(by_phase.round(2).to_string())

    gated_window = (args.anchor_ts is not None and args.window_start_min is not None
                    and args.window_end_min is not None)
    if gated_window:
        print("\n--- Gated-window report ---")
        win_start = args.anchor_ts + args.window_start_min * 60
        win_end = args.anchor_ts + args.window_end_min * 60
        um = minutes.sort_values(["ticker", "minute_ts"]).copy()
        um["prev_mid"] = um.groupby("ticker")["mid"].shift(1)
        in_window = um[(um["minute_ts"] >= win_start) & (um["minute_ts"] < win_end)]
        gated = in_window[(in_window["prev_mid"] >= args.band_lo)
                          & (in_window["prev_mid"] <= args.band_hi)]
        invalid = gated["state_majority"].isin(["CROSSED", "EMPTY_ASK", "EMPTY_BOTH"])
        print(f"Gated market-minutes (window [{args.window_start_min},{args.window_end_min}) min, "
              f"prev_mid in [{args.band_lo},{args.band_hi}]): n={len(gated)}")
        print(f"  CROSSED | EMPTY_ASK | EMPTY_BOTH (no ask-side liquidity to join): "
              f"{invalid.mean()*100:.1f}%")
        four_way = (gated["state_majority"].value_counts(normalize=True)
                    .reindex(STATES).fillna(0) * 100)
        print("  4-way breakdown of the same gated set:")
        print(four_way.round(1).to_string())

    # =====================================================================
    # TASK 2 -- spread + depth on valid books
    # =====================================================================
    print("\n" + "=" * 100)
    print("TASK 2 -- SPREAD + DEPTH ON VALID BOOKS (tick level)")
    print("=" * 100)

    valid_ticks = ticks[ticks["state"] == "VALID"].copy()
    valid_ticks["band"] = band_of(valid_ticks["mid_tick"], args.band_lo, args.band_hi)
    print(f"\nValid ticks: {len(valid_ticks):,} / {len(ticks):,} total "
          f"({len(valid_ticks) / max(1, len(ticks)) * 100:.1f}%)")

    print("\nSpread (yes_ask - yes_bid), cents, overall on VALID ticks:")
    sp = valid_ticks["spread"] * 100
    print(q(sp).round(2).to_string())

    print("\nSpread by band:")
    valid_ticks["spread_c"] = valid_ticks["spread"] * 100
    by_band_spread = valid_ticks.groupby("band", observed=True)["spread_c"].apply(q).unstack()
    print(by_band_spread.round(2).to_string())

    print("\nDepth at touch -- bid (contracts), VALID ticks:")
    print(q(valid_ticks["depth_bid_eff"]).round(1).to_string())
    print("\nDepth at touch -- ask (contracts), VALID ticks:")
    print(q(valid_ticks["depth_ask_eff"]).round(1).to_string())

    if "depth_bid2_eff" in valid_ticks.columns:
        print("\nDepth level-2 -- bid (contracts), VALID ticks:")
        print(q(valid_ticks["depth_bid2_eff"]).round(1).to_string())
        print("\nDepth level-2 -- ask (contracts), VALID ticks:")
        print(q(valid_ticks["depth_ask2_eff"]).round(1).to_string())

    buckets = [0, 1, 5, 20, 100, np.inf]
    blabels = ["<=1", "(1,5]", "(5,20]", "(20,100]", ">100"]
    print("\nDepth-at-touch bucket distribution (fraction of VALID ticks, ask side):")
    print((pd.cut(valid_ticks["depth_ask_eff"], bins=buckets, labels=blabels)
           .value_counts(normalize=True).reindex(blabels) * 100).round(1).to_string())
    print("\nDepth-at-touch bucket distribution (fraction of VALID ticks, bid side):")
    print((pd.cut(valid_ticks["depth_bid_eff"], bins=buckets, labels=blabels)
           .value_counts(normalize=True).reindex(blabels) * 100).round(1).to_string())

    # =====================================================================
    # TASK 3 -- stability / rest dynamics
    # =====================================================================
    print("\n" + "=" * 100)
    print("TASK 3 -- STABILITY / REST DYNAMICS (per-second grid)")
    print("=" * 100)

    grid["band"] = band_of(grid["mid_lkm"], args.band_lo, args.band_hi)

    run_records = []
    transition_counts: dict[str, int] = {}
    tape_hours: dict[str, float] = {}
    grid_by_ticker = {t: g.reset_index(drop=True) for t, g in grid.groupby("ticker", sort=False)}
    for ticker, g in grid_by_ticker.items():
        states = g["state"].to_numpy()
        runs = run_length_encode(states)
        tape_hours[ticker] = len(g) / 3600.0
        n_valid_to_invalid = 0
        for i, (st, ln, _s_idx, _e_idx) in enumerate(runs):
            run_records.append({"ticker": ticker, "state": st, "length_s": ln})
            if st == "VALID" and i + 1 < len(runs):
                n_valid_to_invalid += 1
        transition_counts[ticker] = n_valid_to_invalid

    runs_df = pd.DataFrame(run_records)
    if len(runs_df):
        valid_runs = runs_df[runs_df["state"] == "VALID"]
        print(f"\nPooled VALID run length (seconds) across {len(grid_by_ticker)} markets: "
              f"median={valid_runs['length_s'].median():.1f}  "
              f"p75={valid_runs['length_s'].quantile(.75):.1f}  "
              f"p90={valid_runs['length_s'].quantile(.90):.1f}  "
              f"max={valid_runs['length_s'].max():.0f}")

        per_mkt_stability = valid_runs.groupby("ticker")["length_s"].agg(
            longest_valid_run_s="max", median_valid_run_s="median", n_valid_runs="count")
        per_mkt_stability["transitions_valid_to_invalid"] = pd.Series(transition_counts)
        per_mkt_stability["tape_hours"] = pd.Series(tape_hours)
        per_mkt_stability["transitions_per_hour"] = (
            per_mkt_stability["transitions_valid_to_invalid"] / per_mkt_stability["tape_hours"])
        print("\nPer-market stability table:")
        print(per_mkt_stability.round(2).to_string())

        print(f"\nOverall transition rate (VALID->invalid), pooled: "
              f"{sum(transition_counts.values()) / sum(tape_hours.values()):.2f} / market-hour")

    print("\nFraction of tape-seconds belonging to a VALID run of length >= T, by band "
          "(denominator = ALL seconds in that band; numerator = seconds within a qualifying "
          "VALID run):")
    t_list = [5, 15, 30, 60, 120]
    band_totals: dict[str, int] = {}
    for lbl in [f"<{args.band_lo}", f"{args.band_lo}-{args.band_hi}", f">{args.band_hi}"]:
        band_totals[lbl] = 0
    band_qualify = {t: dict(band_totals) for t in t_list}
    for _ticker, g in grid_by_ticker.items():
        states = g["state"].to_numpy()
        bands = g["band"].to_numpy(dtype=object)
        runs = run_length_encode(states)
        run_len_per_sec = np.zeros(len(states), dtype=np.int64)
        for st, ln, s_idx, e_idx in runs:
            run_len_per_sec[s_idx:e_idx] = ln if st == "VALID" else 0
        for b in band_totals:
            mask_b = bands == b
            band_totals[b] += mask_b.sum()
            for t in t_list:
                band_qualify[t][b] += int(((run_len_per_sec >= t) & mask_b).sum())

    rest_table = pd.DataFrame({
        f"T>={t}s": {b: (band_qualify[t][b] / band_totals[b] * 100 if band_totals[b] else np.nan)
                    for b in band_totals}
        for t in t_list
    })
    rest_table["n_seconds_in_band"] = pd.Series(band_totals)
    total_secs = sum(band_totals.values())
    if total_secs:
        rest_table.loc["ALL"] = [sum(band_qualify[t].values()) / total_secs * 100
                                  for t in t_list] + [total_secs]
    print(rest_table.round(2).to_string())

    # =====================================================================
    # SAVE OUTPUTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("SAVING OUTPUTS")
    print("=" * 100)

    out_minutes = out_dir / "book_shape_minutes.parquet"
    minutes.to_parquet(out_minutes, index=False)
    print(f"Wrote {out_minutes} ({len(minutes):,} rows)")

    out_gaps = out_dir / "book_shape_gaps.parquet"
    runs_df.to_parquet(out_gaps, index=False)
    print(f"Wrote {out_gaps} ({len(runs_df):,} rows)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
