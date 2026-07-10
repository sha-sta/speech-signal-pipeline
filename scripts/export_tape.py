#!/usr/bin/env python3
"""Export a recorded book-tick parquet tape to the C++ engine's binary tape format.

    uv run python scripts/export_tape.py data/sports/<EVENT>/book_ticks.parquet out.tape [TICKER]

The output is the "PMTAPE01" layout read by cpp/include/pmlab_engine/tape.hpp: 8-byte magic,
u64 tick count, then packed 24-byte records (i64 recv_utc, f64 yes_bid, f64 yes_ask), one file
per market. With no TICKER argument the most-ticked market in the file is exported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAGIC = b"PMTAPE01"
RECORD = np.dtype([("ts", "<i8"), ("yes_bid", "<f8"), ("yes_ask", "<f8")])


def export(parquet_path: Path, out_path: Path, ticker: str | None = None) -> int:
    df = pd.read_parquet(parquet_path)
    if ticker is None:
        ticker = str(df["ticker"].value_counts().idxmax())
    one = df[df["ticker"] == ticker].sort_values("recv_utc")
    if one.empty:
        raise SystemExit(f"no ticks for ticker {ticker!r} in {parquet_path}")
    rec = np.empty(len(one), dtype=RECORD)
    rec["ts"] = one["recv_utc"].to_numpy(dtype=np.int64)
    rec["yes_bid"] = one["yes_bid"].to_numpy(dtype=np.float64)
    rec["yes_ask"] = one["yes_ask"].to_numpy(dtype=np.float64)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(np.uint64(len(rec)).tobytes())
        rec.tofile(f)
    print(f"{ticker}: {len(rec):,} ticks -> {out_path}")
    return len(rec)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    export(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else None)
