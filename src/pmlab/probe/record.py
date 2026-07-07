"""Parquet recorders + schemas for the probe: record everything to parquet, so any downstream
analysis (spike detection, alignment QA, microstructure) reads from a durable tape instead of a
live process.

Artifacts per occasion, under ``<data_dir>/<subdir>/<occasion_id>/``:
* ``book_ticks.parquet`` — every top-of-book observation.
* ``transcript.parquet``  — the final streaming transcript.
* ``run.json``           — the per-occasion run summary (counts, timings, host/whisper facts).

Writers are append-safe (read-modify-write parquet) so a crash-restart mid-occasion keeps prior
rows. Schemas are explicit column lists asserted by :func:`validate_frame`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from pmlab.probe.audio import TRANSCRIPT_COLUMNS

log = logging.getLogger("pmlab.probe.record")

BOOK_TICK_COLUMNS = [
    "ticker", "recv_utc", "seq", "yes_bid", "yes_ask", "depth_bid", "depth_ask",
    # New columns append at the END only: a crash-restart concat over a pre-existing tape aligns
    # old rows (NaN-filled) with this order.
    "yes_bid2", "yes_ask2", "depth_bid2", "depth_ask2",
]


def validate_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Assert exact column set/order (the recorder-schema contract)."""
    if list(df.columns) != columns:
        raise ValueError(f"schema mismatch: {list(df.columns)} != {columns}")
    return df


@dataclass
class RunSummary:
    """Per-occasion run facts for notifications and any offline analysis; the recorded frames
    carry the detail, this is the human-readable header."""

    occasion_id: str
    speaker: str
    event_type: str
    host: str                       # "mac-launchd" | "vps" | "manual"
    whisper_model: str
    whisper_rtf: float = float("nan")  # measured real-time factor (self-test)
    quote_source: str = "ws"        # "ws" | "rest_poll"
    armed_utc: int = 0
    capture_start_utc: int = 0
    capture_end_utc: int = 0
    speech_start_utc: int = 0
    speech_end_utc: int = 0
    n_markets_armed: int = 0
    n_ticks: int = 0
    n_words: int = 0
    crashed: bool = False
    note: str = ""
    stop_reason: str = ""  # "speech_ended" | "max_duration" | "" (crashed/other)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


@dataclass
class LiveState:
    """Per-occasion live state: survives a relaunch so a bounce or re-invocation never resets the
    capture clock. ``occurrence_ts`` is the ORIGINAL scheduled/resolved start — resetting it to
    "now" on a relaunch would silently shrink the recorded elapsed-time-since-start for every
    consumer of this tape."""

    occurrence_ts: int
    attempt: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> LiveState:
        d = json.loads(raw)
        return cls(occurrence_ts=int(d["occurrence_ts"]), attempt=int(d["attempt"]))


@dataclass
class ProbeStore:
    """On-disk layout for one occasion's run. ``subdir`` defaults to the speech-probe layout
    (``<root>/probe/<occasion_id>/``); the record-only sports/broadcast capture passes
    ``subdir="sports"`` to keep its tape in a separate tree."""

    occasion_id: str
    root: Path
    subdir: str = "probe"
    _dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self._dir = self.root / self.subdir / self.occasion_id
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def ticks_path(self) -> Path:
        return self._dir / "book_ticks.parquet"

    @property
    def transcript_path(self) -> Path:
        return self._dir / "transcript.parquet"

    @property
    def summary_path(self) -> Path:
        return self._dir / "run.json"

    @property
    def live_state_path(self) -> Path:
        return self._dir / "live_state.json"

    def _append(self, path: Path, df: pd.DataFrame, columns: list[str]) -> int:
        validate_frame(df, columns)
        if df.empty:
            return 0
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.to_parquet(path, index=False)
        return len(df)

    def append_ticks(self, df: pd.DataFrame) -> int:
        """Append top-of-book ticks; returns the new total row count."""
        return self._append(self.ticks_path, df, BOOK_TICK_COLUMNS)

    def write_transcript(self, df: pd.DataFrame) -> Path:
        """Persist the final streaming transcript."""
        validate_frame(df, TRANSCRIPT_COLUMNS)
        df.to_parquet(self.transcript_path, index=False)
        return self.transcript_path

    def load_transcript(self) -> pd.DataFrame:
        if self.transcript_path.exists():
            return validate_frame(pd.read_parquet(self.transcript_path), TRANSCRIPT_COLUMNS)
        return pd.DataFrame(columns=TRANSCRIPT_COLUMNS)

    def write_summary(self, summary: RunSummary) -> Path:
        self.summary_path.write_text(summary.to_json(), encoding="utf-8")
        return self.summary_path

    def load_summary(self) -> dict[str, object]:
        return dict(json.loads(self.summary_path.read_text())) if self.summary_path.exists() else {}

    def load_live_state(self) -> LiveState | None:
        """``None`` on a fresh occasion (attempt 1); a relaunch/re-invocation reads the ORIGINAL
        ``occurrence_ts`` back. A corrupt/truncated file (e.g. a crash mid-write) is treated as
        absent rather than raised — a torn state file must never wedge
        :func:`~pmlab.probe.run.capture`'s startup."""
        if not self.live_state_path.exists():
            return None
        try:
            return LiveState.from_json(self.live_state_path.read_text())
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            log.warning("live_state.json corrupt/truncated (%s: %s) — treating as absent",
                       type(exc).__name__, exc)
            return None

    def write_live_state(self, state: LiveState) -> Path:
        """Atomic write (tmp sibling + ``os.replace``) so a crash mid-write can never leave a
        torn/truncated ``live_state.json`` for :meth:`load_live_state` to trip over."""
        tmp = self.live_state_path.with_name(self.live_state_path.name + ".tmp")
        tmp.write_text(state.to_json(), encoding="utf-8")
        os.replace(tmp, self.live_state_path)
        return self.live_state_path
