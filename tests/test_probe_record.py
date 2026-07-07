"""Recorder schema + notification dry-run tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pmlab.probe.notify import Event, Notification, Notifier
from pmlab.probe.record import (
    BOOK_TICK_COLUMNS,
    LiveState,
    ProbeStore,
    RunSummary,
    validate_frame,
)


def test_validate_frame_rejects_wrong_columns():
    bad = pd.DataFrame({"nope": [1]})
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_frame(bad, BOOK_TICK_COLUMNS)


def test_store_roundtrip_ticks_summary(tmp_path: Path):
    store = ProbeStore("OCC", tmp_path)
    ticks = pd.DataFrame(
        [{"ticker": "T-X", "recv_utc": 1, "seq": 1, "yes_bid": 0.2, "yes_ask": 0.24,
          "depth_bid": 300.0, "depth_ask": 150.0,
          "yes_bid2": 0.19, "yes_ask2": 0.25, "depth_bid2": 80.0, "depth_ask2": 40.0}],
        columns=BOOK_TICK_COLUMNS,
    )
    assert store.append_ticks(ticks) == 1
    assert store.append_ticks(ticks) == 2  # append-safe

    summary = RunSummary(occasion_id="OCC", speaker="Jane Doe", event_type="rally",
                         host="mac-launchd", whisper_model="base")
    store.write_summary(summary)
    assert store.load_summary()["speaker"] == "Jane Doe"
    assert store.ticks_path.exists() and store.summary_path.exists()


def test_probe_store_default_subdir_is_probe(tmp_path: Path):
    store = ProbeStore("OCC", tmp_path)
    assert store.dir == tmp_path / "probe" / "OCC"


def test_probe_store_sports_subdir_isolates_the_tree(tmp_path: Path):
    """Record-only broadcast capture passes subdir="sports" so its tape lands outside the speech
    probe's tree."""
    store = ProbeStore("EVENT-EXAMPLE", tmp_path, subdir="sports")
    assert store.dir == tmp_path / "sports" / "EVENT-EXAMPLE"
    assert store.dir.exists()
    assert not (tmp_path / "probe").exists()


def test_live_state_roundtrip_and_absent_on_first_attempt(tmp_path: Path):
    """Attempt 1 has no live_state.json; a write/read roundtrip carries the original
    occurrence_ts exactly."""
    store = ProbeStore("OCC", tmp_path)
    assert store.load_live_state() is None

    store.write_live_state(LiveState(occurrence_ts=1_800_000_000, attempt=1))
    loaded = store.load_live_state()
    assert loaded is not None
    assert (loaded.occurrence_ts, loaded.attempt) == (1_800_000_000, 1)

    store.write_live_state(LiveState(occurrence_ts=1_800_000_000, attempt=2))
    loaded2 = store.load_live_state()
    assert loaded2 is not None and loaded2.attempt == 2


def test_notifier_dry_run_does_not_send():
    n = Notifier(topic="", dry=True)  # no topic → dry regardless
    out = n.notify(Event.SIGNAL, "T-X 0.28")
    assert isinstance(out, Notification) and out.sent is False
    assert out.title == "capture signal" and out.priority == 4


def test_notifier_covers_every_event():
    n = Notifier(topic="", dry=True)
    for ev in Event:
        out = n.notify(ev, "msg")
        assert out.sent is False and out.title and 1 <= out.priority <= 5
