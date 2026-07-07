"""Streaming-ASR pure logic: LocalAgreement-2 commit, UTC stamping, engine-consumable frame,
and the bounded-uncommitted-window guarantees (force-commit / head-drop)."""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from pmlab.corpus.store import PhraseSpec, parse_phrase_spec
from pmlab.probe.audio import (
    SAMPLE_RATE,
    TRANSCRIPT_COLUMNS,
    Hyp,
    LocalAgreement,
    StreamingWhisper,
    TranscriptBuffer,
    Word,
    words_to_frame,
)


# Inlined from pmlab.signals.align (kept import-free here so this module only depends on
# pmlab.probe.audio + pmlab.corpus.store): the generic phrase-in-transcript matcher used to prove
# a live transcript frame satisfies the same contract an offline aligned transcript does.
def _phrase_occurrence_times(words: pd.DataFrame, spec: PhraseSpec) -> list[int]:
    if not spec.matchable or spec.pattern is None or words.empty:
        return []
    toks = words["word"].astype(str).tolist()
    ends = words["t_end_utc"].astype("int64").tolist()
    starts: list[int] = []
    pos = 0
    for w in toks:
        starts.append(pos)
        pos += len(w) + 1
    text = " ".join(toks)
    times: list[int] = []
    for m in spec.pattern.finditer(text):
        last_char = max(m.start(), m.end() - 1)
        wi = bisect.bisect_right(starts, last_char) - 1
        wi = min(max(wi, 0), len(ends) - 1)
        times.append(ends[wi])
    return sorted(times)


def _matcher_said_time(words: pd.DataFrame, spec: PhraseSpec) -> int | None:
    times = _phrase_occurrence_times(words, spec)
    if len(times) >= spec.min_count:
        return times[spec.min_count - 1]
    return None


def test_local_agreement_commits_only_agreed_prefix():
    la = LocalAgreement()
    assert la.insert([("The", 0.0, 0.5), ("Fed", 0.5, 1.0), ("will", 1.0, 1.5)]) == []
    # second hypothesis: tail revised ("will" → "raised"); only the agreed prefix commits
    newly = la.insert([("The", 0.0, 0.5), ("Fed", 0.5, 1.0), ("raised", 1.0, 1.6)])
    assert [w[0] for w in newly] == ["The", "Fed"]
    # rolling window (older words dropped): agreement over the new region commits "raised"
    assert [w[0] for w in la.insert([("raised", 1.0, 1.6), ("rates", 1.6, 2.0)])] == ["raised"]
    assert [w[0] for w in la.insert([("raised", 1.0, 1.6), ("rates", 1.6, 2.0)])] == ["rates"]
    assert [w[0] for w in la.committed] == ["The", "Fed", "raised", "rates"]


def test_local_agreement_no_false_commit_on_disagreement():
    la = LocalAgreement()
    la.insert([("alpha", 0.0, 0.5)])
    assert la.insert([("beta", 0.0, 0.5)]) == []  # disagree at position 0 → nothing commits
    assert la.committed == []


def test_words_to_frame_schema_and_utc():
    cap = 1_800_000_000
    words = [
        Word("Iran", cap + 10, cap + 11, conf=0.9, emit_utc=cap + 13),
        Word("policy", cap + 11, cap + 12, conf=0.8, emit_utc=cap + 13),
    ]
    df = words_to_frame(words, "OCC")
    assert list(df.columns) == TRANSCRIPT_COLUMNS
    assert df["word_idx"].tolist() == [0, 1]
    assert df.iloc[0]["t_end_utc"] == cap + 11 and df.iloc[0]["source"] == "whisper-live"


def test_frame_is_consumable_by_a_phrase_matcher():
    """The live transcript frame must satisfy a generic matcher contract (word, t_end_utc)."""
    cap = 1_800_000_000
    words = [Word(w, cap + i, cap + i + 1, conf=1.0, emit_utc=cap + 20)
             for i, w in enumerate(["we", "will", "discuss", "Iran", "today"])]
    df = words_to_frame(words, "OCC")
    said = _matcher_said_time(df, parse_phrase_spec("Iran"))
    assert said == cap + 4  # t_end_utc of the "Iran" token


def test_transcript_buffer_grows_contiguously():
    buf = TranscriptBuffer("OCC")
    cap = 1_800_000_000
    buf.add([Word("a", cap, cap + 1, 1.0, cap + 2)])
    buf.add([Word("b", cap + 1, cap + 2, 1.0, cap + 3)])
    assert buf.n_words == 2
    assert buf.frame()["word_idx"].tolist() == [0, 1]


# --- bounded uncommitted window (StreamingWhisper.feed, stubbed _transcribe) ----------------------


def _one_second() -> np.ndarray:
    return np.zeros(SAMPLE_RATE, dtype="float32")


def _sw(**kwargs: float) -> StreamingWhisper:
    return StreamingWhisper(capture_start_utc=1_800_000_000, **kwargs)  # type: ignore[arg-type]


def test_agreement_path_unchanged_when_under_bound():
    sw = _sw(min_step_s=1.0)
    hyp: list[Hyp] = [("hello", 0.0, 0.5), ("world", 0.5, 1.0)]
    sw._transcribe = lambda: hyp  # type: ignore[method-assign]
    first = sw.feed(_one_second(), now_utc=1_800_000_001)
    second = sw.feed(_one_second(), now_utc=1_800_000_002)
    assert first == []  # LocalAgreement-2 needs two consecutive hypotheses
    assert [w.word for w in second] == ["hello", "world"]
    assert all(w.conf != w.conf for w in second)  # NaN conf = agreed words
    assert sw.n_force_commits == 0 and sw.n_head_drops == 0


def test_bound_force_commits_when_hypotheses_never_agree():
    sw = _sw(window_s=4.0, min_step_s=1.0, max_uncommitted_s=5.0, force_hold_back_s=2.0)
    calls = {"n": 0}

    def never_agree() -> list[Hyp]:
        calls["n"] += 1
        end = sw._buf_start_s + len(sw._buf) / SAMPLE_RATE
        return [(f"w{calls['n']}_{i}", float(i), i + 0.5) for i in range(int(end))]

    sw._transcribe = never_agree  # type: ignore[method-assign]
    got: list[Word] = []
    for _ in range(12):
        got += sw.feed(_one_second(), now_utc=1_800_000_100)
    assert sw.n_force_commits >= 2, "the bound must keep firing, not just once"
    assert got, "a wedged ASR must still emit words"
    assert all(w.conf == -1.0 for w in got)  # nothing agreed → everything forced
    ends = [w.t_end_utc for w in got]
    assert ends == sorted(ends)  # committed timeline stays monotonic
    assert sw.uncommitted_s <= sw.max_uncommitted_s + 1.0
    assert len(sw._buf) / SAMPLE_RATE <= sw.max_uncommitted_s + sw.window_s + 1.0


def test_bound_head_drops_on_empty_hypotheses():
    sw = _sw(window_s=3.0, min_step_s=1.0, max_uncommitted_s=4.0)
    sw._transcribe = list  # type: ignore[method-assign] - always-empty hypothesis (VAD-silence)
    for i in range(15):
        assert sw.feed(_one_second(), now_utc=1_800_000_100 + i) == []
    assert sw.n_head_drops >= 2
    assert sw.n_force_commits == 0
    assert sw.uncommitted_s <= sw.max_uncommitted_s + 1.0
    assert len(sw._buf) / SAMPLE_RATE <= sw.max_uncommitted_s + sw.window_s + 1.0


def test_transcribe_sample_leaves_streaming_state_untouched():
    """The join-verification pass must not disturb the rolling commit stream."""
    sw = _sw(min_step_s=1.0)
    sw._transcribe = lambda: [("hello", 0.0, 0.5), ("world", 0.5, 1.0)]  # type: ignore[method-assign]
    sw.feed(_one_second(), now_utc=1_800_000_001)
    sw.feed(_one_second(), now_utc=1_800_000_002)
    buf_before = sw._buf.copy()
    committed_before = list(sw._agree.committed)
    counts_before = (sw.n_force_commits, sw.n_head_drops)

    class _FakeWord:
        def __init__(self, word: str, start: float, end: float) -> None:
            self.word, self.start, self.end = word, start, end

    class _FakeSeg:
        def __init__(self, words: list[_FakeWord]) -> None:
            self.words = words

    class _FakeModel:
        def transcribe(self, pcm: np.ndarray, **kwargs: object) -> tuple[list[_FakeSeg], None]:
            return [_FakeSeg([_FakeWord("sample", 0.0, 0.4), _FakeWord("text", 0.4, 0.9)])], None

    sw._ensure_model = lambda: _FakeModel()  # type: ignore[method-assign]
    hyp = sw.transcribe_sample(np.zeros(SAMPLE_RATE, dtype="float32"))
    assert [w for w, _s, _e in hyp] == ["sample", "text"]
    assert np.array_equal(sw._buf, buf_before)
    assert sw._agree.committed == committed_before
    assert (sw.n_force_commits, sw.n_head_drops) == counts_before


# --- PcmStream.kill tolerates a None _ytdlp leg (device_pcm has no yt-dlp process) ----------------


def test_pcmstream_kill_tolerates_none_ytdlp_and_already_exited_ffmpeg():
    import subprocess

    from pmlab.probe.audio import PcmStream

    ffmpeg = subprocess.Popen(["true"])  # noqa: S603,S607 - fixed, no shell, exits immediately
    ffmpeg.wait()
    stream = PcmStream(chunks=iter([]), _ytdlp=None, _ffmpeg=ffmpeg)
    stream.kill()  # must not raise despite _ytdlp is None and ffmpeg already reaped


def test_pcmstream_kill_kills_a_running_ffmpeg_with_no_ytdlp_leg():
    import subprocess

    from pmlab.probe.audio import PcmStream

    ffmpeg = subprocess.Popen(["sleep", "5"])  # noqa: S603,S607 - fixed, no shell
    stream = PcmStream(chunks=iter([]), _ytdlp=None, _ffmpeg=ffmpeg)
    stream.kill()
    ffmpeg.wait(timeout=2)
    assert ffmpeg.poll() is not None


def test_bound_never_fires_on_healthy_stream():
    """A stream that keeps committing must never hit the bound (byte-identical to old behavior)."""
    sw = _sw(window_s=6.0, min_step_s=1.0, max_uncommitted_s=8.0)
    state = {"end": 0.0}

    def steady() -> list[Hyp]:
        end = sw._buf_start_s + len(sw._buf) / SAMPLE_RATE
        state["end"] = end
        # stable hypothesis: same words every call → agreement commits continuously
        return [(f"tok{i}", float(i), i + 0.5) for i in range(int(end))]

    sw._transcribe = steady  # type: ignore[method-assign]
    total: list[Word] = []
    for _ in range(20):
        total += sw.feed(_one_second(), now_utc=1_800_000_200)
    assert sw.n_force_commits == 0 and sw.n_head_drops == 0
    assert all(w.conf != w.conf for w in total)  # all agreed
    assert sw.uncommitted_s <= sw.max_uncommitted_s
