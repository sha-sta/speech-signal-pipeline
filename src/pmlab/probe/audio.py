"""Live audio → local streaming Whisper word stream.

Path: ``yt-dlp`` pipes the live stream's audio → ``ffmpeg`` decodes to 16 kHz mono float32 → a
rolling buffer is transcribed by **faster-whisper** (base/small, word timestamps) → a
LocalAgreement-2 policy commits only words that two consecutive hypotheses agree on (so the running
transcript prefix a downstream consumer reads never rewrites history). Committed words are stamped
on the stream clock ``t_end_utc = capture_start_utc + word_end_seconds`` — the live release anchor
(audio t=0 == capture start), the same role ``signals/align.py``'s release anchor plays offline.
The residual stream-buffer latency (spoken-vs-received) is bounded; any downstream comparison that
gates on ``t_end_utc`` stays causal on that one clock regardless.

A local model keeps this $0 at any capture volume. faster-whisper is lazy-imported so this module
(and the test suite) never require the model; the pure pieces (LocalAgreement commit, UTC
stamping, the running transcript frame) are unit-tested.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("pmlab.probe.audio")

SAMPLE_RATE = 16_000
TRANSCRIPT_COLUMNS = [
    "occasion_id", "word_idx", "word", "t_start_utc", "t_end_utc", "source", "asr_conf",
]

# A hypothesis word on the stream clock (seconds since capture start).
Hyp = tuple[str, float, float]  # (word, t_start_s, t_end_s)


@dataclass(frozen=True)
class Word:
    """One committed word, stamped to absolute UTC on the stream clock."""

    word: str
    t_start_utc: int
    t_end_utc: int
    conf: float
    emit_utc: int  # wall-clock the word was COMMITTED (an ASR-emission-latency metric input)


def _norm(w: str) -> str:
    return w.strip().lower()


@dataclass
class LocalAgreement:
    """LocalAgreement-2 (Ufal whisper_streaming): commit the longest uncommitted prefix that the two
    most recent hypotheses agree on (same word, near-equal time). Stable against Whisper revising a
    rolling window's tail."""

    max_dt: float = 2.0
    committed: list[Hyp] = field(default_factory=list)
    _prev: list[Hyp] = field(default_factory=list)

    def insert(self, hyp: list[Hyp]) -> list[Hyp]:
        """Feed the current window hypothesis (stream-time); return newly committed words."""
        last_end = self.committed[-1][2] if self.committed else -1.0
        cur = [w for w in hyp if w[2] > last_end + 1e-6]
        prev = [w for w in self._prev if w[2] > last_end + 1e-6]
        newly: list[Hyp] = []
        for a, b in zip(prev, cur, strict=False):
            if _norm(a[0]) == _norm(b[0]) and abs(a[1] - b[1]) <= self.max_dt:
                newly.append(b)
            else:
                break
        self.committed.extend(newly)
        self._prev = hyp
        return newly


@dataclass
class TranscriptBuffer:
    """Accumulates committed :class:`Word`s into the running transcript frame (columns match the
    offline transcript schema in ``signals/align.py``, so a downstream consumer sees one shape
    regardless of live vs. offline source)."""

    occasion_id: str
    source: str = "whisper-live"
    _words: list[Word] = field(default_factory=list)

    def add(self, words: list[Word]) -> None:
        self._words.extend(words)

    @property
    def n_words(self) -> int:
        return len(self._words)

    def frame(self) -> pd.DataFrame:
        rows = [
            {"occasion_id": self.occasion_id, "word_idx": i, "word": w.word,
             "t_start_utc": w.t_start_utc, "t_end_utc": w.t_end_utc,
             "source": self.source, "asr_conf": w.conf}
            for i, w in enumerate(self._words)
        ]
        return pd.DataFrame(rows, columns=TRANSCRIPT_COLUMNS)


def words_to_frame(
    words: list[Word], occasion_id: str, *, source: str = "whisper-live"
) -> pd.DataFrame:
    buf = TranscriptBuffer(occasion_id, source)
    buf.add(words)
    return buf.frame()


# --- live streaming (lazy faster-whisper + subprocess audio; not exercised in tests) -------------


@dataclass
class StreamingWhisper:
    """Rolling-window faster-whisper with LocalAgreement-2 commit. ``capture_start_utc`` anchors the
    stream clock. Call :meth:`feed` with float32 PCM chunks; it transcribes once enough new audio
    has accrued and returns the words committed so far this call.

    The uncommitted window is **bounded by construction**: with zero commits the trim anchor never
    advances, the buffer grows, each decode pass slows, and commits can starve in a spiral —
    unbounded, this can wedge the transcript for the whole remainder of a capture. Past
    ``max_uncommitted_s``, the current hypothesis (minus the still-revising tail) is committed
    without two-hypothesis agreement (``conf=-1.0`` marks those words in the tape), or — when
    there are no decodable words at all — the buffer head is dropped and re-anchored."""

    capture_start_utc: int
    model_name: str = "base"
    window_s: float = 15.0
    min_step_s: float = 2.0
    max_uncommitted_s: float = 75.0  # B1 bound; a wedged ASR is impossible, not just detected
    force_hold_back_s: float = 10.0  # forced commits never take the newest (still-revising) tail
    n_force_commits: int = 0
    n_head_drops: int = 0
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="float32"))
    _buf_start_s: float = 0.0        # stream-time of buf[0]
    _last_asr_end_s: float = 0.0
    _dropped_to_s: float = 0.0       # head-drop re-anchor (audio before this is gone, uncommitted)
    _agree: LocalAgreement = field(default_factory=LocalAgreement)
    _model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: only the live path needs it
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def _transcribe(self) -> list[Hyp]:
        model = self._ensure_model()
        segments, _info = model.transcribe(  # type: ignore[attr-defined]
            self._buf, language="en", word_timestamps=True, vad_filter=True,
        )
        hyp: list[Hyp] = []
        for seg in segments:
            for w in getattr(seg, "words", None) or []:
                hyp.append((w.word, self._buf_start_s + w.start, self._buf_start_s + w.end))
        return hyp

    def transcribe_sample(self, pcm: np.ndarray) -> list[Hyp]:
        """One-shot transcribe of a caller-supplied buffer (join-time verification) — the same
        faster-whisper call :meth:`_transcribe` uses (``_ensure_model``,
        ``language="en", word_timestamps=True, vad_filter=True``), but on ``pcm`` directly, with
        offsets zero-based to the sample's own start (it is not part of the rolling window).

        Does not touch ``_buf``/``_agree``/the commit counters: this is a side verification pass
        for the operator-veto ntfy, never fed into the LocalAgreement commit stream itself (the
        caller re-feeds the same raw chunks through :meth:`feed` afterward, so no audio is lost)."""
        model = self._ensure_model()
        segments, _info = model.transcribe(  # type: ignore[attr-defined]
            pcm, language="en", word_timestamps=True, vad_filter=True,
        )
        hyp: list[Hyp] = []
        for seg in segments:
            for w in getattr(seg, "words", None) or []:
                hyp.append((w.word, w.start, w.end))
        return hyp

    def _anchor_s(self) -> float:
        """Stream-time before which everything is settled: the last committed word's end, or the
        last head-drop point when nothing has committed."""
        last = self._agree.committed[-1][2] if self._agree.committed else 0.0
        return max(last, self._dropped_to_s)

    @property
    def uncommitted_s(self) -> float:
        """Audio seconds pending commit — the B1 ``commit_lag`` health metric."""
        return max(0.0, self._buf_start_s + len(self._buf) / SAMPLE_RATE - self._anchor_s())

    def feed(self, pcm: np.ndarray, *, now_utc: int) -> list[Word]:
        """Append audio; transcribe if ≥ ``min_step_s`` new audio; return newly committed words."""
        self._buf = np.concatenate([self._buf, pcm.astype("float32")])
        cur_end_s = self._buf_start_s + len(self._buf) / SAMPLE_RATE
        if cur_end_s - self._last_asr_end_s < self.min_step_s:
            return []
        self._last_asr_end_s = cur_end_s
        hyp = self._transcribe()
        committed = self._agree.insert(hyp)
        forced: list[Hyp] = []
        if cur_end_s - self._anchor_s() > self.max_uncommitted_s:
            # B1 bound exceeded — force progress. A possibly-noisier word beats a frozen
            # transcript; forced words carry conf=-1.0 so the tape shows them.
            anchor = self._anchor_s()
            cutoff = cur_end_s - self.force_hold_back_s
            forced = [w for w in hyp if anchor + 1e-6 < w[2] <= cutoff]
            if forced:
                self._agree.committed.extend(forced)
                self.n_force_commits += 1
                log.warning("ASR force-commit: %d words (uncommitted %.0fs > %.0fs bound)",
                            len(forced), cur_end_s - anchor, self.max_uncommitted_s)
            else:
                # No decodable words at all (VAD-empty / non-speech): drop the head, re-anchor,
                # keep a window of context. Nothing is lost — there were no words to lose.
                self._dropped_to_s = max(self._dropped_to_s, cur_end_s - self.window_s)
                self.n_head_drops += 1
                log.warning("ASR head-drop: re-anchored to %.0fs (no words to force-commit)",
                            self._dropped_to_s)
        # trim audio the committed (or re-anchored) prefix has consumed, keeping window context
        keep_from = max(0.0, self._anchor_s() - self.window_s)
        if keep_from > self._buf_start_s:
            drop = int((keep_from - self._buf_start_s) * SAMPLE_RATE)
            self._buf = self._buf[drop:]
            self._buf_start_s = keep_from
        return [
            Word(word=w, t_start_utc=self.capture_start_utc + int(t0),
                 t_end_utc=self.capture_start_utc + int(t1), conf=conf, emit_utc=now_utc)
            for words, conf in ((committed, float("nan")), (forced, -1.0))
            for (w, t0, t1) in words
        ]


@dataclass
class PcmStream:
    """Handle for a live :func:`stream_pcm`: ``.chunks`` is the float32-PCM iterator, driven by
    callers via ``loop.run_in_executor(None, next, stream.chunks, None)``.

    A caller that abandons that future (S2 :func:`verify_stream`'s first-chunk timeout, or S3
    :func:`_bounce_audio` cancelling the audio task) leaves the executor thread blocked forever in
    ``ffmpeg.stdout.read()`` — the generator's own ``finally`` never runs because nothing is
    driving the generator to exhaustion. ``.kill()`` lets the abandon path reach in directly:
    killing ffmpeg makes that blocked ``read()`` return EOF, so the thread unblocks and returns to
    the pool. Idempotent (swallows "already dead") so it is safe to call from every abandon site,
    including ones that raced with the generator's own normal-exhaustion cleanup."""

    chunks: Iterator[np.ndarray]
    _ytdlp: subprocess.Popen[bytes] | None  # None for a device_pcm stream (no yt-dlp leg)
    _ffmpeg: subprocess.Popen[bytes]

    def kill(self) -> None:
        for proc in (self._ffmpeg, self._ytdlp):
            if proc is None:
                continue
            try:
                if proc.poll() is None:
                    proc.kill()
            except OSError:
                pass  # already reaped / raced with a normal exit — nothing to do


def stream_pcm(url: str, *, chunk_s: float = 1.0) -> PcmStream:
    """Open a live stream URL via ``yt-dlp | ffmpeg`` (16 kHz mono) and return a :class:`PcmStream`
    handle wrapping the chunk iterator. Live path only — requires ``ffmpeg`` on the host. Chunks
    are decoded s16le → float32 in [-1, 1]. ``--wait-for-video`` blocks (polling ~30 s) until a
    scheduled stream actually goes live — auto-starts the capture the moment it does; the caller
    anchors the audio clock at first-chunk arrival, not at process start."""
    # "bestaudio/best": live HLS streams often expose NO audio-only format (bestaudio alone fails
    # with "Requested format is not available"); fall back to the muxed stream — ffmpeg drops the
    # video track when decoding to PCM.
    ytdlp = subprocess.Popen(  # noqa: S603 - fixed interpreter, url is not shell-interpolated
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio/best", "--quiet",
         "--wait-for-video", "30", "-o", "-", url],
        stdout=subprocess.PIPE,
    )
    ffmpeg = subprocess.Popen(  # noqa: S603,S607 - ffmpeg from PATH, args fixed
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        stdin=ytdlp.stdout, stdout=subprocess.PIPE,
    )
    if ytdlp.stdout is not None:
        ytdlp.stdout.close()  # allow ytdlp to receive SIGPIPE if ffmpeg exits
    n_bytes = int(SAMPLE_RATE * chunk_s) * 2  # int16

    def _chunks() -> Iterator[np.ndarray]:
        try:
            assert ffmpeg.stdout is not None
            while True:
                raw = ffmpeg.stdout.read(n_bytes)
                if not raw:
                    break
                yield np.frombuffer(raw, dtype="int16").astype("float32") / 32768.0
        finally:
            # Backstop for the normal-exhaustion path (stream ended on its own); PcmStream.kill()
            # is the abandon-path cleanup and may have already reaped these.
            for proc in (ffmpeg, ytdlp):
                if proc.poll() is None:
                    proc.terminate()

    return PcmStream(chunks=_chunks(), _ytdlp=ytdlp, _ffmpeg=ffmpeg)


def device_pcm(device: str, *, chunk_s: float = 1.0) -> PcmStream:
    """Open a Mac system-audio loopback device via ffmpeg's ``avfoundation`` input alone — no
    yt-dlp, no stream resolution. Intended for a loopback device (e.g. BlackHole) fed by a
    browser tab or another local audio source; live path, not exercised in tests."""
    ffmpeg = subprocess.Popen(  # noqa: S603,S607 - ffmpeg from PATH; device name is operator-given
        ["ffmpeg", "-loglevel", "quiet", "-f", "avfoundation", "-i", f":{device}",
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        stdout=subprocess.PIPE,
    )
    n_bytes = int(SAMPLE_RATE * chunk_s) * 2  # int16

    def _chunks() -> Iterator[np.ndarray]:
        try:
            assert ffmpeg.stdout is not None
            while True:
                raw = ffmpeg.stdout.read(n_bytes)
                if not raw:
                    break
                yield np.frombuffer(raw, dtype="int16").astype("float32") / 32768.0
        finally:
            if ffmpeg.poll() is None:
                ffmpeg.terminate()

    return PcmStream(chunks=_chunks(), _ytdlp=None, _ffmpeg=ffmpeg)
