"""Transcript acquisition: YouTube auto-captions → word stream (relative video time).

Fetchable auto-captions exist for most live/archived YouTube streams, so no local ASR is strictly
required. Primary path: ``yt-dlp`` pulls the ``json3`` auto-caption track, which carries per-word
timing (``event.tStartMs`` + ``seg.tOffsetMs``). This module fetches + parses that into a word
list on **video-relative** milliseconds; ``signals/align.py`` then maps video-relative → absolute
UTC via an anchor and emits the :class:`~pmlab.signals.align.Transcript` frame.

Whisper-on-video is the documented fallback for occasions without a usable caption track; it plugs
in behind the same :class:`RawWord` interface. Network I/O lives in ``fetch_*``; ``parse_json3`` is
pure and unit-tested. (Courtesy: use a neutral UA / rate-limit any scraping built on top of this.)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("pmlab.signals.transcripts")

# Caption tokens that are not spoken words (crowd/audio cues, speaker-change carets, music).
_MARKER_RE = re.compile(r"^\s*(\[[^\]]*\]|>>+|♪+|\(.*\))\s*$")
_LEAD_CARET = re.compile(r"^>>+\s*")
_DEFAULT_WORD_MS = 300  # fallback per-word duration when the next start is missing/absurd


@dataclass(frozen=True)
class RawWord:
    """One caption token on video-relative milliseconds (start of speech = align.py's job)."""

    word: str
    t_start_ms: int
    t_end_ms: int


def _clean(tok: str) -> str:
    return _LEAD_CARET.sub("", tok).strip()


def parse_json3(data: dict[str, Any]) -> list[RawWord]:
    """Parse a YouTube ``json3`` caption blob into spoken words on video-relative ms.

    Word start = ``event.tStartMs + seg.tOffsetMs``; end = the next word's start (clamped to a sane
    max), else ``+_DEFAULT_WORD_MS``. Audio-cue / speaker-caret / music tokens are dropped."""
    raw: list[tuple[str, int]] = []  # (word, start_ms)
    for ev in data.get("events", []):
        t0 = ev.get("tStartMs")
        segs = ev.get("segs")
        if t0 is None or not segs:
            continue
        for s in segs:
            tok = s.get("utf8", "")
            if not tok or _MARKER_RE.match(tok):
                continue
            cleaned = _clean(tok)
            if not cleaned:
                continue
            raw.append((cleaned, int(t0) + int(s.get("tOffsetMs", 0) or 0)))
    raw.sort(key=lambda w: w[1])
    out: list[RawWord] = []
    for i, (word, start) in enumerate(raw):
        nxt = raw[i + 1][1] if i + 1 < len(raw) else start + _DEFAULT_WORD_MS
        end = nxt if start < nxt <= start + 4000 else start + _DEFAULT_WORD_MS
        out.append(RawWord(word=word, t_start_ms=start, t_end_ms=end))
    return out


def _yt_dlp(args: list[str], *, timeout: float = 180) -> subprocess.CompletedProcess[str]:
    # Invoke via the running interpreter's module (``python -m yt_dlp``) so it works regardless of
    # whether the ``yt-dlp`` console script is on PATH (it's in the venv bin, not always on PATH).
    return subprocess.run(  # noqa: S603 - fixed interpreter, args are not shell-interpolated
        [sys.executable, "-m", "yt_dlp", *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


@dataclass(frozen=True)
class VideoMeta:
    """One search candidate. ``release_ts`` (unix s, UTC) is the live-broadcast start for
    ``was_live`` videos — an *absolute-time anchor* for align.py (caption t=0 == stream start),
    independent of the candle-cross anchors. Upload-time for non-live videos is useless as an
    anchor, so ``release_ts`` is only trusted when ``was_live`` is True."""

    video_id: str
    duration_s: int
    title: str
    channel: str
    release_ts: int | None
    was_live: bool


def search_speech_videos(
    query: str, *, n: int = 6, min_dur_s: int = 900, max_dur_s: int = 43200
) -> list[VideoMeta]:
    """YouTube candidates for a speech query, ranked: live-with-release-timestamp first (absolute
    anchor available), then duration nearest a plausible full speech (~75 min). Non-live videos are
    capped at 3 h (recap/compilation guard); live archives may run to ``max_dur_s`` (pre-show holds
    are fine — caption times stay stream-relative and the release anchor absorbs them)."""
    fields = "%(id)s\t%(duration)s\t%(release_timestamp)s\t%(was_live)s\t%(channel)s\t%(title)s"
    r = _yt_dlp([
        "--skip-download", "--no-warnings", "--socket-timeout", "30",
        "--print", fields, f"ytsearch{n}:{query}",
    ])
    out: list[VideoMeta] = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        vid, dur, rel, live, channel = (p.strip() for p in parts[:5])
        title = "\t".join(parts[5:]).strip()
        try:
            d = int(float(dur))
        except ValueError:
            continue
        was_live = live.lower() == "true"
        try:
            rel_ts: int | None = int(float(rel))
        except ValueError:
            rel_ts = None
        if not (min_dur_s <= d <= (max_dur_s if was_live else 10800)):
            continue
        out.append(VideoMeta(vid, d, title, channel, rel_ts, was_live))
    out.sort(key=lambda v: (not (v.was_live and v.release_ts is not None),
                            abs(v.duration_s - 4500)))
    return out


def search_speech_video(
    query: str, *, min_dur_s: int = 900, max_dur_s: int = 10800
) -> str | None:
    """Back-compat single-id search (best candidate id, or None)."""
    hits = search_speech_videos(query, min_dur_s=min_dur_s, max_dur_s=max_dur_s)
    return hits[0].video_id if hits else None


def fetch_video_captions(video_id: str, out_dir: Path, *, lang: str = "en") -> list[RawWord]:
    """Fetch + parse a video's auto-caption ``json3`` into :class:`RawWord`s (empty if none)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / video_id
    _yt_dlp([
        "--skip-download", "--write-auto-subs", "--sub-langs", lang, "--sub-format", "json3",
        "--no-warnings", "--socket-timeout", "30", "-o", f"{stem}.%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ])
    hits = list(out_dir.glob(f"{video_id}*.json3"))
    if not hits:
        log.warning("no json3 captions for %s", video_id)
        return []
    return parse_json3(json.loads(hits[0].read_text(encoding="utf-8")))


def fetch_vod_audio(video_id: str, out_dir: Path, *, timeout: float = 1800) -> Path | None:
    """Download a VOD's audio track (``bestaudio``) for offline Whisper; idempotent (an existing
    ``vod_<id>.*`` media file short-circuits the download). Returns the media path, or None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.glob(f"vod_{video_id}.*")
                if p.suffix not in (".json", ".part", ".log")]
    if existing:
        return existing[0]
    _yt_dlp([
        "-f", "bestaudio/best", "--no-warnings", "--socket-timeout", "30",
        "-o", f"{out_dir / f'vod_{video_id}'}.%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ], timeout=timeout)
    hits = [p for p in out_dir.glob(f"vod_{video_id}.*")
            if p.suffix not in (".json", ".part", ".log")]
    if not hits:
        log.warning("vod audio download produced no file for %s", video_id)
        return None
    return hits[0]


def whisper_vod_words(
    audio_path: Path, *, model_name: str = "base", language: str = "en"
) -> list[RawWord]:
    """Batch faster-whisper over a downloaded VOD audio file → :class:`RawWord`s on video-relative
    ms — the module-docstring Whisper fallback behind the caption interface. Decode settings match
    the live streaming path (word timestamps, VAD, en) so a retrospective transcript is
    like-for-like with what a healthy live capture of the same model would have committed."""
    from faster_whisper import WhisperModel  # lazy: mirrors pmlab.probe.audio

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path), language=language, word_timestamps=True, vad_filter=True,
    )
    out: list[RawWord] = []
    for seg in segments:
        for w in getattr(seg, "words", None) or []:
            tok = w.word.strip()
            if tok:
                out.append(RawWord(word=tok, t_start_ms=int(w.start * 1000),
                                   t_end_ms=int(w.end * 1000)))
    return out


def text_preview(words: list[RawWord], n: int = 40) -> str:
    return " ".join(w.word for w in words[:n])
