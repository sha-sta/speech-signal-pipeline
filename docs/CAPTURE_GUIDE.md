# Capture guide

Operational runbook for the two capture paths: local device audio and a resolved live/archived
stream. Both record a synchronized transcript + order-book tape to parquet and make no trading
decision — see `docs/ARCHITECTURE.md` for the data-flow and design rationale.

## Device-audio capture (macOS)

Use this path when you want to tape whatever audio is playing on the capture machine itself — a
browser tab, a TV/cable app, any local source — rather than resolving a stream URL.

### 1. Install a loopback device

macOS has no built-in way to route an application's audio output back in as an input. Install a
virtual loopback driver — [BlackHole](https://existential.audio/blackhole/) is a free, widely used
one (2-channel build is sufficient). After installing, it appears as both an input and an output
device in **Audio MIDI Setup**.

### 2. Build a Multi-Output Device

You want to *hear* the audio and *capture* it. In **Audio MIDI Setup** → `+` → **Create Multi-Output
Device**, check both your normal speakers/headphones and the loopback device, then set that
Multi-Output Device as your system output while the source audio is playing. The loopback device
now carries a copy of everything your speakers play.

### 3. Never trust a synthesized voice as a test signal

A common way to burn an hour: using the OS's built-in text-to-speech (`say` or similar) to "test"
that the loopback device is receiving audio. Depending on the OS audio routing, a synthesized
voice can bypass the configured output device entirely and never reach the loopback driver — so a
"no audio" test result doesn't mean the driver is broken, it means the test signal never left the
process that generated it. Always test with audio actually playing through the system output (a
real video/stream tab), not a synthesized announcement.

### 4. Pre-flight: an RMS check before every capture

Before launching a real capture, confirm the loopback device is actually receiving signal — not
silence, not the wrong channel. A short RMS (root-mean-square amplitude) probe on a few seconds of
audio from the device is enough: near-zero RMS while source audio is audibly playing means the
device isn't routed correctly (check the Multi-Output Device is still the active output, and that
the source tab/app isn't muted or on a different output). This is a cheap, mechanical check — run
it every time, not just on first setup, since device selection is easy to silently drop after a
system audio restart (see below).

### 5. Capture

```bash
uv run pmlab probe record-game <EVENT_TICKER> --audio-device "BlackHole 2ch"
```

`--kickoff` lets you schedule a future start (`HH:MM` local or ISO); omitted, capture starts
immediately. The command blocks until the configured quiet-rail or max-duration limit trips, or
until you kill it. Artifacts land under `data/sports/<EVENT_TICKER>/`.

## Stream capture (yt-dlp)

Use this path when the audio source is a resolvable YouTube stream rather than something playing
locally.

### Stream resolution

A capture needs a specific video to open. Resolution has a precedence order: an operator-pinned
video id (most reliable — skips search entirely), a linked stream discovered from the venue's own
market metadata where available, and a search fallback as the last resort. Pin the video id
whenever you have it; search-based resolution is inherently the least reliable path (title
matching against a live/scheduled stream list is brittle around near-duplicate or re-run content).

### Join-time verification sampling

Before committing to a resolved stream for the full capture, the pipeline opens it, pulls a short
audio sample, transcribes it with the already-loaded ASR model, and surfaces the sample text for
review — the automated equivalent of a human glancing at the stream to confirm it's the right one
before walking away. A sample that produces no audio within a timeout is treated as dead (the
pipeline advances to the next candidate, or retries the same pin — a pin is assumed correct, so a
dead pin is treated as a transient join issue rather than a wrong-stream signal). A sample that
transcribes to very few words is "quiet," not dead — early joins on a stream are often
pre-program music or crowd noise, and quiet is not itself a failure.

### Capture

```bash
uv run pmlab probe run-today <EVENT_TICKER>
```

This polls for the stream to actually go live (or the scheduled window to open), joins, verifies,
and then records until the quiet-rail or max-duration limit trips, with bounded automatic
relaunches if a program starts late.

## Common failure modes and their signatures

- **Silent ASR head-drops.** A rolling decode buffer that never accrues two agreeing hypotheses
  (silence, noise, or a persistently revising tail) is bounded rather than allowed to grow forever
  — past a configured limit, the pipeline either force-commits its current best guess (marked with
  a distinct low-confidence value in the tape, so it's visible on review) or, if there's nothing
  decodable at all, drops the stale buffer head and re-anchors. Neither path loses live audio
  going forward; both are visible after the fact as a run of unusually-marked words in the
  transcript.
- **Wedged decode.** If the order book is visibly moving but the transcript has gone silent for
  longer than the ASR pipeline's own commit-lag bound should allow, that's a wedge, not a quiet
  broadcast — a cross-channel watchdog detects exactly this signature and restarts only the
  audio/ASR task, leaving the order-book recorder and the capture's overall timing untouched.
- **A stream that starts at event time, not at your join time.** A resolved stream's audio clock
  is anchored at the moment your capture actually receives its first audio chunk, not at the
  configured start time or the process's own start time — a scheduled stream can leave the
  capture waiting for minutes before video actually arrives, and every word timestamp is relative
  to when audio genuinely began, not to when the capture process was launched.
- **Device renumbering after an audio subsystem restart.** Recovering a wedged system-audio setup
  by restarting the OS's core audio service is sometimes tempting — don't do it as a first
  resort. It can renumber device IDs, silently pointing a capture at the wrong hardware if
  anything in your setup selects a device by index rather than by name. Always capture by device
  **name**, not index, and re-run the RMS pre-flight after any audio-subsystem restart before
  trusting the routing again.
- **Silent mute/output drift.** Because the loopback path depends on a specific system output
  selection staying active, anything that changes it mid-capture (a system alert switching audio
  output, a Bluetooth device connecting) can silently drop the loopback device out of the signal
  path. The RMS pre-flight catches this before a capture starts; nothing currently catches it
  mid-capture, so a spot-check partway through a long capture is worth the ten seconds it costs.
