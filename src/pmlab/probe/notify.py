"""Push notifications: occasion armed, capture started, each event of note, run complete, and any
crash/stall.

Uses ntfy.sh — a free, no-account HTTP pub-sub: publishing is a plain ``POST {base}/{topic}`` with
the message as the body and ``Title``/``Priority``/``Tags`` headers (subscribe on the phone app by
topic name). $0, no key. When no topic is configured (or ``dry=True``) the notifier logs the
payload instead of hitting the network — this is also the default so a misconfigured run never
blocks on the network.

The ntfy topic is a public channel name, not a credential (pick an unguessable one) — no secrets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import httpx

log = logging.getLogger("pmlab.probe.notify")


class Event(StrEnum):
    ARMED = "armed"
    CAPTURE_STARTED = "capture_started"
    SIGNAL = "signal"
    COMPLETE = "complete"
    CRASH = "crash"
    HEALTH = "health"  # degraded-but-alive (self-healed): ASR bound hit, WS lag, etc.


# ntfy priority (1..5) + tag emoji per event — purely cosmetic on the phone.
_META: dict[Event, tuple[int, str, str]] = {
    Event.ARMED: (3, "bell", "capture armed"),
    Event.CAPTURE_STARTED: (3, "microphone", "capture started"),
    Event.SIGNAL: (4, "chart_with_upwards_trend", "capture signal"),
    Event.COMPLETE: (3, "checkered_flag", "capture complete"),
    Event.CRASH: (5, "rotating_light", "capture crash/stall"),
    Event.HEALTH: (4, "warning", "capture health"),
}


@dataclass(frozen=True)
class Notification:
    """A resolved notification payload (returned so callers/tests can assert without a network)."""

    event: Event
    title: str
    message: str
    priority: int
    tags: str
    sent: bool


@dataclass
class Notifier:
    """Fire-and-forget ntfy publisher. Never raises on a network failure — a dropped notification
    must not take down a capture run (the failure is logged and the run continues)."""

    topic: str = ""
    base: str = "https://ntfy.sh"
    dry: bool = False
    timeout_s: float = 10.0

    def notify(self, event: Event, message: str) -> Notification:
        priority, tags, title = _META[event]
        payload = Notification(
            event=event, title=title, message=message,
            priority=priority, tags=tags, sent=False,
        )
        if self.dry or not self.topic:
            log.info("notify[dry] %s: %s", title, message)
            return payload
        try:
            httpx.post(
                f"{self.base.rstrip('/')}/{self.topic}",
                content=message.encode("utf-8"),
                headers={"Title": title, "Priority": str(priority), "Tags": tags},
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:  # network hiccup must not crash the run
            log.warning("notify failed (%s): %s", event.value, exc)
            return payload
        return Notification(event, title, message, priority, tags, sent=True)
