"""Kalshi authenticated WS orderbook stream + REST top-of-book fallback.

Read-only market data ONLY. The API key authenticates the WS handshake (and, if ever needed, signed
REST); **no order endpoint is imported, referenced, or reachable from this module**.

Two drift-prone pieces are pure and unit-tested here (the live socket is not exercised in tests):

* :class:`KalshiSigner` — RSA-PSS request signing per the current docs: sign ``{ts}{METHOD}{path}``
  where ``ts`` is Unix **milliseconds**, ``path`` carries the ``/trade-api/v2`` prefix and **no**
  query string; PSS(MGF1(SHA-256), salt=digest=32); base64. Headers ``KALSHI-ACCESS-{KEY,SIGNATURE,
  TIMESTAMP}``.
* :class:`OrderBook` — reconstructs best YES bid/ask + touch depth from the post-migration
  ``orderbook_snapshot`` / ``orderbook_delta`` payloads (``yes_dollars_fp``/``no_dollars_fp`` arrays
  of ``[price_dollars, count_fp]`` strings; best YES ask = ``1 − best NO bid``). Parses defensively
  and falls back to legacy integer-cents fields if the live socket still emits them — assert the
  real keys from one logged frame before trusting a prod run (docs note the schema may still drift).

The live loop lazy-imports ``websockets`` so importing this module never requires it.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from pmlab.config import Settings, get_settings
from pmlab.probe.bars import BookTick

log = logging.getLogger("pmlab.probe.book")

WS_PATH = "/trade-api/ws/v2"


# --- RSA-PSS request signing ---------------------------------------------------------------------


def _load_private_key(path: Path) -> RSAPrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, RSAPrivateKey):
        raise TypeError(f"{path} is not an RSA private key")
    return key


@dataclass
class KalshiSigner:
    """Signs Kalshi authenticated requests. ``now_ms`` is injectable for deterministic tests."""

    key_id: str
    key: RSAPrivateKey
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> KalshiSigner:
        s = settings or get_settings()
        if not s.kalshi_api_key_id or s.kalshi_private_key_path is None:
            raise SystemExit(
                "Kalshi API key not configured — set PMLAB_KALSHI_API_KEY_ID + "
                "PMLAB_KALSHI_PRIVATE_KEY_PATH (see .env.example)."
            )
        return cls(s.kalshi_api_key_id, _load_private_key(Path(s.kalshi_private_key_path)))

    @staticmethod
    def signed_message(timestamp_ms: int, method: str, path: str) -> str:
        """The exact string signed: ``ts + METHOD + path`` with the query string stripped."""
        path = path.split("?", 1)[0]
        return f"{timestamp_ms}{method.upper()}{path}"

    def _sign(self, message: str) -> str:
        sig = self.key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("ascii")

    def headers(self, method: str, path: str) -> dict[str, str]:
        ts = self.now_ms()
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(self.signed_message(ts, method, path)),
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
        }


# --- orderbook reconstruction --------------------------------------------------------------------


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _levels(raw: Any) -> dict[float, float]:
    """Parse a ``[[price, count], ...]`` array into a ``{price_dollars: count}`` map. Prices already
    in dollars (strings like ``"0.08"``) are kept; bare integer cents (legacy) divide by 100."""
    out: dict[float, float] = {}
    for pair in raw or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        price, count = _f(pair[0]), _f(pair[1])
        if price > 1.0:  # legacy integer-cents encoding
            price /= 100.0
        if price == price and count > 0:  # NaN-safe; zero-count snapshot levels are phantom tops
            out[round(price, 4)] = count
    return out


@dataclass
class OrderBook:
    """Best-bid/ask + touch depth for one market, updated by snapshot then deltas."""

    market_ticker: str
    yes: dict[float, float] = field(default_factory=dict)  # YES bids: price → count
    no: dict[float, float] = field(default_factory=dict)   # NO bids:  price → count

    def apply_snapshot(self, msg: dict[str, Any]) -> None:
        self.yes = _levels(msg.get("yes_dollars_fp", msg.get("yes")))
        self.no = _levels(msg.get("no_dollars_fp", msg.get("no")))

    def apply_delta(self, msg: dict[str, Any]) -> None:
        side = str(msg.get("side", "")).lower()
        book = self.yes if side == "yes" else self.no if side == "no" else None
        if book is None:
            return
        price = _f(msg.get("price_dollars", msg.get("price")))
        if price > 1.0:
            price /= 100.0
        delta = _f(msg.get("delta_fp", msg.get("delta")))
        if price != price or delta != delta:
            return
        key = round(price, 4)
        book[key] = book.get(key, 0.0) + delta
        if book[key] <= 0:
            book.pop(key, None)

    def best_yes_bid(self) -> tuple[float, float]:
        if not self.yes:
            return float("nan"), float("nan")
        p = max(self.yes)
        return p, self.yes[p]

    def best_yes_ask(self) -> tuple[float, float]:
        """YES ask = 1 − best NO bid; available size = the NO-bid size at that level."""
        if not self.no:
            return float("nan"), float("nan")
        p = max(self.no)
        return round(1.0 - p, 4), self.no[p]

    def second_yes_bid(self) -> tuple[float, float]:
        """Next YES bid level below the touch (touch + one level deeper)."""
        if len(self.yes) < 2:
            return float("nan"), float("nan")
        p = sorted(self.yes)[-2]
        return p, self.yes[p]

    def second_yes_ask(self) -> tuple[float, float]:
        """Next YES ask level above the touch = 1 − second-highest NO bid."""
        if len(self.no) < 2:
            return float("nan"), float("nan")
        p = sorted(self.no)[-2]
        return round(1.0 - p, 4), self.no[p]

    def tick(self, recv_utc: int) -> BookTick:
        bid, depth_bid = self.best_yes_bid()
        ask, depth_ask = self.best_yes_ask()
        bid2, depth_bid2 = self.second_yes_bid()
        ask2, depth_ask2 = self.second_yes_ask()
        return BookTick(ts=recv_utc, yes_bid=bid, yes_ask=ask,
                        depth_bid=depth_bid, depth_ask=depth_ask,
                        yes_bid2=bid2, yes_ask2=ask2,
                        depth_bid2=depth_bid2, depth_ask2=depth_ask2)


def subscribe_cmd(tickers: Sequence[str], *, cmd_id: int = 1) -> str:
    """The ``orderbook_delta`` subscribe command (field names per the current docs)."""
    return json.dumps({
        "id": cmd_id, "cmd": "subscribe",
        "params": {"channels": ["orderbook_delta"], "market_tickers": list(tickers)},
    })


@dataclass
class BookUpdate:
    """One reconstructed top-of-book after applying a WS message."""

    market_ticker: str
    seq: int
    tick: BookTick


def process_frame(
    books: dict[str, OrderBook], frame: dict[str, Any], recv_utc: int
) -> BookUpdate | None:
    """Apply one parsed WS frame to the book set; None for non-book / unknown-ticker frames.

    ``recv_utc`` is stamped by the caller AT SOCKET READ, before JSON parse and book rebuild —
    stamping after processing can silently hide event-loop backlog: tick timestamps must measure
    the network edge, not the event loop's health."""
    kind = frame.get("type")
    msg = frame.get("msg", {})
    ticker = str(msg.get("market_ticker", ""))
    book = books.get(ticker)
    if book is None or kind not in ("orderbook_snapshot", "orderbook_delta"):
        return None
    if kind == "orderbook_snapshot":
        book.apply_snapshot(msg)
    else:
        book.apply_delta(msg)
    return BookUpdate(ticker, int(frame.get("seq", 0)), book.tick(recv_utc))


# --- live WS stream (lazy websockets) ------------------------------------------------------------


async def stream_orderbook(
    tickers: Sequence[str], *, settings: Settings | None = None,
    on_raw: Callable[[dict[str, Any]], None] | None = None,
) -> AsyncIterator[BookUpdate]:
    """Connect (authenticated), subscribe to ``orderbook_delta`` for ``tickers``, and yield a
    :class:`BookUpdate` per snapshot/delta. ``on_raw`` (optional) receives the first raw frames so a
    run can log-and-assert the live schema before trusting the parser (docs caveat). Reconnection is
    the caller's job (``run.py`` wraps this with backoff)."""
    import websockets  # lazy: only the live path needs it

    s = settings or get_settings()
    signer = KalshiSigner.from_settings(s)
    books: dict[str, OrderBook] = {t: OrderBook(t) for t in tickers}
    async with websockets.connect(
        s.kalshi_ws_base, additional_headers=signer.headers("GET", WS_PATH), max_size=None,
        open_timeout=30,  # live handshakes have exceeded the 10s default under load
    ) as ws:
        await ws.send(subscribe_cmd(tickers))
        async for raw in ws:
            recv_utc = int(time.time())  # B2a: stamp at socket read, before parse/rebuild
            frame: dict[str, Any] = json.loads(raw)
            if on_raw is not None:
                on_raw(frame)
            upd = process_frame(books, frame, recv_utc)
            if upd is not None:
                yield upd


# --- REST top-of-book fallback (unauth; public quotes) -------------------------------------------


def rest_top_of_book(
    tickers: Iterable[str], *, settings: Settings | None = None
) -> dict[str, BookTick]:
    """1-Hz-friendly batch poll: ``GET /markets?tickers=`` → best YES bid/ask + touch sizes from the
    ``*_dollars`` / ``*_size_fp`` fields. A fallback for when the WS stream is unavailable or
    misbehaving. Public data — no auth required."""
    from pmlab.http import ThrottledClient

    s = settings or get_settings()
    client = ThrottledClient(
        base_url=s.kalshi_base, rps=s.kalshi_rps, cache_dir=s.cache_dir,
        user_agent=s.user_agent, timeout=s.http_timeout_s,
    )
    tk = list(tickers)
    out: dict[str, BookTick] = {}
    now = int(time.time())
    for i in range(0, len(tk), 100):
        group = tk[i : i + 100]
        data = client.get_json("/markets", {"tickers": ",".join(group)}, cache_ttl=0.0)
        for m in data.get("markets", []):
            ticker = str(m.get("ticker", ""))
            out[ticker] = BookTick(
                ts=now,
                yes_bid=_f(m.get("yes_bid_dollars")),
                yes_ask=_f(m.get("yes_ask_dollars")),
                depth_bid=_f(m.get("yes_bid_size_fp")),
                depth_ask=_f(m.get("yes_ask_size_fp")),
            )
    return out
