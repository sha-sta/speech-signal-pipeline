"""RSA-PSS signing recipe + orderbook reconstruction (the two drift-prone, socket-free pieces)."""

from __future__ import annotations

import base64
import json

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from pmlab.probe.book import KalshiSigner, OrderBook, process_frame, subscribe_cmd


def _signer(ts_ms: int = 1_700_000_000_000) -> tuple[KalshiSigner, rsa.RSAPrivateKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return KalshiSigner("key-uuid", key, now_ms=lambda: ts_ms), key


def test_signed_message_strips_query_and_orders_fields():
    msg = KalshiSigner.signed_message(1_700_000_000_000, "get", "/trade-api/v2/markets?tickers=A,B")
    assert msg == "1700000000000GET/trade-api/v2/markets"  # ms + UPPER method + path, no query


def test_headers_signature_verifies_against_public_key():
    signer, key = _signer()
    path = "/trade-api/v2/markets?tickers=A,B"
    h = signer.headers("GET", path)
    assert h["KALSHI-ACCESS-KEY"] == "key-uuid"
    assert h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"  # milliseconds
    msg = KalshiSigner.signed_message(1_700_000_000_000, "GET", path)
    # verify() raises on a bad signature — so a clean return proves the exact PSS/SHA-256 recipe
    key.public_key().verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_ws_handshake_signs_get_ws_path():
    signer, key = _signer()
    h = signer.headers("GET", "/trade-api/ws/v2")
    msg = KalshiSigner.signed_message(1_700_000_000_000, "GET", "/trade-api/ws/v2")
    assert msg == "1700000000000GET/trade-api/ws/v2"
    key.public_key().verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_orderbook_snapshot_best_bid_ask_and_depth():
    b = OrderBook("T-X")
    b.apply_snapshot({
        "market_ticker": "T-X",
        "yes_dollars_fp": [["0.0800", "300.00"], ["0.2200", "333.00"]],
        "no_dollars_fp": [["0.5400", "20.00"], ["0.5600", "146.00"]],
    })
    assert b.best_yes_bid() == (0.22, 333.0)          # highest YES bid
    assert b.best_yes_ask() == (0.44, 146.0)          # 1 − best NO bid (0.56); size = NO-bid size
    tick = b.tick(recv_utc=1_800_000_000)
    assert tick.yes_bid == 0.22 and tick.yes_ask == 0.44
    assert tick.depth_bid == 333.0 and tick.depth_ask == 146.0
    # touch + one level deeper (§2.3): second bid level and second ask level (1 − 0.54)
    assert tick.yes_bid2 == 0.08 and tick.depth_bid2 == 300.0
    assert tick.yes_ask2 == 0.46 and tick.depth_ask2 == 20.0


def test_orderbook_snapshot_drops_zero_count_levels():
    b = OrderBook("T-X")
    b.apply_snapshot({
        "yes_dollars_fp": [["0.0800", "300.00"], ["0.2200", "0.00"]],   # phantom 0-depth top
        "no_dollars_fp": [["0.5600", "0.00"]],
    })
    assert b.best_yes_bid() == (0.08, 300.0)          # zero-count 0.22 never enters the book
    ask, depth = b.best_yes_ask()
    assert ask != ask and depth != depth              # NO side all-phantom → empty (NaN)


def test_orderbook_second_level_nan_when_book_thin():
    b = OrderBook("T-X")
    b.apply_snapshot({"yes_dollars_fp": [["0.2200", "333.00"]],
                      "no_dollars_fp": [["0.5600", "146.00"]]})
    tick = b.tick(recv_utc=1_800_000_000)
    assert tick.yes_bid2 != tick.yes_bid2 and tick.depth_bid2 != tick.depth_bid2   # NaN
    assert tick.yes_ask2 != tick.yes_ask2 and tick.depth_ask2 != tick.depth_ask2   # NaN


def test_orderbook_delta_add_and_remove_level():
    b = OrderBook("T-X")
    b.apply_snapshot({"yes_dollars_fp": [["0.0800", "300.00"], ["0.2200", "333.00"]],
                      "no_dollars_fp": [["0.5600", "146.00"]]})
    b.apply_delta({"side": "yes", "price_dollars": "0.2200", "delta_fp": "-333.00"})
    assert b.best_yes_bid() == (0.08, 300.0)          # 0.22 level emptied → drops out
    b.apply_delta({"side": "yes", "price_dollars": "0.3000", "delta_fp": "100.00"})
    assert b.best_yes_bid() == (0.30, 100.0)          # new higher level


def test_orderbook_legacy_integer_cents_fallback():
    b = OrderBook("T-X")
    b.apply_snapshot({"yes": [[8, 300], [22, 333]], "no": [[56, 146]]})  # legacy cents
    assert b.best_yes_bid() == (0.22, 333.0)
    assert b.best_yes_ask() == (0.44, 146.0)


def test_subscribe_cmd_shape():
    cmd = json.loads(subscribe_cmd(["A", "B"]))
    assert cmd["cmd"] == "subscribe"
    assert cmd["params"]["channels"] == ["orderbook_delta"]
    assert cmd["params"]["market_tickers"] == ["A", "B"]


# --- process_frame (B2a): frame routing + recv stamped at socket read, not at processing ----------


def _snapshot_frame(ticker: str = "T", seq: int = 7) -> dict:
    return {"type": "orderbook_snapshot", "seq": seq,
            "msg": {"market_ticker": ticker,
                    "yes_dollars_fp": [["0.2200", "333.00"]],
                    "no_dollars_fp": [["0.7000", "146.00"]]}}


def test_process_frame_snapshot_carries_caller_recv_stamp():
    books = {"T": OrderBook("T")}
    upd = process_frame(books, _snapshot_frame(), recv_utc=1234)
    assert upd is not None
    assert upd.tick.ts == 1234  # the caller's socket-read stamp, verbatim
    assert upd.seq == 7
    assert upd.tick.yes_bid == 0.22 and upd.tick.yes_ask == 0.30


def test_process_frame_delta_updates_book():
    books = {"T": OrderBook("T")}
    process_frame(books, _snapshot_frame(), recv_utc=1)
    upd = process_frame(books, {"type": "orderbook_delta", "seq": 8,
                                "msg": {"market_ticker": "T", "side": "yes",
                                        "price_dollars": "0.2500", "delta_fp": "10.00"}},
                        recv_utc=2)
    assert upd is not None
    assert upd.tick.ts == 2
    assert upd.tick.yes_bid == 0.25


def test_process_frame_ignores_non_book_and_unknown_ticker():
    books = {"T": OrderBook("T")}
    assert process_frame(books, {"type": "subscribed", "msg": {}}, recv_utc=1) is None
    assert process_frame(books, _snapshot_frame(ticker="OTHER"), recv_utc=1) is None
