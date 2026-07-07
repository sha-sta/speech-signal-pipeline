"""Polymarket public read clients (unauthenticated) — DATA ONLY, forever (US person; ToS §0.5).

Verified live 2026-07-02 (see IMPLEMENTATION_PLAN §2.2):
  * gamma ``/markets`` returns a BARE JSON list; offset pagination.
  * ``clobTokenIds`` is a JSON-encoded string array ``'["<yes>","<no>"]'`` whose order matches the
    ``outcomes`` array — index 0 is the YES token (confirmed: YES midpoint == outcomePrices[0]).
  * clob ``/prices-history`` takes camelCase ``startTs``/``endTs`` (snake_case 400s) and returns
    ``{"history": [{"t": unix_s, "p": float}]}``.
  * clob ``/midpoint`` returns ``{"mid": "<dollar string>"}``.
  * Cloudflare 403s default user-agents — the shared client always sends one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd

from pmlab.config import Settings, get_settings
from pmlab.http import CACHE_FOREVER, ThrottledClient

_LIVE_TTL = 900.0  # 15 min for live market listings


def parse_token_ids(market: dict[str, Any]) -> list[str]:
    """Parse a gamma market's ``clobTokenIds`` (JSON-string or list) into a list of token ids."""
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        parsed: list[str] = json.loads(raw)
        return parsed
    return list(raw or [])


def yes_token_id(market: dict[str, Any]) -> str | None:
    """The YES clob token id (``clobTokenIds[0]``, aligned with ``outcomes[0] == "Yes"``)."""
    tokens = parse_token_ids(market)
    return tokens[0] if tokens else None


class GammaClient:
    def __init__(
        self, client: ThrottledClient | None = None, settings: Settings | None = None
    ) -> None:
        s = settings or get_settings()
        self._c = client or ThrottledClient(
            base_url=s.poly_gamma_base,
            rps=s.poly_rps,
            cache_dir=s.cache_dir,
            user_agent=s.user_agent,
            timeout=s.http_timeout_s,
        )

    def iter_markets(
        self, *, closed: bool, updated_after: str | None = None
    ) -> Iterator[dict[str, Any]]:
        offset = 0
        limit = 100  # gamma caps page size at 100 regardless of a larger request
        while True:
            params: dict[str, Any] = {"closed": closed, "limit": limit, "offset": offset}
            if updated_after is not None:
                params["updated_after"] = updated_after
            page = self._c.get_json("/markets", params, cache_ttl=_LIVE_TTL)
            items = page if isinstance(page, list) else page.get("markets", [])
            if not items:
                return
            yield from items
            # Step by the ACTUAL page length so a server-side limit cap can't skip rows.
            offset += len(items)


class ClobPublic:
    def __init__(
        self, client: ThrottledClient | None = None, settings: Settings | None = None
    ) -> None:
        s = settings or get_settings()
        self._c = client or ThrottledClient(
            base_url=s.poly_clob_base,
            rps=s.poly_rps,
            cache_dir=s.cache_dir,
            user_agent=s.user_agent,
            timeout=s.http_timeout_s,
        )

    def prices_history(
        self, token_id: str, start_ts: int, end_ts: int, fidelity_min: int = 1
    ) -> pd.DataFrame:
        data = self._c.get_json(
            "/prices-history",
            {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity_min},
            cache_ttl=CACHE_FOREVER,
        )
        hist = data.get("history", [])
        return pd.DataFrame(
            [{"ts": int(h["t"]), "p": float(h["p"])} for h in hist], columns=["ts", "p"]
        )

    def midpoint(self, token_id: str) -> float:
        """Live YES midpoint in dollars. Not cached (used as a freshness/semantics sanity check)."""
        data = self._c.get_json("/midpoint", {"token_id": token_id}, cache_ttl=None)
        return float(data["mid"])
