"""Runtime configuration (pydantic-settings). Override any field via PMLAB_* env or .env.

All venue API keys here are for READ-ONLY market-data access (public REST + authenticated
WS quote streams) — nothing in this package imports, wraps, or calls an order-placement endpoint.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PMLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    data_dir: Path = Field(default=Path("data"))

    # Self-throttle rates (req/s). Public unauth limits are undocumented; be conservative.
    kalshi_rps: float = 4.0
    poly_rps: float = 8.0

    # Kalshi API auth (read-only WS market data; no order endpoints are ever called with it).
    # Key ID is a UUID; the private key is the downloaded PEM, referenced by path (never inline,
    # never committed — secrets/ and *.pem are gitignored).
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: Path | None = None

    # Live capture probe. Read-only market data + local ASR; $0 API.
    kalshi_ws_base: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    whisper_model: str = "base"  # faster-whisper size (base|small); $0 local ASR
    ntfy_topic: str = ""         # ntfy.sh topic — public channel name, not a secret ("" → dry)
    ntfy_base: str = "https://ntfy.sh"
    probe_host: str = ""         # "mac-launchd" | "vps" | "manual" — chosen at first arm

    # Study window. ISO date; collectors clip to this lower bound.
    lookback_start: str = "2024-01-01"

    # Endpoints.
    kalshi_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    poly_gamma_base: str = "https://gamma-api.polymarket.com"
    poly_clob_base: str = "https://clob.polymarket.com"

    # Polymarket sits behind Cloudflare and 403s default UAs — always send a real one.
    user_agent: str = "pmlab/0.1 (research; Polymarket->Kalshi lead-lag study)"
    http_timeout_s: float = 30.0

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
