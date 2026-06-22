"""Centralised configuration for DMAI, loaded from environment / .env file."""

from __future__ import annotations

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - fallback when pydantic-settings absent
    from pydantic import BaseSettings  # type: ignore


class DMAISettings(BaseSettings):
    """Strongly-typed application settings.

    Values are read from environment variables (case-insensitive) and, when
    present, from a local ``.env`` file. Every field has a safe default so the
    system can boot in development without any configuration.
    """

    # Core
    dmai_env: str = "development"
    master_key: str = "DMAI_MASTER_2026"
    api_secret_key: str = "change-me"

    # DB
    database_url: str = "sqlite:///data/dmai.db"
    redis_url: str = "redis://localhost:6379"

    # AI APIs
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    deepseek_api_key: str = ""
    xai_api_key: str = ""

    # Trading
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Funding
    self_funding_mode: str = "paper"
    spend_limit_daily: float = 50.0

    # Ports
    api_port: int = 8000
    flask_port: int = 5001

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = DMAISettings()
