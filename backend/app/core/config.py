"""Application configuration loaded from environment variables.

Uses **pydantic-settings** to validate and expose every config value as a
typed attribute.  The ``.env`` file in the project root is loaded
automatically via *python-dotenv* so secrets never need to be hard-coded.

Usage::

    from backend.app.core.config import settings

    settings.GROQ_API_KEY
    settings.cors_origins
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Load .env before the Settings class is instantiated ─────────────────────
# Locate the .env file at the project root (three levels up from this file).
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


class Settings(BaseSettings):
    """Centralised, validated application settings.

    All values are read from environment variables (or the ``.env`` file).
    Required variables that are missing will cause an immediate
    ``ValidationError`` at import time so mis-configurations surface early.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── General ─────────────────────────────────────────────────────────
    app_name: str = Field(
        default="AI Project Builder",
        description="Display name of the application.",
    )
    environment: str = Field(
        default="development",
        description="Runtime environment (development | staging | production).",
    )
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        description="Allowed CORS origins (comma-separated string or JSON list).",
    )

    # ── AI / LLM ────────────────────────────────────────────────────────
    GROQ_API_KEY: str | None = Field(
        default=None,
        description=(
            "API key for the Groq LLM service. "
            "Optional."
        ),
    )

    # ── Validators ──────────────────────────────────────────────────────

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Accept a comma-separated string *or* a JSON list of origins."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        return v


settings: Settings = Settings()
