"""Pydantic v2 schemas for AI provider configurations.

Defines configuration settings for local (Ollama) and remote (OpenAI-compatible)
AI providers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AIProviderType(str, Enum):
    """Supported types of AI completion providers."""

    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class AIProviderStatus(str, Enum):
    """Configuration state of AI providers in DeployAI."""

    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"


# ---------------------------------------------------------------------------
# Helper Validators
# ---------------------------------------------------------------------------


def _strip_nonempty(field_name: str, v: Any) -> str:
    """Validate that a value is a non-empty string and strip whitespace."""
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = v.strip()
    if not stripped:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return stripped


def _clean_base_url(v: Any) -> str:
    """Validate and clean the provider base URL."""
    url = _strip_nonempty("base_url", v)
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("base_url must start with 'http://' or 'https://'")
    # Remove trailing slashes consistently
    return url.rstrip("/")


# ---------------------------------------------------------------------------
# Configuration Schemas
# ---------------------------------------------------------------------------


class BaseAIProviderConfig(BaseModel):
    """Base Pydantic model for all provider configurations."""

    config_id: str = Field(
        ...,
        description="Unique identifier for this provider configuration.",
    )
    provider_type: AIProviderType = Field(
        ...,
        description="Type of the AI provider.",
    )
    display_name: str = Field(
        ...,
        description="User-friendly name to display in the UI.",
    )
    model_name: str = Field(
        ...,
        description="Target model name to use on the provider (e.g. llama3, mixtral).",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this provider configuration is enabled.",
    )

    @field_validator("config_id", mode="before")
    @classmethod
    def _validate_config_id(cls, v: Any) -> str:
        return _strip_nonempty("config_id", v)

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name(cls, v: Any) -> str:
        return _strip_nonempty("display_name", v)

    @field_validator("model_name", mode="before")
    @classmethod
    def _validate_model_name(cls, v: Any) -> str:
        return _strip_nonempty("model_name", v)


class OllamaProviderConfig(BaseAIProviderConfig):
    """Configuration schema for a local Ollama service."""

    provider_type: Literal[AIProviderType.OLLAMA] = Field(
        default=AIProviderType.OLLAMA,
        description="Provider type fixed to 'ollama'.",
    )
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the running local Ollama service instance.",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        description="Timeout for request completions in seconds.",
    )

    @field_validator("base_url", mode="before")
    @classmethod
    def _validate_base_url(cls, v: Any) -> str:
        return _clean_base_url(v)

    @field_validator("request_timeout_seconds")
    @classmethod
    def _validate_timeout(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("request_timeout_seconds must be greater than 0")
        return v

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a safe dictionary representation of the config."""
        return self.model_dump(mode="json")


class OpenAICompatibleProviderConfig(BaseAIProviderConfig):
    """Configuration schema for a remote OpenAI-compatible endpoint service."""

    provider_type: Literal[AIProviderType.OPENAI_COMPATIBLE] = Field(
        default=AIProviderType.OPENAI_COMPATIBLE,
        description="Provider type fixed to 'openai_compatible'.",
    )
    base_url: str = Field(
        ...,
        description="Base URL of the OpenAI-compatible service endpoint.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API authorization key or credential token.",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        description="Timeout for request completions in seconds.",
    )

    @field_validator("base_url", mode="before")
    @classmethod
    def _validate_base_url(cls, v: Any) -> str:
        return _clean_base_url(v)

    @field_validator("api_key", mode="before")
    @classmethod
    def _validate_api_key(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return _strip_nonempty("api_key", v)

    @field_validator("request_timeout_seconds")
    @classmethod
    def _validate_timeout(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("request_timeout_seconds must be greater than 0")
        return v

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a safe dictionary representation masking the API key if present."""
        data = self.model_dump(mode="json")
        if self.api_key is not None:
            data["api_key"] = "***"
        else:
            data["api_key"] = None
        return data


# Discriminated Union / Type Alias
AIProviderConfig = Annotated[
    Union[OllamaProviderConfig, OpenAICompatibleProviderConfig],
    Field(discriminator="provider_type"),
]


# ---------------------------------------------------------------------------
# Global AI Provider Settings
# ---------------------------------------------------------------------------


class AIProviderSettings(BaseModel):
    """Global AI provider settings for DeployAI."""

    status: AIProviderStatus = Field(
        default=AIProviderStatus.UNCONFIGURED,
        description="Configuration status representing configured state.",
    )
    active_config_id: Optional[str] = Field(
        default=None,
        description="Identifier of the active provider config.",
    )
    providers: list[AIProviderConfig] = Field(
        default_factory=list,
        description="List of user-configured AI provider profiles.",
    )

    @model_validator(mode="after")
    def _validate_settings_consistency(self) -> AIProviderSettings:
        # 1. Unique config_ids check
        seen_ids = set()
        for idx, provider in enumerate(self.providers):
            cfg_id = provider.config_id
            if cfg_id in seen_ids:
                raise ValueError(
                    f"Duplicate provider config_id detected: '{cfg_id}' at providers[{idx}]"
                )
            seen_ids.add(cfg_id)

        # 2. Empty providers check
        if not self.providers:
            if self.status != AIProviderStatus.UNCONFIGURED:
                raise ValueError(
                    "status must be 'unconfigured' when providers list is empty"
                )
            if self.active_config_id is not None:
                raise ValueError(
                    "active_config_id must be None when providers list is empty"
                )
            return self

        # 3. Non-empty providers checks
        if self.status != AIProviderStatus.CONFIGURED:
            raise ValueError(
                "status must be 'configured' when providers list is not empty"
            )
        if self.active_config_id is None:
            raise ValueError(
                "active_config_id must be specified when providers list is not empty"
            )

        # 4. Find active config ID match
        active_provider = None
        for provider in self.providers:
            if provider.config_id == self.active_config_id:
                active_provider = provider
                break

        if active_provider is None:
            raise ValueError(
                f"active_config_id '{self.active_config_id}' does not match "
                "any configured provider config_id"
            )

        # 5. Enabled check for active provider
        if not active_provider.enabled:
            raise ValueError(
                f"Active provider '{self.active_config_id}' must be enabled"
            )

        return self

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a safe dictionary representation masking nested keys."""
        safe_providers = []
        for provider in self.providers:
            safe_providers.append(provider.to_safe_dict())

        return {
            "status": self.status.value,
            "active_config_id": self.active_config_id,
            "providers": safe_providers,
        }
