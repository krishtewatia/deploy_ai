"""AI provider preset registry for DeployAI.

Stage 8E provides a deterministic preset registry that maps well-known
OpenAI-compatible services to configuration metadata.  Presets are
configuration helpers — they do NOT perform network calls or implement
AI provider adapters.

Architecture:

    AIProviderPresetRegistry
          ↓
    AIProviderPreset (metadata)
          ↓
    OpenAICompatibleProviderConfig
          ↓
    AIProviderFactory
          ↓
    OpenAICompatibleAIProvider (existing real HTTP execution)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.ai_providers.schemas import (
    AIProviderType,
    OpenAICompatibleProviderConfig,
)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class AIProviderPresetId(str, Enum):
    """Identifiers for well-known OpenAI-compatible service presets."""

    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    CUSTOM_OPENAI_COMPATIBLE = "custom_openai_compatible"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class AIProviderPresetRegistryError(Exception):
    """Raised when the preset registry encounters a resolution error."""

    pass


# ---------------------------------------------------------------------------
# Helpers
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
    """Validate and clean a base URL value."""
    url = _strip_nonempty("default_base_url", v)
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("default_base_url must start with 'http://' or 'https://'")
    return url.rstrip("/")


# ---------------------------------------------------------------------------
# Preset Model
# ---------------------------------------------------------------------------


class AIProviderPreset(BaseModel):
    """Immutable metadata describing an OpenAI-compatible service preset."""

    model_config = ConfigDict(frozen=True)

    preset_id: AIProviderPresetId = Field(
        ...,
        description="Unique identifier for this preset.",
    )
    display_name: str = Field(
        ...,
        description="User-friendly display name for the preset.",
    )
    provider_type: AIProviderType = Field(
        ...,
        description="Provider type — always OPENAI_COMPATIBLE for Stage 8E presets.",
    )
    default_base_url: Optional[str] = Field(
        default=None,
        description="Default base URL for the service, or None if user must supply one.",
    )
    requires_api_key: bool = Field(
        ...,
        description="Whether this service requires an API key for authentication.",
    )
    supports_custom_model_name: bool = Field(
        default=True,
        description="Whether the service supports user-specified model names.",
    )
    description: str = Field(
        ...,
        description="Short human-readable description of the service.",
    )

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name(cls, v: Any) -> str:
        return _strip_nonempty("display_name", v)

    @field_validator("description", mode="before")
    @classmethod
    def _validate_description(cls, v: Any) -> str:
        return _strip_nonempty("description", v)

    @field_validator("default_base_url", mode="before")
    @classmethod
    def _validate_default_base_url(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return _clean_base_url(v)


# ---------------------------------------------------------------------------
# Preset Definitions
# ---------------------------------------------------------------------------

_PRESETS: tuple[AIProviderPreset, ...] = (
    AIProviderPreset(
        preset_id=AIProviderPresetId.OPENAI,
        display_name="OpenAI",
        provider_type=AIProviderType.OPENAI_COMPATIBLE,
        default_base_url="https://api.openai.com/v1",
        requires_api_key=True,
        description="OpenAI API — GPT model family via the standard chat completions endpoint.",
    ),
    AIProviderPreset(
        preset_id=AIProviderPresetId.GEMINI,
        display_name="Google Gemini",
        provider_type=AIProviderType.OPENAI_COMPATIBLE,
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        requires_api_key=True,
        description="Google Gemini API — accessed through the OpenAI-compatible gateway.",
    ),
    AIProviderPreset(
        preset_id=AIProviderPresetId.DEEPSEEK,
        display_name="DeepSeek",
        provider_type=AIProviderType.OPENAI_COMPATIBLE,
        default_base_url="https://api.deepseek.com/v1",
        requires_api_key=True,
        description="DeepSeek API — DeepSeek model family via OpenAI-compatible endpoint.",
    ),
    AIProviderPreset(
        preset_id=AIProviderPresetId.XAI,
        display_name="xAI (Grok)",
        provider_type=AIProviderType.OPENAI_COMPATIBLE,
        default_base_url="https://api.x.ai/v1",
        requires_api_key=True,
        description="xAI API — Grok model family via OpenAI-compatible endpoint.",
    ),
    AIProviderPreset(
        preset_id=AIProviderPresetId.GROQ,
        display_name="Groq",
        provider_type=AIProviderType.OPENAI_COMPATIBLE,
        default_base_url="https://api.groq.com/openai/v1",
        requires_api_key=True,
        description="Groq API — fast inference via OpenAI-compatible endpoint.",
    ),
    AIProviderPreset(
        preset_id=AIProviderPresetId.OPENROUTER,
        display_name="OpenRouter",
        provider_type=AIProviderType.OPENAI_COMPATIBLE,
        default_base_url="https://openrouter.ai/api/v1",
        requires_api_key=True,
        description="OpenRouter API — multi-provider routing via OpenAI-compatible endpoint.",
    ),
    AIProviderPreset(
        preset_id=AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE,
        display_name="Custom OpenAI-Compatible",
        provider_type=AIProviderType.OPENAI_COMPATIBLE,
        default_base_url=None,
        requires_api_key=False,
        description=(
            "Custom OpenAI-compatible endpoint — for self-hosted servers, "
            "enterprise gateways, or any service using the OpenAI chat completions protocol."
        ),
    ),
)

# Build an index for O(1) lookup
_PRESET_INDEX: dict[AIProviderPresetId, AIProviderPreset] = {
    preset.preset_id: preset for preset in _PRESETS
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AIProviderPresetRegistry:
    """Deterministic, no-network registry of OpenAI-compatible provider presets.

    The registry provides configuration metadata only.  It does not perform
    network calls, validate API keys, or discover models.
    """

    def list_presets(self) -> list[AIProviderPreset]:
        """Return all registered presets in deterministic order.

        Returns:
            An independent list of preset objects (mutations to the returned
            list do not affect internal registry state).
        """
        return list(_PRESETS)

    def get_preset(self, preset_id: AIProviderPresetId) -> AIProviderPreset:
        """Retrieve a single preset by its identifier.

        Args:
            preset_id: The preset identifier to look up.

        Returns:
            The matching AIProviderPreset.

        Raises:
            AIProviderPresetRegistryError: If the input is not a valid
                AIProviderPresetId or the preset is not found.
        """
        if not isinstance(preset_id, AIProviderPresetId):
            raise AIProviderPresetRegistryError(
                f"Invalid preset identifier type: {type(preset_id).__name__}. "
                "Must be an AIProviderPresetId enum value."
            )
        preset = _PRESET_INDEX.get(preset_id)
        if preset is None:
            raise AIProviderPresetRegistryError(
                f"Unknown preset identifier: '{preset_id.value}'."
            )
        return preset

    def create_config(
        self,
        *,
        preset_id: AIProviderPresetId,
        config_id: str,
        model_name: str,
        api_key: Optional[str] = None,
        display_name: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout_seconds: float = 120.0,
        enabled: bool = True,
    ) -> OpenAICompatibleProviderConfig:
        """Create an OpenAICompatibleProviderConfig from a preset and caller overrides.

        Args:
            preset_id: The preset to use as the configuration base.
            config_id: Unique identifier for the resulting configuration.
            model_name: Model name to use (always required).
            api_key: Optional API key for authentication.
            display_name: Override for the display name (defaults to preset name).
            base_url: Override for the base URL (defaults to preset URL).
            request_timeout_seconds: Request timeout in seconds.
            enabled: Whether the configuration is enabled.

        Returns:
            A fully validated OpenAICompatibleProviderConfig.

        Raises:
            AIProviderPresetRegistryError: If the preset requires an API key
                but none is provided, or if no base URL can be resolved.
        """
        preset = self.get_preset(preset_id)

        # Resolve base URL: explicit override > preset default
        resolved_base_url = base_url if base_url is not None else preset.default_base_url
        if resolved_base_url is None:
            raise AIProviderPresetRegistryError(
                f"No base URL available for preset '{preset_id.value}'. "
                "A base_url must be explicitly provided."
            )

        # Resolve display name: explicit override > preset default
        resolved_display_name = display_name if display_name is not None else preset.display_name

        # Enforce API key requirement
        if preset.requires_api_key and api_key is None:
            raise AIProviderPresetRegistryError(
                f"Preset '{preset_id.value}' requires an API key, but none was provided."
            )

        return OpenAICompatibleProviderConfig(
            config_id=config_id,
            display_name=resolved_display_name,
            model_name=model_name,
            base_url=resolved_base_url,
            api_key=api_key,
            request_timeout_seconds=request_timeout_seconds,
            enabled=enabled,
        )
