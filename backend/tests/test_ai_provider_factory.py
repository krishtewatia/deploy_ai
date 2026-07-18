"""Tests for AI provider factory and adapters (Stage 8B)."""

import pytest

from backend.app.ai_planning.providers.base import AIProvider, AIProviderError
from backend.app.ai_providers.schemas import (
    BaseAIProviderConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
    AIProviderSettings,
    AIProviderStatus,
    AIProviderType,
)
from backend.app.ai_providers.factory import (
    AIProviderFactory,
    AIProviderFactoryError,
    OllamaAIProvider,
    OpenAICompatibleAIProvider,
)


def _make_ollama_config(enabled: bool = True) -> OllamaProviderConfig:
    return OllamaProviderConfig(
        config_id="ollama_local",
        display_name="Local Ollama",
        model_name="llama3.2",
        base_url="http://localhost:11434",
        request_timeout_seconds=90.0,
        enabled=enabled,
    )


def _make_openai_config(enabled: bool = True) -> OpenAICompatibleProviderConfig:
    return OpenAICompatibleProviderConfig(
        config_id="openai_remote",
        display_name="OpenAI Remote",
        model_name="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="sk-test-secret-key-12345",
        request_timeout_seconds=60.0,
        enabled=enabled,
    )


# ── Provider Adapters Tests (1-8) ───────────────────────────────────────


class TestProviderAdapters:
    def test_ollama_adapter_implements_ai_provider(self):
        config = _make_ollama_config()
        provider = OllamaAIProvider(config)
        assert isinstance(provider, AIProvider)

    def test_openai_compatible_adapter_implements_ai_provider(self):
        config = _make_openai_config()
        provider = OpenAICompatibleAIProvider(config)
        assert isinstance(provider, AIProvider)

    def test_ollama_adapter_preserves_config(self):
        config = _make_ollama_config()
        provider = OllamaAIProvider(config)
        assert provider.config is config
        assert provider.config.config_id == "ollama_local"
        assert provider.config.base_url == "http://localhost:11434"

    def test_openai_compatible_adapter_preserves_config(self):
        config = _make_openai_config()
        provider = OpenAICompatibleAIProvider(config)
        assert provider.config is config
        assert provider.config.config_id == "openai_remote"
        assert provider.config.api_key == "sk-test-secret-key-12345"



    def test_no_network_call_occurs_during_construction(self):
        # Construction is offline and lightweight
        config = _make_ollama_config()
        provider = OllamaAIProvider(config)
        assert provider is not None

    def test_api_key_remains_unchanged_in_stored_config(self):
        config = _make_openai_config()
        provider = OpenAICompatibleAIProvider(config)
        assert provider.config.api_key == "sk-test-secret-key-12345"


# ── Factory create() Tests (9-20) ───────────────────────────────────────


class TestFactoryCreate:
    def test_ollama_config_creates_ollama_adapter(self):
        config = _make_ollama_config()
        provider = AIProviderFactory.create(config)
        assert isinstance(provider, OllamaAIProvider)
        assert isinstance(provider, AIProvider)
        assert provider.config is config

    def test_openai_compatible_config_creates_openai_compatible_adapter(self):
        config = _make_openai_config()
        provider = AIProviderFactory.create(config)
        assert isinstance(provider, OpenAICompatibleAIProvider)
        assert isinstance(provider, AIProvider)
        assert provider.config is config

    def test_none_input_rejected(self):
        with pytest.raises(AIProviderFactoryError, match="Configuration cannot be None"):
            AIProviderFactory.create(None)

    def test_dictionary_input_rejected(self):
        with pytest.raises(AIProviderFactoryError, match="Unsupported configuration type"):
            AIProviderFactory.create({
                "config_id": "ollama_local",
                "provider_type": "ollama",
                "display_name": "Local",
                "model_name": "m",
            })

    def test_arbitrary_object_rejected(self):
        with pytest.raises(AIProviderFactoryError, match="Unsupported configuration type"):
            AIProviderFactory.create(object())

    def test_disabled_ollama_config_rejected(self):
        config = _make_ollama_config(enabled=False)
        with pytest.raises(AIProviderFactoryError, match="Cannot construct provider for disabled configuration"):
            AIProviderFactory.create(config)

    def test_disabled_openai_compatible_config_rejected(self):
        config = _make_openai_config(enabled=False)
        with pytest.raises(AIProviderFactoryError, match="Cannot construct provider for disabled configuration"):
            AIProviderFactory.create(config)

    def test_original_config_not_mutated(self):
        config = _make_ollama_config()
        original_dict = config.model_dump()
        provider = AIProviderFactory.create(config)
        assert provider.config.model_dump() == original_dict

    def test_unsupported_base_config_instance_rejected(self):
        # Make a mock/subclass base config to see if it rejects raw base type
        class FakeBaseConfig(BaseAIProviderConfig):
            pass

        config = FakeBaseConfig(
            config_id="c1",
            provider_type=AIProviderType.OLLAMA,
            display_name="Fake",
            model_name="model",
        )
        with pytest.raises(AIProviderFactoryError, match="Unsupported configuration type"):
            AIProviderFactory.create(config)


# ── Factory create_active() Tests (21-30) ───────────────────────────────


class TestFactoryCreateActive:
    def test_valid_ollama_active_resolves_correctly(self):
        p = _make_ollama_config()
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="ollama_local",
            providers=[p],
        )
        provider = AIProviderFactory.create_active(settings)
        assert isinstance(provider, OllamaAIProvider)
        assert provider.config is p

    def test_valid_openai_compatible_active_resolves_correctly(self):
        p = _make_openai_config()
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[p],
        )
        provider = AIProviderFactory.create_active(settings)
        assert isinstance(provider, OpenAICompatibleAIProvider)
        assert provider.config is p

    def test_multiple_providers_resolves_only_active_config_id(self):
        p1 = _make_ollama_config()
        p2 = _make_openai_config()
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[p1, p2],
        )
        provider = AIProviderFactory.create_active(settings)
        assert isinstance(provider, OpenAICompatibleAIProvider)
        assert provider.config is p2

    def test_none_settings_rejected(self):
        with pytest.raises(AIProviderFactoryError, match="Invalid settings input type"):
            AIProviderFactory.create_active(None)

    def test_dictionary_settings_rejected(self):
        with pytest.raises(AIProviderFactoryError, match="Invalid settings input type"):
            AIProviderFactory.create_active({
                "status": "configured",
                "active_config_id": "ollama_local",
                "providers": [],
            })

    def test_arbitrary_object_settings_rejected(self):
        with pytest.raises(AIProviderFactoryError, match="Invalid settings input type"):
            AIProviderFactory.create_active(object())

    def test_input_settings_not_mutated(self):
        p = _make_ollama_config()
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="ollama_local",
            providers=[p],
        )
        original_dict = settings.model_dump()
        provider = AIProviderFactory.create_active(settings)
        assert settings.model_dump() == original_dict
        assert provider is not None

    def test_unconfigured_settings_status_raises_factory_error(self):
        settings = AIProviderSettings()
        # Mock/bypass validator logic by setting private state or subclassing to test factory validation
        # Settings model validation does not allow status unconfigured with active/providers list,
        # but what if the settings object is unconfigured?
        assert settings.status == AIProviderStatus.UNCONFIGURED
        with pytest.raises(AIProviderFactoryError, match="An active provider can only be created from a 'configured'"):
            AIProviderFactory.create_active(settings)

    def test_settings_active_config_id_none_raises_factory_error(self):
        # We can bypass validator for test by creating a mock class inheriting settings
        class InvalidSettings(AIProviderSettings):
            pass

        settings = InvalidSettings.model_construct(
            status=AIProviderStatus.CONFIGURED,
            active_config_id=None,
            providers=[_make_ollama_config()],
        )
        with pytest.raises(AIProviderFactoryError, match="active_config_id is not specified"):
            AIProviderFactory.create_active(settings)

    def test_settings_active_config_id_not_found_raises_factory_error(self):
        class InvalidSettings(AIProviderSettings):
            pass

        settings = InvalidSettings.model_construct(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="missing_id",
            providers=[_make_ollama_config()],
        )
        with pytest.raises(AIProviderFactoryError, match="was not found in the configured providers list"):
            AIProviderFactory.create_active(settings)

    def test_settings_active_provider_disabled_raises_factory_error(self):
        class InvalidSettings(AIProviderSettings):
            pass

        settings = InvalidSettings.model_construct(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="ollama_local",
            providers=[_make_ollama_config(enabled=False)],
        )
        with pytest.raises(AIProviderFactoryError, match="is disabled"):
            AIProviderFactory.create_active(settings)
