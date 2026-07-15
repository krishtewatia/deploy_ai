"""Tests for AI provider configuration schemas (Stage 8A)."""

import json
import pytest

from pydantic import ValidationError, TypeAdapter

from backend.app.ai_providers.schemas import (
    AIProviderType,
    AIProviderStatus,
    BaseAIProviderConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
    AIProviderConfig,
    AIProviderSettings,
)


# ── AIProviderType Tests (1-2) ──────────────────────────────────────────


class TestAIProviderType:
    def test_enum_values(self):
        assert AIProviderType.OLLAMA.value == "ollama"
        assert AIProviderType.OPENAI_COMPATIBLE.value == "openai_compatible"

    def test_serialization_as_string(self):
        assert json.dumps(AIProviderType.OLLAMA) == '"ollama"'
        assert json.dumps(AIProviderType.OPENAI_COMPATIBLE) == '"openai_compatible"'


# ── AIProviderStatus Tests (3-4) ────────────────────────────────────────


class TestAIProviderStatus:
    def test_enum_values(self):
        assert AIProviderStatus.UNCONFIGURED.value == "unconfigured"
        assert AIProviderStatus.CONFIGURED.value == "configured"

    def test_serialization_as_string(self):
        assert json.dumps(AIProviderStatus.UNCONFIGURED) == '"unconfigured"'
        assert json.dumps(AIProviderStatus.CONFIGURED) == '"configured"'


# ── BaseAIProviderConfig Tests (5-9) ────────────────────────────────────


class TestBaseAIProviderConfig:
    def test_valid_config(self):
        config = BaseAIProviderConfig(
            config_id="c1",
            provider_type=AIProviderType.OLLAMA,
            display_name="Local Llama",
            model_name="llama3",
            enabled=True,
        )
        assert config.config_id == "c1"
        assert config.provider_type == AIProviderType.OLLAMA
        assert config.display_name == "Local Llama"
        assert config.model_name == "llama3"
        assert config.enabled is True

    def test_string_normalization_stripping(self):
        config = BaseAIProviderConfig(
            config_id="   c1   ",
            provider_type=AIProviderType.OLLAMA,
            display_name="   Local Llama   ",
            model_name="   llama3   ",
        )
        assert config.config_id == "c1"
        assert config.display_name == "Local Llama"
        assert config.model_name == "llama3"

    def test_empty_config_id_rejected(self):
        with pytest.raises(ValidationError, match="config_id"):
            BaseAIProviderConfig(
                config_id="  ",
                provider_type=AIProviderType.OLLAMA,
                display_name="Local",
                model_name="llama3",
            )

    def test_empty_display_name_rejected(self):
        with pytest.raises(ValidationError, match="display_name"):
            BaseAIProviderConfig(
                config_id="c1",
                provider_type=AIProviderType.OLLAMA,
                display_name="",
                model_name="llama3",
            )

    def test_empty_model_name_rejected(self):
        with pytest.raises(ValidationError, match="model_name"):
            BaseAIProviderConfig(
                config_id="c1",
                provider_type=AIProviderType.OLLAMA,
                display_name="Local",
                model_name="   ",
            )

    def test_non_string_types_rejected(self):
        with pytest.raises(ValidationError, match="config_id"):
            BaseAIProviderConfig(
                config_id=123,  # type: ignore
                provider_type=AIProviderType.OLLAMA,
                display_name="Local",
                model_name="llama3",
            )


# ── OllamaProviderConfig Tests (10-22) ──────────────────────────────────


class TestOllamaProviderConfig:
    def test_minimal_valid_config(self):
        config = OllamaProviderConfig(
            config_id="ollama_local",
            display_name="Ollama Local",
            model_name="llama3",
        )
        assert config.provider_type == AIProviderType.OLLAMA
        assert config.base_url == "http://localhost:11434"
        assert config.request_timeout_seconds == 120.0
        assert config.enabled is True

    def test_custom_values(self):
        config = OllamaProviderConfig(
            config_id="ollama_custom",
            display_name="Ollama Remote",
            model_name="llama3-70b",
            base_url="https://ollama.myhost.com",
            request_timeout_seconds=60.5,
            enabled=False,
        )
        assert config.base_url == "https://ollama.myhost.com"
        assert config.request_timeout_seconds == 60.5
        assert config.enabled is False

    def test_provider_type_fixed_to_ollama(self):
        with pytest.raises(ValidationError, match="provider_type"):
            # Should not allow overriding provider_type to something else
            OllamaProviderConfig(
                config_id="c1",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,  # type: ignore
                display_name="Name",
                model_name="llama3",
            )

    def test_http_url_accepted(self):
        config = OllamaProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="http://my-ollama-host:11434",
        )
        assert config.base_url == "http://my-ollama-host:11434"

    def test_https_url_accepted(self):
        config = OllamaProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="https://secure-ollama.io",
        )
        assert config.base_url == "https://secure-ollama.io"

    def test_trailing_slash_removed(self):
        config = OllamaProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="http://localhost:11434/",
        )
        assert config.base_url == "http://localhost:11434"

    def test_multiple_trailing_slashes_removed(self):
        config = OllamaProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="http://localhost:11434///",
        )
        assert config.base_url == "http://localhost:11434"

    def test_empty_base_url_rejected(self):
        with pytest.raises(ValidationError, match="base_url"):
            OllamaProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                base_url="   ",
            )

    def test_invalid_url_scheme_rejected(self):
        with pytest.raises(ValidationError, match="base_url"):
            OllamaProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                base_url="ftp://localhost:11434",
            )

    def test_zero_timeout_rejected(self):
        with pytest.raises(ValidationError, match="request_timeout_seconds"):
            OllamaProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                request_timeout_seconds=0.0,
            )

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValidationError, match="request_timeout_seconds"):
            OllamaProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                request_timeout_seconds=-10.0,
            )

    def test_to_safe_dict_works(self):
        config = OllamaProviderConfig(
            config_id="ollama_local",
            display_name="Ollama Local",
            model_name="llama3",
            base_url="http://localhost:11434",
            request_timeout_seconds=45.0,
        )
        safe = config.to_safe_dict()
        assert isinstance(safe, dict)
        assert safe["config_id"] == "ollama_local"
        assert safe["base_url"] == "http://localhost:11434"
        assert safe["request_timeout_seconds"] == 45.0


# ── OpenAICompatibleProviderConfig Tests (23-38) ─────────────────────────


class TestOpenAICompatibleProviderConfig:
    def test_valid_config_with_api_key(self):
        config = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI Remote",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-12345",
            request_timeout_seconds=60.0,
        )
        assert config.provider_type == AIProviderType.OPENAI_COMPATIBLE
        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key == "sk-12345"
        assert config.request_timeout_seconds == 60.0
        assert config.enabled is True

    def test_valid_config_without_api_key(self):
        config = OpenAICompatibleProviderConfig(
            config_id="local_lm_studio",
            display_name="LM Studio",
            model_name="meta-llama",
            base_url="http://localhost:1234/v1",
            api_key=None,
        )
        assert config.api_key is None
        assert config.request_timeout_seconds == 120.0

    def test_provider_type_fixed_to_openai_compatible(self):
        with pytest.raises(ValidationError, match="provider_type"):
            OpenAICompatibleProviderConfig(
                config_id="c1",
                provider_type=AIProviderType.OLLAMA,  # type: ignore
                display_name="N",
                model_name="m",
                base_url="https://api.openai.com/v1",
            )

    def test_http_url_accepted(self):
        config = OpenAICompatibleProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="http://localhost:1234/v1",
        )
        assert config.base_url == "http://localhost:1234/v1"

    def test_https_url_accepted(self):
        config = OpenAICompatibleProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="https://api.openai.com/v1",
        )
        assert config.base_url == "https://api.openai.com/v1"

    def test_trailing_slash_removed(self):
        config = OpenAICompatibleProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="https://api.openai.com/v1/",
        )
        assert config.base_url == "https://api.openai.com/v1"

    def test_empty_base_url_rejected(self):
        with pytest.raises(ValidationError, match="base_url"):
            OpenAICompatibleProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                base_url="   ",
            )

    def test_invalid_url_scheme_rejected(self):
        with pytest.raises(ValidationError, match="base_url"):
            OpenAICompatibleProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                base_url="ws://localhost:1234/v1",
            )

    def test_api_key_whitespace_stripped(self):
        config = OpenAICompatibleProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="https://api.openai.com/v1",
            api_key="   sk-key-value   ",
        )
        assert config.api_key == "sk-key-value"

    def test_empty_api_key_rejected(self):
        with pytest.raises(ValidationError, match="api_key"):
            OpenAICompatibleProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                base_url="https://api.openai.com/v1",
                api_key="",
            )

    def test_whitespace_only_api_key_rejected(self):
        with pytest.raises(ValidationError, match="api_key"):
            OpenAICompatibleProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                base_url="https://api.openai.com/v1",
                api_key="   ",
            )

    def test_zero_timeout_rejected(self):
        with pytest.raises(ValidationError, match="request_timeout_seconds"):
            OpenAICompatibleProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                base_url="https://api.openai.com/v1",
                request_timeout_seconds=0.0,
            )

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValidationError, match="request_timeout_seconds"):
            OpenAICompatibleProviderConfig(
                config_id="c1",
                display_name="N",
                model_name="m",
                base_url="https://api.openai.com/v1",
                request_timeout_seconds=-0.5,
            )

    def test_to_safe_dict_masks_real_api_key(self):
        config = OpenAICompatibleProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="https://api.openai.com/v1",
            api_key="sk-my-secret-key",
        )
        safe = config.to_safe_dict()
        assert safe["api_key"] == "***"
        assert safe["base_url"] == "https://api.openai.com/v1"

    def test_to_safe_dict_preserves_none_api_key(self):
        config = OpenAICompatibleProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="https://api.openai.com/v1",
            api_key=None,
        )
        safe = config.to_safe_dict()
        assert safe["api_key"] is None

    def test_normal_model_dump_still_contains_raw_api_key(self):
        config = OpenAICompatibleProviderConfig(
            config_id="c1",
            display_name="N",
            model_name="m",
            base_url="https://api.openai.com/v1",
            api_key="sk-secret",
        )
        dump = config.model_dump()
        assert dump["api_key"] == "sk-secret"

        dump_json = config.model_dump_json()
        assert '"api_key":"sk-secret"' in dump_json.replace(" ", "")


# ── AIProviderConfig Union Tests (39-42) ────────────────────────────────


class TestAIProviderConfigUnion:
    def test_ollama_config_validates_through_union(self):
        adapter = TypeAdapter(AIProviderConfig)
        raw = {
            "config_id": "ollama_id",
            "provider_type": "ollama",
            "display_name": "Ollama name",
            "model_name": "llama3",
            "base_url": "http://localhost:11434/",
        }
        res = adapter.validate_python(raw)
        assert isinstance(res, OllamaProviderConfig)
        assert res.base_url == "http://localhost:11434"

    def test_openai_compatible_config_validates_through_union(self):
        adapter = TypeAdapter(AIProviderConfig)
        raw = {
            "config_id": "openai_id",
            "provider_type": "openai_compatible",
            "display_name": "OpenAI name",
            "model_name": "gpt-4",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-1",
        }
        res = adapter.validate_python(raw)
        assert isinstance(res, OpenAICompatibleProviderConfig)
        assert res.api_key == "sk-1"

    def test_invalid_provider_type_rejected(self):
        adapter = TypeAdapter(AIProviderConfig)
        raw = {
            "config_id": "c1",
            "provider_type": "unknown_provider",
            "display_name": "Display",
            "model_name": "model",
        }
        with pytest.raises(ValidationError):
            adapter.validate_python(raw)


# ── AIProviderSettings Tests (43-61) ────────────────────────────────────


class TestAIProviderSettings:
    def test_valid_unconfigured_settings(self):
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        assert settings.status == AIProviderStatus.UNCONFIGURED
        assert settings.active_config_id is None
        assert len(settings.providers) == 0

    def test_valid_settings_with_one_ollama_provider(self):
        provider = OllamaProviderConfig(
            config_id="p1",
            display_name="Local",
            model_name="llama3",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="p1",
            providers=[provider],
        )
        assert settings.status == AIProviderStatus.CONFIGURED
        assert settings.active_config_id == "p1"
        assert len(settings.providers) == 1

    def test_valid_settings_with_one_openai_compatible_provider(self):
        provider = OpenAICompatibleProviderConfig(
            config_id="p1",
            display_name="Remote",
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="p1",
            providers=[provider],
        )
        assert settings.active_config_id == "p1"
        assert len(settings.providers) == 1

    def test_valid_settings_with_multiple_providers(self):
        p1 = OllamaProviderConfig(config_id="ollama_local", display_name="Local", model_name="llama3")
        p2 = OpenAICompatibleProviderConfig(config_id="openai_remote", display_name="Remote",
                                             model_name="gpt-4", base_url="https://api.openai.com/v1",
                                             enabled=True)
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[p1, p2],
        )
        assert len(settings.providers) == 2
        assert settings.active_config_id == "openai_remote"

    def test_default_providers_list_is_empty(self):
        settings = AIProviderSettings()
        assert settings.providers == []
        assert settings.status == AIProviderStatus.UNCONFIGURED
        assert settings.active_config_id is None

    def test_two_settings_instances_do_not_share_providers_list(self):
        s1 = AIProviderSettings()
        s2 = AIProviderSettings()
        assert s1.providers is not s2.providers

    def test_duplicate_config_ids_rejected(self):
        p1 = OllamaProviderConfig(config_id="dup", display_name="P1", model_name="m")
        p2 = OllamaProviderConfig(config_id="dup", display_name="P2", model_name="m")
        with pytest.raises(ValidationError, match="Duplicate provider config_id"):
            AIProviderSettings(
                status=AIProviderStatus.CONFIGURED,
                active_config_id="dup",
                providers=[p1, p2],
            )

    def test_empty_providers_with_configured_status_rejected(self):
        with pytest.raises(ValidationError, match="status must be 'unconfigured'"):
            AIProviderSettings(
                status=AIProviderStatus.CONFIGURED,
                active_config_id=None,
                providers=[],
            )

    def test_empty_providers_with_active_config_id_rejected(self):
        with pytest.raises(ValidationError, match="active_config_id must be None"):
            AIProviderSettings(
                status=AIProviderStatus.UNCONFIGURED,
                active_config_id="some_id",
                providers=[],
            )

    def test_non_empty_providers_with_unconfigured_status_rejected(self):
        p = OllamaProviderConfig(config_id="p1", display_name="P", model_name="m")
        with pytest.raises(ValidationError, match="status must be 'configured'"):
            AIProviderSettings(
                status=AIProviderStatus.UNCONFIGURED,
                active_config_id="p1",
                providers=[p],
            )

    def test_non_empty_providers_without_active_config_id_rejected(self):
        p = OllamaProviderConfig(config_id="p1", display_name="P", model_name="m")
        with pytest.raises(ValidationError, match="active_config_id must be specified"):
            AIProviderSettings(
                status=AIProviderStatus.CONFIGURED,
                active_config_id=None,
                providers=[p],
            )

    def test_unknown_active_config_id_rejected(self):
        p = OllamaProviderConfig(config_id="p1", display_name="P", model_name="m")
        with pytest.raises(ValidationError, match="does not match any configured provider"):
            AIProviderSettings(
                status=AIProviderStatus.CONFIGURED,
                active_config_id="unknown_id",
                providers=[p],
            )

    def test_disabled_active_provider_rejected(self):
        p = OllamaProviderConfig(config_id="p1", display_name="P", model_name="m", enabled=False)
        with pytest.raises(ValidationError, match="must be enabled"):
            AIProviderSettings(
                status=AIProviderStatus.CONFIGURED,
                active_config_id="p1",
                providers=[p],
            )

    def test_disabled_non_active_provider_accepted(self):
        p1 = OllamaProviderConfig(config_id="p1", display_name="Active", model_name="m", enabled=True)
        p2 = OllamaProviderConfig(config_id="p2", display_name="Disabled", model_name="m", enabled=False)
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="p1",
            providers=[p1, p2],
        )
        assert len(settings.providers) == 2
        assert settings.active_config_id == "p1"

    def test_to_safe_dict_safely_serializes_ollama_provider(self):
        p = OllamaProviderConfig(config_id="ollama", display_name="O", model_name="m")
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="ollama",
            providers=[p],
        )
        safe = settings.to_safe_dict()
        assert safe["status"] == "configured"
        assert safe["active_config_id"] == "ollama"
        assert len(safe["providers"]) == 1
        assert safe["providers"][0]["config_id"] == "ollama"
        assert safe["providers"][0]["provider_type"] == "ollama"

    def test_to_safe_dict_masks_nested_openai_compatible_api_key(self):
        p = OpenAICompatibleProviderConfig(
            config_id="openai",
            display_name="OpenAI",
            model_name="m",
            base_url="https://api.openai.com/v1",
            api_key="sk-my-secret-key",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai",
            providers=[p],
        )
        safe = settings.to_safe_dict()
        assert safe["providers"][0]["api_key"] == "***"

    def test_model_dump_works(self):
        p = OllamaProviderConfig(config_id="ollama", display_name="O", model_name="m")
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="ollama",
            providers=[p],
        )
        dump = settings.model_dump()
        assert dump["status"] == AIProviderStatus.CONFIGURED
        assert dump["providers"][0]["config_id"] == "ollama"

    def test_model_dump_json_works(self):
        p = OllamaProviderConfig(config_id="ollama", display_name="O", model_name="m")
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="ollama",
            providers=[p],
        )
        dump_json = settings.model_dump_json()
        assert '"status":"configured"' in dump_json.replace(" ", "")

    def test_nested_provider_enums_serialize_correctly(self):
        p = OllamaProviderConfig(config_id="ollama", display_name="O", model_name="m")
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="ollama",
            providers=[p],
        )
        dump = settings.model_dump()
        # Verify enum values are serialized as standard strings
        assert isinstance(dump["status"], str)
        assert dump["status"] == "configured"
        assert dump["providers"][0]["provider_type"] == "ollama"
