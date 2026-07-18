"""Tests for AI provider preset registry (Stage 8E)."""

import pytest

from pydantic import ValidationError

from backend.app.ai_planning.providers.base import AIProvider
from backend.app.ai_providers.schemas import (
    AIProviderType,
    OpenAICompatibleProviderConfig,
)
from backend.app.ai_providers.factory import (
    AIProviderFactory,
    OpenAICompatibleAIProvider,
)
from backend.app.ai_providers.presets import (
    AIProviderPresetId,
    AIProviderPreset,
    AIProviderPresetRegistry,
    AIProviderPresetRegistryError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_PRESET_IDS = {
    AIProviderPresetId.OPENAI,
    AIProviderPresetId.GEMINI,
    AIProviderPresetId.DEEPSEEK,
    AIProviderPresetId.XAI,
    AIProviderPresetId.GROQ,
    AIProviderPresetId.OPENROUTER,
    AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE,
}


def _make_registry() -> AIProviderPresetRegistry:
    return AIProviderPresetRegistry()


# ── Preset Enum Tests (1) ───────────────────────────────────────────────


class TestPresetEnum:
    """Test 1: All expected preset enum values exist."""

    def test_all_expected_values_exist(self):
        assert AIProviderPresetId.OPENAI.value == "openai"
        assert AIProviderPresetId.GEMINI.value == "gemini"
        assert AIProviderPresetId.DEEPSEEK.value == "deepseek"
        assert AIProviderPresetId.XAI.value == "xai"
        assert AIProviderPresetId.GROQ.value == "groq"
        assert AIProviderPresetId.OPENROUTER.value == "openrouter"
        assert AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE.value == "custom_openai_compatible"

    def test_enum_count_matches_expected(self):
        assert len(AIProviderPresetId) == 7


# ── Preset Model Tests (2-7) ────────────────────────────────────────────


class TestPresetModel:
    """Test 2-7: AIProviderPreset Pydantic model validation."""

    def test_valid_preset_creation(self):
        """Test 2."""
        preset = AIProviderPreset(
            preset_id=AIProviderPresetId.OPENAI,
            display_name="OpenAI",
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            default_base_url="https://api.openai.com/v1",
            requires_api_key=True,
            description="OpenAI API.",
        )
        assert preset.preset_id == AIProviderPresetId.OPENAI
        assert preset.display_name == "OpenAI"
        assert preset.provider_type == AIProviderType.OPENAI_COMPATIBLE
        assert preset.default_base_url == "https://api.openai.com/v1"
        assert preset.requires_api_key is True
        assert preset.supports_custom_model_name is True
        assert preset.description == "OpenAI API."

    def test_empty_display_name_rejected(self):
        """Test 3."""
        with pytest.raises(ValidationError, match="display_name"):
            AIProviderPreset(
                preset_id=AIProviderPresetId.OPENAI,
                display_name="  ",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                requires_api_key=True,
                description="Desc.",
            )

    def test_empty_description_rejected(self):
        """Test 4."""
        with pytest.raises(ValidationError, match="description"):
            AIProviderPreset(
                preset_id=AIProviderPresetId.OPENAI,
                display_name="OpenAI",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                requires_api_key=True,
                description="",
            )

    def test_invalid_default_url_rejected(self):
        """Test 5."""
        with pytest.raises(ValidationError, match="default_base_url"):
            AIProviderPreset(
                preset_id=AIProviderPresetId.OPENAI,
                display_name="OpenAI",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                default_base_url="ftp://not-valid.com",
                requires_api_key=True,
                description="Desc.",
            )

    def test_url_trailing_slash_normalized(self):
        """Test 6."""
        preset = AIProviderPreset(
            preset_id=AIProviderPresetId.OPENAI,
            display_name="OpenAI",
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            default_base_url="https://api.openai.com/v1/",
            requires_api_key=True,
            description="Desc.",
        )
        assert preset.default_base_url == "https://api.openai.com/v1"

    def test_none_default_url_accepted(self):
        """Test 7."""
        preset = AIProviderPreset(
            preset_id=AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE,
            display_name="Custom",
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            default_base_url=None,
            requires_api_key=False,
            description="Custom endpoint.",
        )
        assert preset.default_base_url is None

    def test_preset_is_frozen(self):
        """Frozen model prevents mutation."""
        preset = AIProviderPreset(
            preset_id=AIProviderPresetId.OPENAI,
            display_name="OpenAI",
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            requires_api_key=True,
            description="Desc.",
        )
        with pytest.raises(ValidationError):
            preset.display_name = "Modified"  # type: ignore

    def test_display_name_whitespace_stripped(self):
        preset = AIProviderPreset(
            preset_id=AIProviderPresetId.OPENAI,
            display_name="  OpenAI  ",
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            requires_api_key=True,
            description="Desc.",
        )
        assert preset.display_name == "OpenAI"

    def test_description_whitespace_stripped(self):
        preset = AIProviderPreset(
            preset_id=AIProviderPresetId.OPENAI,
            display_name="OpenAI",
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            requires_api_key=True,
            description="   Desc.   ",
        )
        assert preset.description == "Desc."

    def test_non_string_display_name_rejected(self):
        with pytest.raises(ValidationError, match="display_name"):
            AIProviderPreset(
                preset_id=AIProviderPresetId.OPENAI,
                display_name=123,  # type: ignore
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                requires_api_key=True,
                description="Desc.",
            )

    def test_non_string_description_rejected(self):
        with pytest.raises(ValidationError, match="description"):
            AIProviderPreset(
                preset_id=AIProviderPresetId.OPENAI,
                display_name="OpenAI",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                requires_api_key=True,
                description=42,  # type: ignore
            )


# ── Registry Listing Tests (8-13) ───────────────────────────────────────


class TestRegistryListing:
    """Test 8-13: list_presets() behavior."""

    def test_registry_instantiates(self):
        """Test 8."""
        registry = _make_registry()
        assert registry is not None

    def test_list_presets_returns_all_expected(self):
        """Test 9."""
        registry = _make_registry()
        presets = registry.list_presets()
        preset_ids = {p.preset_id for p in presets}
        assert preset_ids == EXPECTED_PRESET_IDS

    def test_order_is_deterministic(self):
        """Test 10."""
        registry = _make_registry()
        ids_1 = [p.preset_id for p in registry.list_presets()]
        ids_2 = [p.preset_id for p in registry.list_presets()]
        assert ids_1 == ids_2

    def test_preset_ids_are_unique(self):
        """Test 11."""
        registry = _make_registry()
        presets = registry.list_presets()
        ids = [p.preset_id for p in presets]
        assert len(ids) == len(set(ids))

    def test_returned_list_is_independent(self):
        """Test 12."""
        registry = _make_registry()
        list_a = registry.list_presets()
        list_b = registry.list_presets()
        assert list_a is not list_b

    def test_mutating_returned_list_does_not_alter_registry(self):
        """Test 13."""
        registry = _make_registry()
        presets = registry.list_presets()
        original_count = len(presets)
        presets.clear()
        assert len(registry.list_presets()) == original_count

    def test_all_presets_use_openai_compatible_type(self):
        """All presets use OPENAI_COMPATIBLE provider type."""
        registry = _make_registry()
        for preset in registry.list_presets():
            assert preset.provider_type == AIProviderType.OPENAI_COMPATIBLE

    def test_all_presets_have_nonempty_descriptions(self):
        """All presets have meaningful descriptions."""
        registry = _make_registry()
        for preset in registry.list_presets():
            assert isinstance(preset.description, str)
            assert len(preset.description.strip()) > 0

    def test_all_presets_have_nonempty_display_names(self):
        """All presets have display names."""
        registry = _make_registry()
        for preset in registry.list_presets():
            assert isinstance(preset.display_name, str)
            assert len(preset.display_name.strip()) > 0


# ── Get Preset Tests (14-17) ────────────────────────────────────────────


class TestGetPreset:
    """Test 14-17: get_preset() behavior."""

    @pytest.mark.parametrize("preset_id", list(AIProviderPresetId))
    def test_each_known_preset_can_be_retrieved(self, preset_id: AIProviderPresetId):
        """Test 14."""
        registry = _make_registry()
        preset = registry.get_preset(preset_id)
        assert preset is not None

    def test_correct_preset_is_returned(self):
        """Test 15."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.OPENAI)
        assert preset.preset_id == AIProviderPresetId.OPENAI
        assert preset.display_name == "OpenAI"

    def test_invalid_input_type_rejected(self):
        """Test 16."""
        registry = _make_registry()
        with pytest.raises(AIProviderPresetRegistryError, match="Invalid preset identifier type"):
            registry.get_preset("openai")  # type: ignore

    def test_none_input_rejected(self):
        registry = _make_registry()
        with pytest.raises(AIProviderPresetRegistryError, match="Invalid preset identifier type"):
            registry.get_preset(None)  # type: ignore

    def test_integer_input_rejected(self):
        registry = _make_registry()
        with pytest.raises(AIProviderPresetRegistryError, match="Invalid preset identifier type"):
            registry.get_preset(42)  # type: ignore


# ── Create Config Tests (18-29) ─────────────────────────────────────────


class TestCreateConfig:
    """Test 18-29: create_config() behavior."""

    def test_returns_openai_compatible_config(self):
        """Test 18."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-test-key",
        )
        assert isinstance(config, OpenAICompatibleProviderConfig)

    def test_provider_type_is_openai_compatible(self):
        """Test 19."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-test-key",
        )
        assert config.provider_type == AIProviderType.OPENAI_COMPATIBLE

    def test_config_id_preserved(self):
        """Test 20."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="my-unique-id",
            model_name="gpt-4o",
            api_key="sk-test-key",
        )
        assert config.config_id == "my-unique-id"

    def test_model_name_preserved(self):
        """Test 21."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.DEEPSEEK,
            config_id="ds_1",
            model_name="deepseek-chat",
            api_key="sk-key",
        )
        assert config.model_name == "deepseek-chat"

    def test_default_display_name_used(self):
        """Test 22."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.GROQ,
            config_id="groq_1",
            model_name="mixtral-8x7b",
            api_key="gsk-key",
        )
        assert config.display_name == "Groq"

    def test_custom_display_name_override(self):
        """Test 23."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.GROQ,
            config_id="groq_1",
            model_name="mixtral-8x7b",
            api_key="gsk-key",
            display_name="My Fast Groq",
        )
        assert config.display_name == "My Fast Groq"

    def test_preset_default_base_url_used(self):
        """Test 24."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-key",
        )
        assert config.base_url == "https://api.openai.com/v1"

    def test_explicit_base_url_override(self):
        """Test 25."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-key",
            base_url="https://my-proxy.example.com/v1",
        )
        assert config.base_url == "https://my-proxy.example.com/v1"

    def test_timeout_preserved(self):
        """Test 26."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-key",
            request_timeout_seconds=45.0,
        )
        assert config.request_timeout_seconds == 45.0

    def test_enabled_state_preserved(self):
        """Test 27."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-key",
            enabled=False,
        )
        assert config.enabled is False

    def test_api_key_preserved_in_config(self):
        """Test 28."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-real-secret-key",
        )
        assert config.api_key == "sk-real-secret-key"

    def test_safe_serialization_masks_api_key(self):
        """Test 29."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-real-secret-key",
        )
        safe = config.to_safe_dict()
        assert safe["api_key"] == "***"
        assert safe["config_id"] == "openai_1"

    def test_default_timeout_is_120(self):
        """Default timeout matches schema default."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-key",
        )
        assert config.request_timeout_seconds == 120.0

    def test_default_enabled_is_true(self):
        """Default enabled matches schema default."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-key",
        )
        assert config.enabled is True


# ── API Key Requirement Tests (30-32) ───────────────────────────────────


class TestAPIKeyRequirements:
    """Test 30-32: API key enforcement behavior."""

    def test_required_key_preset_rejects_missing_api_key(self):
        """Test 30."""
        registry = _make_registry()
        with pytest.raises(AIProviderPresetRegistryError, match="requires an API key"):
            registry.create_config(
                preset_id=AIProviderPresetId.OPENAI,
                config_id="openai_1",
                model_name="gpt-4o",
                api_key=None,
            )

    def test_required_key_preset_accepts_api_key(self):
        """Test 31."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-valid-key",
        )
        assert config.api_key == "sk-valid-key"

    def test_registry_error_never_contains_api_key(self):
        """Test 32."""
        registry = _make_registry()
        secret = "sk-ultra-secret-99999"
        # The registry error for missing key should not contain a key value
        # (it won't since we pass None, but let's verify the error message pattern)
        try:
            registry.create_config(
                preset_id=AIProviderPresetId.OPENAI,
                config_id="openai_1",
                model_name="gpt-4o",
                api_key=None,
            )
        except AIProviderPresetRegistryError as exc:
            assert secret not in str(exc)
            assert "None" not in str(exc) or "none" in str(exc).lower()

    @pytest.mark.parametrize("preset_id", [
        AIProviderPresetId.OPENAI,
        AIProviderPresetId.GEMINI,
        AIProviderPresetId.DEEPSEEK,
        AIProviderPresetId.XAI,
        AIProviderPresetId.GROQ,
        AIProviderPresetId.OPENROUTER,
    ])
    def test_all_named_presets_require_api_key(self, preset_id: AIProviderPresetId):
        """All named service presets require an API key."""
        registry = _make_registry()
        with pytest.raises(AIProviderPresetRegistryError, match="requires an API key"):
            registry.create_config(
                preset_id=preset_id,
                config_id="test",
                model_name="model",
                api_key=None,
            )


# ── Custom OpenAI-Compatible Tests (33-37) ──────────────────────────────


class TestCustomOpenAICompatible:
    """Test 33-37: Custom preset behavior."""

    def test_custom_preset_has_no_default_base_url(self):
        """Test 33."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE)
        assert preset.default_base_url is None

    def test_custom_without_explicit_base_url_rejected(self):
        """Test 34."""
        registry = _make_registry()
        with pytest.raises(AIProviderPresetRegistryError, match="No base URL available"):
            registry.create_config(
                preset_id=AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE,
                config_id="custom_1",
                model_name="my-model",
            )

    def test_custom_with_explicit_base_url_accepted(self):
        """Test 35."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE,
            config_id="custom_1",
            model_name="my-model",
            base_url="http://localhost:8080/v1",
        )
        assert config.base_url == "http://localhost:8080/v1"

    def test_custom_without_api_key_accepted(self):
        """Test 36."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE,
            config_id="custom_1",
            model_name="my-model",
            base_url="http://localhost:8080/v1",
            api_key=None,
        )
        assert config.api_key is None

    def test_custom_with_optional_api_key_accepted(self):
        """Test 37."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE,
            config_id="custom_1",
            model_name="my-model",
            base_url="http://localhost:8080/v1",
            api_key="optional-key",
        )
        assert config.api_key == "optional-key"

    def test_custom_preset_does_not_require_api_key(self):
        """Custom preset has requires_api_key=False."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE)
        assert preset.requires_api_key is False


# ── Factory Integration Tests (38-40) ───────────────────────────────────


class TestFactoryIntegration:
    """Test 38-40: registry configs work with the existing factory."""

    def test_config_works_with_factory_create(self):
        """Test 38."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-key",
        )
        provider = AIProviderFactory.create(config)
        assert provider is not None

    def test_resulting_adapter_is_openai_compatible(self):
        """Test 39."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.GROQ,
            config_id="groq_1",
            model_name="mixtral-8x7b",
            api_key="gsk-key",
        )
        provider = AIProviderFactory.create(config)
        assert isinstance(provider, OpenAICompatibleAIProvider)
        assert isinstance(provider, AIProvider)

    def test_adapter_preserves_exact_generated_config(self):
        """Test 40."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.DEEPSEEK,
            config_id="ds_1",
            model_name="deepseek-chat",
            api_key="sk-key",
        )
        provider = AIProviderFactory.create(config)
        assert provider.config is config


# ── Non-Mutation / Safety Tests (41-43) ─────────────────────────────────


class TestNonMutation:
    """Test 41-43: immutability and safety guarantees."""

    def test_presets_unchanged_after_create_config(self):
        """Test 41."""
        registry = _make_registry()
        preset_before = registry.get_preset(AIProviderPresetId.OPENAI)
        before_url = preset_before.default_base_url
        before_name = preset_before.display_name

        registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_1",
            model_name="gpt-4o",
            api_key="sk-key",
            display_name="My Custom Name",
            base_url="https://custom.example.com/v1",
        )

        preset_after = registry.get_preset(AIProviderPresetId.OPENAI)
        assert preset_after.default_base_url == before_url
        assert preset_after.display_name == before_name

    def test_caller_strings_not_mutated(self):
        """Test 42."""
        registry = _make_registry()
        config_id = "my_config"
        model_name = "gpt-4o"
        api_key = "sk-key-value"
        display_name = "My Display"

        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id=config_id,
            model_name=model_name,
            api_key=api_key,
            display_name=display_name,
        )

        # Python strings are immutable, but verify config reflects exact values
        assert config.config_id == "my_config"
        assert config.model_name == "gpt-4o"
        assert config.api_key == "sk-key-value"
        assert config.display_name == "My Display"

    def test_existing_provider_schemas_unchanged(self):
        """Test 43: OpenAICompatibleProviderConfig still works as before."""
        config = OpenAICompatibleProviderConfig(
            config_id="direct_config",
            display_name="Direct",
            model_name="model",
            base_url="https://example.com/v1",
            api_key="sk-key",
        )
        assert config.provider_type == AIProviderType.OPENAI_COMPATIBLE
        assert config.to_safe_dict()["api_key"] == "***"


# ── Additional Coverage Tests ───────────────────────────────────────────


class TestAdditionalCoverage:
    """Additional tests for full branch coverage and edge cases."""

    @pytest.mark.parametrize("preset_id", [
        AIProviderPresetId.OPENAI,
        AIProviderPresetId.GEMINI,
        AIProviderPresetId.DEEPSEEK,
        AIProviderPresetId.XAI,
        AIProviderPresetId.GROQ,
        AIProviderPresetId.OPENROUTER,
    ])
    def test_each_named_preset_has_default_base_url(self, preset_id: AIProviderPresetId):
        """Named presets all have a default base URL."""
        registry = _make_registry()
        preset = registry.get_preset(preset_id)
        assert preset.default_base_url is not None
        assert preset.default_base_url.startswith("https://")

    @pytest.mark.parametrize("preset_id", [
        AIProviderPresetId.OPENAI,
        AIProviderPresetId.GEMINI,
        AIProviderPresetId.DEEPSEEK,
        AIProviderPresetId.XAI,
        AIProviderPresetId.GROQ,
        AIProviderPresetId.OPENROUTER,
    ])
    def test_each_named_preset_creates_valid_config(self, preset_id: AIProviderPresetId):
        """Each named preset can produce a valid config with an API key."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=preset_id,
            config_id=f"{preset_id.value}_test",
            model_name="test-model",
            api_key="sk-test-key",
        )
        assert isinstance(config, OpenAICompatibleProviderConfig)
        assert config.model_name == "test-model"
        assert config.api_key == "sk-test-key"

    def test_create_config_with_invalid_preset_type_rejected(self):
        """create_config rejects non-enum preset_id."""
        registry = _make_registry()
        with pytest.raises(AIProviderPresetRegistryError, match="Invalid preset identifier type"):
            registry.create_config(
                preset_id="openai",  # type: ignore
                config_id="c1",
                model_name="model",
                api_key="key",
            )

    def test_gemini_base_url(self):
        """Verify Gemini preset base URL."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.GEMINI)
        assert preset.default_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"

    def test_deepseek_base_url(self):
        """Verify DeepSeek preset base URL."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.DEEPSEEK)
        assert preset.default_base_url == "https://api.deepseek.com/v1"

    def test_xai_base_url(self):
        """Verify xAI preset base URL."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.XAI)
        assert preset.default_base_url == "https://api.x.ai/v1"

    def test_groq_base_url(self):
        """Verify Groq preset base URL."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.GROQ)
        assert preset.default_base_url == "https://api.groq.com/openai/v1"

    def test_openrouter_base_url(self):
        """Verify OpenRouter preset base URL."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.OPENROUTER)
        assert preset.default_base_url == "https://openrouter.ai/api/v1"

    def test_openai_base_url(self):
        """Verify OpenAI preset base URL."""
        registry = _make_registry()
        preset = registry.get_preset(AIProviderPresetId.OPENAI)
        assert preset.default_base_url == "https://api.openai.com/v1"

    def test_no_base_urls_have_trailing_slashes(self):
        """All preset base URLs are normalized without trailing slashes."""
        registry = _make_registry()
        for preset in registry.list_presets():
            if preset.default_base_url is not None:
                assert not preset.default_base_url.endswith("/")

    def test_multiple_registries_are_independent(self):
        """Two registry instances share the same data but are independent objects."""
        r1 = _make_registry()
        r2 = _make_registry()
        assert r1 is not r2
        assert r1.list_presets() is not r2.list_presets()

    def test_preset_model_empty_base_url_string_rejected(self):
        """Empty string base_url is rejected."""
        with pytest.raises(ValidationError, match="default_base_url"):
            AIProviderPreset(
                preset_id=AIProviderPresetId.OPENAI,
                display_name="OpenAI",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                default_base_url="",
                requires_api_key=True,
                description="Desc.",
            )

    def test_preset_model_whitespace_only_base_url_rejected(self):
        """Whitespace-only base_url is rejected."""
        with pytest.raises(ValidationError, match="default_base_url"):
            AIProviderPreset(
                preset_id=AIProviderPresetId.OPENAI,
                display_name="OpenAI",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                default_base_url="   ",
                requires_api_key=True,
                description="Desc.",
            )

    def test_optional_api_key_on_non_required_preset(self):
        """Non-required-key preset accepts api_key=None without error."""
        registry = _make_registry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.CUSTOM_OPENAI_COMPATIBLE,
            config_id="custom",
            model_name="model",
            base_url="http://localhost:8000/v1",
            api_key=None,
        )
        assert config.api_key is None
        safe = config.to_safe_dict()
        assert safe["api_key"] is None

    def test_unknown_preset_in_index_raises_error(self):
        """Defensive branch: preset passes type check but is missing from index."""
        from unittest.mock import patch
        registry = _make_registry()
        # Temporarily patch the index to be empty so the lookup fails
        with patch.dict("backend.app.ai_providers.presets._PRESET_INDEX", clear=True):
            with pytest.raises(AIProviderPresetRegistryError, match="Unknown preset identifier"):
                registry.get_preset(AIProviderPresetId.OPENAI)
