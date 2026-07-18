"""Tests for AIProviderSettingsStore persistence (Stage 8F)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from backend.app.ai_providers.schemas import (
    AIProviderSettings,
    AIProviderStatus,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
)
from backend.app.ai_providers.secret_store import (
    InMemorySecretStore,
    SecretStore,
    SecretStoreError,
)
from backend.app.ai_providers.settings_store import (
    AIProviderSettingsStore,
    AIProviderSettingsStoreError,
)
from backend.app.ai_providers.factory import (
    AIProviderFactory,
    OpenAICompatibleAIProvider,
)


class TestSettingsStoreConstructorAndBasic:
    """Tests 21-28: Constructor validation, directory creation, exists checks."""

    def test_constructor_accepts_valid_path_and_secret_store(self, tmp_path: Path):
        """Test 21: Constructor accepts valid Path and SecretStore."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)
        assert store.file_path == file_path
        assert store.secret_store == secret_store

    def test_invalid_file_path_type_rejected(self):
        """Test 22: Non-Path file_path raises TypeError."""
        secret_store = InMemorySecretStore()
        with pytest.raises(TypeError, match="file_path must be a pathlib.Path instance"):
            AIProviderSettingsStore(file_path="settings.json", secret_store=secret_store)  # type: ignore

    def test_invalid_secret_store_type_rejected(self, tmp_path: Path):
        """Test 23: Non-SecretStore secret_store raises TypeError."""
        file_path = tmp_path / "settings.json"
        with pytest.raises(TypeError, match="secret_store must be a SecretStore instance"):
            AIProviderSettingsStore(file_path=file_path, secret_store=object())  # type: ignore

    def test_save_accepts_valid_empty_unconfigured_settings(self, tmp_path: Path):
        """Test 24: save() accepts valid empty/unconfigured settings."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings)
        assert store.exists()
        loaded = store.load()
        assert loaded.status == AIProviderStatus.UNCONFIGURED
        assert len(loaded.providers) == 0

    def test_save_creates_missing_parent_directory(self, tmp_path: Path):
        """Test 25: save() automatically creates missing parent directory."""
        file_path = tmp_path / "nested" / "dir" / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings)
        assert file_path.exists()

    def test_save_creates_json_file(self, tmp_path: Path):
        """Test 26: save() creates a JSON file containing serialized data."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "unconfigured"

    def test_exists_checks(self, tmp_path: Path):
        """Test 27-28: exists() is false before save, true after save."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)
        assert not store.exists()  # Test 27
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings)
        assert store.exists()  # Test 28


class TestSettingsStoreRoundtrip:
    """Tests 29-41: Provider configurations round-trip serialization and field preservation."""

    def test_ollama_config_roundtrip(self, tmp_path: Path):
        """Test 29: Ollama configuration round-trip works without using SecretStore."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        ollama_cfg = OllamaProviderConfig(
            config_id="ollama_local",
            display_name="Local Ollama",
            model_name="llama3.2",
            base_url="http://localhost:11434",
            request_timeout_seconds=90.0,
            enabled=True,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="ollama_local",
            providers=[ollama_cfg],
        )

        store.save(settings=settings)
        loaded = store.load()

        assert len(loaded.providers) == 1
        prov = loaded.providers[0]
        assert isinstance(prov, OllamaProviderConfig)
        assert prov.config_id == "ollama_local"
        assert prov.display_name == "Local Ollama"
        assert prov.model_name == "llama3.2"
        assert prov.base_url == "http://localhost:11434"
        assert prov.request_timeout_seconds == 90.0
        assert prov.enabled is True

    def test_openai_compatible_config_without_api_key_roundtrip(self, tmp_path: Path):
        """Test 30: OpenAICompatibleProviderConfig without an API key round-trip works."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_free",
            display_name="Free Gateway",
            model_name="gpt-4",
            base_url="http://localhost:8000/v1",
            api_key=None,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_free",
            providers=[openai_cfg],
        )

        store.save(settings=settings)
        loaded = store.load()

        assert len(loaded.providers) == 1
        prov = loaded.providers[0]
        assert isinstance(prov, OpenAICompatibleProviderConfig)
        assert prov.config_id == "openai_free"
        assert prov.api_key is None

    def test_openai_compatible_config_with_api_key_roundtrip(self, tmp_path: Path):
        """Test 31: OpenAICompatibleProviderConfig with an API key round-trip works."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI Remote",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-my-secret-api-key",
            request_timeout_seconds=45.0,
            enabled=True,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[openai_cfg],
        )

        store.save(settings=settings)
        loaded = store.load()

        assert len(loaded.providers) == 1
        prov = loaded.providers[0]
        assert isinstance(prov, OpenAICompatibleProviderConfig)
        assert prov.config_id == "openai_remote"
        assert prov.api_key == "sk-my-secret-api-key"

    def test_multiple_providers_and_fields_roundtrip(self, tmp_path: Path):
        """Tests 32-41: Multiple providers, ordering, status, and config fields are preserved."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        p1 = OllamaProviderConfig(
            config_id="ollama_local",
            display_name="Local Ollama",
            model_name="llama3.2",
            base_url="http://localhost:11434",
            request_timeout_seconds=90.0,
            enabled=True,
        )
        p2 = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI Remote",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-my-secret-api-key",
            request_timeout_seconds=45.0,
            enabled=True,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[p1, p2],  # Test 32: Multiple providers
        )

        store.save(settings=settings)
        loaded = store.load()

        # Test 33: Provider ordering preserved
        assert [p.config_id for p in loaded.providers] == ["ollama_local", "openai_remote"]
        # Test 34: active_config_id preserved
        assert loaded.active_config_id == "openai_remote"
        # Test 35: status preserved
        assert loaded.status == AIProviderStatus.CONFIGURED

        # Test fields of p1 (Ollama)
        loaded_p1 = loaded.providers[0]
        # Test 36: model_name preserved
        assert loaded_p1.model_name == "llama3.2"
        # Test 37: display_name preserved
        assert loaded_p1.display_name == "Local Ollama"
        # Test 38: base_url preserved
        assert loaded_p1.base_url == "http://localhost:11434"
        # Test 39: timeout preserved
        assert loaded_p1.request_timeout_seconds == 90.0
        # Test 40: enabled preserved
        assert loaded_p1.enabled is True
        # Test 41: provider_type preserved
        assert loaded_p1.provider_type == "ollama"

        # Test fields of p2 (OpenAI)
        loaded_p2 = loaded.providers[1]
        assert loaded_p2.model_name == "gpt-4o"
        assert loaded_p2.display_name == "OpenAI Remote"
        assert loaded_p2.base_url == "https://api.openai.com/v1"
        assert loaded_p2.request_timeout_seconds == 45.0
        assert loaded_p2.enabled is True
        assert loaded_p2.provider_type == "openai_compatible"
        assert loaded_p2.api_key == "sk-my-secret-api-key"


class TestSettingsStoreSecurityAndSafety:
    """Tests 42-46: API key secure storage and object immutability checks."""

    def test_raw_api_key_is_absent_from_persisted_json(self, tmp_path: Path):
        """Test 42: Raw API key is absent from the persisted JSON text file."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        secret_key = "sk-super-secret-secret-token"
        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=secret_key,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[openai_cfg],
        )

        store.save(settings=settings)

        # Read persisted file as plain text
        file_content = file_path.read_text(encoding="utf-8")
        assert secret_key not in file_content
        # Confirm api_key field is either null or absent
        parsed = json.loads(file_content)
        provider_data = parsed["providers"][0]
        assert provider_data["api_key"] is None

    def test_api_key_stored_in_secret_store(self, tmp_path: Path):
        """Test 43: API key is delegated and stored in the SecretStore."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        secret_key = "sk-stored-in-store"
        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=secret_key,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[openai_cfg],
        )

        store.save(settings=settings)
        # Verify it exists in the SecretStore under config_id
        assert secret_store.get_secret(config_id="openai_remote") == secret_key

    def test_api_key_restored_during_load(self, tmp_path: Path):
        """Test 44: API key is restored from SecretStore during load()."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        secret_key = "sk-restored-api-key"
        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=secret_key,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[openai_cfg],
        )

        store.save(settings=settings)
        loaded = store.load()
        assert loaded.providers[0].api_key == secret_key

    def test_save_does_not_mutate_settings(self, tmp_path: Path):
        """Test 45: save() does not mutate the passed AIProviderSettings object."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        secret_key = "sk-no-mutation"
        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=secret_key,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[openai_cfg],
        )

        original_dump = settings.model_dump()
        store.save(settings=settings)

        assert settings.model_dump() == original_dump
        assert settings.providers[0].api_key == secret_key

    def test_load_returns_new_object(self, tmp_path: Path):
        """Test 46: load() constructs and returns a new object rather than reusing references."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )

        store.save(settings=settings)
        loaded1 = store.load()
        loaded2 = store.load()

        assert loaded1 is not loaded2
        assert loaded1 is not settings


class TestSettingsStoreErrorHandling:
    """Tests 47-53: Corrupt JSON, invalid schema, read/write error encapsulation."""

    def test_corrupted_json_raises_settings_store_error(self, tmp_path: Path):
        """Test 47: Corrupted JSON data in settings file raises AIProviderSettingsStoreError."""
        file_path = tmp_path / "settings.json"
        file_path.write_text("invalid json { content", encoding="utf-8")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())

        with pytest.raises(AIProviderSettingsStoreError, match="Settings file is not valid JSON"):
            store.load()

    def test_invalid_persisted_schema_raises_settings_store_error(self, tmp_path: Path):
        """Test 48: Persisted settings data that violates schemas raises AIProviderSettingsStoreError."""
        file_path = tmp_path / "settings.json"
        invalid_data = {
            "status": "configured",
            "active_config_id": "missing_provider",
            "providers": [],
        }
        file_path.write_text(json.dumps(invalid_data), encoding="utf-8")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())

        with pytest.raises(AIProviderSettingsStoreError, match="Failed to reconstruct AIProviderSettings"):
            store.load()

    def test_missing_settings_file_load_raises_settings_store_error(self, tmp_path: Path):
        """Test 49: Loading when settings file does not exist raises AIProviderSettingsStoreError."""
        file_path = tmp_path / "non_existent.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())

        with pytest.raises(AIProviderSettingsStoreError, match="Settings file not found"):
            store.load()

    def test_secret_store_failure_during_save_is_wrapped(self, tmp_path: Path):
        """Test 50: SecretStore exceptions during save() are wrapped in AIProviderSettingsStoreError."""
        file_path = tmp_path / "settings.json"
        mock_secret_store = MagicMock(spec=SecretStore)
        mock_secret_store.set_secret.side_effect = SecretStoreError("keyring set failure")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=mock_secret_store)

        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[openai_cfg],
        )

        with pytest.raises(AIProviderSettingsStoreError, match="SecretStore failure during save") as exc_info:
            store.save(settings=settings)
        assert isinstance(exc_info.value.__cause__, SecretStoreError)

    def test_secret_store_failure_during_load_is_wrapped(self, tmp_path: Path):
        """Test 51: SecretStore exceptions during load() are wrapped in AIProviderSettingsStoreError."""
        file_path = tmp_path / "settings.json"
        mock_secret_store = MagicMock(spec=SecretStore)
        mock_secret_store.get_secret.side_effect = SecretStoreError("keyring get failure")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=mock_secret_store)

        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=None,
        )
        # write pre-saved settings with api_key=None to avoid save() triggering set_secret error
        settings_data = {
            "status": "configured",
            "active_config_id": "openai_remote",
            "providers": [
                {
                    "config_id": "openai_remote",
                    "provider_type": "openai_compatible",
                    "display_name": "OpenAI",
                    "model_name": "gpt-4o",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": None,
                }
            ],
        }
        file_path.write_text(json.dumps(settings_data), encoding="utf-8")

        with pytest.raises(AIProviderSettingsStoreError, match="SecretStore failure during load") as exc_info:
            store.load()
        assert isinstance(exc_info.value.__cause__, SecretStoreError)

    def test_file_write_failure_is_wrapped(self, tmp_path: Path):
        """Test 52: OS/filesystem write errors during save() are wrapped in AIProviderSettingsStoreError."""
        # Using a directory path as file_path triggers an OSError when writing
        file_path = tmp_path / "some_dir"
        file_path.mkdir()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )

        with pytest.raises(AIProviderSettingsStoreError, match="Failed to save settings atomically"):
            store.save(settings=settings)

    def test_file_read_failure_is_wrapped(self, tmp_path: Path):
        """Test 53: OS/filesystem read errors during load() are wrapped in AIProviderSettingsStoreError."""
        file_path = tmp_path / "settings.json"
        file_path.touch()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(AIProviderSettingsStoreError, match="Failed to read settings file") as exc_info:
                store.load()
            assert isinstance(exc_info.value.__cause__, PermissionError)


class TestSettingsStoreOperations:
    """Tests 54-64: delete behavior, repeated saves, custom providers, atomic replacement, exception messages."""

    def test_delete_removes_settings_file(self, tmp_path: Path):
        """Test 54: delete() removes the settings JSON file."""
        file_path = tmp_path / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings)
        assert store.exists()
        store.delete()
        assert not store.exists()

    def test_delete_removes_only_associated_secrets(self, tmp_path: Path):
        """Test 55: delete() removes associated secrets from the SecretStore, but leaves unrelated secrets."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        # Unrelated secret
        secret_store.set_secret(config_id="unrelated", secret="sk-unrelated")

        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-openai",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[openai_cfg],
        )

        store.save(settings=settings)
        assert secret_store.get_secret(config_id="openai_remote") == "sk-openai"

        store.delete()
        assert secret_store.get_secret(config_id="openai_remote") is None
        assert secret_store.get_secret(config_id="unrelated") == "sk-unrelated"

    def test_delete_with_missing_file_is_safe(self, tmp_path: Path):
        """Test 56: delete() on a non-existent settings file behaves safely."""
        file_path = tmp_path / "non_existent.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        # Should not raise exception
        store.delete()

    def test_delete_handles_multiple_provider_secrets(self, tmp_path: Path):
        """Test 57: delete() successfully deletes secrets for multiple providers config_ids."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        p1 = OpenAICompatibleProviderConfig(
            config_id="openai_1",
            display_name="OpenAI 1",
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-1",
        )
        p2 = OpenAICompatibleProviderConfig(
            config_id="openai_2",
            display_name="OpenAI 2",
            model_name="gpt-3.5",
            base_url="https://api.openai.com/v1",
            api_key="sk-2",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p1, p2],
        )

        store.save(settings=settings)
        assert secret_store.get_secret(config_id="openai_1") == "sk-1"
        assert secret_store.get_secret(config_id="openai_2") == "sk-2"

        store.delete()
        assert secret_store.get_secret(config_id="openai_1") is None
        assert secret_store.get_secret(config_id="openai_2") is None

    def test_repeated_save_updates_persisted_settings(self, tmp_path: Path):
        """Test 58: Repeated calls to save() update the persisted JSON values."""
        file_path = tmp_path / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())

        settings1 = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings1)

        p = OllamaProviderConfig(config_id="o1", display_name="Local", model_name="m")
        settings2 = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="o1",
            providers=[p],
        )
        store.save(settings=settings2)

        loaded = store.load()
        assert loaded.status == AIProviderStatus.CONFIGURED
        assert loaded.active_config_id == "o1"

    def test_repeated_save_updates_stored_api_key(self, tmp_path: Path):
        """Test 59: Repeated calls to save() update the stored API key in SecretStore."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        p = OpenAICompatibleProviderConfig(
            config_id="openai_1",
            display_name="OpenAI",
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-first-key",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p],
        )

        store.save(settings=settings)
        assert secret_store.get_secret(config_id="openai_1") == "sk-first-key"

        # Update API key
        p.api_key = "sk-second-key"
        # Config needs to be rebuilt because model validation ensures values match
        settings_updated = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p],
        )
        store.save(settings=settings_updated)
        assert secret_store.get_secret(config_id="openai_1") == "sk-second-key"

    def test_saving_api_key_none_does_not_create_secret(self, tmp_path: Path):
        """Test 60: Saving an OpenAI-compatible config with api_key=None does not create a secret store entry."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        p = OpenAICompatibleProviderConfig(
            config_id="openai_1",
            display_name="OpenAI",
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key=None,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p],
        )

        store.save(settings=settings)
        assert secret_store.get_secret(config_id="openai_1") is None

    def test_json_output_is_valid_json(self, tmp_path: Path):
        """Test 61: Persisted configuration JSON file is valid JSON."""
        file_path = tmp_path / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings)
        # If parsing fails, json.loads will raise JSONDecodeError
        data = json.loads(file_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_persisted_configuration_can_be_reconstructed_through_pydantic(self, tmp_path: Path):
        """Test 62: Loaded and reconstructed settings are valid Pydantic models."""
        file_path = tmp_path / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings)
        loaded = store.load()
        assert isinstance(loaded, AIProviderSettings)

    def test_atomic_replacement_leaves_final_destination_file_valid(self, tmp_path: Path):
        """Test 63: Atomic replace is used and successfully leaves final destination file valid."""
        file_path = tmp_path / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )

        # Mock os.replace to verify it's called
        with patch("os.replace", wraps=os.replace) as mock_replace:
            store.save(settings=settings)
            mock_replace.assert_called_once()

        assert file_path.exists()
        loaded = store.load()
        assert loaded.status == AIProviderStatus.UNCONFIGURED

    def test_api_key_never_appears_in_persistence_exception_messages(self, tmp_path: Path):
        """Test 64: API key never appears in exceptions raised during settings store operations."""
        file_path = tmp_path / "settings.json"
        mock_store = MagicMock(spec=SecretStore)
        secret_key = "sk-super-secret-credentials-99"
        mock_store.set_secret.side_effect = SecretStoreError("Error saving secret")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=mock_store)

        openai_cfg = OpenAICompatibleProviderConfig(
            config_id="openai_remote",
            display_name="OpenAI",
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=secret_key,
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[openai_cfg],
        )

        with pytest.raises(AIProviderSettingsStoreError) as exc_info:
            store.save(settings=settings)
        assert secret_key not in str(exc_info.value)


class TestStaleSecretCleanup:
    """Tests 65-67: Cleanup of stale credentials."""

    def test_existing_secret_removed_when_provider_api_key_changes_to_none(self, tmp_path: Path):
        """Test 65: When saving an existing provider configuration with api_key=None, its old secret is removed."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        p = OpenAICompatibleProviderConfig(
            config_id="openai_1",
            display_name="OpenAI",
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-original",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p],
        )

        store.save(settings=settings)
        assert secret_store.get_secret(config_id="openai_1") == "sk-original"

        # Re-save with key set to None
        p.api_key = None
        settings_updated = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p],
        )
        store.save(settings=settings_updated)

        assert secret_store.get_secret(config_id="openai_1") is None

    def test_secret_removed_when_provider_configuration_is_removed(self, tmp_path: Path):
        """Test 66: When a provider configuration is removed from settings list, its secret is cleaned up during save()."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        p1 = OpenAICompatibleProviderConfig(
            config_id="openai_1",
            display_name="OpenAI 1",
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-1",
        )
        p2 = OpenAICompatibleProviderConfig(
            config_id="openai_2",
            display_name="OpenAI 2",
            model_name="gpt-3.5",
            base_url="https://api.openai.com/v1",
            api_key="sk-2",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p1, p2],
        )

        store.save(settings=settings)
        assert secret_store.get_secret(config_id="openai_1") == "sk-1"
        assert secret_store.get_secret(config_id="openai_2") == "sk-2"

        # Save again with p2 removed
        settings_updated = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p1],
        )
        store.save(settings=settings_updated)

        assert secret_store.get_secret(config_id="openai_1") == "sk-1"
        assert secret_store.get_secret(config_id="openai_2") is None  # Cleaned up

    def test_unrelated_secret_remains_untouched(self, tmp_path: Path):
        """Test 67: Stale secret cleanup doesn't delete secrets belonging to other unrelated config_ids in SecretStore."""
        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        # Set unrelated secret in store
        secret_store.set_secret(config_id="unrelated", secret="sk-unrelated")

        p = OpenAICompatibleProviderConfig(
            config_id="openai_1",
            display_name="OpenAI",
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-1",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_1",
            providers=[p],
        )

        store.save(settings=settings)

        # Re-save with provider removed (empty list)
        settings_empty = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )
        store.save(settings=settings_empty)

        assert secret_store.get_secret(config_id="openai_1") is None  # Cleaned up
        assert secret_store.get_secret(config_id="unrelated") == "sk-unrelated"  # Untouched


class TestFactoryAndSettingsStoreIntegration:
    """Integration with existing AIProviderFactory and model validator rules."""

    def test_config_created_by_registry_reconstructs_and_creates_provider(self, tmp_path: Path):
        """Reconstruct config via load() and verify with AIProviderFactory."""
        from backend.app.ai_providers.presets import AIProviderPresetRegistry, AIProviderPresetId

        file_path = tmp_path / "settings.json"
        secret_store = InMemorySecretStore()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=secret_store)

        registry = AIProviderPresetRegistry()
        config = registry.create_config(
            preset_id=AIProviderPresetId.OPENAI,
            config_id="openai_preset",
            model_name="gpt-4",
            api_key="sk-preset",
        )
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_preset",
            providers=[config],
        )

        store.save(settings=settings)
        loaded = store.load()

        loaded_config = loaded.providers[0]
        provider = AIProviderFactory.create(loaded_config)
        assert isinstance(provider, OpenAICompatibleAIProvider)
        assert provider.config is loaded_config
        assert provider.config.api_key == "sk-preset"


class TestSettingsStoreCoverageBoost:
    """Extra tests designed specifically to cover all error/exception branches in settings_store.py."""

    def test_save_invalid_settings_type_rejected(self, tmp_path: Path):
        """Line 75: save() rejects non-AIProviderSettings objects with TypeError."""
        file_path = tmp_path / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        with pytest.raises(TypeError, match="settings must be an AIProviderSettings instance"):
            store.save(settings=object())  # type: ignore

    def test_save_failed_parent_directory_creation(self, tmp_path: Path):
        """Line 80-81: parent directory creation failure is caught and wrapped."""
        file_path = tmp_path / "nested" / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )

        with patch("pathlib.Path.mkdir", side_effect=OSError("Read-only file system")):
            with pytest.raises(AIProviderSettingsStoreError, match="Failed to create parent directory") as exc_info:
                store.save(settings=settings)
            assert isinstance(exc_info.value.__cause__, OSError)

    def test_save_failed_serialization(self, tmp_path: Path):
        """Line 126-127: model_dump serialization failure is caught and wrapped."""
        file_path = tmp_path / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )

        with patch("backend.app.ai_providers.schemas.AIProviderSettings.model_dump", side_effect=ValueError("Dump error")):
            with pytest.raises(AIProviderSettingsStoreError, match="Failed to serialize settings data") as exc_info:
                store.save(settings=settings)
            assert isinstance(exc_info.value.__cause__, ValueError)

    def test_save_temp_file_unlink_failure(self, tmp_path: Path):
        """Line 140-141: temp file unlink failure is ignored safely when atomic save fails."""
        file_path = tmp_path / "settings.json"
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        settings = AIProviderSettings(
            status=AIProviderStatus.UNCONFIGURED,
            active_config_id=None,
            providers=[],
        )

        # Force write failure so except block is entered, and patch os.unlink to fail as well
        with patch("os.replace", side_effect=OSError("Replace error")):
            with patch("os.unlink", side_effect=OSError("Unlink error")):
                with pytest.raises(AIProviderSettingsStoreError, match="Failed to save settings atomically"):
                    store.save(settings=settings)

    def test_load_json_not_dictionary(self, tmp_path: Path):
        """Line 166: loaded JSON that parses into a non-dict structure is rejected."""
        file_path = tmp_path / "settings.json"
        file_path.write_text(json.dumps(["list", "not", "dict"]), encoding="utf-8")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())

        with pytest.raises(AIProviderSettingsStoreError, match="Settings JSON is not an object/dict"):
            store.load()

    def test_load_provider_not_dictionary(self, tmp_path: Path):
        """Line 172: Loaded provider entry that is not a dictionary is ignored safely (skipped during secret injection but fails final validation)."""
        file_path = tmp_path / "settings.json"
        corrupted_data = {
            "status": "configured",
            "active_config_id": "ollama_local",
            "providers": ["not-a-provider-dict", {"config_id": "ollama_local", "provider_type": "ollama", "display_name": "Local", "model_name": "m"}],
        }
        file_path.write_text(json.dumps(corrupted_data), encoding="utf-8")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        with pytest.raises(AIProviderSettingsStoreError, match="Failed to reconstruct AIProviderSettings"):
            store.load()

    def test_load_provider_missing_config_id(self, tmp_path: Path):
        """Line 176: OpenAI provider dictionary with missing config_id is safely skipped."""
        file_path = tmp_path / "settings.json"
        corrupted_data = {
            "status": "configured",
            "active_config_id": "openai_remote",
            "providers": [
                {
                    "provider_type": "openai_compatible",
                    # missing config_id
                    "display_name": "OpenAI",
                    "model_name": "gpt-4",
                    "base_url": "https://api.openai.com/v1",
                },
                {
                    "config_id": "openai_remote",
                    "provider_type": "openai_compatible",
                    "display_name": "OpenAI 2",
                    "model_name": "gpt-4",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }
        file_path.write_text(json.dumps(corrupted_data), encoding="utf-8")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())
        # The first item will be skipped for secret lookup, but Pydantic validation will ultimately fail on it
        # because config_id is required in BaseAIProviderConfig. Let's assert it raises settings store validation error.
        with pytest.raises(AIProviderSettingsStoreError, match="Failed to reconstruct AIProviderSettings"):
            store.load()

    def test_delete_corrupted_json_handled_gracefully(self, tmp_path: Path):
        """Line 210-212: delete() behaves safely and deletes the settings file even if reading it for secret deletion fails."""
        file_path = tmp_path / "settings.json"
        file_path.write_text("{ corrupted json", encoding="utf-8")
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())

        assert store.exists()
        store.delete()
        assert not store.exists()

    def test_delete_file_unlink_failure_is_wrapped(self, tmp_path: Path):
        """Line 216-217: delete() wraps unlink failures in AIProviderSettingsStoreError."""
        file_path = tmp_path / "settings.json"
        file_path.touch()
        store = AIProviderSettingsStore(file_path=file_path, secret_store=InMemorySecretStore())

        with patch("pathlib.Path.unlink", side_effect=PermissionError("Cannot delete")):
            with pytest.raises(AIProviderSettingsStoreError, match="Failed to delete settings file") as exc_info:
                store.delete()
            assert isinstance(exc_info.value.__cause__, PermissionError)
