"""AI provider settings persistence store for DeployAI.

Stage 8F introduces the AIProviderSettingsStore to handle saving and loading
provider settings while delegating API keys to a SecretStore.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Set

from backend.app.ai_providers.schemas import (
    AIProviderSettings,
    AIProviderType,
)
from backend.app.ai_providers.secret_store import SecretStore


class AIProviderSettingsStoreError(Exception):
    """Raised when settings storage or retrieval operations fail."""

    pass


class AIProviderSettingsStore:
    """Handles persistence of AIProviderSettings.

    Separates non-secret config fields (persisted to a JSON file) from API keys
    (stored via the SecretStore abstraction).
    """

    def __init__(self, *, file_path: Path, secret_store: SecretStore) -> None:
        """Initialize with target settings file path and secret store backend.

        Args:
            file_path: pathlib.Path target to the JSON settings file.
            secret_store: SecretStore backend for secure credential storage.

        Raises:
            TypeError: If arguments are of incorrect type.
        """
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be a pathlib.Path instance")
        if not isinstance(secret_store, SecretStore):
            raise TypeError("secret_store must be a SecretStore instance")

        self.file_path = file_path
        self.secret_store = secret_store

    def exists(self) -> bool:
        """Check if the persisted settings file exists.

        Returns:
            True if the JSON file exists, False otherwise.
        """
        return self.file_path.exists()

    def save(self, *, settings: AIProviderSettings) -> None:
        """Persist settings to disk and API keys to the secret store.

        Cleans up stale secrets from any removed provider configurations or
        configurations whose API key changed to None.

        Args:
            settings: The AIProviderSettings instance to persist.

        Raises:
            AIProviderSettingsStoreError: If saving fails or secret store errors occur.
            TypeError: If settings is not an AIProviderSettings instance.
        """
        if not isinstance(settings, AIProviderSettings):
            raise TypeError("settings must be an AIProviderSettings instance")

        try:
            # Create parent directory if missing
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise AIProviderSettingsStoreError("Failed to create parent directory") from exc

        # Identify previously persisted configurations for stale secret cleanup
        old_config_ids: Set[str] = set()
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                if isinstance(old_data, dict) and "providers" in old_data:
                    for p in old_data["providers"]:
                        if isinstance(p, dict) and "config_id" in p:
                            old_config_ids.add(p["config_id"])
            except Exception:
                # Ignore read failures here; clean up using best effort
                pass

        # Identify newly saved configurations
        new_config_ids = {p.config_id for p in settings.providers}
        removed_ids = old_config_ids - new_config_ids

        # Perform SecretStore updates (wrapping them inside the store error boundary)
        try:
            # 1. Delete secrets of removed configurations
            for rid in removed_ids:
                self.secret_store.delete_secret(config_id=rid)

            # 2. Update new provider secrets
            for provider in settings.providers:
                if provider.provider_type == AIProviderType.OPENAI_COMPATIBLE:
                    if provider.api_key is not None:
                        self.secret_store.set_secret(
                            config_id=provider.config_id, secret=provider.api_key
                        )
                    else:
                        # Stale secret cleanup: key changed to None
                        self.secret_store.delete_secret(config_id=provider.config_id)
        except Exception as exc:
            raise AIProviderSettingsStoreError("SecretStore failure during save") from exc

        # Construct JSON data excluding raw API keys
        try:
            data = settings.model_dump(mode="json")
            for provider in data.get("providers", []):
                if provider.get("provider_type") == "openai_compatible":
                    provider["api_key"] = None
        except Exception as exc:
            raise AIProviderSettingsStoreError("Failed to serialize settings data") from exc

        # Write atomically using a temporary file in the same directory
        dir_name = self.file_path.parent
        temp_fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as tmp:
                json.dump(data, tmp, indent=2)
            os.replace(temp_path, self.file_path)
        except Exception as exc:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise AIProviderSettingsStoreError("Failed to save settings atomically") from exc

    def load(self) -> AIProviderSettings:
        """Load and reconstruct settings from disk, fetching secrets from the store.

        Returns:
            A fully reconstructed and validated AIProviderSettings instance.

        Raises:
            AIProviderSettingsStoreError: If loading, parsing, secret fetching,
                or validation fails.
        """
        # Read and parse JSON file
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise AIProviderSettingsStoreError("Settings file not found") from exc
        except json.JSONDecodeError as exc:
            raise AIProviderSettingsStoreError("Settings file is not valid JSON") from exc
        except Exception as exc:
            raise AIProviderSettingsStoreError("Failed to read settings file") from exc

        if not isinstance(data, dict):
            raise AIProviderSettingsStoreError("Settings JSON is not an object/dict")

        # Re-inject secrets from the secret store
        try:
            for provider in data.get("providers", []):
                if not isinstance(provider, dict):
                    continue
                if provider.get("provider_type") == "openai_compatible":
                    config_id = provider.get("config_id")
                    if not config_id:
                        continue
                    # Retrieve the secret (missing secret gets None)
                    api_key = self.secret_store.get_secret(config_id=config_id)
                    provider["api_key"] = api_key
        except Exception as exc:
            raise AIProviderSettingsStoreError("SecretStore failure during load") from exc

        # Reconstruct and validate
        try:
            return AIProviderSettings.model_validate(data)
        except Exception as exc:
            raise AIProviderSettingsStoreError(
                "Failed to reconstruct AIProviderSettings from loaded data"
            ) from exc

    def delete(self) -> None:
        """Delete the settings file and all associated secrets.

        If the file does not exist, behaves safely and does not raise an error.

        Raises:
            AIProviderSettingsStoreError: If deletion fails.
        """
        if not self.file_path.exists():
            return

        # Attempt to load provider configs to delete their secrets
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "providers" in data:
                for p in data["providers"]:
                    if isinstance(p, dict) and "config_id" in p:
                        self.secret_store.delete_secret(config_id=p["config_id"])
        except Exception:
            # If load fails (e.g. corrupted JSON), clean up file anyway; best effort for secrets
            pass

        try:
            self.file_path.unlink()
        except Exception as exc:
            raise AIProviderSettingsStoreError("Failed to delete settings file") from exc
