"""Secret store implementations for DeployAI.

Stage 8F introduces the SecretStore abstraction to securely manage API keys.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional
import keyring
import keyring.errors


class SecretStoreError(Exception):
    """Raised when secret storage operations fail."""

    pass


class SecretStore(abc.ABC):
    """Abstract contract for storing and retrieving provider API keys/secrets.

    The interface is strictly separate from configuration models.
    """

    @abc.abstractmethod
    def set_secret(
        self,
        *,
        config_id: str,
        secret: str,
    ) -> None:
        """Securely store an API key for a provider config_id.

        Args:
            config_id: Non-empty, non-whitespace-only string identifier.
            secret: Non-empty, non-whitespace-only string secret value.

        Raises:
            SecretStoreError: If storage fails.
            TypeError/ValueError: For invalid arguments.
        """
        ...

    @abc.abstractmethod
    def get_secret(
        self,
        *,
        config_id: str,
    ) -> Optional[str]:
        """Retrieve a stored API key for a provider config_id.

        Args:
            config_id: Non-empty, non-whitespace-only string identifier.

        Returns:
            The stored secret string or None if not found.

        Raises:
            SecretStoreError: If retrieval fails.
            TypeError/ValueError: For invalid arguments.
        """
        ...

    @abc.abstractmethod
    def delete_secret(
        self,
        *,
        config_id: str,
    ) -> None:
        """Delete a stored API key for a provider config_id.

        Deleting a missing key is safe and does not raise an error.

        Args:
            config_id: Non-empty, non-whitespace-only string identifier.

        Raises:
            SecretStoreError: If deletion fails.
            TypeError/ValueError: For invalid arguments.
        """
        ...


def _validate_config_id(config_id: Any) -> str:
    """Validate and clean config_id.

    Args:
        config_id: The config identifier.

    Returns:
        The stripped config_id string.

    Raises:
        TypeError: If config_id is not a string.
        ValueError: If config_id is empty or whitespace-only.
    """
    if not isinstance(config_id, str):
        raise TypeError("config_id must be a string")
    stripped = config_id.strip()
    if not stripped:
        raise ValueError("config_id cannot be empty or whitespace-only")
    return stripped


def _validate_secret(secret: Any) -> str:
    """Validate secret string.

    Args:
        secret: The secret string.

    Returns:
        The validated secret string.

    Raises:
        TypeError: If secret is not a string.
        ValueError: If secret is empty or whitespace-only.
    """
    if not isinstance(secret, str):
        raise TypeError("secret must be a string")
    # Preserve the original valid secret value exactly, do not strip it!
    # But check if it is empty or whitespace-only.
    if not secret:
        raise ValueError("secret cannot be empty")
    if not secret.strip():
        raise ValueError("secret cannot be whitespace-only")
    return secret


class InMemorySecretStore(SecretStore):
    """In-memory secret store implementation for tests and dev/in-memory use.

    State is isolated per instance.
    """

    def __init__(self) -> None:
        self._secrets: Dict[str, str] = {}

    def set_secret(
        self,
        *,
        config_id: str,
        secret: str,
    ) -> None:
        cleaned_id = _validate_config_id(config_id)
        validated_secret = _validate_secret(secret)
        self._secrets[cleaned_id] = validated_secret

    def get_secret(
        self,
        *,
        config_id: str,
    ) -> Optional[str]:
        cleaned_id = _validate_config_id(config_id)
        return self._secrets.get(cleaned_id)

    def delete_secret(
        self,
        *,
        config_id: str,
    ) -> None:
        cleaned_id = _validate_config_id(config_id)
        self._secrets.pop(cleaned_id, None)


class SystemSecretStore(SecretStore):
    """Production secret store implementing OS-level keyring storage via keyring."""

    SERVICE_NAME = "DeployAI"

    def set_secret(
        self,
        *,
        config_id: str,
        secret: str,
    ) -> None:
        cleaned_id = _validate_config_id(config_id)
        validated_secret = _validate_secret(secret)
        try:
            keyring.set_password(self.SERVICE_NAME, cleaned_id, validated_secret)
        except Exception as exc:
            # Mask sensitive values in error messages.
            raise SecretStoreError("SystemSecretStore failed to set secret") from exc

    def get_secret(
        self,
        *,
        config_id: str,
    ) -> Optional[str]:
        cleaned_id = _validate_config_id(config_id)
        try:
            return keyring.get_password(self.SERVICE_NAME, cleaned_id)
        except Exception as exc:
            raise SecretStoreError("SystemSecretStore failed to get secret") from exc

    def delete_secret(
        self,
        *,
        config_id: str,
    ) -> None:
        cleaned_id = _validate_config_id(config_id)
        try:
            keyring.delete_password(self.SERVICE_NAME, cleaned_id)
        except keyring.errors.PasswordDeleteError:
            # safe deletion when credentials are not found
            pass
        except Exception as exc:
            raise SecretStoreError("SystemSecretStore failed to delete secret") from exc
