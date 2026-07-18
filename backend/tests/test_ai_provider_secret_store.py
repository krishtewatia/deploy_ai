"""Tests for secure secret store implementations (Stage 8F)."""

from __future__ import annotations

import pytest
from unittest.mock import patch

import keyring
import keyring.errors

from backend.app.ai_providers.secret_store import (
    SecretStore,
    SecretStoreError,
    InMemorySecretStore,
    SystemSecretStore,
)


class TestSecretStoreContractAndInMemory:
    """Tests 1-13: Abstract class contract and InMemorySecretStore implementation."""

    def test_secret_store_is_abstract(self):
        """Test 1: SecretStore is an abstract base class and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SecretStore()  # type: ignore

    def test_in_memory_set_get_works(self):
        """Test 2: InMemorySecretStore set and get works."""
        store = InMemorySecretStore()
        store.set_secret(config_id="openai_1", secret="sk-test-123")
        assert store.get_secret(config_id="openai_1") == "sk-test-123"

    def test_in_memory_delete_works(self):
        """Test 3: InMemorySecretStore delete removes the stored secret."""
        store = InMemorySecretStore()
        store.set_secret(config_id="openai_1", secret="sk-test-123")
        assert store.get_secret(config_id="openai_1") == "sk-test-123"
        store.delete_secret(config_id="openai_1")
        assert store.get_secret(config_id="openai_1") is None

    def test_missing_secret_returns_none(self):
        """Test 4: Requesting a missing secret returns None."""
        store = InMemorySecretStore()
        assert store.get_secret(config_id="non_existent") is None

    def test_delete_missing_secret_is_safe(self):
        """Test 5: Deleting a missing secret is safe and does not raise an error."""
        store = InMemorySecretStore()
        # Should not raise any exception
        store.delete_secret(config_id="non_existent")

    def test_empty_config_id_rejected(self):
        """Test 6: Empty config_id is rejected."""
        store = InMemorySecretStore()
        with pytest.raises(ValueError, match="config_id cannot be empty"):
            store.set_secret(config_id="", secret="sk-test")
        with pytest.raises(ValueError, match="config_id cannot be empty"):
            store.get_secret(config_id="")
        with pytest.raises(ValueError, match="config_id cannot be empty"):
            store.delete_secret(config_id="")

    def test_whitespace_only_config_id_rejected(self):
        """Test 7: Whitespace-only config_id is rejected."""
        store = InMemorySecretStore()
        with pytest.raises(ValueError, match="config_id cannot be empty or whitespace-only"):
            store.set_secret(config_id="   ", secret="sk-test")
        with pytest.raises(ValueError, match="config_id cannot be empty or whitespace-only"):
            store.get_secret(config_id=" \t\n ")
        with pytest.raises(ValueError, match="config_id cannot be empty or whitespace-only"):
            store.delete_secret(config_id="   ")

    def test_non_string_config_id_rejected(self):
        """Test 8: Non-string config_id is rejected with TypeError."""
        store = InMemorySecretStore()
        with pytest.raises(TypeError, match="config_id must be a string"):
            store.set_secret(config_id=123, secret="sk-test")  # type: ignore
        with pytest.raises(TypeError, match="config_id must be a string"):
            store.get_secret(config_id=None)  # type: ignore
        with pytest.raises(TypeError, match="config_id must be a string"):
            store.delete_secret(config_id=[])  # type: ignore

    def test_empty_secret_rejected(self):
        """Test 9: Empty secret is rejected."""
        store = InMemorySecretStore()
        with pytest.raises(ValueError, match="secret cannot be empty"):
            store.set_secret(config_id="openai_1", secret="")

    def test_whitespace_only_secret_rejected(self):
        """Test 10: Whitespace-only secret is rejected."""
        store = InMemorySecretStore()
        with pytest.raises(ValueError, match="secret cannot be whitespace-only"):
            store.set_secret(config_id="openai_1", secret="   ")

    def test_non_string_secret_rejected(self):
        """Test 11: Non-string secret is rejected with TypeError."""
        store = InMemorySecretStore()
        with pytest.raises(TypeError, match="secret must be a string"):
            store.set_secret(config_id="openai_1", secret=123.45)  # type: ignore

    def test_valid_secret_is_preserved_exactly(self):
        """Test 12: Valid secret (even with leading/trailing spaces if it is not just whitespace) is preserved exactly."""
        store = InMemorySecretStore()
        secret_with_spaces = "  sk-test-with-padding  "
        store.set_secret(config_id="openai_1", secret=secret_with_spaces)
        assert store.get_secret(config_id="openai_1") == secret_with_spaces

    def test_two_in_memory_secret_store_instances_do_not_share_state(self):
        """Test 13: Different InMemorySecretStore instances do not share state."""
        store1 = InMemorySecretStore()
        store2 = InMemorySecretStore()
        store1.set_secret(config_id="id", secret="sec1")
        store2.set_secret(config_id="id", secret="sec2")
        assert store1.get_secret(config_id="id") == "sec1"
        assert store2.get_secret(config_id="id") == "sec2"


class TestSystemSecretStore:
    """Tests 14-20: SystemSecretStore implementation mocking the keyring package."""

    @patch("keyring.set_password")
    def test_system_secret_store_calls_keyring_set_password_correctly(self, mock_set):
        """Test 14: SystemSecretStore.set_secret calls keyring.set_password with correct args."""
        store = SystemSecretStore()
        store.set_secret(config_id="openai_1", secret="sk-secret")
        mock_set.assert_called_once_with("DeployAI", "openai_1", "sk-secret")

    @patch("keyring.get_password")
    def test_system_secret_store_calls_keyring_get_password_correctly(self, mock_get):
        """Test 15: SystemSecretStore.get_secret calls keyring.get_password with correct args."""
        mock_get.return_value = "sk-secret-retrieved"
        store = SystemSecretStore()
        val = store.get_secret(config_id="openai_1")
        mock_get.assert_called_once_with("DeployAI", "openai_1")
        assert val == "sk-secret-retrieved"

    @patch("keyring.delete_password")
    def test_system_secret_store_calls_keyring_delete_password_correctly(self, mock_delete):
        """Test 16: SystemSecretStore.delete_secret calls keyring.delete_password with correct args."""
        store = SystemSecretStore()
        store.delete_secret(config_id="openai_1")
        mock_delete.assert_called_once_with("DeployAI", "openai_1")

    @patch("keyring.delete_password")
    def test_system_secret_store_delete_missing_secret_is_safe(self, mock_delete):
        """Test 16 (continued): delete_secret behaves safely when keyring raises PasswordDeleteError."""
        mock_delete.side_effect = keyring.errors.PasswordDeleteError("Password not found")
        store = SystemSecretStore()
        # Should not raise any error
        store.delete_secret(config_id="non_existent")

    @patch("keyring.set_password")
    def test_keyring_set_failure_wrapped_in_secret_store_error(self, mock_set):
        """Test 17: Errors raised by keyring.set_password are wrapped in SecretStoreError."""
        mock_set.side_effect = RuntimeError("OS keyring error")
        store = SystemSecretStore()
        with pytest.raises(SecretStoreError) as exc_info:
            store.set_secret(config_id="openai_1", secret="sk-secret")
        assert exc_info.value.__cause__ is not None
        assert "OS keyring error" in str(exc_info.value.__cause__)

    @patch("keyring.get_password")
    def test_keyring_get_failure_wrapped_in_secret_store_error(self, mock_get):
        """Test 18: Errors raised by keyring.get_password are wrapped in SecretStoreError."""
        mock_get.side_effect = RuntimeError("OS keyring read error")
        store = SystemSecretStore()
        with pytest.raises(SecretStoreError) as exc_info:
            store.get_secret(config_id="openai_1")
        assert exc_info.value.__cause__ is not None
        assert "OS keyring read error" in str(exc_info.value.__cause__)

    @patch("keyring.delete_password")
    def test_keyring_delete_failure_wrapped_in_secret_store_error(self, mock_delete):
        """Test 19: Non-delete errors raised by keyring.delete_password are wrapped in SecretStoreError."""
        mock_delete.side_effect = RuntimeError("OS keyring write error")
        store = SystemSecretStore()
        with pytest.raises(SecretStoreError) as exc_info:
            store.delete_secret(config_id="openai_1")
        assert exc_info.value.__cause__ is not None
        assert "OS keyring write error" in str(exc_info.value.__cause__)

    @patch("keyring.set_password")
    def test_secret_never_appears_in_wrapped_error_messages(self, mock_set):
        """Test 20: The raw secret value never appears in SecretStoreError exception messages."""
        secret_val = "sk-sensitive-api-key-value-123"
        mock_set.side_effect = RuntimeError("Keyring error occurred")
        store = SystemSecretStore()
        with pytest.raises(SecretStoreError) as exc_info:
            store.set_secret(config_id="openai_1", secret=secret_val)
        assert secret_val not in str(exc_info.value)
