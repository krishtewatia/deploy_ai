"""Tests for backend.app.ai_engine.groq_client — targeting 100 % coverage.

Every test uses mocking; **no real Groq API calls are made**.

Covers:
* Successful response via injected mock client
* Missing API key raises ValueError
* Explicit api_key parameter works
* AuthenticationError  → GroqAuthenticationError
* APIConnectionError   → GroqNetworkError
* APIStatusError       → GroqAPIError
* Generic Exception    → GroqClientError
* None content fallback (empty string)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from groq import APIConnectionError, APIStatusError, AuthenticationError

from backend.app.ai_engine.groq_client import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    GroqAPIError,
    GroqAuthenticationError,
    GroqClient,
    GroqClientError,
    GroqNetworkError,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_mock_client(content: str | None = "mock response") -> MagicMock:
    """Return a ``MagicMock`` that mimics ``groq.Groq``."""
    mock = MagicMock()
    mock.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return mock


# ── Initialisation ──────────────────────────────────────────────────────────


class TestGroqClientInit:
    """Constructor and dependency-injection tests."""

    def test_injected_client_skips_key_resolution(self) -> None:
        mock = _make_mock_client()
        client = GroqClient(client=mock)
        # Should not raise, even without env var
        assert client._client is mock

    def test_missing_api_key_raises_value_error(self) -> None:
        with patch("backend.app.ai_engine.groq_client.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = ""
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                GroqClient()

    @patch("backend.app.ai_engine.groq_client.Groq")
    def test_explicit_api_key_parameter(self, mock_groq_cls: MagicMock) -> None:
        GroqClient(api_key="test-key-123")
        mock_groq_cls.assert_called_once_with(api_key="test-key-123")

    @patch("backend.app.ai_engine.groq_client.Groq")
    def test_settings_api_key(self, mock_groq_cls: MagicMock) -> None:
        with patch("backend.app.ai_engine.groq_client.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "settings-key-456"
            GroqClient()
            mock_groq_cls.assert_called_once_with(api_key="settings-key-456")


# ── Successful response ────────────────────────────────────────────────────


class TestGenerateRecommendationsSuccess:
    """Happy-path tests."""

    def test_returns_raw_text(self) -> None:
        mock = _make_mock_client(content='{"cleaning_plan": {}}')
        client = GroqClient(client=mock)

        result = client.generate_recommendations("test prompt")

        assert result == '{"cleaning_plan": {}}'
        mock.chat.completions.create.assert_called_once_with(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            messages=[{"role": "user", "content": "test prompt"}],
        )

    def test_none_content_returns_empty_string(self) -> None:
        mock = _make_mock_client(content=None)
        client = GroqClient(client=mock)

        result = client.generate_recommendations("test prompt")
        assert result == ""


# ── Error handling ──────────────────────────────────────────────────────────


class TestGenerateRecommendationsErrors:
    """Each Groq SDK exception maps to the correct custom exception."""

    def test_authentication_error(self) -> None:
        mock = _make_mock_client()
        mock.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body=None,
        )
        client = GroqClient(client=mock)

        with pytest.raises(GroqAuthenticationError, match="Authentication"):
            client.generate_recommendations("prompt")

    def test_network_error(self) -> None:
        mock = _make_mock_client()
        mock.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock(),
        )
        client = GroqClient(client=mock)

        with pytest.raises(GroqNetworkError, match="connect"):
            client.generate_recommendations("prompt")

    def test_api_status_error(self) -> None:
        mock = _make_mock_client()
        mock.chat.completions.create.side_effect = APIStatusError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        client = GroqClient(client=mock)

        with pytest.raises(GroqAPIError, match="API returned an error"):
            client.generate_recommendations("prompt")

    def test_unexpected_exception(self) -> None:
        mock = _make_mock_client()
        mock.chat.completions.create.side_effect = RuntimeError("boom")
        client = GroqClient(client=mock)

        with pytest.raises(GroqClientError, match="Unexpected error"):
            client.generate_recommendations("prompt")
