"""Tests for AI provider abstract contract (Stage 7D2)."""

import pytest

from backend.app.ai_planning.providers.base import AIProvider, AIProviderError


# ── Fake Provider for Testing ──────────────────────────────────────────


class FakeAIProvider(AIProvider):
    """Concrete test provider that returns a canned response."""

    def __init__(self, response: str = "{}") -> None:
        self._response = response

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._response


class FailingAIProvider(AIProvider):
    """Provider that always raises AIProviderError."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise AIProviderError("Provider failure simulated.")


# ── Tests ──────────────────────────────────────────────────────────────


class TestAIProviderContract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AIProvider()

    def test_fake_provider_returns_response(self):
        provider = FakeAIProvider(response='{"key": "value"}')
        result = provider.generate(system_prompt="sys", user_prompt="usr")
        assert result == '{"key": "value"}'

    def test_fake_provider_default_response(self):
        provider = FakeAIProvider()
        result = provider.generate(system_prompt="s", user_prompt="u")
        assert result == "{}"

    def test_failing_provider_raises_error(self):
        provider = FailingAIProvider()
        with pytest.raises(AIProviderError, match="Provider failure simulated"):
            provider.generate(system_prompt="s", user_prompt="u")

    def test_provider_error_is_exception(self):
        err = AIProviderError("test message")
        assert isinstance(err, Exception)
        assert str(err) == "test message"

    def test_fake_provider_is_instance_of_abstract(self):
        provider = FakeAIProvider()
        assert isinstance(provider, AIProvider)

    def test_generate_uses_keyword_only_args(self):
        provider = FakeAIProvider(response="ok")
        # Must use keyword args
        with pytest.raises(TypeError):
            provider.generate("sys", "usr")

    def test_provider_subclass_must_implement_generate(self):
        class IncompleteProvider(AIProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()
