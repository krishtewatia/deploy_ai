"""Tests for OpenAI-compatible AI provider implementation (Stage 8D)."""

import pytest
from unittest.mock import patch, MagicMock
import httpx

from backend.app.ai_planning.providers.base import AIProvider, AIProviderError
from backend.app.ai_providers.schemas import (
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
    AIProviderSettings,
    AIProviderStatus,
)
from backend.app.ai_providers.factory import (
    AIProviderFactory,
    OllamaAIProvider,
    OpenAICompatibleAIProvider,
)


def _make_openai_config(
    *,
    api_key: str | None = "sk-test-secret-key-12345",
    base_url: str = "https://api.example.com/v1",
    model_name: str = "gpt-4",
    request_timeout_seconds: float = 60.0,
) -> OpenAICompatibleProviderConfig:
    return OpenAICompatibleProviderConfig(
        config_id="openai_remote",
        display_name="OpenAI Remote",
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        request_timeout_seconds=request_timeout_seconds,
    )


def _make_openai_success_response(content: str = "AI response here") -> dict:
    """Return a valid OpenAI-compatible chat completion response shape."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _make_mock_response(
    status_code: int = 200,
    json_data: object = None,
    json_raises: Exception | None = None,
    raise_for_status_exc: Exception | None = None,
) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if json_raises:
        mock_resp.json.side_effect = json_raises
    else:
        mock_resp.json.return_value = json_data
    if raise_for_status_exc:
        mock_resp.raise_for_status.side_effect = raise_for_status_exc
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ── Input Validation Tests (1-7) ────────────────────────────────────────


class TestOpenAICompatibleInputValidation:
    """Test 1-7: prompt input validation."""

    def test_valid_prompts_accepted(self):
        """Test 1: Valid system and user prompts accepted."""
        config = _make_openai_config()
        provider = OpenAICompatibleAIProvider(config)
        mock_resp = _make_mock_response(json_data=_make_openai_success_response("OK"))
        with patch("httpx.post", return_value=mock_resp):
            res = provider.generate(system_prompt="Valid sys", user_prompt="Valid user")
            assert res == "OK"

    def test_empty_system_prompt_rejected(self):
        """Test 2: Empty system prompt rejected."""
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="system_prompt cannot be empty"):
            provider.generate(system_prompt="", user_prompt="user")

    def test_whitespace_only_system_prompt_rejected(self):
        """Test 3: Whitespace-only system prompt rejected."""
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="system_prompt cannot be empty"):
            provider.generate(system_prompt="   \n   ", user_prompt="user")

    def test_non_string_system_prompt_rejected(self):
        """Test 4: Non-string system prompt rejected."""
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="system_prompt must be a string"):
            provider.generate(system_prompt=123, user_prompt="user")  # type: ignore

    def test_empty_user_prompt_rejected(self):
        """Test 5: Empty user prompt rejected."""
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="user_prompt cannot be empty"):
            provider.generate(system_prompt="sys", user_prompt="")

    def test_whitespace_only_user_prompt_rejected(self):
        """Test 6: Whitespace-only user prompt rejected."""
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="user_prompt cannot be empty"):
            provider.generate(system_prompt="sys", user_prompt="   \t   ")

    def test_non_string_user_prompt_rejected(self):
        """Test 7: Non-string user prompt rejected."""
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="user_prompt must be a string"):
            provider.generate(system_prompt="sys", user_prompt=None)  # type: ignore


# ── Request Construction Tests (8-14) ───────────────────────────────────


class TestOpenAICompatibleRequestConstruction:
    """Test 8-14: HTTP request payload and endpoint construction."""

    @patch("httpx.post")
    def test_correct_endpoint_used(self, mock_post: MagicMock):
        """Test 8: Correct endpoint is used."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(base_url="https://api.example.com/v1")
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")

        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.example.com/v1/chat/completions"

    @patch("httpx.post")
    def test_correct_model_name_sent(self, mock_post: MagicMock):
        """Test 9: Correct model name is sent."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(model_name="mixtral-8x7b")
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "mixtral-8x7b"

    @patch("httpx.post")
    def test_system_message_included(self, mock_post: MagicMock):
        """Test 10: System message is included correctly."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        provider.generate(system_prompt="System Prompt Text", user_prompt="User Prompt Text")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["messages"][0] == {"role": "system", "content": "System Prompt Text"}

    @patch("httpx.post")
    def test_user_message_included(self, mock_post: MagicMock):
        """Test 11: User message is included correctly."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        provider.generate(system_prompt="System Prompt Text", user_prompt="User Prompt Text")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["messages"][1] == {"role": "user", "content": "User Prompt Text"}

    @patch("httpx.post")
    def test_stream_is_false(self, mock_post: MagicMock):
        """Test 12: stream is false."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        provider.generate(system_prompt="sys", user_prompt="user")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["stream"] is False

    @patch("httpx.post")
    def test_configured_timeout_passed(self, mock_post: MagicMock):
        """Test 13: Configured timeout is passed exactly."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(request_timeout_seconds=45.0)
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")

        assert mock_post.call_args.kwargs["timeout"] == 45.0

    @patch("httpx.post")
    def test_prompts_transmitted_without_mutation(self, mock_post: MagicMock):
        """Test 14: Valid prompts are transmitted without mutation."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        provider = OpenAICompatibleAIProvider(_make_openai_config())

        original_sys = "  System with   spaces  "
        original_usr = "  User with   spaces  "
        provider.generate(system_prompt=original_sys, user_prompt=original_usr)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["messages"][0]["content"] == original_sys
        assert payload["messages"][1]["content"] == original_usr


# ── Authentication Tests (15-18) ────────────────────────────────────────


class TestOpenAICompatibleAuthentication:
    """Test 15-18: Authorization header behavior."""

    @patch("httpx.post")
    def test_authorization_header_sent_with_api_key(self, mock_post: MagicMock):
        """Test 15: Authorization header sent when api_key exists."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(api_key="sk-my-key")
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")

        headers = mock_post.call_args.kwargs["headers"]
        assert "Authorization" in headers

    @patch("httpx.post")
    def test_authorization_header_uses_bearer_format(self, mock_post: MagicMock):
        """Test 16: Authorization header uses Bearer format."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(api_key="sk-my-key")
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-my-key"

    @patch("httpx.post")
    def test_authorization_header_omitted_when_no_api_key(self, mock_post: MagicMock):
        """Test 17: Authorization header omitted when api_key is None."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(api_key=None)
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")

        headers = mock_post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    @patch("httpx.post")
    def test_api_key_not_mutated(self, mock_post: MagicMock):
        """Test 18: API key is not mutated."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(api_key="sk-original-key-789")
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")

        assert config.api_key == "sk-original-key-789"


# ── Success Response Tests (19-22) ──────────────────────────────────────


class TestOpenAICompatibleSuccessResponse:
    """Test 19-22: successful response content extraction."""

    @patch("httpx.post")
    def test_valid_response_content_returned(self, mock_post: MagicMock):
        """Test 19: Valid response content returned."""
        mock_post.return_value = _make_mock_response(
            json_data=_make_openai_success_response("Generated assistant content")
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == "Generated assistant content"

    @patch("httpx.post")
    def test_surrounding_whitespace_stripped(self, mock_post: MagicMock):
        """Test 20: Surrounding whitespace in returned content is stripped."""
        mock_post.return_value = _make_mock_response(
            json_data=_make_openai_success_response("   Padded content   ")
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == "Padded content"

    @patch("httpx.post")
    def test_json_looking_content_returned_as_raw_string(self, mock_post: MagicMock):
        """Test 21: JSON-looking AI content is returned as raw string."""
        json_str = '{"proposal_set_id": "ps_1", "summary": "Looks good"}'
        mock_post.return_value = _make_mock_response(
            json_data=_make_openai_success_response(json_str)
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == json_str

    @patch("httpx.post")
    def test_markdown_fenced_content_returned_as_raw_string(self, mock_post: MagicMock):
        """Test 22: Markdown-fenced content is returned as raw string."""
        fence_str = "```json\n{'folds': 3}\n```"
        mock_post.return_value = _make_mock_response(
            json_data=_make_openai_success_response(fence_str)
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == fence_str


# ── Malformed Response Tests (23-33) ────────────────────────────────────


class TestOpenAICompatibleMalformedResponses:
    """Test 23-33: defensive response validation."""

    @patch("httpx.post")
    def test_non_dictionary_top_level_rejected(self, mock_post: MagicMock):
        """Test 23: Non-dictionary top-level JSON rejected."""
        mock_post.return_value = _make_mock_response(json_data=[{"item": "val"}])
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="Top-level JSON value is not an object"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_missing_choices_rejected(self, mock_post: MagicMock):
        """Test 24: Missing choices rejected."""
        mock_post.return_value = _make_mock_response(json_data={"id": "abc"})
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="Response JSON is missing 'choices' key"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_choices_not_a_list_rejected(self, mock_post: MagicMock):
        """Test 25: choices not a list rejected."""
        mock_post.return_value = _make_mock_response(json_data={"choices": "not_a_list"})
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'choices' is not a list"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_empty_choices_rejected(self, mock_post: MagicMock):
        """Test 26: Empty choices rejected."""
        mock_post.return_value = _make_mock_response(json_data={"choices": []})
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'choices' list is empty"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_first_choice_not_a_dictionary_rejected(self, mock_post: MagicMock):
        """Test 27: First choice not a dictionary rejected."""
        mock_post.return_value = _make_mock_response(json_data={"choices": ["not_a_dict"]})
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="First choice is not an object/dict"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_missing_message_rejected(self, mock_post: MagicMock):
        """Test 28: Missing message rejected."""
        mock_post.return_value = _make_mock_response(
            json_data={"choices": [{"index": 0}]}
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="First choice is missing 'message' key"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_message_not_a_dictionary_rejected(self, mock_post: MagicMock):
        """Test 29: message not a dictionary rejected."""
        mock_post.return_value = _make_mock_response(
            json_data={"choices": [{"message": "not_a_dict"}]}
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'message' is not an object/dict"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_missing_content_rejected(self, mock_post: MagicMock):
        """Test 30: Missing content rejected."""
        mock_post.return_value = _make_mock_response(
            json_data={"choices": [{"message": {"role": "assistant"}}]}
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="Response message is missing 'content' key"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_content_not_a_string_rejected(self, mock_post: MagicMock):
        """Test 31: content not a string rejected."""
        mock_post.return_value = _make_mock_response(
            json_data={"choices": [{"message": {"role": "assistant", "content": 12345}}]}
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'content' must be a string"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_empty_content_rejected(self, mock_post: MagicMock):
        """Test 32: Empty content rejected."""
        mock_post.return_value = _make_mock_response(
            json_data=_make_openai_success_response("")
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'content' cannot be empty or whitespace-only"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_whitespace_only_content_rejected(self, mock_post: MagicMock):
        """Test 33: Whitespace-only content rejected."""
        mock_post.return_value = _make_mock_response(
            json_data=_make_openai_success_response("   \n   ")
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'content' cannot be empty or whitespace-only"):
            provider.generate(system_prompt="sys", user_prompt="user")


# ── Network / HTTP Error Tests (34-41) ──────────────────────────────────


class TestOpenAICompatibleNetworkErrors:
    """Test 34-41: transport and HTTP status error handling."""

    @patch("httpx.post")
    def test_timeout_wrapped_as_ai_provider_error(self, mock_post: MagicMock):
        """Test 34: Timeout wrapped as AIProviderError."""
        mock_post.side_effect = httpx.TimeoutException("Read timed out")
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="request timed out"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_connection_error_wrapped_as_ai_provider_error(self, mock_post: MagicMock):
        """Test 35: Connection error wrapped as AIProviderError."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="Failed to connect to OpenAI-compatible provider"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_protocol_error_wrapped_as_ai_provider_error(self, mock_post: MagicMock):
        """Test 36: Protocol/request error wrapped as AIProviderError."""
        mock_post.side_effect = httpx.RequestError("Some protocol error")
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="General request error contacting OpenAI-compatible provider"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_http_400_wrapped_as_ai_provider_error(self, mock_post: MagicMock):
        """Test 37: HTTP 400 wrapped as AIProviderError."""
        req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        resp = httpx.Response(400, request=req)
        mock_post.side_effect = httpx.HTTPStatusError("Bad Request", request=req, response=resp)
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="returned HTTP 400"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_http_401_wrapped_as_ai_provider_error(self, mock_post: MagicMock):
        """Test 38: HTTP 401 wrapped as AIProviderError."""
        req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        resp = httpx.Response(401, request=req)
        mock_post.side_effect = httpx.HTTPStatusError("Unauthorized", request=req, response=resp)
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="returned HTTP 401"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_http_404_wrapped_as_ai_provider_error(self, mock_post: MagicMock):
        """Test 39: HTTP 404 wrapped as AIProviderError."""
        req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        resp = httpx.Response(404, request=req)
        mock_post.side_effect = httpx.HTTPStatusError("Not Found", request=req, response=resp)
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="returned HTTP 404"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_http_500_wrapped_as_ai_provider_error(self, mock_post: MagicMock):
        """Test 40: HTTP 500 wrapped as AIProviderError."""
        req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        resp = httpx.Response(500, request=req)
        mock_post.side_effect = httpx.HTTPStatusError("Server Error", request=req, response=resp)
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="returned HTTP 500"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_original_exception_preserved_as_cause(self, mock_post: MagicMock):
        """Test 41: Original exception is preserved as __cause__."""
        original_exc = httpx.ConnectError("Connection refused")
        mock_post.side_effect = original_exc
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError) as exc_info:
            provider.generate(system_prompt="sys", user_prompt="user")
        assert exc_info.value.__cause__ is original_exc


# ── Configuration Tests (42-46) ─────────────────────────────────────────


class TestOpenAICompatibleConfiguration:
    """Test 42-46: configuration immutability and preservation."""

    def test_exact_config_object_preserved(self):
        """Test 42: Exact config object is preserved."""
        config = _make_openai_config()
        provider = OpenAICompatibleAIProvider(config)
        assert provider.config is config

    @patch("httpx.post")
    def test_base_url_not_mutated(self, mock_post: MagicMock):
        """Test 43: base_url is not mutated."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(base_url="https://custom.server.com/v1")
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")
        assert config.base_url == "https://custom.server.com/v1"

    @patch("httpx.post")
    def test_model_name_not_mutated(self, mock_post: MagicMock):
        """Test 44: model_name is not mutated."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(model_name="my-model-v2")
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")
        assert config.model_name == "my-model-v2"

    @patch("httpx.post")
    def test_timeout_not_mutated(self, mock_post: MagicMock):
        """Test 45: timeout is not mutated."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(request_timeout_seconds=75.0)
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")
        assert config.request_timeout_seconds == 75.0

    @patch("httpx.post")
    def test_api_key_not_mutated_after_generation(self, mock_post: MagicMock):
        """Test 46: api_key is not mutated."""
        mock_post.return_value = _make_mock_response(json_data=_make_openai_success_response())
        config = _make_openai_config(api_key="sk-immutable-key")
        original_dict = config.model_dump()
        provider = OpenAICompatibleAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")
        assert config.model_dump() == original_dict
        assert config.api_key == "sk-immutable-key"


# ── Factory Integration Tests (47-50) ───────────────────────────────────


class TestOpenAICompatibleFactoryIntegration:
    """Test 47-50: factory and cross-provider integration."""

    def test_factory_creates_openai_compatible_provider(self):
        """Test 47: AIProviderFactory still creates OpenAICompatibleAIProvider."""
        config = _make_openai_config()
        provider = AIProviderFactory.create(config)
        assert isinstance(provider, OpenAICompatibleAIProvider)

    def test_created_provider_implements_ai_provider(self):
        """Test 48: Created provider implements AIProvider."""
        config = _make_openai_config()
        provider = AIProviderFactory.create(config)
        assert isinstance(provider, AIProvider)

    def test_ollama_provider_behavior_unchanged(self):
        """Test 49: Ollama provider behavior remains unchanged."""
        config = OllamaProviderConfig(
            config_id="ollama_local",
            display_name="Local Ollama",
            model_name="llama3.2",
            base_url="http://localhost:11434",
            request_timeout_seconds=90.0,
        )
        provider = AIProviderFactory.create(config)
        assert isinstance(provider, OllamaAIProvider)
        assert isinstance(provider, AIProvider)
        assert provider.config is config

    def test_active_provider_resolution_works(self):
        """Test 50: Active provider resolution still works."""
        config = _make_openai_config()
        settings = AIProviderSettings(
            status=AIProviderStatus.CONFIGURED,
            active_config_id="openai_remote",
            providers=[config],
        )
        provider = AIProviderFactory.create_active(settings)
        assert isinstance(provider, OpenAICompatibleAIProvider)
        assert provider.config is config


# ── Additional Coverage Tests ───────────────────────────────────────────


class TestOpenAICompatibleAdditionalCoverage:
    """Additional tests for full branch coverage and edge cases."""

    @patch("httpx.post")
    def test_invalid_json_response_wrapped(self, mock_post: MagicMock):
        """response.json() failure wrapped as AIProviderError with cause."""
        json_exc = ValueError("Expecting value: line 1 column 1 (char 0)")
        mock_post.return_value = _make_mock_response(json_raises=json_exc)
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="Response body is not valid JSON") as exc_info:
            provider.generate(system_prompt="sys", user_prompt="user")
        assert exc_info.value.__cause__ is json_exc

    @patch("httpx.post")
    def test_no_request_made_during_construction(self, mock_post: MagicMock):
        """No HTTP request occurs during provider construction."""
        _make_openai_config()
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        mock_post.assert_not_called()
        assert provider is not None

    @patch("httpx.post")
    def test_config_unmodified_after_failed_generation(self, mock_post: MagicMock):
        """Config is not modified even after a failed generation."""
        mock_post.side_effect = httpx.TimeoutException("timeout")
        config = _make_openai_config()
        original_dict = config.model_dump()
        provider = OpenAICompatibleAIProvider(config)
        with pytest.raises(AIProviderError):
            provider.generate(system_prompt="sys", user_prompt="user")
        assert config.model_dump() == original_dict

    @patch("httpx.post")
    def test_no_api_key_in_error_message(self, mock_post: MagicMock):
        """API key must not appear in error messages."""
        mock_post.side_effect = httpx.ConnectError("Failed to connect")
        config = _make_openai_config(api_key="sk-super-secret-99999")
        provider = OpenAICompatibleAIProvider(config)
        try:
            provider.generate(system_prompt="sys", user_prompt="user")
        except AIProviderError as exc:
            err_msg = str(exc)
            assert "sk-super-secret-99999" not in err_msg

    @patch("httpx.post")
    def test_raw_httpx_exception_does_not_escape(self, mock_post: MagicMock):
        """Raw httpx exceptions never escape the provider boundary."""
        mock_post.side_effect = httpx.ConnectTimeout("Connect timeout")
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        try:
            provider.generate(system_prompt="sys", user_prompt="user")
        except Exception as exc:
            assert isinstance(exc, AIProviderError)
            assert not isinstance(exc, httpx.HTTPError)

    @patch("httpx.post")
    def test_unexpected_exception_wrapped(self, mock_post: MagicMock):
        """Unexpected non-httpx exceptions also wrapped as AIProviderError."""
        mock_post.side_effect = RuntimeError("Crash")
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="Unexpected transport failure"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_choices_none_rejected(self, mock_post: MagicMock):
        """choices as None is rejected."""
        mock_post.return_value = _make_mock_response(json_data={"choices": None})
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'choices' is not a list"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_message_none_rejected(self, mock_post: MagicMock):
        """message as None is rejected."""
        mock_post.return_value = _make_mock_response(
            json_data={"choices": [{"message": None}]}
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'message' is not an object/dict"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_content_none_rejected(self, mock_post: MagicMock):
        """content as None is rejected."""
        mock_post.return_value = _make_mock_response(
            json_data={"choices": [{"message": {"role": "assistant", "content": None}}]}
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'content' must be a string"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_content_list_rejected(self, mock_post: MagicMock):
        """content as list is rejected."""
        mock_post.return_value = _make_mock_response(
            json_data={"choices": [{"message": {"role": "assistant", "content": ["text"]}}]}
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="'content' must be a string"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_multiline_content_preserved(self, mock_post: MagicMock):
        """Multiline content is preserved as-is."""
        multiline = "Line 1\nLine 2\nLine 3"
        mock_post.return_value = _make_mock_response(
            json_data=_make_openai_success_response(multiline)
        )
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == multiline

    @patch("httpx.post")
    def test_top_level_string_json_rejected(self, mock_post: MagicMock):
        """Top-level JSON string value rejected."""
        mock_post.return_value = _make_mock_response(json_data="just a string")
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="Top-level JSON value is not an object"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_timeout_cause_preserved(self, mock_post: MagicMock):
        """Timeout exception cause is preserved."""
        original_exc = httpx.ReadTimeout("Read timed out")
        mock_post.side_effect = original_exc
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError) as exc_info:
            provider.generate(system_prompt="sys", user_prompt="user")
        assert exc_info.value.__cause__ is original_exc

    @patch("httpx.post")
    def test_http_status_cause_preserved(self, mock_post: MagicMock):
        """HTTP status error cause is preserved."""
        req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        resp = httpx.Response(429, request=req)
        original_exc = httpx.HTTPStatusError("Too Many Requests", request=req, response=resp)
        mock_post.side_effect = original_exc
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError) as exc_info:
            provider.generate(system_prompt="sys", user_prompt="user")
        assert exc_info.value.__cause__ is original_exc

    def test_non_string_both_prompts_rejected_independently(self):
        """Non-string system_prompt is caught before user_prompt is checked."""
        provider = OpenAICompatibleAIProvider(_make_openai_config())
        with pytest.raises(AIProviderError, match="system_prompt must be a string"):
            provider.generate(system_prompt=42, user_prompt=42)  # type: ignore
