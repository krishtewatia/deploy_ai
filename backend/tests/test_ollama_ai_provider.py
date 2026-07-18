"""Tests for Ollama AI provider implementation (Stage 8C)."""

import pytest
from unittest.mock import patch, MagicMock
import httpx

from backend.app.ai_planning.providers.base import AIProvider, AIProviderError
from backend.app.ai_providers.schemas import (
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
)
from backend.app.ai_providers.factory import (
    AIProviderFactory,
    OllamaAIProvider,
    OpenAICompatibleAIProvider,
)


def _make_ollama_config() -> OllamaProviderConfig:
    return OllamaProviderConfig(
        config_id="ollama_local",
        display_name="Local Ollama",
        model_name="llama3.2",
        base_url="http://localhost:11434",
        request_timeout_seconds=90.0,
    )


def make_mock_response(status_code=200, json_data=None, json_raises=None, raise_for_status_exc=None):
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


class TestOllamaInputValidation:
    def test_valid_prompts_accepted(self):
        config = _make_ollama_config()
        provider = OllamaAIProvider(config)
        
        # Test input validation only (we mock httpx.post to succeed)
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": "OK"}})
        with patch("httpx.post", return_value=mock_resp):
            res = provider.generate(system_prompt="Valid sys", user_prompt="Valid user")
            assert res == "OK"

    def test_empty_system_prompt_rejected(self):
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="system_prompt cannot be empty"):
            provider.generate(system_prompt="", user_prompt="user")

    def test_whitespace_only_system_prompt_rejected(self):
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="system_prompt cannot be empty"):
            provider.generate(system_prompt="   \n   ", user_prompt="user")

    def test_non_string_system_prompt_rejected(self):
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="system_prompt must be a string"):
            provider.generate(system_prompt=123, user_prompt="user")  # type: ignore

    def test_empty_user_prompt_rejected(self):
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="user_prompt cannot be empty"):
            provider.generate(system_prompt="sys", user_prompt="")

    def test_whitespace_only_user_prompt_rejected(self):
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="user_prompt cannot be empty"):
            provider.generate(system_prompt="sys", user_prompt="   \t   ")

    def test_non_string_user_prompt_rejected(self):
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="user_prompt must be a string"):
            provider.generate(system_prompt="sys", user_prompt=None)  # type: ignore


# ── Request Construction Tests (8-16) ───────────────────────────────────


class TestOllamaRequestConstruction:
    @patch("httpx.post")
    def test_request_details(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": "OK"}})
        mock_post.return_value = mock_resp

        config = _make_ollama_config()
        provider = OllamaAIProvider(config)
        
        provider.generate(system_prompt="System Prompt Text", user_prompt="User Prompt Text")

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Endpoint check
        assert args[0] == "http://localhost:11434/api/chat"
        # Payload checks
        payload = kwargs["json"]
        assert payload["model"] == "llama3.2"
        assert payload["stream"] is False
        assert payload["messages"] == [
            {"role": "system", "content": "System Prompt Text"},
            {"role": "user", "content": "User Prompt Text"},
        ]
        # Timeout check
        assert kwargs["timeout"] == 90.0

    @patch("httpx.post")
    def test_no_request_before_generate_called(self, mock_post):
        config = _make_ollama_config()
        OllamaAIProvider(config)
        mock_post.assert_not_called()


# ── Success Response Tests (17-21) ──────────────────────────────────────


class TestOllamaSuccessResponse:
    @patch("httpx.post")
    def test_valid_response_extraction(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": "   Generated assistant content   "}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == "Generated assistant content"

    @patch("httpx.post")
    def test_multiline_content_preserved(self, mock_post):
        multiline = "Line 1\nLine 2\nLine 3"
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": multiline}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == multiline

    @patch("httpx.post")
    def test_json_looking_content_preserved_as_raw_text(self, mock_post):
        json_str = '{"proposal_set_id": "ps_1", "summary": "Looks good"}'
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": json_str}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == json_str

    @patch("httpx.post")
    def test_markdown_fenced_content_preserved_as_raw_text(self, mock_post):
        fence_str = "```json\n{'folds': 3}\n```"
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": fence_str}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        res = provider.generate(system_prompt="sys", user_prompt="user")
        assert res == fence_str


# ── Network / HTTP Error Tests (22-28) ──────────────────────────────────


class TestOllamaNetworkErrors:
    @patch("httpx.post")
    def test_connection_failure_handling(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Failed to connect to Ollama provider"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_timeout_handling(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("Read timed out")
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Ollama request timed out"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_http_400_handling(self, mock_post):
        req = httpx.Request("POST", "http://localhost/api/chat")
        resp = httpx.Response(400, request=req)
        mock_post.side_effect = httpx.HTTPStatusError("Bad Request", request=req, response=resp)

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Ollama provider returned HTTP 400"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_http_404_handling(self, mock_post):
        req = httpx.Request("POST", "http://localhost/api/chat")
        resp = httpx.Response(404, request=req)
        mock_post.side_effect = httpx.HTTPStatusError("Not Found", request=req, response=resp)

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Ollama provider returned HTTP 404"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_http_500_handling(self, mock_post):
        req = httpx.Request("POST", "http://localhost/api/chat")
        resp = httpx.Response(500, request=req)
        mock_post.side_effect = httpx.HTTPStatusError("Internal Server Error", request=req, response=resp)

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Ollama provider returned HTTP 500"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_general_request_error_handling(self, mock_post):
        mock_post.side_effect = httpx.RequestError("Some protocol error")
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="General request error contacting Ollama provider"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_general_unexpected_exception_handling(self, mock_post):
        mock_post.side_effect = RuntimeError("Crash")
        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Unexpected transport failure"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_raw_httpx_exception_does_not_escape(self, mock_post):
        mock_post.side_effect = httpx.ConnectTimeout("Connect timeout")
        provider = OllamaAIProvider(_make_ollama_config())
        try:
            provider.generate(system_prompt="sys", user_prompt="user")
        except Exception as exc:
            # Check it's an AIProviderError and not raw httpx exception
            assert isinstance(exc, AIProviderError)
            assert not isinstance(exc, httpx.HTTPError)


# ── Malformed Responses Tests (29-39) ───────────────────────────────────


class TestOllamaMalformedResponses:
    @patch("httpx.post")
    def test_invalid_json_rejected(self, mock_post):
        mock_resp = make_mock_response(json_raises=ValueError("Bad JSON"))
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Response body is not valid JSON"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_top_level_list_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data=[{"item": "val"}])
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Top-level JSON value is not an object"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_missing_message_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"other_key": "val"})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Response JSON is missing 'message' key"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_message_none_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": None})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="'message' key is not an object/dict"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_message_list_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": ["list"]})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="'message' key is not an object/dict"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_missing_content_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant"}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="Response message JSON is missing 'content' key"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_content_none_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": None}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="'content' must be a string"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_numeric_content_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": 12345}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="'content' must be a string"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_list_content_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": ["list"]}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="'content' must be a string"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_empty_content_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": ""}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="'content' cannot be empty or whitespace-only"):
            provider.generate(system_prompt="sys", user_prompt="user")

    @patch("httpx.post")
    def test_whitespace_only_content_rejected(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": "   \n   "}})
        mock_post.return_value = mock_resp

        provider = OllamaAIProvider(_make_ollama_config())
        with pytest.raises(AIProviderError, match="'content' cannot be empty or whitespace-only"):
            provider.generate(system_prompt="sys", user_prompt="user")


# ── Compatibility / Safety Tests (40-46) ───────────────────────────────


class TestOllamaCompatibility:
    def test_provider_implements_ai_provider_interface(self):
        config = _make_ollama_config()
        provider = OllamaAIProvider(config)
        assert isinstance(provider, AIProvider)

    def test_factory_creates_ollama_provider(self):
        config = _make_ollama_config()
        provider = AIProviderFactory.create(config)
        assert isinstance(provider, OllamaAIProvider)

    def test_exact_config_object_preserved(self):
        config = _make_ollama_config()
        provider = OllamaAIProvider(config)
        assert provider.config is config

    @patch("httpx.post")
    def test_config_unmodified_after_successful_generation(self, mock_post):
        mock_resp = make_mock_response(json_data={"message": {"role": "assistant", "content": "OK"}})
        mock_post.return_value = mock_resp

        config = _make_ollama_config()
        original_dict = config.model_dump()
        provider = OllamaAIProvider(config)
        provider.generate(system_prompt="sys", user_prompt="user")
        
        assert config.model_dump() == original_dict

    @patch("httpx.post")
    def test_config_unmodified_after_failed_generation(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timeout")

        config = _make_ollama_config()
        original_dict = config.model_dump()
        provider = OllamaAIProvider(config)
        with pytest.raises(AIProviderError):
            provider.generate(system_prompt="sys", user_prompt="user")

        assert config.model_dump() == original_dict


    @patch("httpx.post")
    def test_no_sensitive_data_in_error_message(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Failed to connect")
        provider = OllamaAIProvider(_make_ollama_config())
        
        system_secret = "SYSTEM_SECRET_KEY_12345"
        user_secret = "USER_SECRET_KEY_54321"
        try:
            provider.generate(system_prompt=system_secret, user_prompt=user_secret)
        except AIProviderError as exc:
            err_msg = str(exc)
            assert system_secret not in err_msg
            assert user_secret not in err_msg
