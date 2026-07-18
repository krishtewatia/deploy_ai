"""Tests for AIProviderConnectionValidator connection and model validation (Stage 8G)."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

import httpx
from pydantic import ValidationError

from backend.app.ai_planning.providers.base import AIProviderError
from backend.app.ai_providers.schemas import (
    AIProviderType,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
)
from backend.app.ai_providers.factory import AIProviderFactory
from backend.app.ai_providers.validation import (
    AIProviderConnectionValidator,
    AIProviderConnectionValidatorError,
    AIProviderValidationIssue,
    AIProviderValidationResult,
    AIProviderValidationStatus,
)


def _make_ollama_config(enabled: bool = True) -> OllamaProviderConfig:
    return OllamaProviderConfig(
        config_id="ollama_local",
        display_name="Local Ollama",
        model_name="llama3.2",
        base_url="http://localhost:11434",
        request_timeout_seconds=90.0,
        enabled=enabled,
    )


def _make_openai_config(enabled: bool = True, api_key: str | None = "sk-secret-key-12345") -> OpenAICompatibleProviderConfig:
    return OpenAICompatibleProviderConfig(
        config_id="openai_remote",
        display_name="OpenAI Remote",
        model_name="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        request_timeout_seconds=60.0,
        enabled=enabled,
    )


# ── Schema and Enum Tests (1-20) ────────────────────────────────────────


class TestValidationSchemaAndEnum:
    """Tests 1-20: Tests verification of status enum, validation issue, and validation result validation rules."""

    def test_validation_status_enum_values(self):
        """Test 1: All AIProviderValidationStatus enum values exist and match specs."""
        assert AIProviderValidationStatus.READY.value == "ready"
        assert AIProviderValidationStatus.UNREACHABLE.value == "unreachable"
        assert AIProviderValidationStatus.AUTHENTICATION_FAILED.value == "authentication_failed"
        assert AIProviderValidationStatus.MODEL_UNAVAILABLE.value == "model_unavailable"
        assert AIProviderValidationStatus.INVALID_RESPONSE.value == "invalid_response"
        assert AIProviderValidationStatus.PROVIDER_ERROR.value == "provider_error"

    def test_valid_validation_issue(self):
        """Test 2: Creating a valid AIProviderValidationIssue works."""
        issue = AIProviderValidationIssue(code="SOME_CODE", message="Some description msg")
        assert issue.code == "SOME_CODE"
        assert issue.message == "Some description msg"

    def test_empty_issue_code_rejected(self):
        """Test 3: Empty code rejected for validation issue."""
        with pytest.raises(ValidationError, match="code"):
            AIProviderValidationIssue(code="", message="msg")

    def test_whitespace_only_issue_code_rejected(self):
        """Test 4: Whitespace-only code rejected for validation issue."""
        with pytest.raises(ValidationError, match="code"):
            AIProviderValidationIssue(code="   \n   ", message="msg")

    def test_empty_issue_message_rejected(self):
        """Test 5: Empty message rejected for validation issue."""
        with pytest.raises(ValidationError, match="message"):
            AIProviderValidationIssue(code="CODE", message="")

    def test_whitespace_only_issue_message_rejected(self):
        """Test 6: Whitespace-only message rejected for validation issue."""
        with pytest.raises(ValidationError, match="message"):
            AIProviderValidationIssue(code="CODE", message="  \t  ")

    def test_non_string_issue_code_rejected(self):
        """Test non-string code rejected with ValueError wrapped in ValidationError."""
        with pytest.raises(ValidationError, match="code must be a string"):
            AIProviderValidationIssue(code=123, message="msg")  # type: ignore

    def test_non_string_issue_message_rejected(self):
        """Test non-string message rejected with ValueError wrapped in ValidationError."""
        with pytest.raises(ValidationError, match="message must be a string"):
            AIProviderValidationIssue(code="CODE", message=True)  # type: ignore

    def test_issue_strings_normalize_surrounding_whitespace(self):
        """Test 7: Surrounding whitespace on code/message is stripped."""
        issue = AIProviderValidationIssue(code="   CODE_STRIPPED   ", message="  Msg stripped  ")
        assert issue.code == "CODE_STRIPPED"
        assert issue.message == "Msg stripped"

    def test_valid_ready_result(self):
        """Test 8: Valid validation result with status=READY and is_ready=True works."""
        res = AIProviderValidationResult(
            config_id="c1",
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            model_name="gpt-4",
            status=AIProviderValidationStatus.READY,
            is_ready=True,
            message="READY",
            issues=[],
        )
        assert res.is_ready is True
        assert len(res.issues) == 0

    def test_valid_non_ready_result(self):
        """Test 9: Valid validation result with status!=READY and is_ready=False works."""
        issue = AIProviderValidationIssue(code="ERR", message="details")
        res = AIProviderValidationResult(
            config_id="c1",
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            model_name="gpt-4",
            status=AIProviderValidationStatus.UNREACHABLE,
            is_ready=False,
            message="Offline",
            issues=[issue],
        )
        assert res.is_ready is False
        assert len(res.issues) == 1

    def test_ready_with_is_ready_false_rejected(self):
        """Test 10: status=READY but is_ready=False is rejected."""
        with pytest.raises(ValidationError, match="is_ready must be True when status is ready"):
            AIProviderValidationResult(
                config_id="c1",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                model_name="gpt-4",
                status=AIProviderValidationStatus.READY,
                is_ready=False,
                message="READY",
                issues=[],
            )

    def test_non_ready_with_is_ready_true_rejected(self):
        """Test 11: status!=READY but is_ready=True is rejected."""
        with pytest.raises(ValidationError, match="is_ready must be False when status is not ready"):
            AIProviderValidationResult(
                config_id="c1",
                provider_type=AIProviderType.OPENAI_COMPATIBLE,
                model_name="gpt-4",
                status=AIProviderValidationStatus.UNREACHABLE,
                is_ready=True,
                message="READY",
                issues=[],
            )

    def test_empty_config_id_rejected(self):
        """Test 12: Empty config_id is rejected."""
        with pytest.raises(ValidationError, match="config_id"):
            AIProviderValidationResult(
                config_id="  ",
                provider_type=AIProviderType.OLLAMA,
                model_name="gpt-4",
                status=AIProviderValidationStatus.READY,
                is_ready=True,
                message="READY",
                issues=[],
            )

    def test_empty_model_name_rejected(self):
        """Test 13: Empty model_name is rejected."""
        with pytest.raises(ValidationError, match="model_name"):
            AIProviderValidationResult(
                config_id="c1",
                provider_type=AIProviderType.OLLAMA,
                model_name=" ",
                status=AIProviderValidationStatus.READY,
                is_ready=True,
                message="READY",
                issues=[],
            )

    def test_empty_result_message_rejected(self):
        """Test 14: Empty result message is rejected."""
        with pytest.raises(ValidationError, match="message"):
            AIProviderValidationResult(
                config_id="c1",
                provider_type=AIProviderType.OLLAMA,
                model_name="m1",
                status=AIProviderValidationStatus.READY,
                is_ready=True,
                message=" \t\n ",
                issues=[],
            )

    def test_result_strings_normalize_surrounding_whitespace(self):
        """Test 15: Surrounding whitespace on config_id, model_name, and message is stripped."""
        res = AIProviderValidationResult(
            config_id="   c1   ",
            provider_type=AIProviderType.OLLAMA,
            model_name="   m1   ",
            status=AIProviderValidationStatus.READY,
            is_ready=True,
            message="   READY MSG   ",
            issues=[],
        )
        assert res.config_id == "c1"
        assert res.model_name == "m1"
        assert res.message == "READY MSG"

    def test_default_issues_list_is_empty(self):
        """Test 16: Result issues defaults to an empty list."""
        res = AIProviderValidationResult(
            config_id="c1",
            provider_type=AIProviderType.OLLAMA,
            model_name="m1",
            status=AIProviderValidationStatus.READY,
            is_ready=True,
            message="READY",
        )
        assert res.issues == []

    def test_two_results_do_not_share_issues_list(self):
        """Test 17: Different validation result instances do not share issues reference."""
        res1 = AIProviderValidationResult(
            config_id="c1",
            provider_type=AIProviderType.OLLAMA,
            model_name="m1",
            status=AIProviderValidationStatus.READY,
            is_ready=True,
            message="READY",
        )
        res2 = AIProviderValidationResult(
            config_id="c1",
            provider_type=AIProviderType.OLLAMA,
            model_name="m1",
            status=AIProviderValidationStatus.READY,
            is_ready=True,
            message="READY",
        )
        assert res1.issues is not res2.issues

    def test_model_dump_works(self):
        """Test 18: model_dump() works successfully."""
        res = AIProviderValidationResult(
            config_id="c1",
            provider_type=AIProviderType.OLLAMA,
            model_name="m1",
            status=AIProviderValidationStatus.READY,
            is_ready=True,
            message="READY",
            issues=[AIProviderValidationIssue(code="CODE", message="msg")],
        )
        dump = res.model_dump()
        assert dump["config_id"] == "c1"
        assert dump["issues"][0]["code"] == "CODE"

    def test_model_dump_json_works(self):
        """Test 19: model_dump_json() works successfully."""
        res = AIProviderValidationResult(
            config_id="c1",
            provider_type=AIProviderType.OLLAMA,
            model_name="m1",
            status=AIProviderValidationStatus.READY,
            is_ready=True,
            message="READY",
            issues=[],
        )
        dump_json = res.model_dump_json()
        assert '"config_id":"c1"' in dump_json.replace(" ", "")

    def test_enum_json_serialization_uses_string_values(self):
        """Test 20: Enum values serialize as plain string values in JSON."""
        res = AIProviderValidationResult(
            config_id="c1",
            provider_type=AIProviderType.OLLAMA,
            model_name="m1",
            status=AIProviderValidationStatus.READY,
            is_ready=True,
            message="READY",
            issues=[],
        )
        dump_json = res.model_dump_json()
        assert '"status":"ready"' in dump_json.replace(" ", "")


# ── Validator Input Tests (21-26) ───────────────────────────────────────


class TestValidatorInput:
    """Tests 21-26: Connection validator input check rules."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_valid_ollama_config_accepted(self, mock_create):
        """Test 21: Valid Ollama config accepted by validator."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_ollama_config())
        assert res.status == AIProviderValidationStatus.READY

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_valid_openai_compatible_config_accepted(self, mock_create):
        """Test 22: Valid OpenAI-compatible config accepted by validator."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.READY

    def test_none_config_rejected(self):
        """Test 23: None config rejected with AIProviderConnectionValidatorError."""
        validator = AIProviderConnectionValidator()
        with pytest.raises(AIProviderConnectionValidatorError, match="config cannot be None"):
            validator.validate(config=None)  # type: ignore

    def test_dict_rejected(self):
        """Test 24: Dict config rejected with AIProviderConnectionValidatorError."""
        validator = AIProviderConnectionValidator()
        with pytest.raises(AIProviderConnectionValidatorError, match="Unsupported configuration type"):
            validator.validate(config={"config_id": "c1"})  # type: ignore

    def test_arbitrary_object_rejected(self):
        """Test 25: Arbitrary object rejected with AIProviderConnectionValidatorError."""
        validator = AIProviderConnectionValidator()
        with pytest.raises(AIProviderConnectionValidatorError, match="Unsupported configuration type"):
            validator.validate(config=object())  # type: ignore

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_config_remains_unmodified_after_validation(self, mock_create):
        """Test 26: The input configuration object remains unmodified after validation."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        config = _make_openai_config()
        original_dump = config.model_dump()

        validator = AIProviderConnectionValidator()
        validator.validate(config=config)

        assert config.model_dump() == original_dump


# ── Success Tests (27-34) ───────────────────────────────────────────────


class TestValidatorSuccess:
    """Tests 27-34: validate() successful verification results."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_ollama_exact_ok_returns_ready(self, mock_create):
        """Test 27: Ollama returning exact 'DEPLOYAI_OK' maps to status=READY."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_ollama_config())
        assert res.status == AIProviderValidationStatus.READY
        assert res.is_ready is True
        assert len(res.issues) == 0

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_openai_compatible_exact_ok_returns_ready(self, mock_create):
        """Test 28: OpenAI-compatible returning exact 'DEPLOYAI_OK' maps to status=READY."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.READY
        assert res.is_ready is True
        assert len(res.issues) == 0

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_surrounding_whitespace_around_ok_accepted(self, mock_create):
        """Test 29: Surrounding whitespace around validation response is stripped and accepted."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "   DEPLOYAI_OK   \n"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.READY

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_ready_result_fields_populated_correctly(self, mock_create):
        """Tests 30-34: Result fields populated correctly for successful verification."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        config = _make_openai_config()
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=config)

        # Test 30: READY result contains correct config_id
        assert res.config_id == "openai_remote"
        # Test 31: READY result contains correct provider_type
        assert res.provider_type == AIProviderType.OPENAI_COMPATIBLE
        # Test 32: READY result contains correct model_name
        assert res.model_name == "gpt-4"
        # Test 33: READY result has is_ready=True
        assert res.is_ready is True
        # Test 34: READY result has empty issues list
        assert res.issues == []


# ── Invalid Response Tests (35-41) ──────────────────────────────────────


class TestValidatorInvalidResponse:
    """Tests 35-41: Invalid response outcomes (wrong content/malformed responses)."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_wrong_text_returns_invalid_response(self, mock_create):
        """Test 35: Non-matching text returns status=INVALID_RESPONSE."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "SOMETHING_ELSE"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.INVALID_RESPONSE
        assert res.is_ready is False
        assert res.issues[0].code == "UNEXPECTED_VALIDATION_RESPONSE"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_lowercase_ok_is_rejected(self, mock_create):
        """Test 36: Lowercase 'deployai_ok' is rejected and returns status=INVALID_RESPONSE."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "deployai_ok"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.INVALID_RESPONSE

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_partial_response_is_rejected(self, mock_create):
        """Test 37: Partial match response is rejected."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI"
        mock_create.return_value = mock_provider
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.INVALID_RESPONSE

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_markdown_wrapped_response_is_rejected(self, mock_create):
        """Test 38: Markdown-wrapped response (e.g. `DEPLOYAI_OK`) is rejected."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "`DEPLOYAI_OK`"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.INVALID_RESPONSE

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_additional_text_is_rejected(self, mock_create):
        """Test 39: Additional text (e.g. 'Here is: DEPLOYAI_OK') is rejected."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Here is the response: DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.INVALID_RESPONSE

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_unexpected_response_does_not_appear_fully_in_result_message(self, mock_create):
        """Test 40: Unexpected response content is not leaked/exposed in the result message."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "super-secret-model-output-123456"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.INVALID_RESPONSE
        assert "super-secret-model-output-123456" not in res.message
        assert "super-secret-model-output-123456" not in res.issues[0].message

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_invalid_provider_response_structure_maps_to_invalid_response(self, mock_create):
        """Test 41: Adapter response structural failure (such as bad JSON) maps to status=INVALID_RESPONSE."""
        mock_provider = MagicMock()
        # Raise AIProviderError matching the adapter JSON parse error
        mock_provider.generate.side_effect = AIProviderError("Response body is not valid JSON.")
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.INVALID_RESPONSE
        assert res.issues[0].code == "INVALID_PROVIDER_RESPONSE"


# ── Authentication Tests (42-47) ────────────────────────────────────────


class TestValidatorAuthentication:
    """Tests 42-47: Authentication operational failure mapping and secrecy rules."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_http_401_maps_to_authentication_failed(self, mock_create):
        """Test 42: HTTP status code 401 (Unauthorized) maps to status=AUTHENTICATION_FAILED."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=401, request=req)
        # Create chained exception as in actual provider execution
        cause = httpx.HTTPStatusError("Unauthorized", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 401.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config(api_key="sk-secret-key-to-hide"))
        assert res.status == AIProviderValidationStatus.AUTHENTICATION_FAILED
        assert res.is_ready is False
        assert res.issues[0].code == "AUTHENTICATION_FAILED"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_http_403_maps_to_authentication_failed(self, mock_create):
        """Test 43: HTTP status code 403 (Forbidden) maps to status=AUTHENTICATION_FAILED."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=403, request=req)
        cause = httpx.HTTPStatusError("Forbidden", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 403.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.AUTHENTICATION_FAILED

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_authentication_failure_is_ready_false(self, mock_create):
        """Test 44: Authentication failures always have is_ready=False."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=401, request=req)
        cause = httpx.HTTPStatusError("Unauthorized", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 401.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.is_ready is False

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_authentication_issue_code_is_stable(self, mock_create):
        """Test 45: Authentication issue has stable code 'AUTHENTICATION_FAILED'."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=401, request=req)
        cause = httpx.HTTPStatusError("Unauthorized", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 401.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.issues[0].code == "AUTHENTICATION_FAILED"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_api_key_does_not_appear_in_result_or_issues_messages(self, mock_create):
        """Tests 46-47: The raw API key value does not leak or appear in result messages or issue descriptions."""
        mock_provider = MagicMock()
        secret_key = "sk-super-secret-key-never-leak-this"
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {secret_key}"})
        resp = httpx.Response(status_code=401, request=req)
        cause = httpx.HTTPStatusError("Unauthorized", request=req, response=resp)
        # Exception string could possibly contain the key if printed raw
        exc = AIProviderError(f"OpenAI provider returned HTTP 401 for key {secret_key}.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config(api_key=secret_key))

        assert secret_key not in res.message
        assert secret_key not in res.issues[0].message


# ── Model Availability Tests (48-52) ────────────────────────────────────


class TestValidatorModelAvailability:
    """Tests 48-52: Model unavailable operational failure mappings."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_http_404_maps_to_model_unavailable(self, mock_create):
        """Test 48: HTTP status code 404 (Not Found) maps to status=MODEL_UNAVAILABLE."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=404, request=req)
        cause = httpx.HTTPStatusError("Not Found", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 404.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config(api_key="sk-test"))
        assert res.status == AIProviderValidationStatus.MODEL_UNAVAILABLE
        assert res.is_ready is False
        assert res.issues[0].code == "MODEL_UNAVAILABLE"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_model_unavailable_has_is_ready_false(self, mock_create):
        """Test 49: Model unavailable validation result always has is_ready=False."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=404, request=req)
        cause = httpx.HTTPStatusError("Not Found", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 404.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.is_ready is False

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_model_unavailable_issue_code_is_stable(self, mock_create):
        """Test 50: Model unavailable has stable issue code 'MODEL_UNAVAILABLE'."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=404, request=req)
        cause = httpx.HTTPStatusError("Not Found", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 404.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.issues[0].code == "MODEL_UNAVAILABLE"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_configured_model_name_preserved_in_model_unavailable_result(self, mock_create):
        """Test 51: Tested model_name is preserved in the validation result when unavailable."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=404, request=req)
        cause = httpx.HTTPStatusError("Not Found", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 404.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        config = _make_openai_config()
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=config)
        assert res.model_name == "gpt-4"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_no_raw_provider_response_body_is_exposed_in_model_unavailable(self, mock_create):
        """Test 52: The raw HTTP response body of the 404 error is not exposed in the validation result."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=404, request=req, content=b"Detailed API vendor error dump")
        cause = httpx.HTTPStatusError("Not Found", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 404.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert "Detailed API vendor error dump" not in res.message
        assert "Detailed API vendor error dump" not in res.issues[0].message


# ── Connectivity and Timeout Tests (53-58) ──────────────────────────────


class TestValidatorConnectivity:
    """Tests 53-58: Connectivity, timeouts, request errors operational mappings."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_connect_error_maps_to_unreachable(self, mock_create):
        """Test 53: httpx.ConnectError maps to status=UNREACHABLE."""
        mock_provider = MagicMock()
        cause = httpx.ConnectError("Connection refused")
        exc = AIProviderError("Failed to connect to OpenAI-compatible provider.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.UNREACHABLE
        assert res.is_ready is False
        assert res.issues[0].code == "PROVIDER_UNREACHABLE"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_timeout_exception_maps_to_unreachable(self, mock_create):
        """Test 54: httpx.TimeoutException maps to status=UNREACHABLE."""
        mock_provider = MagicMock()
        cause = httpx.TimeoutException("Read timed out")
        exc = AIProviderError("OpenAI-compatible provider request timed out.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.UNREACHABLE
        assert res.is_ready is False
        assert res.issues[0].code == "PROVIDER_TIMEOUT"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_generic_request_error_maps_to_unreachable(self, mock_create):
        """Test 55: Generic httpx.RequestError maps to status=UNREACHABLE."""
        mock_provider = MagicMock()
        cause = httpx.RequestError("Protocol error")
        exc = AIProviderError("General request error contacting OpenAI-compatible provider.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.UNREACHABLE
        assert res.is_ready is False
        assert res.issues[0].code == "PROVIDER_UNREACHABLE"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_timeout_uses_provider_timeout_issue_code(self, mock_create):
        """Test 56: Timeout exception returns the specific issue code 'PROVIDER_TIMEOUT'."""
        mock_provider = MagicMock()
        cause = httpx.TimeoutException("Read timed out")
        exc = AIProviderError("Request timed out.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.issues[0].code == "PROVIDER_TIMEOUT"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_connect_error_uses_provider_unreachable_issue_code(self, mock_create):
        """Test 57: Connect error returns the specific issue code 'PROVIDER_UNREACHABLE'."""
        mock_provider = MagicMock()
        cause = httpx.ConnectError("DNS failed")
        exc = AIProviderError("DNS failure.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.issues[0].code == "PROVIDER_UNREACHABLE"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_connectivity_failure_is_ready_false(self, mock_create):
        """Test 58: Connectivity/unreachable results always have is_ready=False."""
        mock_provider = MagicMock()
        cause = httpx.ConnectError("DNS failed")
        exc = AIProviderError("DNS failure.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.is_ready is False


# ── Generic Provider Error Tests (59-64) ────────────────────────────────


class TestValidatorProviderErrors:
    """Tests 59-64: Generic operational error mapping rules."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_http_400_maps_to_provider_error(self, mock_create):
        """Test 59: HTTP status code 400 (Bad Request) maps to status=PROVIDER_ERROR."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=400, request=req)
        cause = httpx.HTTPStatusError("Bad Request", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 400.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.PROVIDER_ERROR
        assert res.is_ready is False
        assert res.issues[0].code == "PROVIDER_ERROR"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_http_429_maps_to_provider_error(self, mock_create):
        """Test 60: HTTP status code 429 (Too Many Requests) maps to status=PROVIDER_ERROR."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=429, request=req)
        cause = httpx.HTTPStatusError("Too Many Requests", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 429.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.PROVIDER_ERROR
        assert res.is_ready is False

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_http_500_maps_to_provider_error(self, mock_create):
        """Test 61: HTTP status code 500 (Internal Server Error) maps to status=PROVIDER_ERROR."""
        mock_provider = MagicMock()
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=500, request=req)
        cause = httpx.HTTPStatusError("Internal Server Error", request=req, response=resp)
        exc = AIProviderError("OpenAI provider returned HTTP 500.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.PROVIDER_ERROR

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_unknown_ai_provider_error_maps_to_provider_error(self, mock_create):
        """Test 62: Generic/unknown AIProviderError instances map to status=PROVIDER_ERROR."""
        mock_provider = MagicMock()
        exc = AIProviderError("Some obscure internal adapter crash.")
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.PROVIDER_ERROR
        assert res.issues[0].code == "PROVIDER_ERROR"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_provider_error_has_stable_issue_code(self, mock_create):
        """Test 63: Generic provider error returns issue code 'PROVIDER_ERROR'."""
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = AIProviderError("Generic wrapper crash")
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.issues[0].code == "PROVIDER_ERROR"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_provider_error_does_not_leak_secrets(self, mock_create):
        """Test 64: Generic provider errors do not leak the configured API key in their error or issue descriptions."""
        mock_provider = MagicMock()
        secret_key = "sk-leaky-secret-key-12345"
        mock_provider.generate.side_effect = AIProviderError(f"Crashed with secret {secret_key}")
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config(api_key=secret_key))
        assert secret_key not in res.message
        assert secret_key not in res.issues[0].message


# ── Disabled Configuration Tests (65-68) ────────────────────────────────


class TestValidatorDisabledConfig:
    """Tests 65-68: Disabled configuration behavior rules."""

    def test_disabled_ollama_config_returns_provider_error(self):
        """Test 65: A disabled Ollama configuration immediately returns status=PROVIDER_ERROR."""
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_ollama_config(enabled=False))
        assert res.status == AIProviderValidationStatus.PROVIDER_ERROR
        assert res.is_ready is False
        assert res.issues[0].code == "PROVIDER_DISABLED"

    def test_disabled_openai_compatible_config_returns_provider_error(self):
        """Test 66: A disabled OpenAI-compatible configuration immediately returns status=PROVIDER_ERROR."""
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config(enabled=False))
        assert res.status == AIProviderValidationStatus.PROVIDER_ERROR
        assert res.is_ready is False
        assert res.issues[0].code == "PROVIDER_DISABLED"

    def test_disabled_config_uses_provider_disabled_issue_code(self):
        """Test 67: A disabled configuration returns specific issue code 'PROVIDER_DISABLED'."""
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config(enabled=False))
        assert res.issues[0].code == "PROVIDER_DISABLED"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_disabled_config_does_not_invoke_generate(self, mock_create):
        """Test 68: A disabled configuration is validated immediately without creating/calling the adapter."""
        validator = AIProviderConnectionValidator()
        validator.validate(config=_make_openai_config(enabled=False))
        mock_create.assert_not_called()


# ── Factory Integration Tests (69-75) ───────────────────────────────────


class TestValidatorFactoryIntegration:
    """Tests 69-75: Factory construction delegation checks."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validator_delegates_provider_construction_to_factory(self, mock_create):
        """Test 69: The validator calls AIProviderFactory.create to build the provider."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        config = _make_openai_config()
        validator = AIProviderConnectionValidator()
        validator.validate(config=config)
        mock_create.assert_called_once_with(config)

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validator_uses_provider_returned_by_factory(self, mock_create):
        """Test 70: Validator calls generate() on the exact adapter instance returned by the factory."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        validator.validate(config=_make_openai_config())
        mock_provider.generate.assert_called_once()

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validation_calls_generate_exactly_once(self, mock_create):
        """Test 71: Validator triggers exactly one provider.generate() call during check."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        validator.validate(config=_make_openai_config())
        assert mock_provider.generate.call_count == 1

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validation_uses_non_empty_prompts(self, mock_create):
        """Test 72-73: Validator passes valid non-empty system and user prompts."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        validator.validate(config=_make_openai_config())

        args, kwargs = mock_provider.generate.call_args
        assert len(kwargs["system_prompt"].strip()) > 0  # Test 72: Non-empty system prompt
        assert len(kwargs["user_prompt"].strip()) > 0  # Test 73: Non-empty user prompt

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validation_does_not_mutate_provider_config_fields(self, mock_create):
        """Test 74: Provider config remains unmodified during validation workflow."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        config = _make_openai_config()
        original_dict = config.model_dump()

        validator = AIProviderConnectionValidator()
        validator.validate(config=config)
        assert config.model_dump() == original_dict

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_factory_operational_failure_is_safely_classified(self, mock_create):
        """Test 75: Errors during factory construction are caught and wrapped as PROVIDER_ERROR validation results."""
        mock_create.side_effect = Exception("Factory failed to resolve backend.")

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())

        assert res.status == AIProviderValidationStatus.PROVIDER_ERROR
        assert res.is_ready is False
        assert res.issues[0].code == "PROVIDER_ERROR"


# ── Security Tests (76-81) ──────────────────────────────────────────────


class TestValidatorSecurityLeaks:
    """Tests 76-81: Leakage prevention validation for credentials."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_api_key_absent_from_successful_result_serialization(self, mock_create):
        """Test 76: API key value never appears in serializations of a successful READY result."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        secret_key = "sk-successful-secret-key"
        config = _make_openai_config(api_key=secret_key)
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=config)

        serialized = res.model_dump_json()
        assert secret_key not in serialized

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_api_key_absent_from_auth_failure_serialization(self, mock_create):
        """Test 77: API key value never appears in serializations of an AUTHENTICATION_FAILED result."""
        mock_provider = MagicMock()
        secret_key = "sk-leaky-secret-auth-failed"
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=401, request=req)
        cause = httpx.HTTPStatusError("Unauthorized", request=req, response=resp)
        exc = AIProviderError(f"Crashed with key: {secret_key}")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        config = _make_openai_config(api_key=secret_key)
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=config)

        serialized = res.model_dump_json()
        assert secret_key not in serialized

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_api_key_absent_from_connectivity_failure_serialization(self, mock_create):
        """Test 78: API key value never appears in serializations of an UNREACHABLE result."""
        mock_provider = MagicMock()
        secret_key = "sk-leaky-secret-unreachable"
        cause = httpx.ConnectError("Connection refused")
        exc = AIProviderError(f"DNS check crash for {secret_key}")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        config = _make_openai_config(api_key=secret_key)
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=config)

        serialized = res.model_dump_json()
        assert secret_key not in serialized

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_api_key_absent_from_provider_error_serialization(self, mock_create):
        """Test 79: API key value never appears in serializations of a generic PROVIDER_ERROR result."""
        mock_provider = MagicMock()
        secret_key = "sk-leaky-secret-generic-error"
        exc = AIProviderError(f"Generic wrapper crash showing key {secret_key}")
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        config = _make_openai_config(api_key=secret_key)
        validator = AIProviderConnectionValidator()
        res = validator.validate(config=config)

        serialized = res.model_dump_json()
        assert secret_key not in serialized

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_raw_authorization_header_never_included(self, mock_create):
        """Test 80: Raw 'Authorization' header strings do not appear in the validation messages."""
        mock_provider = MagicMock()
        secret_key = "sk-secret-key-1234"
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {secret_key}"})
        resp = httpx.Response(status_code=401, request=req)
        cause = httpx.HTTPStatusError("Unauthorized", request=req, response=resp)
        exc = AIProviderError("Unauthorized.")
        exc.__cause__ = cause
        mock_provider.generate.side_effect = exc
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config(api_key=secret_key))
        assert f"Bearer {secret_key}" not in res.message
        assert f"Bearer {secret_key}" not in res.issues[0].message
        assert "Authorization" not in res.message
        assert "Authorization" not in res.issues[0].message

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_raw_provider_response_is_not_stored_in_result(self, mock_create):
        """Test 81: Raw connection response/payload is not persisted or saved in the validation result object."""
        mock_provider = MagicMock()
        raw_model_response = "DEPLOYAI_OK (Verified)"
        mock_provider.generate.return_value = raw_model_response
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        assert res.status == AIProviderValidationStatus.INVALID_RESPONSE
        # Ensure raw model output is not leaked
        assert raw_model_response not in res.message
        assert raw_model_response not in res.issues[0].message


# ── Additional Behavior Tests (82-90) ───────────────────────────────────


class TestValidatorAdditionalBehaviors:
    """Tests 82-90: Result serialization, independence, determinism, validator scope boundary limits."""

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_result_json_is_valid_json(self, mock_create):
        """Test 82: Result serializes into completely valid JSON structure."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        serialized = res.model_dump_json()

        # Try parsing back
        parsed = json.loads(serialized)
        assert parsed["status"] == "ready"
        assert parsed["is_ready"] is True

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_nested_issues_serialize_correctly(self, mock_create):
        """Test 83: Validation result's nested issues list serializes and deserializes accurately."""
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = AIProviderError("Connection failed")
        mock_provider.generate.side_effect.__cause__ = httpx.ConnectError("DNS failed")
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res = validator.validate(config=_make_openai_config())
        serialized = res.model_dump_json()

        parsed = json.loads(serialized)
        assert len(parsed["issues"]) == 1
        assert parsed["issues"][0]["code"] == "PROVIDER_UNREACHABLE"

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_multiple_result_instances_are_independent(self, mock_create):
        """Test 84: Issues array references are independent across validation results."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res1 = validator.validate(config=_make_openai_config())
        res2 = validator.validate(config=_make_openai_config())
        assert res1.issues is not res2.issues

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validation_is_deterministic_for_identical_mocked_outcomes(self, mock_create):
        """Test 85: Given identical outcomes, validation results are completely identical in data."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        res1 = validator.validate(config=_make_openai_config())
        res2 = validator.validate(config=_make_openai_config())
        assert res1.model_dump() == res2.model_dump()

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_existing_provider_config_object_identity_remains_unchanged(self, mock_create):
        """Test 86: Validation does not change the memory reference/identity of the configuration object."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        config = _make_openai_config()
        config_id_before = id(config)

        validator = AIProviderConnectionValidator()
        validator.validate(config=config)
        assert id(config) == config_id_before

    @patch("backend.app.ai_providers.settings_store.AIProviderSettingsStore.save")
    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validator_does_not_call_settings_persistence(self, mock_create, mock_save):
        """Test 87: Validation does not perform any calls to save/load settings to disk."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        validator.validate(config=_make_openai_config())
        mock_save.assert_not_called()

    @patch("backend.app.ai_providers.secret_store.InMemorySecretStore.set_secret")
    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validator_does_not_call_secret_store(self, mock_create, mock_secret_set):
        """Test 88: Validation does not interact with the SecretStore during checks."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        validator.validate(config=_make_openai_config())
        mock_secret_set.assert_not_called()

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validator_does_not_perform_model_discovery(self, mock_create):
        """Test 89: Validation tests the configured model directly and does not perform model listing/discovery calls."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "DEPLOYAI_OK"
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        validator.validate(config=_make_openai_config())
        # The generate call was called once on the configured model, no separate list/discovery methods called
        assert mock_provider.generate.call_count == 1

    @patch("backend.app.ai_providers.factory.AIProviderFactory.create")
    def test_validator_does_not_perform_retries(self, mock_create):
        """Test 90: Validator does not perform network retry loops on failure."""
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = AIProviderError("Connection error")
        mock_provider.generate.side_effect.__cause__ = httpx.ConnectError("Refused")
        mock_create.return_value = mock_provider

        validator = AIProviderConnectionValidator()
        validator.validate(config=_make_openai_config())
        assert mock_provider.generate.call_count == 1
