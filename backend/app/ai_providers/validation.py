"""AI provider validation models and services for DeployAI.

Stage 8G provides operational validation for configured AI models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional
import httpx

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.ai_planning.providers.base import AIProviderError
from backend.app.ai_providers.schemas import (
    AIProviderConfig,
    AIProviderType,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
)
from backend.app.ai_providers.factory import AIProviderFactory


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AIProviderValidationStatus(str, Enum):
    """The operational readiness state of an AI provider."""

    READY = "ready"
    UNREACHABLE = "unreachable"
    AUTHENTICATION_FAILED = "authentication_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER_ERROR = "provider_error"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AIProviderConnectionValidatorError(Exception):
    """Raised for validation misuse or internal validator invariant failure."""

    pass


# ---------------------------------------------------------------------------
# Helper Validators
# ---------------------------------------------------------------------------


def _strip_nonempty(field_name: str, v: Any) -> str:
    """Validate that a value is a non-empty string and strip whitespace."""
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = v.strip()
    if not stripped:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return stripped


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AIProviderValidationIssue(BaseModel):
    """Represents a specific validation issue discovered during provider check."""

    code: str = Field(
        ...,
        description="Stable machine-readable code identifier for the issue.",
    )
    message: str = Field(
        ...,
        description="Safe human-readable explanation of the issue.",
    )

    @field_validator("code", mode="before")
    @classmethod
    def _validate_code(cls, v: Any) -> str:
        return _strip_nonempty("code", v)

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, v: Any) -> str:
        return _strip_nonempty("message", v)


class AIProviderValidationResult(BaseModel):
    """Structured operational readiness check result for an AI provider."""

    config_id: str = Field(
        ...,
        description="Unique identifier of the validated configuration.",
    )
    provider_type: AIProviderType = Field(
        ...,
        description="Type of the AI provider.",
    )
    model_name: str = Field(
        ...,
        description="Target model name tested.",
    )
    status: AIProviderValidationStatus = Field(
        ...,
        description="Validation status.",
    )
    is_ready: bool = Field(
        ...,
        description="True if status is READY, False otherwise.",
    )
    message: str = Field(
        ...,
        description="Summary validation message.",
    )
    issues: List[AIProviderValidationIssue] = Field(
        default_factory=list,
        description="List of specific issues found.",
    )

    @field_validator("config_id", mode="before")
    @classmethod
    def _validate_config_id(cls, v: Any) -> str:
        return _strip_nonempty("config_id", v)

    @field_validator("model_name", mode="before")
    @classmethod
    def _validate_model_name(cls, v: Any) -> str:
        return _strip_nonempty("model_name", v)

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, v: Any) -> str:
        return _strip_nonempty("message", v)

    @model_validator(mode="after")
    def _validate_is_ready_consistency(self) -> AIProviderValidationResult:
        if self.status == AIProviderValidationStatus.READY:
            if self.is_ready is not True:
                raise ValueError("is_ready must be True when status is ready")
        else:
            if self.is_ready is not False:
                raise ValueError("is_ready must be False when status is not ready")
        return self


# ---------------------------------------------------------------------------
# Connection Validator
# ---------------------------------------------------------------------------


class AIProviderConnectionValidator:
    """Validator for testing connection and model availability for AI providers."""

    def validate(self, *, config: AIProviderConfig) -> AIProviderValidationResult:
        """Validate the AI provider connection and model using provider.generate().

        Args:
            config: Concrete provider configuration (Ollama or OpenAI-compatible).

        Returns:
            Structured AIProviderValidationResult containing classification and issues.

        Raises:
            AIProviderConnectionValidatorError: For invalid inputs.
        """
        if config is None:
            raise AIProviderConnectionValidatorError("config cannot be None")

        if not isinstance(config, (OllamaProviderConfig, OpenAICompatibleProviderConfig)):
            raise AIProviderConnectionValidatorError(
                f"Unsupported configuration type: {type(config).__name__}. "
                "Must be a validated OllamaProviderConfig or OpenAICompatibleProviderConfig."
            )

        if not config.enabled:
            return AIProviderValidationResult(
                config_id=config.config_id,
                provider_type=config.provider_type,
                model_name=config.model_name,
                status=AIProviderValidationStatus.PROVIDER_ERROR,
                is_ready=False,
                message="Provider configuration is disabled.",
                issues=[
                    AIProviderValidationIssue(
                        code="PROVIDER_DISABLED",
                        message=f"Configuration '{config.config_id}' is disabled."
                    )
                ]
            )

        try:
            provider = AIProviderFactory.create(config)
        except Exception as exc:
            # Factory construction failure
            return AIProviderValidationResult(
                config_id=config.config_id,
                provider_type=config.provider_type,
                model_name=config.model_name,
                status=AIProviderValidationStatus.PROVIDER_ERROR,
                is_ready=False,
                message=f"Failed to create provider adapter: {str(exc)}",
                issues=[
                    AIProviderValidationIssue(
                        code="PROVIDER_ERROR",
                        message=f"Factory resolution failure: {str(exc)}"
                    )
                ]
            )

        sys_prompt = "You are validating an AI provider connection. Respond with exactly: DEPLOYAI_OK"
        user_prompt = "Respond with exactly: DEPLOYAI_OK"

        try:
            response = provider.generate(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
            )
        except AIProviderError as exc:
            status = AIProviderValidationStatus.PROVIDER_ERROR
            issues: List[AIProviderValidationIssue] = []
            msg = "Validation failed due to a provider error."

            cause = exc.__cause__
            if isinstance(cause, httpx.HTTPStatusError):
                status_code = cause.response.status_code
                if status_code in (401, 403):
                    status = AIProviderValidationStatus.AUTHENTICATION_FAILED
                    msg = "Provider authentication failed."
                    issues.append(
                        AIProviderValidationIssue(
                            code="AUTHENTICATION_FAILED",
                            message=f"Authentication rejected by provider with status code {status_code}."
                        )
                    )
                elif status_code == 404:
                    status = AIProviderValidationStatus.MODEL_UNAVAILABLE
                    msg = "Configured model or endpoint is unavailable."
                    issues.append(
                        AIProviderValidationIssue(
                            code="MODEL_UNAVAILABLE",
                            message=f"Model '{config.model_name}' or endpoint not found (HTTP 404)."
                        )
                    )
                else:
                    status = AIProviderValidationStatus.PROVIDER_ERROR
                    msg = f"Provider returned HTTP status {status_code}."
                    issues.append(
                        AIProviderValidationIssue(
                            code="PROVIDER_ERROR",
                            message=msg
                        )
                    )
            elif isinstance(cause, httpx.ConnectError):
                status = AIProviderValidationStatus.UNREACHABLE
                msg = "Failed to connect to the provider endpoint."
                issues.append(
                    AIProviderValidationIssue(
                        code="PROVIDER_UNREACHABLE",
                        message="Connection refused or DNS resolution failed."
                    )
                )
            elif isinstance(cause, httpx.TimeoutException):
                status = AIProviderValidationStatus.UNREACHABLE
                msg = "Request to provider timed out."
                issues.append(
                    AIProviderValidationIssue(
                        code="PROVIDER_TIMEOUT",
                        message="The request or connection timed out."
                    )
                )
            elif isinstance(cause, httpx.RequestError):
                status = AIProviderValidationStatus.UNREACHABLE
                msg = "Network request error contacting provider."
                issues.append(
                    AIProviderValidationIssue(
                        code="PROVIDER_UNREACHABLE",
                        message="Network request failed."
                    )
                )
            else:
                # Malformed JSON or format parsing failure wrapping in AIProviderError
                err_msg = str(exc)
                if any(
                    k in err_msg
                    for k in [
                        "JSON",
                        "json",
                        "choices",
                        "message",
                        "content",
                        "Response",
                        "response",
                        "First choice",
                    ]
                ):
                    status = AIProviderValidationStatus.INVALID_RESPONSE
                    msg = "Provider returned an invalid or malformed response structure."
                    issues.append(
                        AIProviderValidationIssue(
                            code="INVALID_PROVIDER_RESPONSE",
                            message="Provider response structure was malformed."
                        )
                    )
                else:
                    status = AIProviderValidationStatus.PROVIDER_ERROR
                    msg = "Unexpected provider operational error."
                    issues.append(
                        AIProviderValidationIssue(
                            code="PROVIDER_ERROR",
                            message="Unexpected error occurred."
                        )
                    )

            # Ensure API keys never leak into the validation message/issues
            if config.provider_type == AIProviderType.OPENAI_COMPATIBLE and config.api_key:
                clean_api_key = config.api_key
                msg = msg.replace(clean_api_key, "***")
                for issue in issues:
                    issue.message = issue.message.replace(clean_api_key, "***")

            return AIProviderValidationResult(
                config_id=config.config_id,
                provider_type=config.provider_type,
                model_name=config.model_name,
                status=status,
                is_ready=False,
                message=msg,
                issues=issues,
            )

        # Check response content
        normalized_response = response.strip()
        if normalized_response == "DEPLOYAI_OK":
            return AIProviderValidationResult(
                config_id=config.config_id,
                provider_type=config.provider_type,
                model_name=config.model_name,
                status=AIProviderValidationStatus.READY,
                is_ready=True,
                message="Provider connection and model validation succeeded.",
                issues=[],
            )
        else:
            return AIProviderValidationResult(
                config_id=config.config_id,
                provider_type=config.provider_type,
                model_name=config.model_name,
                status=AIProviderValidationStatus.INVALID_RESPONSE,
                is_ready=False,
                message="Unexpected validation response from provider.",
                issues=[
                    AIProviderValidationIssue(
                        code="UNEXPECTED_VALIDATION_RESPONSE",
                        message="The provider responded successfully, but the text response did not match the expected validation token."
                    )
                ],
            )
