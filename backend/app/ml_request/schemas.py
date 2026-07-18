"""Pydantic v2 schemas and enums for user machine learning requests.

These models capture the user's intent, goal, and preferences for the automated
ML pipeline. They represent user input/intent only, without verifying target column
existence in a dataset, metric compatibility, or dataset properties.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProblemTypePreference(str, Enum):
    """User preference for the machine learning problem type."""

    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class AutomationLevel(str, Enum):
    """Whether decisions should be made automatically or guide the user."""

    AUTOMATIC = "automatic"
    GUIDED = "guided"


class ComputePreference(str, Enum):
    """User performance/resource usage preference."""

    BALANCED = "balanced"
    FAST = "fast"
    THOROUGH = "thorough"


# ---------------------------------------------------------------------------
# UserMLRequest Model
# ---------------------------------------------------------------------------


class UserMLRequest(BaseModel):
    """Represents a structured user request defining intent and parameters."""

    request_id: str = Field(
        ...,
        description="Unique identifier for this ML request.",
    )
    goal: str = Field(
        ...,
        description="Natural-language description of what the user wants to predict or build.",
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Optional name of the target column to predict.",
    )
    problem_type: ProblemTypePreference = Field(
        default=ProblemTypePreference.AUTO,
        description="User preference for the problem type (auto, classification, or regression).",
    )
    primary_metric: Optional[str] = Field(
        default=None,
        description="Optional primary metric requested by the user.",
    )
    automation_level: AutomationLevel = Field(
        default=AutomationLevel.AUTOMATIC,
        description="Level of automation desired (automatic or guided).",
    )
    compute_preference: ComputePreference = Field(
        default=ComputePreference.BALANCED,
        description="User compute resource/time preference.",
    )
    excluded_columns: list[str] = Field(
        default_factory=list,
        description="Columns to explicitly exclude from model training/features.",
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Optional additional domain or business context.",
    )

    # --- Validators ---

    @field_validator("request_id", mode="before")
    @classmethod
    def _validate_request_id(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("request_id must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("request_id cannot be empty or whitespace-only")
        return stripped

    @field_validator("goal", mode="before")
    @classmethod
    def _validate_goal(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("goal must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("goal cannot be empty or whitespace-only")
        return stripped

    @field_validator("target_column", mode="before")
    @classmethod
    def _validate_target_column(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("target_column must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("target_column cannot be empty or whitespace-only")
        return stripped

    @field_validator("primary_metric", mode="before")
    @classmethod
    def _validate_primary_metric(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("primary_metric must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("primary_metric cannot be empty or whitespace-only")
        return stripped

    @field_validator("additional_context", mode="before")
    @classmethod
    def _validate_additional_context(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("additional_context must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("additional_context cannot be empty or whitespace-only")
        return stripped

    @field_validator("excluded_columns", mode="before")
    @classmethod
    def _validate_excluded_columns(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("excluded_columns must be a list of strings")
        seen = set()
        cleaned = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("All excluded column names must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("Excluded column names cannot be empty or whitespace-only")
            if stripped in seen:
                raise ValueError(f"Duplicate excluded column name found: {stripped}")
            seen.add(stripped)
            cleaned.append(stripped)
        return cleaned

    @model_validator(mode="after")
    def _check_target_not_excluded(self) -> UserMLRequest:
        if self.target_column is not None and self.target_column in self.excluded_columns:
            raise ValueError(
                f"Target column '{self.target_column}' cannot be in the list of excluded columns"
            )
        return self
