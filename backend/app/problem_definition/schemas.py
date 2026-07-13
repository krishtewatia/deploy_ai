"""Pydantic v2 schemas and enums for resolved problem definitions.

These models represent the resolved machine learning problem, specifying
problem type, features, targets, exclusions, and warnings/confirmation items.
This is the unified contract consumed by downstream ML planning.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProblemType(str, Enum):
    """Resolved problem type (auto is not allowed here)."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TargetSource(str, Enum):
    """Source of the target selection (supplied by user or inferred by logic)."""

    USER = "user"
    INFERRED = "inferred"


class ResolutionStatus(str, Enum):
    """Resolution status of the problem definition."""

    RESOLVED = "resolved"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------


class ProblemWarning(BaseModel):
    """Non-fatal warning detected during problem definition resolution."""

    code: str = Field(
        ...,
        description="Warning classification code.",
    )
    message: str = Field(
        ...,
        description="Human-readable warning message.",
    )
    column_name: Optional[str] = Field(
        default=None,
        description="Optional column name associated with the warning.",
    )

    @field_validator("code", "message", "column_name", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped


class ConfirmationItem(BaseModel):
    """A decision or assumption that requires explicit user confirmation."""

    key: str = Field(
        ...,
        description="Unique identifier key for the confirmation item.",
    )
    question: str = Field(
        ...,
        description="Human-readable confirmation question.",
    )
    reason: str = Field(
        ...,
        description="Reasoning context behind the confirmation request.",
    )

    @field_validator("key", "question", "reason", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped


# ---------------------------------------------------------------------------
# Main ProblemDefinition Model
# ---------------------------------------------------------------------------


class ProblemDefinition(BaseModel):
    """Resolved ML problem definition prepared for planning."""

    definition_id: str = Field(
        ...,
        description="Unique identifier for this resolved problem definition.",
    )
    request_id: str = Field(
        ...,
        description="Identifier linking back to the UserMLRequest.",
    )
    dataset_id: str = Field(
        ...,
        description="Identifier linking back to the DatasetContext.",
    )
    goal: str = Field(
        ...,
        description="Resolved objective description.",
    )
    problem_type: ProblemType = Field(
        ...,
        description="Confirmed problem type (classification or regression).",
    )
    target_column: str = Field(
        ...,
        description="Resolved target column.",
    )
    target_source: TargetSource = Field(
        ...,
        description="Whether target column was user-supplied or inferred.",
    )
    feature_columns: list[str] = Field(
        ...,
        min_length=1,
        description="List of columns approved as feature inputs (minimum one).",
    )
    excluded_columns: list[str] = Field(
        default_factory=list,
        description="List of columns excluded from features.",
    )
    primary_metric: str = Field(
        ...,
        description="Resolved evaluation metric.",
    )
    status: ResolutionStatus = Field(
        default=ResolutionStatus.RESOLVED,
        description="Current resolution status.",
    )
    warnings: list[ProblemWarning] = Field(
        default_factory=list,
        description="List of non-blocking or blocking warning items.",
    )
    confirmation_items: list[ConfirmationItem] = Field(
        default_factory=list,
        description="Questions requiring user confirmation.",
    )

    # --- Validators ---

    @field_validator(
        "definition_id",
        "request_id",
        "dataset_id",
        "goal",
        "target_column",
        "primary_metric",
        mode="before",
    )
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _validate_feature_columns(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("feature_columns must be a list of strings")
        if len(v) == 0:
            raise ValueError("feature_columns must contain at least one column")
        seen = set()
        cleaned = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("All column names must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("Column name cannot be empty or whitespace-only")
            if stripped in seen:
                raise ValueError(
                    f"Duplicate column name found in feature_columns: {stripped}"
                )
            seen.add(stripped)
            cleaned.append(stripped)
        return cleaned

    @field_validator("excluded_columns", mode="before")
    @classmethod
    def _validate_excluded_columns(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("excluded_columns must be a list of strings")
        seen = set()
        cleaned = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("All column names must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("Column name cannot be empty or whitespace-only")
            if stripped in seen:
                raise ValueError(
                    f"Duplicate column name found in excluded_columns: {stripped}"
                )
            seen.add(stripped)
            cleaned.append(stripped)
        return cleaned

    @model_validator(mode="after")
    def _validate_consistency(self) -> ProblemDefinition:
        # 1. Target column must not be in feature_columns or excluded_columns
        if self.target_column in self.feature_columns:
            raise ValueError(
                f"Target column '{self.target_column}' cannot be in feature_columns"
            )
        if self.target_column in self.excluded_columns:
            raise ValueError(
                f"Target column '{self.target_column}' cannot be in excluded_columns"
            )

        # 2. Columns cannot be in both features and excluded
        overlap = set(self.feature_columns) & set(self.excluded_columns)
        if overlap:
            raise ValueError(
                f"Columns cannot be in both feature_columns and excluded_columns: {sorted(overlap)}"
            )

        # 3. Status consistency rules
        if self.status == ResolutionStatus.RESOLVED:
            if len(self.confirmation_items) > 0:
                raise ValueError(
                    "confirmation_items must be empty when status is 'resolved'"
                )
        elif self.status == ResolutionStatus.NEEDS_CONFIRMATION:
            if len(self.confirmation_items) == 0:
                raise ValueError(
                    "confirmation_items must contain at least one item when status is 'needs_confirmation'"
                )

        return self
