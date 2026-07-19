"""Pydantic v2 schemas for AI-assisted ML planning proposals.

These models define structured, declarative AI proposals that describe
potential improvements to a deterministic baseline MLPlan.  They do NOT
contain executable code, scripts, or complete MLPlan replacements.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.ml_plan.schemas import (
    FeatureEngineeringOperation,
    FeatureSelectionMethod,
    MLPlan,
    ModelFamily,
    PreprocessingOperation,
    SearchStrategy,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AIDecisionConfidence(str, Enum):
    """Confidence level of an AI planning decision."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProposalAction(str, Enum):
    """Type of modification proposed by the AI."""

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


# ---------------------------------------------------------------------------
# Helper validators
# ---------------------------------------------------------------------------


def _check_json_serializable(name: str, val: Any) -> None:
    """Validate that the provided value can be correctly dumped to JSON."""
    try:
        json.dumps(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc


def _strip_nonempty(field_name: str, v: Any) -> str:
    """Strip and reject empty/whitespace-only strings."""
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = v.strip()
    if not stripped:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return stripped


def _validate_columns_list(name: str, v: Any) -> list[str]:
    """Validate list elements are stripped, non-empty, and unique."""
    if not isinstance(v, list):
        raise ValueError(f"{name} must be a list of strings")
    processed: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(v):
        if not isinstance(item, str):
            raise ValueError(f"{name} at index {idx} must be a string")
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{name} at index {idx} cannot be empty or whitespace-only")
        if stripped in seen:
            raise ValueError(f"Duplicate column name detected in {name}: '{stripped}'")
        seen.add(stripped)
        processed.append(stripped)
    return processed


# ---------------------------------------------------------------------------
# Individual Proposal Models
# ---------------------------------------------------------------------------


class AIPreprocessingProposal(BaseModel):
    """A single proposed preprocessing change."""

    proposal_id: str = Field(
        ...,
        description="Unique identifier for this proposal.",
    )
    action: ProposalAction = Field(
        ...,
        description="Whether to add, remove, or replace a preprocessing step.",
    )
    operation: PreprocessingOperation = Field(
        ...,
        description="Preprocessing operation type.",
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Dataset column names affected.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Operation parameters.",
    )
    reason: str = Field(
        ...,
        description="Rationale for the proposal.",
    )
    confidence: AIDecisionConfidence = Field(
        ...,
        description="Confidence level of the proposal.",
    )

    @field_validator("proposal_id", "reason", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        return _strip_nonempty("field", v)

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: Any) -> Any:
        _check_json_serializable("parameters", v)
        return v

    @field_validator("columns", mode="before")
    @classmethod
    def _validate_columns(cls, v: Any) -> list[str]:
        """Require concrete columns for a proposal that creates a plan step."""
        columns = _validate_columns_list("columns", v)
        if not columns:
            raise ValueError("columns must contain at least one column")
        return columns


class AIFeatureEngineeringProposal(BaseModel):
    """A single proposed feature engineering change."""

    proposal_id: str = Field(
        ...,
        description="Unique identifier for this proposal.",
    )
    action: ProposalAction = Field(
        ...,
        description="Whether to add, remove, or replace a feature engineering step.",
    )
    operation: FeatureEngineeringOperation = Field(
        ...,
        description="Feature engineering operation type.",
    )
    input_columns: list[str] = Field(
        default_factory=list,
        description="Columns consumed as input.",
    )
    output_columns: list[str] = Field(
        default_factory=list,
        description="Newly generated output columns.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Operation parameters.",
    )
    reason: str = Field(
        ...,
        description="Rationale for the proposal.",
    )
    confidence: AIDecisionConfidence = Field(
        ...,
        description="Confidence level of the proposal.",
    )

    @field_validator("proposal_id", "reason", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        return _strip_nonempty("field", v)

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: Any) -> Any:
        _check_json_serializable("parameters", v)
        return v

    @field_validator("input_columns", "output_columns", mode="before")
    @classmethod
    def _validate_columns(cls, v: Any) -> list[str]:
        """Require concrete input and output columns for a feature step."""
        columns = _validate_columns_list("feature engineering columns", v)
        if not columns:
            raise ValueError("feature engineering columns must contain at least one column")
        return columns


class AIModelCandidateProposal(BaseModel):
    """A single proposed model candidate change."""

    proposal_id: str = Field(
        ...,
        description="Unique identifier for this proposal.",
    )
    action: ProposalAction = Field(
        ...,
        description="Whether to add, remove, or replace a model candidate.",
    )
    model_family: ModelFamily = Field(
        ...,
        description="Classical learning algorithm family.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic fixed training hyperparameters.",
    )
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.NONE,
        description="Hyperparameter search algorithm.",
    )
    search_space: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameter search boundaries.",
    )
    reason: str = Field(
        ...,
        description="Rationale for the proposal.",
    )
    confidence: AIDecisionConfidence = Field(
        ...,
        description="Confidence level of the proposal.",
    )

    @field_validator("proposal_id", "reason", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        return _strip_nonempty("field", v)

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: Any) -> Any:
        _check_json_serializable("parameters", v)
        return v

    @field_validator("search_space")
    @classmethod
    def _validate_search_space(cls, v: Any) -> Any:
        _check_json_serializable("search_space", v)
        return v


class AIFeatureSelectionProposal(BaseModel):
    """Proposed feature selection configuration."""

    method: FeatureSelectionMethod = Field(
        ...,
        description="Feature selection algorithm.",
    )
    candidate_columns: list[str] = Field(
        default_factory=list,
        description="Columns eligible for filtering.",
    )
    max_features: Optional[int] = Field(
        default=None,
        description="Maximum number of columns to select.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Feature selector specific inputs.",
    )
    reason: str = Field(
        ...,
        description="Rationale for the proposal.",
    )
    confidence: AIDecisionConfidence = Field(
        ...,
        description="Confidence level of the proposal.",
    )

    @field_validator("reason", mode="before")
    @classmethod
    def _strip_and_validate_reason(cls, v: Any) -> str:
        return _strip_nonempty("reason", v)

    @field_validator("candidate_columns", mode="before")
    @classmethod
    def _validate_candidates(cls, v: Any) -> list[str]:
        if isinstance(v, list) and len(v) == 0:
            return v
        return _validate_columns_list("candidate_columns", v)

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: Any) -> Any:
        _check_json_serializable("parameters", v)
        return v

    @model_validator(mode="after")
    def _validate_max_features(self) -> AIFeatureSelectionProposal:
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be >= 1")
        return self


class AIEvaluationProposal(BaseModel):
    """Proposed evaluation configuration changes."""

    primary_metric: Optional[str] = Field(
        default=None,
        description="Proposed primary metric (None means no change).",
    )
    secondary_metrics: list[str] = Field(
        default_factory=list,
        description="Proposed secondary metrics.",
    )
    cross_validation_folds: Optional[int] = Field(
        default=None,
        description="Proposed CV folds (None means no change).",
    )
    reason: str = Field(
        ...,
        description="Rationale for the proposal.",
    )
    confidence: AIDecisionConfidence = Field(
        ...,
        description="Confidence level of the proposal.",
    )

    @field_validator("reason", mode="before")
    @classmethod
    def _strip_and_validate_reason(cls, v: Any) -> str:
        return _strip_nonempty("reason", v)

    @field_validator("primary_metric", mode="before")
    @classmethod
    def _strip_primary_metric(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return _strip_nonempty("primary_metric", v)

    @field_validator("cross_validation_folds")
    @classmethod
    def _validate_cv_folds(cls, v: Any) -> Optional[int]:
        if v is not None and v < 2:
            raise ValueError("cross_validation_folds must be >= 2")
        return v


class AIDecisionWarning(BaseModel):
    """Warning generated by AI planning decisions."""

    code: str = Field(
        ...,
        description="Warning classification code.",
    )
    message: str = Field(
        ...,
        description="Human-readable warning message.",
    )

    @field_validator("code", "message", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        return _strip_nonempty("field", v)


# ---------------------------------------------------------------------------
# Main AIDecisionProposal
# ---------------------------------------------------------------------------


class AIDecisionProposal(BaseModel):
    """Top-level structured AI proposal for improving a baseline MLPlan."""

    proposal_set_id: str = Field(
        ...,
        description="Unique identifier for this proposal set.",
    )
    baseline_plan_id: str = Field(
        ...,
        description="ID of the baseline plan being improved.",
    )
    dataset_id: str = Field(
        ...,
        description="Linkage ID to DatasetContext.",
    )
    request_id: str = Field(
        ...,
        description="Linkage ID to UserMLRequest.",
    )
    problem_definition_id: str = Field(
        ...,
        description="Linkage ID to ProblemDefinition.",
    )
    compute_capability_id: str = Field(
        ...,
        description="Linkage ID to ComputeCapabilities.",
    )
    preprocessing_proposals: list[AIPreprocessingProposal] = Field(
        default_factory=list,
        description="Proposed preprocessing changes.",
    )
    feature_engineering_proposals: list[AIFeatureEngineeringProposal] = Field(
        default_factory=list,
        description="Proposed feature engineering changes.",
    )
    model_candidate_proposals: list[AIModelCandidateProposal] = Field(
        default_factory=list,
        description="Proposed model candidate changes.",
    )
    feature_selection_proposal: Optional[AIFeatureSelectionProposal] = Field(
        default=None,
        description="Proposed feature selection configuration.",
    )
    evaluation_proposal: Optional[AIEvaluationProposal] = Field(
        default=None,
        description="Proposed evaluation configuration.",
    )
    warnings: list[AIDecisionWarning] = Field(
        default_factory=list,
        description="AI-generated warnings.",
    )
    summary: str = Field(
        ...,
        description="Brief summary of the proposed changes.",
    )

    @field_validator(
        "proposal_set_id",
        "baseline_plan_id",
        "dataset_id",
        "request_id",
        "problem_definition_id",
        "compute_capability_id",
        "summary",
        mode="before",
    )
    @classmethod
    def _strip_and_validate_ids(cls, v: Any) -> str:
        return _strip_nonempty("field", v)

    @model_validator(mode="after")
    def _validate_unique_proposal_ids(self) -> AIDecisionProposal:
        """Ensure all proposal IDs are unique across all proposal categories."""
        all_ids: list[str] = []
        for p in self.preprocessing_proposals:
            all_ids.append(p.proposal_id)
        for p in self.feature_engineering_proposals:
            all_ids.append(p.proposal_id)
        for p in self.model_candidate_proposals:
            all_ids.append(p.proposal_id)

        seen: set[str] = set()
        for pid in all_ids:
            if pid in seen:
                raise ValueError(f"Duplicate proposal_id detected across categories: '{pid}'")
            seen.add(pid)

        return self


class AIAssistedPlanningResult(BaseModel):
    """The result of the AI-assisted ML planning orchestration."""

    baseline_plan_id: str = Field(
        ...,
        description="ID of the baseline plan that the AI analyzed.",
    )
    proposal: AIDecisionProposal = Field(
        ...,
        description="The structured AI decision proposal.",
    )
    final_plan: MLPlan = Field(
        ...,
        description="The final validated MLPlan (with proposed changes merged if applicable).",
    )
    applied: bool = Field(
        ...,
        description="True if the AI proposed non-trivial changes that were successfully merged.",
    )

