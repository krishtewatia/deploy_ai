"""Pydantic v2 schemas and enums for optimization actions and results."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from backend.app.ml_plan.schemas import MLPlan


class OptimizationActionType(str, Enum):
    """The type of deterministic plan modification action."""

    ADD_PREPROCESSING = "ADD_PREPROCESSING"
    REMOVE_PREPROCESSING = "REMOVE_PREPROCESSING"
    REPLACE_PREPROCESSING = "REPLACE_PREPROCESSING"
    ADD_MODEL = "ADD_MODEL"
    REMOVE_MODEL = "REMOVE_MODEL"
    REPLACE_MODEL = "REPLACE_MODEL"
    CHANGE_FEATURE_SELECTION = "CHANGE_FEATURE_SELECTION"
    CHANGE_CV_FOLDS = "CHANGE_CV_FOLDS"
    CHANGE_SEARCH_STRATEGY = "CHANGE_SEARCH_STRATEGY"
    CHANGE_SEARCH_SPACE = "CHANGE_SEARCH_SPACE"
    ADD_WARNING = "ADD_WARNING"
    NO_ACTION = "NO_ACTION"


def _clean_string(field_name: str, v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = v.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return cleaned


class OptimizationAction(BaseModel):
    """A deterministic modification action parsed from a recommendation."""

    model_config = ConfigDict(use_enum_values=True)

    action_id: str = Field(..., description="Unique identifier for this action.")
    action_type: OptimizationActionType = Field(..., description="The category of action to take.")
    target: Optional[str] = Field(None, description="The configuration element to change or remove.")
    replacement: Optional[str] = Field(None, description="The new configuration element to introduce.")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Action-specific parameters.")
    reason: str = Field(..., description="Rationale detailing why this recommendation is mapped.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The confidence rating for this optimization action.")

    @field_validator("action_id", "reason", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        return _clean_string(info.field_name, v)

    @field_validator("target", "replacement", mode="before")
    @classmethod
    def _validate_optional_strings(cls, v: Any, info: Any) -> Optional[str]:
        if v is None:
            return None
        return _clean_string(info.field_name, v)


class OptimizationResult(BaseModel):
    """The outcome of optimizing a baseline MLPlan."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    optimization_id: str = Field(..., description="Unique identifier for this optimization run.")
    baseline_plan_id: str = Field(..., description="The reference baseline MLPlan identifier.")
    optimized_plan: MLPlan = Field(..., description="The new mutated (copied) MLPlan configuration.")
    actions: List[OptimizationAction] = Field(..., description="The list of recommendation actions applied.")
    summary: str = Field(..., description="Overall executive summary of changes.")

    @field_validator("optimization_id", "baseline_plan_id", "summary", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        return _clean_string(info.field_name, v)
