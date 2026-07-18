"""Pydantic v2 schemas for the Champion Comparator and Model Governance Engine."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from backend.app.ml_execution.execution_report import ExecutionReport


class Winner(str, Enum):
    """Indicates which model configuration won the comparison."""

    BASELINE = "BASELINE"
    RETRAINED = "RETRAINED"
    TIE = "TIE"


def _clean_string(field_name: str, v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = v.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return cleaned


class ChampionDecision(BaseModel):
    """Details the outcomes and metadata of a champion model comparison run."""

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    decision_id: str = Field(..., description="Unique decision run tracker ID.")
    baseline_report_id: str = Field(..., description="Identifier of the baseline execution report.")
    retrained_report_id: str = Field(..., description="Identifier of the retrained execution report.")
    winner: Winner = Field(..., description="The chosen champion model configuration.")
    winner_report: ExecutionReport = Field(..., description="The ExecutionReport of the winning model configuration.")
    improvement_detected: bool = Field(..., description="True if retrained model metric improves over baseline.")
    metric_name: str = Field(..., description="Name of the primary metric compared.")
    baseline_metric: float = Field(..., description="Primary metric score of the baseline model.")
    retrained_metric: float = Field(..., description="Primary metric score of the retrained model.")
    relative_improvement: float = Field(..., description="Percentage or relative improvement score.")
    decision_reason: str = Field(..., description="Deterministic explanation of why the winner was selected.")
    production_ready: bool = Field(..., description="True if winner is production ready according to critic.")
    comparison_timestamp: str = Field(..., description="UTC timestamp in ISO format indicating run time.")

    @field_validator("decision_id", "baseline_report_id", "retrained_report_id", "metric_name", "decision_reason", "comparison_timestamp", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        return _clean_string(info.field_name, v)
