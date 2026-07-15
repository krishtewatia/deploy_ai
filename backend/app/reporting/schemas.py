"""Pydantic schemas for the Executive Report Generator."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_string(field_name: str, v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = v.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return cleaned


class ExecutiveReport(BaseModel):
    """The final consolidated report contract presented to the end user."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    report_id: str = Field(..., description="Unique executive report tracker ID.")
    title: str = Field(..., description="Report title.")
    generated_timestamp: str = Field(..., description="ISO UTC timestamp when report was generated.")
    problem_summary: Dict[str, Any] = Field(..., description="Summary details of the target problem.")
    dataset_summary: Dict[str, Any] = Field(..., description="Summary stats of the dataset.")
    pipeline_summary: Dict[str, Any] = Field(..., description="ML processing pipeline details.")
    models_summary: List[Dict[str, Any]] = Field(..., description="Performance profile of all trained models.")
    champion_summary: Dict[str, Any] = Field(..., description="Metadata of the selected winner champion model.")
    optimization_summary: Dict[str, Any] = Field(..., description="List of recommendation actions mapped.")
    ai_review: Dict[str, Any] = Field(..., description="Repackaged critique review from AI Model Critic.")
    governance_summary: Dict[str, Any] = Field(..., description="Final governance champion selection results.")
    deployment_summary: Dict[str, Any] = Field(..., description="Production readiness and deployment details.")
    warnings: List[str] = Field(..., description="Consolidated list of all warning messages.")
    recommendations: List[str] = Field(..., description="List of recommendations for the pipeline.")
    executive_summary: str = Field(..., description="High-level summary summary text.")

    @field_validator("report_id", "title", "generated_timestamp", "executive_summary", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        return _clean_string(info.field_name, v)
