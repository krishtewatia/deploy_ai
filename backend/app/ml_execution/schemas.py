"""Pydantic v2 schemas for ML execution contract.

Defines schemas and enums used to track execution state, progress,
artifacts, and validation rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from backend.app.compute_capabilities import AcceleratorType, ComputeTier


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExecutionStatus(str, Enum):
    """The status of an ML execution task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExecutionStage(str, Enum):
    """The current phase of the ML execution pipeline."""

    INITIALIZATION = "INITIALIZATION"
    DATA_SPLITTING = "DATA_SPLITTING"
    PREPROCESSING = "PREPROCESSING"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    FEATURE_SELECTION = "FEATURE_SELECTION"
    MODEL_TRAINING = "MODEL_TRAINING"
    MODEL_EVALUATION = "MODEL_EVALUATION"
    MODEL_SELECTION = "MODEL_SELECTION"
    FINISHED = "FINISHED"


class ArtifactType(str, Enum):
    """The type of output artifact produced by an ML execution."""

    MODEL = "MODEL"
    PREPROCESSOR = "PREPROCESSOR"
    FEATURE_SELECTOR = "FEATURE_SELECTOR"
    METRICS = "METRICS"
    PREDICTIONS = "PREDICTIONS"
    CONFUSION_MATRIX = "CONFUSION_MATRIX"
    ROC_CURVE = "ROC_CURVE"
    FEATURE_IMPORTANCE = "FEATURE_IMPORTANCE"
    REPORT = "REPORT"


class WarningSeverity(str, Enum):
    """The severity levels for execution-related warnings."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Helper Validators
# ---------------------------------------------------------------------------


def _clean_string(field_name: str, v: Any) -> str:
    """Validate that a value is a non-empty string and strip whitespace."""
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = v.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return cleaned


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ExecutionArtifact(BaseModel):
    """Metadata describing a generated ML execution output artifact."""

    model_config = ConfigDict(use_enum_values=True)

    artifact_id: str = Field(
        ...,
        description="Unique identifier for the artifact.",
    )
    artifact_type: ArtifactType = Field(
        ...,
        description="Type category of the artifact.",
    )
    name: str = Field(
        ...,
        description="User-friendly name of the artifact.",
    )
    path: str = Field(
        ...,
        description="Non-empty file path or location URI of the artifact.",
    )
    description: str = Field(
        ...,
        description="Description of the artifact's purpose/contents.",
    )

    @field_validator("artifact_id", "name", "path", "description", mode="before")
    @classmethod
    def _validate_strings(cls, v: Any, info: ValidationInfo) -> str:
        return _clean_string(info.field_name, v)


class ExecutionWarning(BaseModel):
    """A warning or informational notice generated during ML execution."""

    model_config = ConfigDict(use_enum_values=True)

    code: str = Field(
        ...,
        description="Stable alphanumeric warning code.",
    )
    message: str = Field(
        ...,
        description="Human-readable warning message.",
    )
    severity: WarningSeverity = Field(
        ...,
        description="Severity classification of the warning.",
    )

    @field_validator("code", "message", mode="before")
    @classmethod
    def _validate_strings(cls, v: Any, info: ValidationInfo) -> str:
        return _clean_string(info.field_name, v)


class TrainingMetrics(BaseModel):
    """Primary and secondary evaluation metrics from model training."""

    primary_metric: str = Field(
        ...,
        description="Name of the primary metric used to optimize/evaluate the model.",
    )
    primary_metric_value: float = Field(
        ...,
        description="Numeric value of the primary metric.",
    )
    secondary_metrics: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping other secondary metric names to values.",
    )

    @field_validator("primary_metric", mode="before")
    @classmethod
    def _validate_primary_metric(cls, v: Any) -> str:
        return _clean_string("primary_metric", v)

    @field_validator("secondary_metrics", mode="before")
    @classmethod
    def _validate_secondary_metrics(cls, v: Any) -> Dict[str, float]:
        if not isinstance(v, dict):
            raise ValueError("secondary_metrics must be a dictionary")
        cleaned_metrics = {}
        for key, val in v.items():
            cleaned_key = _clean_string("secondary_metrics key", key)
            if not isinstance(val, (int, float)):
                raise ValueError("metric values must be numbers")
            cleaned_metrics[cleaned_key] = float(val)
        return cleaned_metrics

    @model_validator(mode="after")
    def _validate_metrics_consistency(self) -> TrainingMetrics:
        if self.primary_metric in self.secondary_metrics:
            raise ValueError(
                f"primary_metric '{self.primary_metric}' cannot appear inside secondary_metrics"
            )
        return self


class ExecutionProgress(BaseModel):
    """The progress state of a running or completed ML execution."""

    model_config = ConfigDict(use_enum_values=True)

    current_stage: ExecutionStage = Field(
        ...,
        description="The active phase of execution.",
    )
    status: ExecutionStatus = Field(
        ...,
        description="The high-level status of the task.",
    )
    percent_complete: int = Field(
        ...,
        ge=0,
        le=100,
        description="Completion percentage (0 to 100).",
    )
    message: str = Field(
        ...,
        description="Status description message.",
    )

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, v: Any) -> str:
        return _clean_string("message", v)


class ExecutionConstraintsSnapshot(BaseModel):
    """Hardware and performance constraints configuration snapshot."""

    model_config = ConfigDict(use_enum_values=True)

    parallel_workers: int = Field(
        ...,
        ge=1,
        description="Number of parallel workers permitted for execution.",
    )
    gpu_enabled: bool = Field(
        ...,
        description="Whether GPU acceleration is snapshot enabled.",
    )
    accelerator_type: AcceleratorType = Field(
        ...,
        description="The type of accelerator configured.",
    )
    compute_tier: ComputeTier = Field(
        ...,
        description="Compute power tier of the node.",
    )


class ExecutionResult(BaseModel):
    """The final or intermediate execution outcome results contract."""

    model_config = ConfigDict(use_enum_values=True)

    execution_id: str = Field(
        ...,
        description="Unique identifier for this execution run.",
    )
    plan_id: str = Field(
        ...,
        description="Identifier of the MLPlan being executed.",
    )
    status: ExecutionStatus = Field(
        ...,
        description="Status outcome of the execution.",
    )
    progress: ExecutionProgress = Field(
        ...,
        description="Progress state snapshot.",
    )
    training_metrics: Optional[TrainingMetrics] = Field(
        default=None,
        description="Evaluation metrics, populated on COMPLETED status.",
    )
    artifacts: List[ExecutionArtifact] = Field(
        default_factory=list,
        description="List of produced execution output artifacts.",
    )
    warnings: List[ExecutionWarning] = Field(
        default_factory=list,
        description="Warnings generated during the run.",
    )
    constraints_snapshot: ExecutionConstraintsSnapshot = Field(
        ...,
        description="Snapshot of constraints configured during execution.",
    )

    @field_validator("execution_id", "plan_id", mode="before")
    @classmethod
    def _validate_strings(cls, v: Any, info: ValidationInfo) -> str:
        return _clean_string(info.field_name, v)
