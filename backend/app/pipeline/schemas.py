"""Pydantic v2 schemas for the unified PipelineContext and stage contracts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


from backend.app.upload.schemas import DatasetMetadata
from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.problem_definition.schemas import ProblemDefinition
from backend.app.hardware.schemas import HardwareProfile
from backend.app.compute_capabilities.schemas import ComputeCapabilities
from backend.app.ai_providers.schemas import AIProviderConfig, AIProviderSettings
from backend.app.ml_plan.schemas import MLPlan
from backend.app.ml_plan.validator import MLPlanValidationResult
from backend.app.ml_execution.schemas import ExecutionResult
from backend.app.ml_execution.orchestrator import MLExecutionResult
from backend.app.ml_execution.execution_report import ExecutionReport
from backend.app.model_governance.schemas import ChampionDecision
from backend.app.ai_model_critic.schemas import ModelCritique
from backend.app.reporting.schemas import ExecutiveReport


class PipelineStage(str, Enum):
    """The 11 core stages of the DeployAI backend workflow."""

    UPLOAD = "upload"
    VALIDATION = "validation"
    DATASET_INTELLIGENCE = "dataset_intelligence"
    PROBLEM_DEFINITION = "problem_definition"
    AI_CONFIGURATION = "ai_configuration"
    PLANNING = "planning"
    EXECUTION = "execution"
    CHAMPION_SELECTION = "champion_selection"
    AI_EXPLANATION = "ai_explanation"
    PDF_GENERATION = "pdf_generation"
    EXPORT = "export"


class PipelineStatus(str, Enum):
    """Execution status of the pipeline orchestrator."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportResult(BaseModel):
    """Container for exported model artifact metadata."""

    export_id: str = Field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:8]}")
    champion_model_id: str = Field(..., description="ID of the exported champion model.")
    export_format: str = Field(..., description="Format of the export (e.g. ONNX, PICKLE, KERAS).")
    artifact_path: str = Field(..., description="Absolute file path to the exported binary artifact.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AIExplanation(BaseModel):
    """Aggregated AI explanation and feature importance metrics."""

    champion_model_id: str = Field(..., description="Target champion model ID.")
    feature_importances: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance rankings mapping column names to relative importance scores."
    )
    shap_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional SHAP value summary breakdown."
    )
    ai_critique: Optional[ModelCritique] = Field(
        default=None,
        description="Qualitative AI critique and evaluation."
    )
    explanation_text: str = Field(
        default="",
        description="Synthesized natural language explanation of model decisions."
    )


class PipelineContext(BaseModel):
    """Unified state context holding accumulated outputs across all pipeline stages.

    Passed through every step in the pipeline. Updated immutably or via stage helper methods.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline_id: str = Field(
        default_factory=lambda: f"pipe-{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the pipeline run."
    )
    status: PipelineStatus = Field(
        default=PipelineStatus.INITIALIZED,
        description="Current execution status of the pipeline."
    )
    current_stage: PipelineStage = Field(
        default=PipelineStage.UPLOAD,
        description="Current active workflow stage."
    )
    completed_stages: List[PipelineStage] = Field(
        default_factory=list,
        description="List of successfully completed pipeline stages."
    )

    # In-memory DataFrame (excluded from JSON serialization)
    raw_dataframe: Optional[Any] = Field(default=None, exclude=True, description="In-memory Pandas DataFrame.")

    # 1. Upload
    dataset_path: Optional[str] = Field(default=None, description="Path to input dataset file.")
    dataset_metadata: Optional[DatasetMetadata] = Field(default=None, description="Extracted dataset metadata.")

    # 2. Validation
    analysis_report: Optional[DatasetAnalysisReport] = Field(default=None, description="Statistical profiling report.")

    # 3. Dataset Intelligence
    dataset_context: Optional[DatasetContext] = Field(default=None, description="AI-ready dataset context snapshot.")

    # 4. Problem Definition
    problem_definition: Optional[ProblemDefinition] = Field(default=None, description="Resolved problem formulation.")

    # 5. AI Configuration
    hardware_profile: Optional[HardwareProfile] = Field(default=None, description="System hardware specs.")
    compute_capabilities: Optional[ComputeCapabilities] = Field(default=None, description="Compute limits and resource tiers.")
    provider_config: Optional[AIProviderConfig] = Field(default=None, description="Active AI provider configuration.")

    # 6. Planning
    ml_plan: Optional[MLPlan] = Field(default=None, description="Validated ML Execution Plan.")
    plan_validation: Optional[MLPlanValidationResult] = Field(default=None, description="Plan validation result.")

    # 7. Execution
    execution_result: Optional[MLExecutionResult] = Field(default=None, exclude=True, description="In-memory trained models and raw execution metrics.")
    execution_report: Optional[ExecutionReport] = Field(default=None, description="Execution summary report.")

    # 8. Champion Selection
    champion_decision: Optional[ChampionDecision] = Field(default=None, description="Selected Champion Model decision.")

    # 9. AI Explanation
    ai_explanation: Optional[AIExplanation] = Field(default=None, description="SHAP feature importances and AI critique.")

    # 10. PDF Generation
    executive_report: Optional[ExecutiveReport] = Field(default=None, description="Synthesized executive report object.")
    pdf_report_path: Optional[str] = Field(default=None, description="Path to rendered PDF report file.")

    # 11. Export
    export_result: Optional[ExportResult] = Field(default=None, description="Exported model binary artifact path.")
    export_format: str = Field(default="PICKLE", description="Desired export format (e.g. PICKLE, JOBLIB, ONNX).")

    # Timestamps & Error Handling
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = Field(default=None, description="Error message if pipeline execution fails.")

    def mark_stage_completed(self, stage: PipelineStage) -> None:
        """Mark a stage as completed and update current stage pointer."""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.updated_at = datetime.now(timezone.utc)
