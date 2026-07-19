"""Unit tests for backend.app.pipeline.schemas.PipelineContext."""

import pytest
from backend.app.pipeline.schemas import (
    AIExplanation,
    ExportResult,
    PipelineContext,
    PipelineStage,
    PipelineStatus,
)


def test_pipeline_context_initialization():
    context = PipelineContext()
    assert context.pipeline_id.startswith("pipe-")
    assert context.status == PipelineStatus.INITIALIZED
    assert context.current_stage == PipelineStage.UPLOAD
    assert context.completed_stages == []
    assert context.dataset_path is None


def test_pipeline_context_mark_stage_completed():
    context = PipelineContext()
    context.mark_stage_completed(PipelineStage.UPLOAD)
    assert PipelineStage.UPLOAD in context.completed_stages
    assert len(context.completed_stages) == 1

    # Re-marking does not duplicate
    context.mark_stage_completed(PipelineStage.UPLOAD)
    assert len(context.completed_stages) == 1


def test_export_result_and_ai_explanation_schemas():
    export = ExportResult(
        champion_model_id="model-123",
        export_format="ONNX",
        artifact_path="/tmp/model.onnx",
    )
    assert export.champion_model_id == "model-123"
    assert export.export_format == "ONNX"

    explanation = AIExplanation(
        champion_model_id="model-123",
        feature_importances={"age": 0.8, "income": 0.2},
        explanation_text="Age is the most important feature.",
    )
    assert explanation.feature_importances["age"] == 0.8
    assert "important" in explanation.explanation_text
