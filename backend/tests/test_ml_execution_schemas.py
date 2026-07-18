"""Tests for ML Execution Schemas (Stage 9)."""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.ml_execution import (
    ExecutionStatus,
    ExecutionStage,
    ArtifactType,
    WarningSeverity,
    ExecutionArtifact,
    ExecutionWarning,
    TrainingMetrics,
    ExecutionProgress,
    ExecutionConstraintsSnapshot,
    ExecutionResult,
)


def _make_valid_constraints() -> ExecutionConstraintsSnapshot:
    return ExecutionConstraintsSnapshot(
        parallel_workers=2,
        gpu_enabled=True,
        accelerator_type=AcceleratorType.CUDA,
        compute_tier=ComputeTier.HIGH,
    )


def _make_valid_progress() -> ExecutionProgress:
    return ExecutionProgress(
        current_stage=ExecutionStage.MODEL_TRAINING,
        status=ExecutionStatus.RUNNING,
        percent_complete=50,
        message="Training in progress",
    )


# ── Enum Tests ─────────────────────────────────────────────────────────────


class TestExecutionEnums:
    def test_execution_status_enum_values(self):
        assert ExecutionStatus.PENDING == "PENDING"
        assert ExecutionStatus.RUNNING == "RUNNING"
        assert ExecutionStatus.COMPLETED == "COMPLETED"
        assert ExecutionStatus.FAILED == "FAILED"
        assert ExecutionStatus.CANCELLED == "CANCELLED"

    def test_execution_stage_enum_values(self):
        assert ExecutionStage.INITIALIZATION == "INITIALIZATION"
        assert ExecutionStage.DATA_SPLITTING == "DATA_SPLITTING"
        assert ExecutionStage.PREPROCESSING == "PREPROCESSING"
        assert ExecutionStage.FEATURE_ENGINEERING == "FEATURE_ENGINEERING"
        assert ExecutionStage.FEATURE_SELECTION == "FEATURE_SELECTION"
        assert ExecutionStage.MODEL_TRAINING == "MODEL_TRAINING"
        assert ExecutionStage.MODEL_EVALUATION == "MODEL_EVALUATION"
        assert ExecutionStage.MODEL_SELECTION == "MODEL_SELECTION"
        assert ExecutionStage.FINISHED == "FINISHED"

    def test_artifact_type_enum_values(self):
        assert ArtifactType.MODEL == "MODEL"
        assert ArtifactType.PREPROCESSOR == "PREPROCESSOR"
        assert ArtifactType.FEATURE_SELECTOR == "FEATURE_SELECTOR"
        assert ArtifactType.METRICS == "METRICS"
        assert ArtifactType.PREDICTIONS == "PREDICTIONS"
        assert ArtifactType.CONFUSION_MATRIX == "CONFUSION_MATRIX"
        assert ArtifactType.ROC_CURVE == "ROC_CURVE"
        assert ArtifactType.FEATURE_IMPORTANCE == "FEATURE_IMPORTANCE"
        assert ArtifactType.REPORT == "REPORT"

    def test_warning_severity_enum_values(self):
        assert WarningSeverity.INFO == "INFO"
        assert WarningSeverity.WARNING == "WARNING"
        assert WarningSeverity.ERROR == "ERROR"


# ── String Trimming and Empty String Tests ──────────────────────────────────


class TestStringTrimmingAndEmptyRejection:
    def test_execution_artifact_string_trimming_and_rejection(self):
        # Trim whitespace
        art = ExecutionArtifact(
            artifact_id="   art_1   ",
            artifact_type=ArtifactType.MODEL,
            name="   My Model   ",
            path="   /path/to/model   ",
            description="   Description info   ",
        )
        assert art.artifact_id == "art_1"
        assert art.name == "My Model"
        assert art.path == "/path/to/model"
        assert art.description == "Description info"

        # Rejects empty string
        with pytest.raises(ValidationError, match="artifact_id"):
            ExecutionArtifact(
                artifact_id="",
                artifact_type=ArtifactType.MODEL,
                name="Model",
                path="/path",
                description="desc",
            )

        # Rejects whitespace-only
        with pytest.raises(ValidationError, match="name"):
            ExecutionArtifact(
                artifact_id="a1",
                artifact_type=ArtifactType.MODEL,
                name="   ",
                path="/path",
                description="desc",
            )

        # Rejects non-string types
        with pytest.raises(ValidationError, match="description must be a string"):
            ExecutionArtifact(
                artifact_id="a1",
                artifact_type=ArtifactType.MODEL,
                name="Model",
                path="/path",
                description=123,  # type: ignore
            )

    def test_execution_warning_string_trimming_and_rejection(self):
        warn = ExecutionWarning(
            code="   W001   ",
            message="   Some warning   ",
            severity=WarningSeverity.WARNING,
        )
        assert warn.code == "W001"
        assert warn.message == "Some warning"

        with pytest.raises(ValidationError, match="code"):
            ExecutionWarning(code="", message="msg", severity=WarningSeverity.WARNING)

        with pytest.raises(ValidationError, match="message"):
            ExecutionWarning(code="W001", message="   ", severity=WarningSeverity.WARNING)


# ── Bounds and Model Validation Tests ──────────────────────────────────────


class TestProgressAndConstraintsValidation:
    def test_percent_complete_bounds(self):
        # 0 is valid
        p0 = ExecutionProgress(
            current_stage=ExecutionStage.INITIALIZATION,
            status=ExecutionStatus.PENDING,
            percent_complete=0,
            message="Starting",
        )
        assert p0.percent_complete == 0

        # 100 is valid
        p100 = ExecutionProgress(
            current_stage=ExecutionStage.FINISHED,
            status=ExecutionStatus.COMPLETED,
            percent_complete=100,
            message="Done",
        )
        assert p100.percent_complete == 100

        # Less than 0 rejected
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            ExecutionProgress(
                current_stage=ExecutionStage.INITIALIZATION,
                status=ExecutionStatus.PENDING,
                percent_complete=-5,
                message="Starting",
            )

        # Greater than 100 rejected
        with pytest.raises(ValidationError, match="Input should be less than or equal to 100"):
            ExecutionProgress(
                current_stage=ExecutionStage.INITIALIZATION,
                status=ExecutionStatus.PENDING,
                percent_complete=105,
                message="Starting",
            )

    def test_constraints_snapshot_parallel_workers_rejected(self):
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
            ExecutionConstraintsSnapshot(
                parallel_workers=0,
                gpu_enabled=False,
                accelerator_type=AcceleratorType.NONE,
                compute_tier=ComputeTier.MINIMAL,
            )


# ── TrainingMetrics Validation Tests ────────────────────────────────────────


class TestTrainingMetricsValidation:
    def test_valid_metrics(self):
        metrics = TrainingMetrics(
            primary_metric="accuracy",
            primary_metric_value=0.92,
            secondary_metrics={"f1_score": 0.91, "precision": 0.93},
        )
        assert metrics.primary_metric == "accuracy"
        assert metrics.secondary_metrics["f1_score"] == 0.91

    def test_duplicate_primary_metric_in_secondary_rejected(self):
        with pytest.raises(ValidationError, match="primary_metric 'accuracy' cannot appear inside secondary_metrics"):
            TrainingMetrics(
                primary_metric="accuracy",
                primary_metric_value=0.92,
                secondary_metrics={"accuracy": 0.92, "f1_score": 0.91},
            )

    def test_secondary_metrics_key_validation(self):
        # Trims spaces in secondary metric keys
        metrics = TrainingMetrics(
            primary_metric="accuracy",
            primary_metric_value=0.92,
            secondary_metrics={"   f1_score   ": 0.91},
        )
        assert "f1_score" in metrics.secondary_metrics
        assert "   f1_score   " not in metrics.secondary_metrics

        # Rejects empty secondary metric key
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            TrainingMetrics(
                primary_metric="accuracy",
                primary_metric_value=0.92,
                secondary_metrics={"": 0.91},
            )

        # Rejects non-dict secondary_metrics type
        with pytest.raises(ValidationError, match="secondary_metrics must be a dictionary"):
            TrainingMetrics(
                primary_metric="accuracy",
                primary_metric_value=0.92,
                secondary_metrics=["not-a-dict"],  # type: ignore
            )

        # Rejects non-numeric metric values
        with pytest.raises(ValidationError, match="metric values must be numbers"):
            TrainingMetrics(
                primary_metric="accuracy",
                primary_metric_value=0.92,
                secondary_metrics={"f1_score": "not-a-number"},  # type: ignore
            )


# ── ExecutionResult Schema Tests ────────────────────────────────────────────


class TestExecutionResultSchema:
    def test_valid_minimal_result(self):
        res = ExecutionResult(
            execution_id="exec_123",
            plan_id="plan_abc",
            status=ExecutionStatus.PENDING,
            progress=_make_progress_pending(),
            constraints_snapshot=_make_valid_constraints(),
        )
        assert res.execution_id == "exec_123"
        assert res.status == ExecutionStatus.PENDING
        assert res.training_metrics is None
        assert res.artifacts == []
        assert res.warnings == []

    def test_valid_full_result_roundtrip(self):
        art1 = ExecutionArtifact(
            artifact_id="a1",
            artifact_type=ArtifactType.MODEL,
            name="Model",
            path="/model.bin",
            description="weights",
        )
        warn1 = ExecutionWarning(
            code="W001",
            message="Slow training warning",
            severity=WarningSeverity.WARNING,
        )
        metrics = TrainingMetrics(
            primary_metric="loss",
            primary_metric_value=0.15,
            secondary_metrics={"val_loss": 0.18},
        )
        res = ExecutionResult(
            execution_id="exec_1",
            plan_id="plan_1",
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(
                current_stage=ExecutionStage.FINISHED,
                status=ExecutionStatus.COMPLETED,
                percent_complete=100,
                message="Done",
            ),
            training_metrics=metrics,
            artifacts=[art1],
            warnings=[warn1],
            constraints_snapshot=_make_valid_constraints(),
        )

        assert res.status == ExecutionStatus.COMPLETED
        assert len(res.artifacts) == 1
        assert len(res.warnings) == 1
        assert res.training_metrics is not None
        assert res.training_metrics.primary_metric_value == 0.15

        # Serialization to JSON round-trip
        data_json = res.model_dump_json()
        parsed = json.loads(data_json)
        # Verify enum values are stored as strings
        assert parsed["status"] == "COMPLETED"
        assert parsed["progress"]["current_stage"] == "FINISHED"
        assert parsed["artifacts"][0]["artifact_type"] == "MODEL"

        # Reconstruct from serialized dict
        reconstructed = ExecutionResult.model_validate(parsed)
        assert reconstructed.execution_id == "exec_1"
        assert reconstructed.artifacts[0].artifact_id == "a1"

    def test_default_factories_isolation(self):
        res1 = ExecutionResult(
            execution_id="e1",
            plan_id="p1",
            status=ExecutionStatus.PENDING,
            progress=_make_progress_pending(),
            constraints_snapshot=_make_valid_constraints(),
        )
        res2 = ExecutionResult(
            execution_id="e2",
            plan_id="p2",
            status=ExecutionStatus.PENDING,
            progress=_make_progress_pending(),
            constraints_snapshot=_make_valid_constraints(),
        )
        assert res1.artifacts is not res2.artifacts
        assert res1.warnings is not res2.warnings


def _make_progress_pending() -> ExecutionProgress:
    return ExecutionProgress(
        current_stage=ExecutionStage.INITIALIZATION,
        status=ExecutionStatus.PENDING,
        percent_complete=0,
        message="Initialized",
    )
