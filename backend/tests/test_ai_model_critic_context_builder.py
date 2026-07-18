"""Unit tests for AIModelCriticContextBuilder."""

from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.ml_plan import ModelFamily, SearchStrategy
from backend.app.problem_definition.schemas import ProblemType
from backend.app.ml_execution.execution_report import (
    CandidateSummary,
    ChampionSummary,
    ExecutionReport,
)
from backend.app.ai_model_critic.context_builder import AIModelCriticContextBuilder


def _make_sample_report() -> ExecutionReport:
    cand = CandidateSummary(
        candidate_id="model_01",
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        primary_metric="f1",
        primary_metric_value=0.92,
        training_duration=0.1,
        evaluation_duration=0.01,
        search_strategy=SearchStrategy.NONE,
        best_parameters={"C": 1.0},
    )
    champ = ChampionSummary(
        candidate_id="model_01",
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        primary_metric="f1",
        primary_metric_value=0.92,
        feature_importance={"feat_a": 0.8},
        training_duration=0.1,
        evaluation_duration=0.01,
    )
    return ExecutionReport(
        report_id="rep_abc",
        dataset_id="ds_xyz",
        request_id="req_123",
        problem_definition_id="pd_456",
        plan_id="plan_789",
        execution_id="exec_999",
        problem_type=ProblemType.CLASSIFICATION,
        target_column="label",
        feature_columns=["feat_a"],
        compute_tier=ComputeTier.STANDARD,
        accelerator_type=AcceleratorType.NONE,
        candidate_summaries=[cand],
        champion_summary=champ,
        training_summary={"goal": "Predict churn"},
        evaluation_summary={"metric": "f1"},
        warnings=["Test warning"],
        execution_duration=0.5,
        created_timestamp="2026-07-15T00:00:00Z",
    )


class TestAIModelCriticContextBuilder:
    """Tests covering the compact JSON extraction from ExecutionReport."""

    def test_context_builder_success(self):
        """Verify report is successfully converted to expected compact JSON."""
        report = _make_sample_report()
        builder = AIModelCriticContextBuilder()

        json_str = builder.build(report)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["report_id"] == "rep_abc"
        assert parsed["plan_id"] == "plan_789"
        assert parsed["problem"]["problem_type"] == "classification"
        assert parsed["target"] == "label"
        assert parsed["feature_count"] == 1
        assert parsed["champion"]["candidate_id"] == "model_01"
        assert parsed["champion"]["feature_importance"] == {"feat_a": 0.8}
        assert "Test warning" in parsed["warnings"]

        # Ensure no datasets or raw predictions are leaked
        assert "dataframe" not in parsed
        assert "predictions" not in parsed
