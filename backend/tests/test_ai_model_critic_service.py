"""Unit tests for AIModelCritic service and orchestrator pipeline."""

from __future__ import annotations

import copy
import json
from unittest.mock import MagicMock, patch
import pytest

from backend.app.ai_planning.providers import AIProvider
from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.ml_plan import ModelFamily, SearchStrategy
from backend.app.problem_definition.schemas import ProblemType
from backend.app.ml_execution.execution_report import (
    CandidateSummary,
    ChampionSummary,
    ExecutionReport,
)
from backend.app.ai_model_critic.schemas import CritiqueGrade, ModelCritique
from backend.app.ai_model_critic.critic_service import (
    AIModelCritic,
    AIModelCriticError,
)


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


def _make_valid_critique_json() -> str:
    return json.dumps({
        "overall_grade": "A+",
        "production_ready": True,
        "confidence": 0.98,
        "strengths": ["Strong", "Reliable"],
        "weaknesses": ["None noticed"],
        "risks": ["Low"],
        "recommendations": ["Retrain"],
        "warnings": ["Check scaling"],
        "summary": "Outstanding classification model.",
    })


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestAIModelCriticService:
    """Tests covering AIModelCritic validation, orchestrator pipeline, and error states."""

    def test_critic_init_validation(self):
        """Verify invalid init args are rejected."""
        with pytest.raises(AIModelCriticError, match="provider cannot be None"):
            AIModelCritic(provider=None)

        bad_provider = MagicMock()
        del bad_provider.generate  # ensure generate is absent
        with pytest.raises(AIModelCriticError, match="must implement 'generate' method"):
            AIModelCritic(provider=bad_provider)

    def test_review_success_pipeline(self):
        """Verify successful expert critique generation pipeline."""
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.generate.return_value = _make_valid_critique_json()

        report = _make_sample_report()

        critic = AIModelCritic(provider=mock_provider)
        critique = critic.review(report)

        assert isinstance(critique, ModelCritique)
        assert critique.report_id == "rep_abc"
        assert critique.overall_grade == CritiqueGrade.A_PLUS
        assert critique.production_ready is True
        assert mock_provider.generate.call_count == 1

    def test_review_rejects_none(self):
        """Verify passing None raises AIModelCriticError."""
        mock_provider = MagicMock(spec=AIProvider)
        critic = AIModelCritic(provider=mock_provider)

        with pytest.raises(AIModelCriticError, match="execution_report cannot be None"):
            critic.review(None)

    def test_review_rejects_wrong_type(self):
        """Verify wrong input types raise AIModelCriticError."""
        mock_provider = MagicMock(spec=AIProvider)
        critic = AIModelCritic(provider=mock_provider)

        with pytest.raises(AIModelCriticError, match="must be an ExecutionReport instance"):
            critic.review("not-a-report")

    def test_review_rejects_empty_candidate_summaries(self):
        """Verify empty candidates list in report raises AIModelCriticError."""
        mock_provider = MagicMock(spec=AIProvider)
        critic = AIModelCritic(provider=mock_provider)

        report = _make_sample_report()
        report.candidate_summaries = []  # Empty summaries list

        with pytest.raises(AIModelCriticError, match="candidate_summaries list cannot be empty"):
            critic.review(report)

    def test_context_builder_failure_handling(self):
        """Verify context builder exceptions are caught and wrapped."""
        mock_provider = MagicMock(spec=AIProvider)
        critic = AIModelCritic(provider=mock_provider)
        
        report = _make_sample_report()

        with patch.object(
            critic.context_builder, "build", side_effect=Exception("Context boom")
        ):
            with pytest.raises(AIModelCriticError, match="Context builder failed"):
                critic.review(report)

    def test_provider_failure_handling(self):
        """Verify LLM provider exceptions are caught and wrapped."""
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.generate.side_effect = RuntimeError("LLM timed out")

        report = _make_sample_report()

        critic = AIModelCritic(provider=mock_provider)
        with pytest.raises(AIModelCriticError, match="AI Provider generation failed"):
            critic.review(report)

    def test_parser_validation_failure_handling(self):
        """Verify critique schema or parsing failures are caught and wrapped."""
        mock_provider = MagicMock(spec=AIProvider)
        # Return invalid JSON
        mock_provider.generate.return_value = "{invalid-json}"

        report = _make_sample_report()

        critic = AIModelCritic(provider=mock_provider)
        with pytest.raises(AIModelCriticError, match="Critique parsing or validation failed"):
            critic.review(report)

    def test_non_mutation(self):
        """Verify report input object is not mutated by the critic pipeline."""
        mock_provider = MagicMock(spec=AIProvider)
        mock_provider.generate.return_value = _make_valid_critique_json()

        report = _make_sample_report()
        report_copy = copy.deepcopy(report)

        critic = AIModelCritic(provider=mock_provider)
        critic.review(report)

        assert report == report_copy
