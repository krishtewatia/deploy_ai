"""Unit tests for AIModelOptimizer service."""

from __future__ import annotations

import copy
import pytest

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.ml_plan import ModelFamily, SearchStrategy
from backend.app.ml_plan.schemas import (
    MLPlan,
    MLPlanStatus,
    ProblemType,
    SplitStrategy,
    DatasetSplitPlan,
    FeatureSelectionPlan,
    FeatureSelectionMethod,
    EvaluationPlan,
    ExecutionConstraints,
    ModelCandidate,
    PreprocessingStep,
    PreprocessingOperation,
)
from backend.app.ai_model_critic.schemas import CritiqueGrade, ModelCritique
from backend.app.ai_model_optimizer.schemas import OptimizationResult
from backend.app.ai_model_optimizer.optimizer_service import (
    AIModelOptimizer,
    AIModelOptimizerError,
)


# ── Helper Builders ───────────────────────────────────────────────────


def _make_sample_plan(plan_id: str = "plan_01") -> MLPlan:
    preprocessing = [
        PreprocessingStep(
            step_id="step_scale",
            operation=PreprocessingOperation.STANDARD_SCALE,
            columns=["feat_a"],
            reason="Scale baseline features",
        )
    ]
    candidates = [
        ModelCandidate(
            candidate_id="model_001",
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            search_strategy=SearchStrategy.NONE,
            reason="Provides baseline",
        ),
    ]

    return MLPlan(
        plan_id=plan_id,
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        problem_type=ProblemType.CLASSIFICATION,
        target_column="species",
        feature_columns=["feat_a"],
        preprocessing_steps=preprocessing,
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["feat_a"],
            max_features=None,
            reason="No selection",
        ),
        split_plan=DatasetSplitPlan(
            strategy=SplitStrategy.RANDOM,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        ),
        model_candidates=candidates,
        evaluation_plan=EvaluationPlan(
            primary_metric="f1",
            secondary_metrics=[],
            cross_validation_folds=2,
        ),
        execution_constraints=ExecutionConstraints(
            parallel_workers=1,
            use_gpu_acceleration=False,
            accelerator_type=AcceleratorType.NONE,
            compute_tier=ComputeTier.STANDARD,
        ),
        status=MLPlanStatus.READY,
    )


def _make_critique(report_id: str = "report_plan_01_abc", recommendations: list[str] | None = None) -> ModelCritique:
    if recommendations is None:
        recommendations = ["Increase cross validation folds"]
    return ModelCritique(
        critique_id="crit_01",
        report_id=report_id,
        overall_grade=CritiqueGrade.B_PLUS,
        production_ready=False,
        confidence=0.85,
        strengths=["Baseline accuracy"],
        weaknesses=["Slight overfitting"],
        risks=["Drift"],
        recommendations=recommendations,
        warnings=[],
        summary="Baseline review.",
    )


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestAIModelOptimizerService:
    """Tests covering inputs validation, identity mismatches, and pipeline execution in optimizer service."""

    def test_optimize_success_pipeline(self):
        """Verify successful end-to-end service execution."""
        plan = _make_sample_plan(plan_id="plan_123")
        critique = _make_critique(report_id="report_plan_123_xyz")

        optimizer = AIModelOptimizer()
        result = optimizer.optimize(critique, plan)

        assert isinstance(result, OptimizationResult)
        assert result.baseline_plan_id == "plan_123"
        assert result.optimized_plan.evaluation_plan.cross_validation_folds == 5
        assert len(result.actions) == 1
        assert "CHANGE_CV_FOLDS" in result.summary

    def test_optimize_success_no_actions(self):
        """Verify pipeline execution with unknown recommendations resulting in no actions."""
        plan = _make_sample_plan(plan_id="plan_123")
        critique = _make_critique(report_id="report_plan_123_xyz", recommendations=["Tune learning rate"])

        optimizer = AIModelOptimizer()
        result = optimizer.optimize(critique, plan)

        assert isinstance(result, OptimizationResult)
        assert "No optimization actions applied" in result.summary

    def test_optimize_rejects_none(self):
        """Verify None inputs raise AIModelOptimizerError."""
        optimizer = AIModelOptimizer()
        plan = _make_sample_plan()
        critique = _make_critique()

        with pytest.raises(AIModelOptimizerError, match="critique cannot be None"):
            optimizer.optimize(None, plan)

        with pytest.raises(AIModelOptimizerError, match="baseline_plan cannot be None"):
            optimizer.optimize(critique, None)

    def test_optimize_rejects_wrong_types(self):
        """Verify incorrect parameter types raise AIModelOptimizerError."""
        optimizer = AIModelOptimizer()
        plan = _make_sample_plan()
        critique = _make_critique()

        with pytest.raises(AIModelOptimizerError, match="must be a ModelCritique instance"):
            optimizer.optimize("not-a-critique", plan)

        with pytest.raises(AIModelOptimizerError, match="must be an MLPlan instance"):
            optimizer.optimize(critique, "not-a-plan")

    def test_optimize_rejects_identity_mismatch(self):
        """Verify report ID vs plan ID mismatches raise AIModelOptimizerError."""
        plan = _make_sample_plan(plan_id="plan_123")
        critique = _make_critique(report_id="report_plan_999_xyz")  # Mismatching plan ID

        optimizer = AIModelOptimizer()
        with pytest.raises(AIModelOptimizerError, match="Identity mismatch"):
            optimizer.optimize(critique, plan)

    def test_optimize_propagates_plan_optimizer_errors(self):
        """Verify plan optimizer errors (e.g. invalid targets) are caught and wrapped."""
        # A plan with no scaler step
        plan = _make_sample_plan(plan_id="plan_123")
        plan.preprocessing_steps = []
        
        # A recommendation requiring replacement of scaler
        critique = _make_critique(report_id="report_plan_123_xyz", recommendations=["Use RobustScaler"])

        optimizer = AIModelOptimizer()
        with pytest.raises(AIModelOptimizerError, match="Plan optimization failed"):
            optimizer.optimize(critique, plan)

    def test_non_mutation(self):
        """Verify the service does not mutate inputs."""
        plan = _make_sample_plan(plan_id="plan_123")
        critique = _make_critique(report_id="report_plan_123_xyz")

        plan_copy = copy.deepcopy(plan)
        critique_copy = copy.deepcopy(critique)

        optimizer = AIModelOptimizer()
        optimizer.optimize(critique, plan)

        assert plan == plan_copy
        assert critique == critique_copy

    def test_optimize_mapper_failure_handling(self):
        """Verify mapper exceptions are caught and wrapped in AIModelOptimizerError."""
        plan = _make_sample_plan(plan_id="plan_123")
        critique = _make_critique(report_id="report_plan_123_xyz")

        optimizer = AIModelOptimizer()
        from unittest.mock import patch
        with patch.object(optimizer.mapper, "map_recommendations", side_effect=Exception("Mapper boom")):
            with pytest.raises(AIModelOptimizerError, match="Recommendation mapping failed"):
                optimizer.optimize(critique, plan)
