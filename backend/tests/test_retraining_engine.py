"""Unit tests for the Automatic Retraining Engine."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from backend.app.compute_capabilities.schemas import ComputeCapabilities, MemoryConstraintLevel
from backend.app.compute_capabilities import AcceleratorType, ComputeTier
# ... imports ...
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)
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
    ModelFamily,
    SearchStrategy,
)
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ResolutionStatus,
    TargetSource,
)
from backend.app.ml_execution.orchestrator import MLExecutionResult
from backend.app.ml_execution.evaluation_engine import EvaluationResult
from backend.app.ml_execution.execution_report import ExecutionReport, ChampionSummary, CandidateSummary
from backend.app.ai_model_optimizer.retraining_engine import (
    RetrainingEngine,
    RetrainingEngineError,
    RetrainingResult,
)


# ── Helper Builders ───────────────────────────────────────────────────


def _make_dataframe(is_regression: bool = False) -> pd.DataFrame:
    if is_regression:
        return pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "species": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        })
    return pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "species": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def _make_dataset_context(dataset_id: str = "ds_01") -> DatasetContext:
    basic_info = DatasetBasicInfo(
        dataset_id=dataset_id,
        file_name="data.csv",
        row_count=10,
        column_count=2,
        memory_usage_bytes=1000,
    )
    col = ColumnContext(
        name="feat_a",
        dtype="float64",
        is_numeric=True,
        is_categorical=False,
        is_datetime=False,
        missing_count=0,
        missing_percentage=0.0,
        unique_count=10,
        unique_percentage=100.0,
        sample_values=[1.0, 2.0],
    )
    col_target = ColumnContext(
        name="species",
        dtype="int64",
        is_numeric=True,
        is_categorical=False,
        is_datetime=False,
        missing_count=0,
        missing_percentage=0.0,
        unique_count=2,
        unique_percentage=20.0,
        sample_values=[0, 1],
    )
    return DatasetContext(
        basic_info=basic_info,
        columns=[col, col_target],
        missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
    )


def _make_problem_definition(
    dataset_id: str = "ds_01",
    request_id: str = "req_01",
    definition_id: str = "pd_01",
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
) -> ProblemDefinition:
    return ProblemDefinition(
        definition_id=definition_id,
        request_id=request_id,
        dataset_id=dataset_id,
        goal="Retraining goal",
        problem_type=problem_type,
        target_column="species",
        target_source=TargetSource.USER,
        feature_columns=["feat_a"],
        excluded_columns=[],
        primary_metric="f1" if problem_type == ProblemType.CLASSIFICATION else "mae",
        status=ResolutionStatus.RESOLVED,
    )


def _make_compute_capabilities(capability_id: str = "cap_01") -> ComputeCapabilities:
    return ComputeCapabilities(
        capability_id=capability_id,
        hardware_profile_id="hw_01",
        compute_tier=ComputeTier.STANDARD,
        memory_constraint=MemoryConstraintLevel.COMFORTABLE,
        cpu_training_available=True,
        gpu_acceleration_available=False,
        accelerator_type=AcceleratorType.NONE,
        safe_parallel_workers=2,
        max_parallel_workers=4,
        available_ram_mb_snapshot=4096,
        total_ram_mb=8192,
        warnings=[],
    )


def _make_optimized_plan(
    dataset_id: str = "ds_01",
    request_id: str = "req_01",
    definition_id: str = "pd_01",
    capability_id: str = "cap_01",
    plan_id: str = "plan_01",
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
) -> MLPlan:
    candidates = [
        ModelCandidate(
            candidate_id="model_001",
            model_family=ModelFamily.LOGISTIC_REGRESSION if problem_type == ProblemType.CLASSIFICATION else ModelFamily.LINEAR_REGRESSION,
            search_strategy=SearchStrategy.NONE,
            reason="Retrain candidate",
        ),
    ]

    return MLPlan(
        plan_id=plan_id,
        dataset_id=dataset_id,
        request_id=request_id,
        problem_definition_id=definition_id,
        compute_capability_id=capability_id,
        problem_type=problem_type,
        target_column="species",
        feature_columns=["feat_a"],
        preprocessing_steps=[],
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
            primary_metric="f1" if problem_type == ProblemType.CLASSIFICATION else "mae",
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


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestRetrainingEngine:
    """Tests covering inputs validation, identity mismatches, successful retraining pipelines, and failures."""

    def test_classification_retraining_success(self):
        """Verify end-to-end retraining success for classification models."""
        df = _make_dataframe()
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        cc = _make_compute_capabilities()
        plan = _make_optimized_plan()

        engine = RetrainingEngine()
        result = engine.retrain(
            dataframe=df,
            dataset_context=ctx,
            problem_definition=prob_def,
            compute_capabilities=cc,
            optimized_plan=plan,
        )

        assert isinstance(result, RetrainingResult)
        assert result.optimized_plan_id == "plan_01"
        assert isinstance(result.execution_result, MLExecutionResult)
        assert isinstance(result.execution_report, ExecutionReport)
        assert result.execution_duration > 0.0
        assert result.execution_report.problem_type == ProblemType.CLASSIFICATION

    def test_regression_retraining_success(self):
        """Verify end-to-end retraining success for regression models."""
        df = _make_dataframe(is_regression=True)
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION)
        cc = _make_compute_capabilities()
        plan = _make_optimized_plan(problem_type=ProblemType.REGRESSION)

        engine = RetrainingEngine()
        result = engine.retrain(
            dataframe=df,
            dataset_context=ctx,
            problem_definition=prob_def,
            compute_capabilities=cc,
            optimized_plan=plan,
        )

        assert isinstance(result, RetrainingResult)
        assert result.execution_report.problem_type == ProblemType.REGRESSION

    def test_validation_rejects_none(self):
        """Verify None inputs raise RetrainingEngineError."""
        df = _make_dataframe()
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        cc = _make_compute_capabilities()
        plan = _make_optimized_plan()

        engine = RetrainingEngine()

        with pytest.raises(RetrainingEngineError, match="dataframe cannot be None"):
            engine.retrain(dataframe=None, dataset_context=ctx, problem_definition=prob_def, compute_capabilities=cc, optimized_plan=plan)

        with pytest.raises(RetrainingEngineError, match="dataset_context cannot be None"):
            engine.retrain(dataframe=df, dataset_context=None, problem_definition=prob_def, compute_capabilities=cc, optimized_plan=plan)

        with pytest.raises(RetrainingEngineError, match="problem_definition cannot be None"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=None, compute_capabilities=cc, optimized_plan=plan)

        with pytest.raises(RetrainingEngineError, match="compute_capabilities cannot be None"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=prob_def, compute_capabilities=None, optimized_plan=plan)

        with pytest.raises(RetrainingEngineError, match="optimized_plan cannot be None"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=prob_def, compute_capabilities=cc, optimized_plan=None)

    def test_validation_rejects_wrong_types(self):
        """Verify wrong input types raise RetrainingEngineError."""
        df = _make_dataframe()
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        cc = _make_compute_capabilities()
        plan = _make_optimized_plan()

        engine = RetrainingEngine()

        with pytest.raises(RetrainingEngineError, match="dataframe must be a pandas DataFrame"):
            engine.retrain(dataframe="not-df", dataset_context=ctx, problem_definition=prob_def, compute_capabilities=cc, optimized_plan=plan)

        with pytest.raises(RetrainingEngineError, match="dataset_context must be a DatasetContext"):
            engine.retrain(dataframe=df, dataset_context="not-ctx", problem_definition=prob_def, compute_capabilities=cc, optimized_plan=plan)

        with pytest.raises(RetrainingEngineError, match="problem_definition must be a ProblemDefinition"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition="not-pd", compute_capabilities=cc, optimized_plan=plan)

        with pytest.raises(RetrainingEngineError, match="compute_capabilities must be a ComputeCapabilities"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=prob_def, compute_capabilities="not-cc", optimized_plan=plan)

        with pytest.raises(RetrainingEngineError, match="optimized_plan must be an MLPlan"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=prob_def, compute_capabilities=cc, optimized_plan="not-plan")

    def test_identity_preservation_mismatches(self):
        """Verify identity mismatches across context, definition, capability, and plan raise RetrainingEngineError."""
        df = _make_dataframe()
        ctx = _make_dataset_context(dataset_id="ds_01")
        prob_def = _make_problem_definition(dataset_id="ds_01", request_id="req_01", definition_id="pd_01")
        cc = _make_compute_capabilities(capability_id="cap_01")
        
        engine = RetrainingEngine()

        # 1. ProblemDefinition dataset_id mismatch
        pd_bad = _make_problem_definition(dataset_id="ds_mismatch")
        plan = _make_optimized_plan(dataset_id="ds_01", request_id="req_01", definition_id="pd_01", capability_id="cap_01")
        with pytest.raises(RetrainingEngineError, match="Identity mismatch.*problem_definition.dataset_id"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=pd_bad, compute_capabilities=cc, optimized_plan=plan)

        # 2. MLPlan dataset_id mismatch
        plan_bad_ds = _make_optimized_plan(dataset_id="ds_mismatch")
        with pytest.raises(RetrainingEngineError, match="Identity mismatch.*optimized_plan.dataset_id"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=prob_def, compute_capabilities=cc, optimized_plan=plan_bad_ds)

        # 3. MLPlan request_id mismatch
        plan_bad_req = _make_optimized_plan(request_id="req_mismatch")
        with pytest.raises(RetrainingEngineError, match="Identity mismatch.*request_id"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=prob_def, compute_capabilities=cc, optimized_plan=plan_bad_req)

        # 4. MLPlan definition_id mismatch
        plan_bad_def = _make_optimized_plan(definition_id="pd_mismatch")
        with pytest.raises(RetrainingEngineError, match="Identity mismatch.*problem_definition_id"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=prob_def, compute_capabilities=cc, optimized_plan=plan_bad_def)

        # 5. MLPlan compute_capability_id mismatch
        plan_bad_cap = _make_optimized_plan(capability_id="cap_mismatch")
        with pytest.raises(RetrainingEngineError, match="Identity mismatch.*compute_capability_id"):
            engine.retrain(dataframe=df, dataset_context=ctx, problem_definition=prob_def, compute_capabilities=cc, optimized_plan=plan_bad_cap)

    def test_validation_rejects_invalid_optimized_plan(self):
        """Verify invalid optimized plans raise RetrainingEngineError."""
        df = _make_dataframe()
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        cc = _make_compute_capabilities()
        
        # A plan with stratified split for regression (invalid for MLPlanValidator)
        plan = _make_optimized_plan(problem_type=ProblemType.REGRESSION)
        plan.split_plan.strategy = SplitStrategy.STRATIFIED

        engine = RetrainingEngine()
        with pytest.raises(RetrainingEngineError, match="Invalid optimized plan"):
            engine.retrain(
                dataframe=df,
                dataset_context=ctx,
                problem_definition=prob_def,
                compute_capabilities=cc,
                optimized_plan=plan,
            )

    def test_subsystem_orchestrator_failure_handling(self):
        """Verify orchestrator failure exceptions are caught and wrapped."""
        df = _make_dataframe()
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        cc = _make_compute_capabilities()
        plan = _make_optimized_plan()

        engine = RetrainingEngine()
        
        # Patch execute to throw error
        with patch.object(engine._orchestrator, "execute", side_effect=Exception("Orchestrator crashed")):
            with pytest.raises(RetrainingEngineError, match="ML Execution Orchestrator pipeline failure"):
                engine.retrain(
                    dataframe=df,
                    dataset_context=ctx,
                    problem_definition=prob_def,
                    compute_capabilities=cc,
                    optimized_plan=plan,
                )

    def test_subsystem_report_builder_failure_handling(self):
        """Verify report builder failure exceptions are caught and wrapped."""
        df = _make_dataframe()
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        cc = _make_compute_capabilities()
        plan = _make_optimized_plan()

        engine = RetrainingEngine()
        
        # Patch build to throw error
        with patch.object(engine._report_builder, "build", side_effect=Exception("Builder crashed")):
            with pytest.raises(RetrainingEngineError, match="Execution Report Builder pipeline failure"):
                engine.retrain(
                    dataframe=df,
                    dataset_context=ctx,
                    problem_definition=prob_def,
                    compute_capabilities=cc,
                    optimized_plan=plan,
                )

    def test_non_mutation(self):
        """Verify the retraining pipeline does not mutate inputs."""
        df = _make_dataframe()
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        cc = _make_compute_capabilities()
        plan = _make_optimized_plan()

        df_copy = df.copy()
        ctx_copy = copy.deepcopy(ctx)
        pd_copy = copy.deepcopy(prob_def)
        cc_copy = copy.deepcopy(cc)
        plan_copy = copy.deepcopy(plan)

        engine = RetrainingEngine()
        engine.retrain(
            dataframe=df,
            dataset_context=ctx,
            problem_definition=prob_def,
            compute_capabilities=cc,
            optimized_plan=plan,
        )

        pd.testing.assert_frame_equal(df, df_copy)
        assert ctx == ctx_copy
        assert prob_def == pd_copy
        assert cc.capability_id == cc_copy.capability_id
        assert plan == plan_copy
