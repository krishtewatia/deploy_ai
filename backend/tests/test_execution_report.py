"""Unit tests for ExecutionReportBuilder."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock
import pytest

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
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
    MLPlanWarning,
)
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ResolutionStatus,
    TargetSource,
    ProblemWarning,
)
from backend.app.ml_execution.orchestrator import MLExecutionResult
from backend.app.ml_execution.evaluation_engine import EvaluationResult
from backend.app.ml_execution.execution_report import (
    ExecutionReport,
    ExecutionReportBuilder,
    ExecutionReportBuilderError,
)


# ── Helper Builders ───────────────────────────────────────────────────


def _make_dataset_context(dataset_id: str = "ds_01") -> DatasetContext:
    basic_info = DatasetBasicInfo(
        dataset_id=dataset_id,
        file_name="data.csv",
        row_count=100,
        column_count=3,
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
        unique_count=50,
        unique_percentage=5.0,
        sample_values=[1.0, 2.0],
    )
    return DatasetContext(
        basic_info=basic_info,
        columns=[col],
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
        goal="Classification goal",
        problem_type=problem_type,
        target_column="species",
        target_source=TargetSource.USER,
        feature_columns=["feat_a", "feat_b"],
        excluded_columns=[],
        primary_metric="f1" if problem_type == ProblemType.CLASSIFICATION else "mae",
        status=ResolutionStatus.RESOLVED,
    )


def _make_plan(
    dataset_id: str = "ds_01",
    request_id: str = "req_01",
    definition_id: str = "pd_01",
    plan_id: str = "plan_01",
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    candidates: list[ModelCandidate] | None = None,
) -> MLPlan:
    if candidates is None:
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
        dataset_id=dataset_id,
        request_id=request_id,
        problem_definition_id=definition_id,
        compute_capability_id="cap_01",
        problem_type=problem_type,
        target_column="species",
        feature_columns=["feat_a", "feat_b"],
        preprocessing_steps=[],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["feat_a", "feat_b"],
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


def _make_eval_result(
    candidate_id: str,
    model_family: ModelFamily,
    primary_metric: str = "f1",
    primary_metric_value: float = 0.95,
) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        model_family=model_family,
        predictions=[0, 1],
        primary_metric=primary_metric,
        primary_metric_value=primary_metric_value,
        all_metrics={primary_metric: primary_metric_value},
        confusion_matrix=[[1, 0], [0, 1]],
        classification_report={},
        feature_importance={"feat_a": 0.6, "feat_b": 0.4},
        cross_validation_scores=[0.94, 0.96],
        evaluation_duration_seconds=0.01,
        train_score=0.98,
        test_score=0.95,
        prediction_count=2,
        training_duration=0.05,
        evaluation_duration=0.01,
        model_parameters={"C": 1.0},
        warnings=["Overfitting warn"],
    )


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestExecutionReportBuilder:
    """Tests covering the ExecutionReportBuilder consolidated report assembly."""

    def test_classification_report_success(self):
        """Verify successful consolidated report generation for classification problem."""
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        prob_def.warnings = [ProblemWarning(code="W_TEST", message="Test definition warning")]
        
        plan = _make_plan()
        plan.warnings = [MLPlanWarning(code="P_TEST", message="Test plan warning")]
        
        # Build execution result
        eval_res = _make_eval_result("model_001", ModelFamily.LOGISTIC_REGRESSION)
        exec_res = MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=prob_def.definition_id,
            candidate_results={"model_001": eval_res},
            best_candidate_id="model_001",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.5,
        )

        builder = ExecutionReportBuilder()
        report = builder.build(
            dataset_context=ctx,
            problem_definition=prob_def,
            plan=plan,
            execution_result=exec_res,
        )

        assert isinstance(report, ExecutionReport)
        assert report.report_id.startswith("report_")
        assert report.dataset_id == "ds_01"
        assert report.plan_id == "plan_01"
        assert report.problem_type == ProblemType.CLASSIFICATION
        assert report.target_column == "species"
        assert report.feature_columns == ["feat_a", "feat_b"]
        assert len(report.candidate_summaries) == 1
        assert report.candidate_summaries[0].candidate_id == "model_001"
        assert report.candidate_summaries[0].search_strategy == SearchStrategy.NONE
        assert report.candidate_summaries[0].best_parameters == {"C": 1.0}
        
        assert report.champion_summary.candidate_id == "model_001"
        assert report.champion_summary.feature_importance == {"feat_a": 0.6, "feat_b": 0.4}
        
        # Verify consolidated warnings list
        assert any("Test definition warning" in w for w in report.warnings)
        assert any("Test plan warning" in w for w in report.warnings)
        assert any("Overfitting warn" in w for w in report.warnings)

    def test_regression_report_success(self):
        """Verify successful consolidated report generation for regression problem."""
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION)
        plan = _make_plan(
            problem_type=ProblemType.REGRESSION,
            candidates=[
                ModelCandidate(
                    candidate_id="model_001",
                    model_family=ModelFamily.LINEAR_REGRESSION,
                    search_strategy=SearchStrategy.NONE,
                    reason="Provides baseline",
                )
            ],
        )

        eval_res = _make_eval_result("model_001", ModelFamily.LINEAR_REGRESSION, primary_metric="mae", primary_metric_value=0.12)
        exec_res = MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=prob_def.definition_id,
            candidate_results={"model_001": eval_res},
            best_candidate_id="model_001",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.4,
        )

        builder = ExecutionReportBuilder()
        report = builder.build(
            dataset_context=ctx,
            problem_definition=prob_def,
            plan=plan,
            execution_result=exec_res,
        )

        assert isinstance(report, ExecutionReport)
        assert report.problem_type == ProblemType.REGRESSION
        assert report.champion_summary.primary_metric_value == 0.12

    def test_multiple_candidates_report_success(self):
        """Verify report creation with multiple algorithm candidates."""
        candidates = [
            ModelCandidate(
                candidate_id="model_001",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="c1",
            ),
            ModelCandidate(
                candidate_id="model_002",
                model_family=ModelFamily.RANDOM_FOREST,
                search_strategy=SearchStrategy.GRID,
                search_space={"n_estimators": [10, 50]},
                reason="c2",
            ),
        ]
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        plan = _make_plan(candidates=candidates)

        eval_res_1 = _make_eval_result("model_001", ModelFamily.LOGISTIC_REGRESSION, primary_metric_value=0.85)
        eval_res_2 = _make_eval_result("model_002", ModelFamily.RANDOM_FOREST, primary_metric_value=0.92)

        exec_res = MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=prob_def.definition_id,
            candidate_results={"model_001": eval_res_1, "model_002": eval_res_2},
            best_candidate_id="model_002",
            best_model=MagicMock(),
            best_evaluation=eval_res_2,
            execution_duration_seconds=1.2,
        )

        builder = ExecutionReportBuilder()
        report = builder.build(
            dataset_context=ctx,
            problem_definition=prob_def,
            plan=plan,
            execution_result=exec_res,
        )

        assert len(report.candidate_summaries) == 2
        assert report.champion_summary.candidate_id == "model_002"
        assert report.champion_summary.primary_metric_value == 0.92

    def test_validation_rejects_none(self):
        """Verify passing None arguments raises ExecutionReportBuilderError."""
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        plan = _make_plan()
        exec_res = MLExecutionResult(
            plan_id="plan_01",
            problem_definition_id="pd_01",
            candidate_results={},
            best_candidate_id="model_001",
            best_model=MagicMock(),
            best_evaluation=MagicMock(),
            execution_duration_seconds=0.1,
        )

        builder = ExecutionReportBuilder()

        with pytest.raises(ExecutionReportBuilderError, match="dataset_context cannot be None"):
            builder.build(dataset_context=None, problem_definition=prob_def, plan=plan, execution_result=exec_res)

        with pytest.raises(ExecutionReportBuilderError, match="problem_definition cannot be None"):
            builder.build(dataset_context=ctx, problem_definition=None, plan=plan, execution_result=exec_res)

        with pytest.raises(ExecutionReportBuilderError, match="plan cannot be None"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=None, execution_result=exec_res)

        with pytest.raises(ExecutionReportBuilderError, match="execution_result cannot be None"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan, execution_result=None)

    def test_validation_rejects_wrong_types(self):
        """Verify wrong object types raise ExecutionReportBuilderError."""
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        plan = _make_plan()
        exec_res = MLExecutionResult(
            plan_id="plan_01",
            problem_definition_id="pd_01",
            candidate_results={},
            best_candidate_id="model_001",
            best_model=MagicMock(),
            best_evaluation=MagicMock(),
            execution_duration_seconds=0.1,
        )

        builder = ExecutionReportBuilder()

        with pytest.raises(ExecutionReportBuilderError, match="dataset_context must be a DatasetContext"):
            builder.build(dataset_context="not-ctx", problem_definition=prob_def, plan=plan, execution_result=exec_res)

        with pytest.raises(ExecutionReportBuilderError, match="problem_definition must be a ProblemDefinition"):
            builder.build(dataset_context=ctx, problem_definition="not-pd", plan=plan, execution_result=exec_res)

        with pytest.raises(ExecutionReportBuilderError, match="plan must be an MLPlan"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan="not-plan", execution_result=exec_res)

        with pytest.raises(ExecutionReportBuilderError, match="execution_result must be an MLExecutionResult"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan, execution_result="not-result")

    def test_identity_consistency_checks(self):
        """Verify validation rules catch ID mismatches between inputs."""
        ctx = _make_dataset_context(dataset_id="ds_01")
        prob_def = _make_problem_definition(dataset_id="ds_01", request_id="req_01", definition_id="pd_01")
        plan = _make_plan(dataset_id="ds_01", request_id="req_01", definition_id="pd_01", plan_id="plan_01")
        
        eval_res = _make_eval_result("model_001", ModelFamily.LOGISTIC_REGRESSION)
        exec_res = MLExecutionResult(
            plan_id="plan_01",
            problem_definition_id="pd_01",
            candidate_results={"model_001": eval_res},
            best_candidate_id="model_001",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.1,
        )

        builder = ExecutionReportBuilder()

        # 1. Dataset ID Mismatch Definition
        prob_def_bad = _make_problem_definition(dataset_id="ds_mismatch")
        with pytest.raises(ExecutionReportBuilderError, match="dataset_id mismatch.*ProblemDefinition"):
            builder.build(dataset_context=ctx, problem_definition=prob_def_bad, plan=plan, execution_result=exec_res)

        # 2. Dataset ID Mismatch Plan
        plan_bad_ds = _make_plan(dataset_id="ds_mismatch")
        with pytest.raises(ExecutionReportBuilderError, match="dataset_id mismatch.*MLPlan"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan_bad_ds, execution_result=exec_res)

        # 3. Request ID Mismatch
        plan_bad_req = _make_plan(request_id="req_mismatch")
        with pytest.raises(ExecutionReportBuilderError, match="request_id mismatch"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan_bad_req, execution_result=exec_res)

        # 4. Definition ID Mismatch Plan
        plan_bad_def = _make_plan(definition_id="pd_mismatch")
        with pytest.raises(ExecutionReportBuilderError, match="problem_definition_id mismatch.*MLPlan"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan_bad_def, execution_result=exec_res)

        # 5. Definition ID Mismatch Execution Result
        exec_bad_def = copy.deepcopy(exec_res)
        exec_bad_def.problem_definition_id = "pd_mismatch"
        with pytest.raises(ExecutionReportBuilderError, match="problem_definition_id mismatch.*MLExecutionResult"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan, execution_result=exec_bad_def)

        # 6. Plan ID Mismatch Execution Result
        exec_bad_plan = copy.deepcopy(exec_res)
        exec_bad_plan.plan_id = "plan_mismatch"
        with pytest.raises(ExecutionReportBuilderError, match="plan_id mismatch"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan, execution_result=exec_bad_plan)

    def test_missing_champion(self):
        """Verify errors are raised for missing champion candidate IDs."""
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        plan = _make_plan()
        
        eval_res = _make_eval_result("model_001", ModelFamily.LOGISTIC_REGRESSION)
        builder = ExecutionReportBuilder()

        # best_candidate_id is empty/None
        exec_res_bad1 = MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=prob_def.definition_id,
            candidate_results={"model_001": eval_res},
            best_candidate_id="",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.1,
        )
        with pytest.raises(ExecutionReportBuilderError, match="best_candidate_id.*cannot be empty"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan, execution_result=exec_res_bad1)

        # best_candidate_id not in candidate results
        exec_res_bad2 = MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=prob_def.definition_id,
            candidate_results={"model_001": eval_res},
            best_candidate_id="model_missing",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.1,
        )
        with pytest.raises(ExecutionReportBuilderError, match="best_candidate_id.*not found in candidate results"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan, execution_result=exec_res_bad2)

    def test_missing_candidate_list(self):
        """Verify validation rules reject missing plan or execution candidates."""
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        
        eval_res = _make_eval_result("model_001", ModelFamily.LOGISTIC_REGRESSION)
        builder = ExecutionReportBuilder()

        # 1. Plan candidate list empty
        plan_empty = _make_plan()
        plan_empty.model_candidates = []
        exec_res = MLExecutionResult(
            plan_id="plan_01",
            problem_definition_id=prob_def.definition_id,
            candidate_results={"model_001": eval_res},
            best_candidate_id="model_001",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.1,
        )
        with pytest.raises(ExecutionReportBuilderError, match="plan.model_candidates.*cannot be empty"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan_empty, execution_result=exec_res)

        # 2. Execution candidate results empty
        plan = _make_plan()
        exec_res_empty = MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=prob_def.definition_id,
            candidate_results={},
            best_candidate_id="model_001",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.1,
        )
        with pytest.raises(ExecutionReportBuilderError, match="execution_result.candidate_results cannot be empty"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan, execution_result=exec_res_empty)

        # 3. Candidate in plan but missing from execution results
        plan_two = _make_plan(candidates=[
            ModelCandidate(candidate_id="model_001", model_family=ModelFamily.LOGISTIC_REGRESSION, search_strategy=SearchStrategy.NONE, reason="c1"),
            ModelCandidate(candidate_id="model_002", model_family=ModelFamily.RANDOM_FOREST, search_strategy=SearchStrategy.NONE, reason="c2"),
        ])
        with pytest.raises(ExecutionReportBuilderError, match="missing from execution_result.candidate_results"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan_two, execution_result=exec_res)

        # 4. Best candidate ID mismatch with plan model candidates list lookup
        exec_res_bad_best = MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=prob_def.definition_id,
            candidate_results={"model_001": eval_res, "model_002": eval_res},
            best_candidate_id="model_002",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.1,
        )
        with pytest.raises(ExecutionReportBuilderError, match="Best candidate ID.*not found in plan.model_candidates"):
            builder.build(dataset_context=ctx, problem_definition=prob_def, plan=plan, execution_result=exec_res_bad_best)

    def test_non_mutation(self):
        """Verify inputs are not mutated by the report builder."""
        ctx = _make_dataset_context()
        prob_def = _make_problem_definition()
        plan = _make_plan()
        
        eval_res = _make_eval_result("model_001", ModelFamily.LOGISTIC_REGRESSION)
        exec_res = MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=prob_def.definition_id,
            candidate_results={"model_001": eval_res},
            best_candidate_id="model_001",
            best_model=MagicMock(),
            best_evaluation=eval_res,
            execution_duration_seconds=0.5,
        )

        ctx_copy = copy.deepcopy(ctx)
        prob_def_copy = copy.deepcopy(prob_def)
        plan_copy = copy.deepcopy(plan)
        exec_res_copy = copy.deepcopy(exec_res)

        builder = ExecutionReportBuilder()
        builder.build(
            dataset_context=ctx,
            problem_definition=prob_def,
            plan=plan,
            execution_result=exec_res,
        )

        assert ctx == ctx_copy
        assert prob_def == prob_def_copy
        assert plan == plan_copy
        assert exec_res.plan_id == exec_res_copy.plan_id
        assert exec_res.best_candidate_id == exec_res_copy.best_candidate_id
