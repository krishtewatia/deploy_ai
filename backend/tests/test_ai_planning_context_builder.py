"""Tests for AI Planning Context Builder (Stage 7D3)."""

import copy
import json
import pytest

from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
    ResourceWarning,
)
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    ColumnStatistics,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)
from backend.app.ml_plan.schemas import (
    DatasetSplitPlan,
    EvaluationPlan,
    ExecutionConstraints,
    FeatureSelectionMethod,
    FeatureSelectionPlan,
    MLPlan,
    MLPlanStatus,
    ModelCandidate,
    ModelFamily,
    PreprocessingOperation,
    PreprocessingStep,
    SearchStrategy,
    SplitStrategy,
)
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ProblemType,
    ProblemWarning,
    ResolutionStatus,
    TargetSource,
)
from backend.app.ai_planning.context_builder import AIPlanningContextBuilder


# ── Fixtures ───────────────────────────────────────────────────────────


def _make_dataset_context() -> DatasetContext:
    columns = [
        ColumnContext(
            name="age", dtype="float64", is_numeric=True, is_categorical=False,
            is_datetime=False, missing_count=10, missing_percentage=1.0,
            unique_count=50, unique_percentage=5.0, sample_values=[25.0, 30.0],
            statistics=ColumnStatistics(mean=35.0, median=34.0, std=10.0, min=18.0, max=80.0),
        ),
        ColumnContext(
            name="department", dtype="object", is_numeric=False, is_categorical=True,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=5, unique_percentage=0.5, sample_values=["sales", "engineering"],
        ),
        ColumnContext(
            name="churn", dtype="int64", is_numeric=True, is_categorical=True,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=2, unique_percentage=0.2, sample_values=[0, 1],
        ),
    ]
    return DatasetContext(
        basic_info=DatasetBasicInfo(dataset_id="ds_01", file_name="data.csv",
                                    row_count=1000, column_count=3, memory_usage_bytes=50000),
        columns=columns,
        missing_data=MissingDataSummary(total_missing_cells=10, columns_with_missing=["age"]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
    )


def _make_user_request() -> UserMLRequest:
    return UserMLRequest(
        request_id="req_01", goal="Predict churn",
        target_column="churn", additional_context="Monthly data.",
        excluded_columns=["id"],
    )


def _make_problem_definition() -> ProblemDefinition:
    return ProblemDefinition(
        definition_id="pd_01", request_id="req_01", dataset_id="ds_01",
        goal="Predict churn", problem_type=ProblemType.CLASSIFICATION,
        target_column="churn", target_source=TargetSource.USER,
        feature_columns=["age", "department"], excluded_columns=[],
        primary_metric="f1", status=ResolutionStatus.RESOLVED,
        warnings=[ProblemWarning(code="W1", message="Minor warning.")],
    )


def _make_compute_capabilities() -> ComputeCapabilities:
    return ComputeCapabilities(
        capability_id="cap_01", hardware_profile_id="hw_01",
        compute_tier=ComputeTier.STANDARD, memory_constraint=MemoryConstraintLevel.MODERATE,
        cpu_training_available=True, gpu_acceleration_available=False,
        accelerator_type=AcceleratorType.NONE, safe_parallel_workers=4,
        max_parallel_workers=8, available_ram_mb_snapshot=4096, total_ram_mb=8192,
        warnings=[ResourceWarning(code="RW1", message="Resource note.")],
    )


def _make_baseline_plan() -> MLPlan:
    return MLPlan(
        plan_id="plan_01", dataset_id="ds_01", request_id="req_01",
        problem_definition_id="pd_01", compute_capability_id="cap_01",
        problem_type=ProblemType.CLASSIFICATION, target_column="churn",
        feature_columns=["age", "department"],
        preprocessing_steps=[
            PreprocessingStep(step_id="prep_001", operation=PreprocessingOperation.IMPUTE_MEDIAN,
                              columns=["age"], parameters={}, reason="Impute age."),
        ],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE, candidate_columns=["age", "department"],
            reason="No selection."),
        split_plan=DatasetSplitPlan(strategy=SplitStrategy.STRATIFIED, test_size=0.2,
                                     stratify_column="churn"),
        model_candidates=[
            ModelCandidate(candidate_id="model_001", model_family=ModelFamily.LOGISTIC_REGRESSION,
                           parameters={"random_state": 42}, search_strategy=SearchStrategy.NONE,
                           search_space={}, reason="Baseline."),
        ],
        evaluation_plan=EvaluationPlan(primary_metric="f1", secondary_metrics=["accuracy"],
                                        cross_validation_folds=5),
        execution_constraints=ExecutionConstraints(
            parallel_workers=4, use_gpu_acceleration=False,
            accelerator_type=AcceleratorType.NONE, compute_tier=ComputeTier.STANDARD),
        status=MLPlanStatus.READY,
    )


# ── Tests ──────────────────────────────────────────────────────────────


class TestAIPlanningContextBuilder:
    def test_returns_dict(self):
        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        assert isinstance(ctx, dict)

    def test_json_serializable(self):
        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        j = json.dumps(ctx)
        assert isinstance(json.loads(j), dict)

    def test_dataset_section(self):
        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        ds = ctx["dataset"]
        assert ds["dataset_id"] == "ds_01"
        assert ds["row_count"] == 1000
        assert ds["column_count"] == 3
        assert len(ds["columns"]) == 3

        age_col = ds["columns"][0]
        assert age_col["name"] == "age"
        assert age_col["is_numeric"] is True
        assert "statistics" in age_col
        assert age_col["statistics"]["mean"] == 35.0

    def test_column_without_statistics(self):
        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        dept_col = ctx["dataset"]["columns"][1]
        assert "statistics" not in dept_col

    def test_user_goal_section(self):
        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        ug = ctx["user_goal"]
        assert ug["goal"] == "Predict churn"
        assert ug["target_column"] == "churn"
        assert ug["additional_context"] == "Monthly data."
        assert ug["excluded_columns"] == ["id"]

    def test_resolved_problem_section(self):
        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        rp = ctx["resolved_problem"]
        assert rp["target_column"] == "churn"
        assert rp["problem_type"] == "classification"
        assert rp["primary_metric"] == "f1"
        assert len(rp["warnings"]) == 1
        assert rp["warnings"][0]["code"] == "W1"

    def test_compute_section(self):
        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        cc = ctx["compute_capabilities"]
        assert cc["compute_tier"] == "standard"
        assert cc["safe_parallel_workers"] == 4
        assert cc["gpu_acceleration_available"] is False
        assert len(cc["warnings"]) == 1

    def test_baseline_plan_section(self):
        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        bp = ctx["baseline_plan"]
        assert len(bp["preprocessing_steps"]) == 1
        assert bp["preprocessing_steps"][0]["operation"] == "impute_median"
        assert bp["feature_engineering_steps"] == []
        assert bp["feature_selection"]["method"] == "none"
        assert bp["split_plan"]["strategy"] == "stratified"
        assert len(bp["model_candidates"]) == 1
        assert bp["model_candidates"][0]["model_family"] == "logistic_regression"
        assert bp["evaluation_plan"]["primary_metric"] == "f1"
        assert bp["execution_constraints"]["parallel_workers"] == 4

    def test_does_not_mutate_inputs(self):
        builder = AIPlanningContextBuilder()
        ds = _make_dataset_context()
        req = _make_user_request()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()
        bp = _make_baseline_plan()

        orig_ds = copy.deepcopy(ds)
        orig_req = copy.deepcopy(req)
        orig_pd = copy.deepcopy(pd)
        orig_cc = copy.deepcopy(cc)
        orig_bp = copy.deepcopy(bp)

        builder.build(dataset_context=ds, user_request=req,
                       problem_definition=pd, compute_capabilities=cc,
                       baseline_plan=bp)

        assert ds == orig_ds
        assert req == orig_req
        assert pd == orig_pd
        assert cc == orig_cc
        assert bp == orig_bp

    def test_deterministic(self):
        builder = AIPlanningContextBuilder()
        args = dict(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )
        ctx1 = builder.build(**args)
        ctx2 = builder.build(**args)
        assert ctx1 == ctx2

    def test_sample_values_limited(self):
        """Ensure sample values are capped at 5 items."""
        cols = [
            ColumnContext(
                name="x", dtype="float64", is_numeric=True, is_categorical=False,
                is_datetime=False, missing_count=0, missing_percentage=0.0,
                unique_count=100, unique_percentage=10.0,
                sample_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ),
            ColumnContext(
                name="y", dtype="int64", is_numeric=True, is_categorical=True,
                is_datetime=False, missing_count=0, missing_percentage=0.0,
                unique_count=2, unique_percentage=0.2, sample_values=[0, 1],
            ),
        ]
        ds = DatasetContext(
            basic_info=DatasetBasicInfo(dataset_id="ds_01", file_name="d.csv",
                                        row_count=100, column_count=2, memory_usage_bytes=1000),
            columns=cols,
            missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
            duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
        )
        builder = AIPlanningContextBuilder()
        pd = _make_problem_definition()
        pd.feature_columns = ["x"]
        pd.target_column = "y"

        bp = MLPlan(
            plan_id="plan_01", dataset_id="ds_01", request_id="req_01",
            problem_definition_id="pd_01", compute_capability_id="cap_01",
            problem_type=ProblemType.CLASSIFICATION, target_column="y",
            feature_columns=["x"],
            feature_selection=FeatureSelectionPlan(
                method=FeatureSelectionMethod.NONE, candidate_columns=["x"], reason="No."),
            split_plan=DatasetSplitPlan(strategy=SplitStrategy.STRATIFIED, test_size=0.2,
                                         stratify_column="y"),
            model_candidates=[ModelCandidate(candidate_id="m1", model_family=ModelFamily.LOGISTIC_REGRESSION,
                                              search_strategy=SearchStrategy.NONE, search_space={}, reason="B.")],
            evaluation_plan=EvaluationPlan(primary_metric="f1", cross_validation_folds=5),
            execution_constraints=ExecutionConstraints(parallel_workers=2, use_gpu_acceleration=False,
                                                        accelerator_type=AcceleratorType.NONE,
                                                        compute_tier=ComputeTier.STANDARD),
            status=MLPlanStatus.READY,
        )

        ctx = builder.build(dataset_context=ds, user_request=_make_user_request(),
                             problem_definition=pd, compute_capabilities=_make_compute_capabilities(),
                             baseline_plan=bp)
        assert len(ctx["dataset"]["columns"][0]["sample_values"]) == 5

    def test_baseline_plan_with_feature_engineering_and_warnings(self):
        """Cover feature engineering loop body and plan warnings loop."""
        from backend.app.ml_plan.schemas import (
            FeatureEngineeringOperation,
            FeatureEngineeringStep,
            MLPlanWarning,
        )
        plan = _make_baseline_plan()
        plan.feature_engineering_steps = [
            FeatureEngineeringStep(
                step_id="fe_001",
                operation=FeatureEngineeringOperation.LOG_TRANSFORM,
                input_columns=["age"],
                output_columns=["log_age"],
                parameters={},
                reason="Log transform age.",
            ),
        ]
        plan.warnings = [
            MLPlanWarning(code="PW1", message="Plan warning note."),
        ]

        builder = AIPlanningContextBuilder()
        ctx = builder.build(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=plan,
        )
        bp = ctx["baseline_plan"]
        assert len(bp["feature_engineering_steps"]) == 1
        assert bp["feature_engineering_steps"][0]["operation"] == "log_transform"
        assert len(bp["warnings"]) == 1
        assert bp["warnings"][0]["code"] == "PW1"

