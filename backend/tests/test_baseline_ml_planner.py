"""Unit tests for BaselineMLPlanner.

Covers all 93 specified test scenarios for the baseline planner.
"""

import copy
import json
import pytest
from unittest.mock import MagicMock

from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
    ResourceWarning,
)
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
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
    MLPlanWarning,
    ModelCandidate,
    ModelFamily,
    PreprocessingOperation,
    PreprocessingStep,
    SearchStrategy,
    SplitStrategy,
)
from backend.app.ml_plan.validator import (
    MLPlanValidationError,
    MLPlanValidationIssue,
    MLPlanValidationResult,
    MLPlanValidator,
    ValidationSeverity,
)
from backend.app.ml_plan.baseline_planner import BaselineMLPlanner, BaselineMLPlannerError
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import (
    ConfirmationItem,
    ProblemDefinition,
    ProblemType,
    ResolutionStatus,
    TargetSource,
)


# ── Helper Builders ───────────────────────────────────────────────────


def _make_dataset_context(
    dataset_id: str = "ds_01",
    columns: list[ColumnContext] = None,
) -> DatasetContext:
    if columns is None:
        columns = [
            ColumnContext(
                name="age",
                dtype="float64",
                is_numeric=True,
                is_categorical=False,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=50,
                unique_percentage=5.0,
                sample_values=[25.0, 30.0],
            ),
            ColumnContext(
                name="department",
                dtype="object",
                is_numeric=False,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=5,
                unique_percentage=0.5,
                sample_values=["sales", "engineering"],
            ),
            ColumnContext(
                name="signup_date",
                dtype="datetime64",
                is_numeric=False,
                is_categorical=False,
                is_datetime=True,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=100,
                unique_percentage=10.0,
                sample_values=["2026-01-01"],
            ),
            ColumnContext(
                name="churn",
                dtype="int64",
                is_numeric=True,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
        ]

    basic_info = DatasetBasicInfo(
        dataset_id=dataset_id,
        file_name="data.csv",
        row_count=1000,
        column_count=len(columns),
        memory_usage_bytes=50000,
    )
    return DatasetContext(
        basic_info=basic_info,
        columns=columns,
        missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
    )


def _make_problem_definition(
    dataset_id: str = "ds_01",
    request_id: str = "req_01",
    definition_id: str = "pd_01",
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    target_column: str = "churn",
    feature_columns: list[str] = None,
    status: ResolutionStatus = ResolutionStatus.RESOLVED,
) -> ProblemDefinition:
    if feature_columns is None:
        feature_columns = ["age", "department"]

    primary_metric = "f1" if problem_type == ProblemType.CLASSIFICATION else "mae"

    confirmation_items = []
    if status == ResolutionStatus.NEEDS_CONFIRMATION:
        confirmation_items = [ConfirmationItem(key="k", question="q", reason="r")]

    return ProblemDefinition(
        definition_id=definition_id,
        request_id=request_id,
        dataset_id=dataset_id,
        goal="Predict customer churn",
        problem_type=problem_type,
        target_column=target_column,
        target_source=TargetSource.USER,
        feature_columns=feature_columns,
        excluded_columns=[],
        primary_metric=primary_metric,
        status=status,
        confirmation_items=confirmation_items,
    )


def _make_user_request(
    request_id: str = "req_01",
    target_column: str = "churn",
) -> UserMLRequest:
    return UserMLRequest(
        request_id=request_id,
        goal="Build baseline predictive model",
        target_column=target_column,
    )


def _make_compute_capabilities(
    capability_id: str = "cap_01",
    hardware_profile_id: str = "hw_01",
    compute_tier: ComputeTier = ComputeTier.STANDARD,
    safe_parallel_workers: int = 4,
    gpu_acceleration_available: bool = False,
    accelerator_type: AcceleratorType = AcceleratorType.NONE,
    warnings: list[ResourceWarning] = None,
) -> ComputeCapabilities:
    return ComputeCapabilities(
        capability_id=capability_id,
        hardware_profile_id=hardware_profile_id,
        compute_tier=compute_tier,
        memory_constraint=MemoryConstraintLevel.MODERATE,
        cpu_training_available=True,
        gpu_acceleration_available=gpu_acceleration_available,
        accelerator_type=accelerator_type,
        safe_parallel_workers=safe_parallel_workers,
        max_parallel_workers=8,
        available_ram_mb_snapshot=4096,
        total_ram_mb=8192,
        warnings=warnings or [],
    )


# ── Tests ──────────────────────────────────────────────────────────────


class TestBaselinePlannerInputs:
    """Scenarios 1-12: Inputs and upstream state validation tests."""

    # 1-4. Non-Pydantic inputs rejected
    def test_invalid_input_types_rejected(self):
        planner = BaselineMLPlanner()
        ds = _make_dataset_context()
        pd = _make_problem_definition()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        with pytest.raises(BaselineMLPlannerError, match="dataset_context must be an instance"):
            planner.create_plan(dataset_context="not-a-ds", problem_definition=pd, user_request=req, compute_capabilities=cc)
        with pytest.raises(BaselineMLPlannerError, match="problem_definition must be an instance"):
            planner.create_plan(dataset_context=ds, problem_definition="not-a-pd", user_request=req, compute_capabilities=cc)
        with pytest.raises(BaselineMLPlannerError, match="user_request must be an instance"):
            planner.create_plan(dataset_context=ds, problem_definition=pd, user_request="not-a-req", compute_capabilities=cc)
        with pytest.raises(BaselineMLPlannerError, match="compute_capabilities must be an instance"):
            planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=req, compute_capabilities="not-a-cc")

    # 5. Invalid validator rejected
    def test_invalid_validator_rejected(self):
        with pytest.raises(BaselineMLPlannerError, match="validator must be an instance of MLPlanValidator"):
            BaselineMLPlanner(validator="not-a-validator")

    # 6. Resolved ProblemDefinition accepted
    def test_resolved_accepted(self):
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(status=ResolutionStatus.RESOLVED),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        assert isinstance(plan, MLPlan)

    # 7-8. Unresolved ProblemDefinition statuses rejected
    @pytest.mark.parametrize("status", [ResolutionStatus.BLOCKED, ResolutionStatus.NEEDS_CONFIRMATION])
    def test_unresolved_problem_definition_rejected(self, status):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition(status=status)
        with pytest.raises(BaselineMLPlannerError, match="Cannot plan for unresolved problem definition"):
            planner.create_plan(
                dataset_context=_make_dataset_context(),
                problem_definition=pd,
                user_request=_make_user_request(),
                compute_capabilities=_make_compute_capabilities(),
            )

    # 9. Dataset ID mismatch
    def test_dataset_id_mismatch_rejected(self):
        planner = BaselineMLPlanner()
        ds = _make_dataset_context(dataset_id="ds_real")
        pd = _make_problem_definition(dataset_id="ds_mismatch")
        with pytest.raises(BaselineMLPlannerError, match="dataset_context ID mismatch"):
            planner.create_plan(
                dataset_context=ds,
                problem_definition=pd,
                user_request=_make_user_request(),
                compute_capabilities=_make_compute_capabilities(),
            )

    # 10. Request ID mismatch
    def test_request_id_mismatch_rejected(self):
        planner = BaselineMLPlanner()
        req = _make_user_request(request_id="req_real")
        pd = _make_problem_definition(request_id="req_mismatch")
        with pytest.raises(BaselineMLPlannerError, match="user_request ID mismatch"):
            planner.create_plan(
                dataset_context=_make_dataset_context(),
                problem_definition=pd,
                user_request=req,
                compute_capabilities=_make_compute_capabilities(),
            )

    # 11. Missing target in DatasetContext
    def test_missing_target_rejected(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition(target_column="missing_col")
        req = _make_user_request(target_column="missing_col")
        with pytest.raises(BaselineMLPlannerError, match="Target column 'missing_col' does not exist"):
            planner.create_plan(
                dataset_context=_make_dataset_context(),
                problem_definition=pd,
                user_request=req,
                compute_capabilities=_make_compute_capabilities(),
            )

    # 12. Missing feature in DatasetContext
    def test_missing_feature_rejected(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition(feature_columns=["age", "missing_feat"])
        with pytest.raises(BaselineMLPlannerError, match="Feature column 'missing_feat' does not exist"):
            planner.create_plan(
                dataset_context=_make_dataset_context(),
                problem_definition=pd,
                user_request=_make_user_request(),
                compute_capabilities=_make_compute_capabilities(),
            )


class TestBaselinePlannerIdentifiersAndCore:
    """Scenarios 13-22: Copy checks and Core values."""

    def test_ids_and_core_plan_properties(self):
        planner = BaselineMLPlanner()
        ds = _make_dataset_context(dataset_id="ds_custom")
        pd = _make_problem_definition(dataset_id="ds_custom", request_id="req_custom", definition_id="pd_custom")
        req = _make_user_request(request_id="req_custom")
        cc = _make_compute_capabilities(capability_id="cap_custom")

        plan = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=req, compute_capabilities=cc)

        # 13. Unique plan ID generated
        assert plan.plan_id.startswith("mlplan_")
        assert len(plan.plan_id) > 10

        # 14-17. Identifier linkage
        assert plan.dataset_id == "ds_custom"
        assert plan.request_id == "req_custom"
        assert plan.problem_definition_id == "pd_custom"
        assert plan.compute_capability_id == "cap_custom"

        # 18-20. Problem description
        assert plan.problem_type == ProblemType.CLASSIFICATION
        assert plan.target_column == "churn"
        assert plan.feature_columns == ["age", "department"]  # Order preserved

        # 21-22. Successful status
        assert plan.status == MLPlanStatus.READY
        assert plan.confirmation_items == []


class TestBaselinePlannerPreprocessing:
    """Scenarios 23-36: Preprocessing rules execution tests."""

    # 23. Missing numeric feature gets median imputation
    def test_missing_numeric_gets_median(self):
        planner = BaselineMLPlanner()
        columns = [
            ColumnContext(
                name="age",
                dtype="float64",
                is_numeric=True,
                is_categorical=False,
                is_datetime=False,
                missing_count=10,
                missing_percentage=1.0,
                unique_count=50,
                unique_percentage=5.0,
                sample_values=[25.0, 30.0],
            ),
            ColumnContext(
                name="churn",
                dtype="int64",
                is_numeric=True,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
        ]
        ds = _make_dataset_context(columns=columns)
        pd = _make_problem_definition(feature_columns=["age"])
        plan = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=_make_user_request(), compute_capabilities=_make_compute_capabilities())

        # Impute median must exist
        median_steps = [s for s in plan.preprocessing_steps if s.operation == PreprocessingOperation.IMPUTE_MEDIAN]
        assert len(median_steps) == 1
        assert median_steps[0].columns == ["age"]

    # 24. Numeric feature without missing values gets no imputation
    def test_numeric_no_missing_no_impute(self):
        planner = BaselineMLPlanner()
        ds = _make_dataset_context()
        pd = _make_problem_definition(feature_columns=["age"])
        plan = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=_make_user_request(), compute_capabilities=_make_compute_capabilities())

        median_steps = [s for s in plan.preprocessing_steps if s.operation == PreprocessingOperation.IMPUTE_MEDIAN]
        assert len(median_steps) == 0

    # 25. Missing categorical feature gets mode imputation
    def test_missing_categorical_gets_mode(self):
        planner = BaselineMLPlanner()
        columns = [
            ColumnContext(
                name="department",
                dtype="object",
                is_numeric=False,
                is_categorical=True,
                is_datetime=False,
                missing_count=5,
                missing_percentage=0.5,
                unique_count=5,
                unique_percentage=0.5,
                sample_values=["sales"],
            ),
            ColumnContext(
                name="churn",
                dtype="int64",
                is_numeric=True,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
        ]
        ds = _make_dataset_context(columns=columns)
        pd = _make_problem_definition(feature_columns=["department"])
        plan = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=_make_user_request(), compute_capabilities=_make_compute_capabilities())

        mode_steps = [s for s in plan.preprocessing_steps if s.operation == PreprocessingOperation.IMPUTE_MODE]
        assert len(mode_steps) == 1
        assert mode_steps[0].columns == ["department"]

    # 26-27. Categorical encoding & sequence order (imputation first, encoding second)
    def test_categorical_encoding_order(self):
        planner = BaselineMLPlanner()
        columns = [
            ColumnContext(
                name="department",
                dtype="object",
                is_numeric=False,
                is_categorical=True,
                is_datetime=False,
                missing_count=5,
                missing_percentage=0.5,
                unique_count=5,
                unique_percentage=0.5,
                sample_values=["sales"],
            ),
            ColumnContext(
                name="churn",
                dtype="int64",
                is_numeric=True,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
        ]
        ds = _make_dataset_context(columns=columns)
        pd = _make_problem_definition(feature_columns=["department"])
        plan = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=_make_user_request(), compute_capabilities=_make_compute_capabilities())

        steps = plan.preprocessing_steps
        assert len(steps) >= 2
        # Check order: impute_mode, then one_hot_encode
        assert steps[0].operation == PreprocessingOperation.IMPUTE_MODE
        assert steps[1].operation == PreprocessingOperation.ONE_HOT_ENCODE

    # 28-29. Numeric scaling under scale-sensitive portfolio (standard normal scaling added)
    def test_numeric_scaling_when_sensitive(self):
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(feature_columns=["age", "department"]),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        scale_steps = [s for s in plan.preprocessing_steps if s.operation == PreprocessingOperation.STANDARD_SCALE]
        assert len(scale_steps) == 1
        # 29. Do not scale categorical columns
        assert scale_steps[0].columns == ["age"]

    # 30. Datetime feature gets datetime extraction
    def test_datetime_feature_extraction(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition(feature_columns=["signup_date"])
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        dt_steps = [s for s in plan.preprocessing_steps if s.operation == PreprocessingOperation.DATETIME_EXTRACT]
        assert len(dt_steps) == 1
        assert dt_steps[0].columns == ["signup_date"]

    # 31. Target column never appears in preprocessing
    def test_target_never_preprocessed(self):
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        for step in plan.preprocessing_steps:
            assert "churn" not in step.columns

    # 32. Excluded/non-feature columns are not preprocessed
    def test_non_feature_never_preprocessed(self):
        # signup_date is a datetime column, but not included in feature columns
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(feature_columns=["age"]),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        for step in plan.preprocessing_steps:
            assert "signup_date" not in step.columns

    # 33. Ambiguous feature type is handled according to current schema contract (passthrough)
    def test_ambiguous_feature_type_passthrough(self):
        planner = BaselineMLPlanner()
        columns = [
            ColumnContext(
                name="strange_col",
                dtype="unknown",
                is_numeric=False,
                is_categorical=False,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
            ColumnContext(
                name="churn",
                dtype="int64",
                is_numeric=True,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
        ]
        ds = _make_dataset_context(columns=columns)
        pd = _make_problem_definition(feature_columns=["strange_col"])
        plan = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=_make_user_request(), compute_capabilities=_make_compute_capabilities())

        passthrough_steps = [s for s in plan.preprocessing_steps if s.operation == PreprocessingOperation.PASSTHROUGH]
        assert len(passthrough_steps) == 1
        assert passthrough_steps[0].columns == ["strange_col"]

    # 34-36. Step IDs uniqueness, determinism, and order
    def test_preprocessing_step_ids_properties(self):
        planner = BaselineMLPlanner()
        columns = [
            ColumnContext(
                name="department",
                dtype="object",
                is_numeric=False,
                is_categorical=True,
                is_datetime=False,
                missing_count=5,
                missing_percentage=0.5,
                unique_count=5,
                unique_percentage=0.5,
                sample_values=["sales"],
            ),
            ColumnContext(
                name="churn",
                dtype="int64",
                is_numeric=True,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
        ]
        ds = _make_dataset_context(columns=columns)
        pd = _make_problem_definition(feature_columns=["department"])
        plan = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=_make_user_request(), compute_capabilities=_make_compute_capabilities())

        step_ids = [s.step_id for s in plan.preprocessing_steps]
        # 34. Unique
        assert len(step_ids) == len(set(step_ids))
        # 35. Deterministic format
        assert step_ids == ["preprocess_001", "preprocess_002"]


class TestBaselinePlannerFeatureEngineeringAndSelection:
    """Scenarios 37-40: FeatureEngineering & Selection baseline rules."""

    # 37. Feature engineering steps are empty
    def test_feature_engineering_is_empty(self):
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        assert plan.feature_engineering_steps == []

    # 38-40. Feature selection plan constraints
    def test_feature_selection_plan_properties(self):
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(feature_columns=["age", "department"]),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        sel = plan.feature_selection
        # 38. Method NONE
        assert sel.method == FeatureSelectionMethod.NONE
        # 39-40. Candidate columns
        assert sel.candidate_columns == ["age", "department"]


class TestBaselinePlannerSplitting:
    """Scenarios 41-48: SplitPlan rules checks."""

    # 41-42. Classification stratification rules
    def test_classification_split_rules(self):
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(problem_type=ProblemType.CLASSIFICATION),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        split = plan.split_plan
        assert split.strategy == SplitStrategy.STRATIFIED
        assert split.stratify_column == "churn"
        assert split.test_size == 0.2
        assert split.validation_size == 0.0
        assert split.shuffle is True
        assert split.random_state == 42

    # 43-47. Regression random split rules
    def test_regression_split_rules(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(target_column="age"),
            compute_capabilities=_make_compute_capabilities(),
        )
        split = plan.split_plan
        assert split.strategy == SplitStrategy.RANDOM
        assert split.stratify_column is None
        assert split.time_column is None
        assert split.test_size == 0.2
        assert split.validation_size == 0.0
        assert split.shuffle is True
        assert split.random_state == 42

    # 48. Time-based split is not automatically selected for datetime features
    def test_datetime_does_not_force_time_split(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition(feature_columns=["signup_date"])
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        assert plan.split_plan.strategy == SplitStrategy.STRATIFIED  # Still stratified classification


class TestBaselinePlannerModelCandidates:
    """Scenarios 49-62: Model candidate portfolios selection and properties."""

    # 49. Minimal classification (two candidates)
    def test_minimal_classification_candidates(self):
        planner = BaselineMLPlanner()
        cc = _make_compute_capabilities(compute_tier=ComputeTier.MINIMAL)
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(problem_type=ProblemType.CLASSIFICATION),
            user_request=_make_user_request(),
            compute_capabilities=cc,
        )
        families = [c.model_family for c in plan.model_candidates]
        assert families == [ModelFamily.LOGISTIC_REGRESSION, ModelFamily.RANDOM_FOREST]

    # 50-51. Standard/High classification (three candidates)
    @pytest.mark.parametrize("tier", [ComputeTier.STANDARD, ComputeTier.HIGH])
    def test_standard_high_classification_candidates(self, tier):
        planner = BaselineMLPlanner()
        cc = _make_compute_capabilities(compute_tier=tier)
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(problem_type=ProblemType.CLASSIFICATION),
            user_request=_make_user_request(),
            compute_capabilities=cc,
        )
        families = [c.model_family for c in plan.model_candidates]
        assert families == [ModelFamily.LOGISTIC_REGRESSION, ModelFamily.RANDOM_FOREST, ModelFamily.GRADIENT_BOOSTING]

    # 52. Minimal regression (two candidates)
    def test_minimal_regression_candidates(self):
        planner = BaselineMLPlanner()
        cc = _make_compute_capabilities(compute_tier=ComputeTier.MINIMAL)
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(target_column="age"),
            compute_capabilities=cc,
        )
        families = [c.model_family for c in plan.model_candidates]
        assert families == [ModelFamily.LINEAR_REGRESSION, ModelFamily.RANDOM_FOREST]

    # 53-54. Standard/High regression (three candidates)
    @pytest.mark.parametrize("tier", [ComputeTier.STANDARD, ComputeTier.HIGH])
    def test_standard_high_regression_candidates(self, tier):
        planner = BaselineMLPlanner()
        cc = _make_compute_capabilities(compute_tier=tier)
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(target_column="age"),
            compute_capabilities=cc,
        )
        families = [c.model_family for c in plan.model_candidates]
        assert families == [ModelFamily.LINEAR_REGRESSION, ModelFamily.RANDOM_FOREST, ModelFamily.GRADIENT_BOOSTING]

    # 55-60. Candidates format properties
    def test_model_candidate_details(self):
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        cands = plan.model_candidates
        # 55-57. IDs uniqueness and format
        assert cands[0].candidate_id == "model_001"
        assert cands[1].candidate_id == "model_002"
        # 58-59. Search NONE
        assert all(c.search_strategy == SearchStrategy.NONE for c in cands)
        assert all(c.search_space == {} for c in cands)
        # 60. Reasons
        assert all(isinstance(c.reason, str) and len(c.reason) > 5 for c in cands)

    # 61-62. Classification/regression candidates pass validation
    def test_candidates_validation(self):
        # Already verified by the validator check at the end of create_plan, but let's test it directly too
        planner = BaselineMLPlanner()
        plan_c = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(problem_type=ProblemType.CLASSIFICATION),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        assert isinstance(plan_c, MLPlan)

        pd_r = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        plan_r = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd_r,
            user_request=_make_user_request(target_column="age"),
            compute_capabilities=_make_compute_capabilities(),
        )
        assert isinstance(plan_r, MLPlan)


class TestBaselinePlannerEvaluation:
    """Scenarios 63-71: Metric selection and CV folds rules."""

    # 63. Primary metric comes from ProblemDefinition
    def test_primary_metric_from_definition(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition()
        pd.primary_metric = "precision"
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        assert plan.evaluation_plan.primary_metric == "precision"

    # 64-65. Classification secondary metrics exclusions
    def test_classification_secondary_metrics_exclusions(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition()
        pd.primary_metric = "recall"  # recall in pool ['accuracy', 'precision', 'recall', 'f1']
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        sec = plan.evaluation_plan.secondary_metrics
        assert "recall" not in sec
        assert sec == ["accuracy", "precision", "f1"]

    # 66-67. Regression secondary metrics exclusions
    def test_regression_secondary_metrics_exclusions(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        pd.primary_metric = "rmse"
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(target_column="age"),
            compute_capabilities=_make_compute_capabilities(),
        )
        sec = plan.evaluation_plan.secondary_metrics
        assert "rmse" not in sec
        assert sec == ["mae", "r2"]

    # 68-70. CV Folds adaptation
    @pytest.mark.parametrize("tier,expected_folds", [
        (ComputeTier.MINIMAL, 3), (ComputeTier.STANDARD, 5), (ComputeTier.HIGH, 5)
    ])
    def test_cv_folds_adaptation(self, tier, expected_folds):
        planner = BaselineMLPlanner()
        cc = _make_compute_capabilities(compute_tier=tier)
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(),
            user_request=_make_user_request(),
            compute_capabilities=cc,
        )
        assert plan.evaluation_plan.cross_validation_folds == expected_folds

    # 71. Metric comparison case-normalized without mutating upstream
    def test_metric_case_normalized_no_mutation(self):
        planner = BaselineMLPlanner()
        pd = _make_problem_definition()
        pd.primary_metric = "  AcCuRaCy  "
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=pd,
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        # Primary metric is AcCuRaCy, case-normalized it should match secondary pool 'accuracy' and exclude it.
        assert "accuracy" not in plan.evaluation_plan.secondary_metrics
        assert pd.primary_metric == "  AcCuRaCy  "  # Unmutated


class TestBaselinePlannerExecutionAndWarnings:
    """Scenarios 72-79: ExecutionConstraints and warning maps."""

    # 72-76. Execution constraint mapping properties
    def test_execution_constraints_mapping(self):
        planner = BaselineMLPlanner()
        cc = _make_compute_capabilities(
            safe_parallel_workers=3,
            gpu_acceleration_available=True,
            accelerator_type=AcceleratorType.CUDA,
        )
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(),
            user_request=_make_user_request(),
            compute_capabilities=cc,
        )
        cons = plan.execution_constraints
        assert cons.parallel_workers == 3
        assert cons.compute_tier == ComputeTier.STANDARD
        assert cons.use_gpu_acceleration is True
        assert cons.accelerator_type == AcceleratorType.CUDA

    # 74. CPU-only capabilities constraints
    def test_cpu_only_capabilities_mapping(self):
        planner = BaselineMLPlanner()
        cc = _make_compute_capabilities(gpu_acceleration_available=False, accelerator_type=AcceleratorType.NONE)
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(),
            user_request=_make_user_request(),
            compute_capabilities=cc,
        )
        cons = plan.execution_constraints
        assert cons.use_gpu_acceleration is False
        assert cons.accelerator_type == AcceleratorType.NONE

    # 77-79. ComputeCapabilities warning propagation
    def test_warning_propagation(self):
        planner = BaselineMLPlanner()
        warn1 = ResourceWarning(code="LIMIT_CPU", message="Parallel performance restricted.")
        warn2 = ResourceWarning(code="LOW_RAM", message="System memory is low.")
        cc = _make_compute_capabilities(warnings=[warn1, warn2])

        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(),
            user_request=_make_user_request(),
            compute_capabilities=cc,
        )

        assert len(plan.warnings) == 2
        # Warning order preserved
        assert plan.warnings[0].code == "LIMIT_CPU"
        assert plan.warnings[1].code == "LOW_RAM"


class TestBaselinePlannerInternalValidation:
    """Scenarios 80-84: Validation gate tests."""

    # 80-81. Generated plan passes validation
    def test_valid_generated_plans(self):
        planner = BaselineMLPlanner()
        plan_c = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(problem_type=ProblemType.CLASSIFICATION),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        assert isinstance(plan_c, MLPlan)

    # 82-84. Invalid generated plan causes BaselineMLPlannerError
    def test_invalid_plan_causes_error(self):
        # Create a mock validator that returns is_valid = False
        mock_validator = MagicMock(spec=MLPlanValidator)
        issue = MLPlanValidationIssue(
            code="PLAN_ERROR_TEST",
            message="Mock error message",
            severity=ValidationSeverity.ERROR,
        )
        result = MLPlanValidationResult(plan_id="p", is_valid=False, errors=[issue], warnings=[])
        mock_validator.validate.return_value = result

        planner = BaselineMLPlanner(validator=mock_validator)
        with pytest.raises(BaselineMLPlannerError, match="Generated baseline plan failed internal validation check. Error codes:.*PLAN_ERROR_TEST"):
            planner.create_plan(
                dataset_context=_make_dataset_context(),
                problem_definition=_make_problem_definition(),
                user_request=_make_user_request(),
                compute_capabilities=_make_compute_capabilities(),
            )
        # 83. Validator was actually invoked
        mock_validator.validate.assert_called_once()


class TestBaselinePlannerBehavior:
    """Scenarios 85-93: Non-mutation, repeatability, serialization."""

    # 85-88. Non-mutation of upstream artifacts
    def test_non_mutation_of_inputs(self):
        planner = BaselineMLPlanner()
        ds = _make_dataset_context()
        pd = _make_problem_definition()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        orig_ds = copy.deepcopy(ds)
        orig_pd = copy.deepcopy(pd)
        orig_req = copy.deepcopy(req)
        orig_cc = copy.deepcopy(cc)

        planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=req, compute_capabilities=cc)

        assert ds == orig_ds
        assert pd == orig_pd
        assert req == orig_req
        assert cc == orig_cc

    # 89. Repeated planning produces equivalent decisions except plan_id
    def test_repeated_planning_produces_equivalent(self):
        planner = BaselineMLPlanner()
        ds = _make_dataset_context()
        pd = _make_problem_definition()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        plan1 = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=req, compute_capabilities=cc)
        plan2 = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=req, compute_capabilities=cc)

        assert plan1.plan_id != plan2.plan_id
        # Copy to compare other fields
        plan2_patched = copy.deepcopy(plan2)
        plan2_patched.plan_id = plan1.plan_id
        assert plan1.model_dump() == plan2_patched.model_dump()

    # 90-92. Deterministic preprocessing, model, metric order
    def test_deterministic_ordering(self):
        # We can construct a dataset with multiple missing columns
        columns = [
            ColumnContext(
                name="b_numeric",
                dtype="float64",
                is_numeric=True,
                is_categorical=False,
                is_datetime=False,
                missing_count=10,
                missing_percentage=1.0,
                unique_count=50,
                unique_percentage=5.0,
                sample_values=[25.0, 30.0],
            ),
            ColumnContext(
                name="a_numeric",
                dtype="float64",
                is_numeric=True,
                is_categorical=False,
                is_datetime=False,
                missing_count=10,
                missing_percentage=1.0,
                unique_count=50,
                unique_percentage=5.0,
                sample_values=[25.0, 30.0],
            ),
            ColumnContext(
                name="churn",
                dtype="int64",
                is_numeric=True,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
        ]
        ds = _make_dataset_context(columns=columns)
        pd = _make_problem_definition(feature_columns=["b_numeric", "a_numeric"])
        planner = BaselineMLPlanner()
        plan = planner.create_plan(dataset_context=ds, problem_definition=pd, user_request=_make_user_request(), compute_capabilities=_make_compute_capabilities())

        # 90. Preprocessing order follows feature_columns order: b_numeric first, then a_numeric, then scale step
        median_steps = [s for s in plan.preprocessing_steps if s.operation == PreprocessingOperation.IMPUTE_MEDIAN]
        assert median_steps[0].columns == ["b_numeric"]
        assert median_steps[1].columns == ["a_numeric"]

        # 91. Model candidates order: logistic_regression, random_forest, gradient_boosting
        model_families = [c.model_family for c in plan.model_candidates]
        assert model_families == [ModelFamily.LOGISTIC_REGRESSION, ModelFamily.RANDOM_FOREST, ModelFamily.GRADIENT_BOOSTING]

        # 92. Metric order
        assert plan.evaluation_plan.secondary_metrics == ["accuracy", "precision", "recall"]

    # 93. JSON serialization of generated MLPlan works
    def test_json_serialization_works(self):
        planner = BaselineMLPlanner()
        plan = planner.create_plan(
            dataset_context=_make_dataset_context(),
            problem_definition=_make_problem_definition(),
            user_request=_make_user_request(),
            compute_capabilities=_make_compute_capabilities(),
        )
        json_str = plan.model_dump_json()
        assert isinstance(json.loads(json_str), dict)
