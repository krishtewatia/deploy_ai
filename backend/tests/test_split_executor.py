"""Unit tests for SplitExecutor."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from pydantic import ValidationError

from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
)
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)
from backend.app.ml_plan.schemas import (
    MLPlan,
    SplitStrategy,
    ProblemType,
    DatasetSplitPlan,
)
from backend.app.ml_plan.baseline_planner import BaselineMLPlanner
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ResolutionStatus,
    TargetSource,
)
from backend.app.ml_execution.split_executor import (
    SplitExecutor,
    SplitExecutorError,
    DatasetSplitResult,
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
        confirmation_items=[],
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


def _make_compute_capabilities() -> ComputeCapabilities:
    return ComputeCapabilities(
        capability_id="cap_01",
        hardware_profile_id="hw_01",
        compute_tier=ComputeTier.STANDARD,
        memory_constraint=MemoryConstraintLevel.MODERATE,
        cpu_training_available=True,
        gpu_acceleration_available=False,
        accelerator_type=AcceleratorType.NONE,
        safe_parallel_workers=4,
        max_parallel_workers=8,
        available_ram_mb_snapshot=4096,
        total_ram_mb=8192,
        warnings=[],
    )


def _make_valid_dataframe(rows: int = 100) -> pd.DataFrame:
    # 50% class 0, 50% class 1 for perfect stratification testing
    churn_values = [0] * (rows // 2) + [1] * (rows - rows // 2)
    return pd.DataFrame(
        {
            "age": [float(i % 50 + 20) for i in range(rows)],
            "department": ["sales" if i % 2 == 0 else "engineering" for i in range(rows)],
            "signup_date": pd.date_range("2026-01-01", periods=rows, freq="D"),
            "churn": churn_values,
        }
    )


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestSplitExecutor:
    def test_random_split_success(self):
        """Verify random split strategy works and respects configurations."""
        df = _make_valid_dataframe(100)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn", "department"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()

        planner = BaselineMLPlanner()
        plan = planner.create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Force split plan to be RANDOM
        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.RANDOM,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify_column=None,
            time_column=None,
        )

        executor = SplitExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert isinstance(res, DatasetSplitResult)
        assert res.X_train.shape == (80, 2)
        assert res.X_test.shape == (20, 2)
        assert res.y_train.shape == (80,)
        assert res.y_test.shape == (20,)
        assert res.feature_columns == ["churn", "department"]
        assert res.target_column == "age"
        assert len(res.train_indices) == 80
        assert len(res.test_indices) == 20
        # Index preservation
        assert list(res.X_train.index) == res.train_indices
        assert list(res.X_test.index) == res.test_indices

    def test_stratified_split_success(self):
        """Verify stratified split strategy works, splits, and stratifies correctly."""
        df = _make_valid_dataframe(100)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.CLASSIFICATION, target_column="churn")
        req = _make_user_request(target_column="churn")
        caps = _make_compute_capabilities()

        planner = BaselineMLPlanner()
        plan = planner.create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        executor = SplitExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert res.X_train.shape == (80, 2)
        assert res.X_test.shape == (20, 2)
        # Verify stratification target ratio (churn=1 is 50% in original, should be 50% in splits)
        assert res.y_train.value_counts()[1] == 40
        assert res.y_test.value_counts()[1] == 10

    def test_time_based_split_success(self):
        """Verify time-based split strategy works and sorts oldest -> newest."""
        df = _make_valid_dataframe(10)
        # Shuffle input dataframe so it's not pre-sorted by date
        df_shuffled = df.sample(frac=1.0, random_state=12).copy()

        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn", "department"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()

        planner = BaselineMLPlanner()
        plan = planner.create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Configure time-based split
        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.3,  # 30% of 10 is 3 rows
            shuffle=False,
            time_column="signup_date",
        )

        executor = SplitExecutor()
        res = executor.execute(dataframe=df_shuffled, dataset_context=context, plan=plan)

        assert res.X_train.shape == (7, 2)
        assert res.X_test.shape == (3, 2)

        # Get sorted times to verify train/test sequencing
        sorted_times = df["signup_date"].sort_values(ascending=True).tolist()
        expected_train_times = sorted_times[:7]
        expected_test_times = sorted_times[7:]

        # signup_date of train/test outputs
        train_times_actual = df_shuffled.loc[res.train_indices, "signup_date"].tolist()
        test_times_actual = df_shuffled.loc[res.test_indices, "signup_date"].tolist()

        assert train_times_actual == expected_train_times
        assert test_times_actual == expected_test_times

    def test_unsupported_strategy_error(self):
        """Verify exception when split strategy is unknown/unsupported."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.CLASSIFICATION)
        req = _make_user_request()
        caps = _make_compute_capabilities()

        planner = BaselineMLPlanner()
        plan = planner.create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Hack plan to set an unsupported strategy (e.g. bypassing Pydantic check via construct or similar)
        # But wait! Pydantic enum check rejects it during construction.
        # We can mock the strategy attribute or config object directly:
        plan.split_plan = MagicMock()
        plan.split_plan.strategy = "k-fold"
        plan.split_plan.test_size = 0.2
        plan.split_plan.random_state = 42
        plan.split_plan.shuffle = True

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="Unsupported split strategy: k-fold"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_none_inputs(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="dataframe cannot be None"):
            executor.execute(dataframe=None, dataset_context=context, plan=plan)  # type: ignore
        with pytest.raises(SplitExecutorError, match="dataset_context cannot be None"):
            executor.execute(dataframe=df, dataset_context=None, plan=plan)  # type: ignore
        with pytest.raises(SplitExecutorError, match="plan cannot be None"):
            executor.execute(dataframe=df, dataset_context=context, plan=None)  # type: ignore

    def test_validation_reject_wrong_types(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="dataframe must be a pandas.DataFrame"):
            executor.execute(dataframe=object(), dataset_context=context, plan=plan)  # type: ignore
        with pytest.raises(SplitExecutorError, match="dataset_context must be a DatasetContext"):
            executor.execute(dataframe=df, dataset_context=object(), plan=plan)  # type: ignore
        with pytest.raises(SplitExecutorError, match="plan must be an MLPlan"):
            executor.execute(dataframe=df, dataset_context=context, plan=object())  # type: ignore

    def test_validation_reject_empty_dataframe(self):
        df_empty = pd.DataFrame()
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="dataframe cannot be empty"):
            executor.execute(dataframe=df_empty, dataset_context=context, plan=plan)

    def test_validation_reject_missing_target_column(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # target column not in df
        plan.target_column = "non_existent_target"

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="Target column 'non_existent_target' does not exist"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_feature_column(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.feature_columns = ["age", "missing_feat"]

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="Feature column 'missing_feat' does not exist"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_stratify_column(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.STRATIFIED,
            test_size=0.2,
            stratify_column="missing_stratify",
        )

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="Stratify column 'missing_stratify' does not exist"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_time_column(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.2,
            shuffle=False,
            time_column="missing_time_col",
        )

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="Time column 'missing_time_col' does not exist"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_feature_columns(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.feature_columns = ["age", "age"]

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="plan feature_columns cannot contain duplicates"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_target_inside_feature_columns(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.feature_columns = ["age", "churn"]

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="target_column cannot appear in feature_columns"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_regression_stratified_inconsistency(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Bypass pydantic validation of DatasetSplitPlan consistency checks using MagicMock
        plan.split_plan = MagicMock()
        plan.split_plan.strategy = SplitStrategy.STRATIFIED
        plan.split_plan.stratify_column = "age"
        plan.split_plan.test_size = 0.2

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="Stratified splits are not allowed for regression tasks"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_stratify_target_mismatch(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.CLASSIFICATION, target_column="churn", feature_columns=["age", "department"])
        req = _make_user_request(target_column="churn")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Bypass stratified target matching using MagicMock
        plan.split_plan = MagicMock()
        plan.split_plan.strategy = SplitStrategy.STRATIFIED
        plan.split_plan.stratify_column = "department"  # Not the target column
        plan.split_plan.test_size = 0.2

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="stratify_column must match target_column"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_non_datetime_time_column(self):
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.2,
            shuffle=False,
            time_column="churn",  # Churn is numeric/categorical, not datetime in context
        )

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="must be a datetime column"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_dataframe_immutability(self):
        """Verify original DataFrame is not mutated during the execution process."""
        df = _make_valid_dataframe(10)
        df_original = df.copy()

        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        executor = SplitExecutor()
        executor.execute(dataframe=df, dataset_context=context, plan=plan)

        pd.testing.assert_frame_equal(df, df_original)

    def test_time_split_failure_due_to_small_dataset(self):
        """Verify SplitExecutorError raised when dataset has fewer than 2 rows for time-based split."""
        df = _make_valid_dataframe(1)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.5,
            shuffle=False,
            time_column="signup_date",
        )

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="Dataset too small to split train and test sets"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_time_split_failure_due_to_general_exception(self):
        """Verify generic exceptions are captured and wrapped inside SplitExecutorError."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.2,
            shuffle=False,
            time_column="signup_date",
        )

        executor = SplitExecutor()
        # Mock sort_values on DataFrame copy to raise a RuntimeError
        with patch("pandas.DataFrame.sort_values", side_effect=RuntimeError("Sort crash")):
            with pytest.raises(SplitExecutorError, match="Split execution failed: Sort crash") as exc_info:
                executor.execute(dataframe=df, dataset_context=context, plan=plan)
            assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_validation_reject_empty_target_column(self):
        """Plan target_column is empty string."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.target_column = ""

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="plan target_column cannot be empty"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_empty_feature_columns(self):
        """Plan feature_columns is empty list."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.feature_columns = []

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="plan feature_columns cannot be empty"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_stratify_column_attr(self):
        """Bypass plan schema validation to test missing stratify_column attribute."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.CLASSIFICATION, target_column="churn")
        req = _make_user_request(target_column="churn")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = MagicMock()
        plan.split_plan.strategy = SplitStrategy.STRATIFIED
        plan.split_plan.stratify_column = None
        plan.split_plan.test_size = 0.2

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="stratify_column is required for stratified split"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_time_column_attr(self):
        """Bypass plan schema validation to test missing time_column attribute."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = MagicMock()
        plan.split_plan.strategy = SplitStrategy.TIME_BASED
        plan.split_plan.time_column = None
        plan.split_plan.test_size = 0.2

        executor = SplitExecutor()
        with pytest.raises(SplitExecutorError, match="time_column is required for time-based split"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_time_based_split_size_rounding(self):
        """Verify time-based split rounds up to 1 if test size rounds to 0."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.01,  # Rounds to 0
            shuffle=False,
            time_column="signup_date",
        )

        executor = SplitExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)
        assert len(res.test_indices) == 1
        assert len(res.train_indices) == 9

    def test_time_based_split_size_rounding_upper(self):
        """Verify time-based split rounds down to n-1 if test size rounds to >= n."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        prob_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        req = _make_user_request(target_column="age")
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.split_plan = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.99,  # Rounds to 10
            shuffle=False,
            time_column="signup_date",
        )

        executor = SplitExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)
        assert len(res.test_indices) == 9
        assert len(res.train_indices) == 1
