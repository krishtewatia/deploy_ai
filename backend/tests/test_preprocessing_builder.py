"""Unit tests for PreprocessingPipelineBuilder."""

from __future__ import annotations

import copy
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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
    PreprocessingOperation,
    PreprocessingStep,
)
from backend.app.ml_plan.baseline_planner import BaselineMLPlanner
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ResolutionStatus,
    TargetSource,
)
from backend.app.ml_execution.preprocessing_builder import (
    PreprocessingPipelineBuilder,
    PreprocessingPipelineBuilderError,
)


# ── Helper Builders ───────────────────────────────────────────────────


def _make_dataset_context() -> DatasetContext:
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
            name="salary",
            dtype="float64",
            is_numeric=True,
            is_categorical=False,
            is_datetime=False,
            missing_count=0,
            missing_percentage=0.0,
            unique_count=100,
            unique_percentage=10.0,
            sample_values=[50000.0, 60000.0],
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
        dataset_id="ds_01",
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


def _make_problem_definition() -> ProblemDefinition:
    return ProblemDefinition(
        definition_id="pd_01",
        request_id="req_01",
        dataset_id="ds_01",
        goal="Predict customer churn",
        problem_type="classification",
        target_column="churn",
        target_source=TargetSource.USER,
        feature_columns=["age", "salary", "department"],
        excluded_columns=[],
        primary_metric="f1",
        status=ResolutionStatus.RESOLVED,
        confirmation_items=[],
    )


def _make_user_request() -> UserMLRequest:
    return UserMLRequest(
        request_id="req_01",
        goal="Build baseline predictive model",
        target_column="churn",
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


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestPreprocessingPipelineBuilder:
    def test_pipeline_type_and_empty_steps(self):
        """Verify build returns a Pipeline even when preprocessing_steps list is empty."""
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()

        planner = BaselineMLPlanner()
        plan = planner.create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Clear preprocessing steps
        plan.preprocessing_steps = []

        builder = PreprocessingPipelineBuilder()
        pipe = builder.build(dataset_context=context, plan=plan)

        assert isinstance(pipe, Pipeline)
        assert len(pipe.steps) == 1
        assert pipe.steps[0][0] == "passthrough"
        assert pipe.steps[0][1] == "passthrough"

        # Verify executing on pandas DataFrame works
        df = pd.DataFrame({"age": [20.0, 30.0]})
        pipe.fit(df)
        transformed = pipe.transform(df)
        pd.testing.assert_frame_equal(df, transformed)

    def test_all_supported_operations_success(self):
        """Verify median, mode imputer, standard & robust scaling, one hot encoding, and passthrough builder success."""
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()

        planner = BaselineMLPlanner()
        plan = planner.create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Build custom steps list with all 6 supported operations
        plan.preprocessing_steps = [
            PreprocessingStep(
                step_id="step_impute_median",
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["age", "salary"],
                reason="Impute age and salary",
            ),
            PreprocessingStep(
                step_id="step_impute_mode",
                operation=PreprocessingOperation.IMPUTE_MODE,
                columns=["department"],
                reason="Impute department",
            ),
            PreprocessingStep(
                step_id="step_ohe",
                operation=PreprocessingOperation.ONE_HOT_ENCODE,
                columns=["department"],
                reason="Encode department",
            ),
            PreprocessingStep(
                step_id="step_std_scale",
                operation=PreprocessingOperation.STANDARD_SCALE,
                columns=["age"],
                reason="Scale age",
            ),
            PreprocessingStep(
                step_id="step_robust_scale",
                operation=PreprocessingOperation.ROBUST_SCALE,
                columns=["salary"],
                reason="Robust scale salary",
            ),
            PreprocessingStep(
                step_id="step_passthrough",
                operation=PreprocessingOperation.PASSTHROUGH,
                columns=["salary"],
                reason="Passthrough salary",
            ),
        ]

        builder = PreprocessingPipelineBuilder()
        pipe = builder.build(dataset_context=context, plan=plan)

        assert isinstance(pipe, Pipeline)
        assert len(pipe.steps) == 6

        # Check Pipeline Order
        step_ids = [step[0] for step in pipe.steps]
        assert step_ids == [
            "step_impute_median",
            "step_impute_mode",
            "step_ohe",
            "step_std_scale",
            "step_robust_scale",
            "step_passthrough",
        ]

        # Verify that executing the pipeline actually works
        df = pd.DataFrame(
            {
                "age": [20.0, None, 30.0],
                "salary": [50000.0, 60000.0, None],
                "department": ["sales", None, "engineering"],
            }
        )

        pipe.fit(df)
        transformed = pipe.transform(df)

        # Output should be pandas DataFrame due to set_output(transform="pandas")
        assert isinstance(transformed, pd.DataFrame)
        assert "department_sales" in transformed.columns
        assert "department_engineering" in transformed.columns
        assert "age" in transformed.columns
        assert "salary" in transformed.columns
        # Make sure no NaNs remain (imputation was executed successfully)
        assert not transformed.isna().any().any()

    def test_validation_reject_none_inputs(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="dataset_context cannot be None"):
            builder.build(dataset_context=None, plan=plan)  # type: ignore
        with pytest.raises(PreprocessingPipelineBuilderError, match="plan cannot be None"):
            builder.build(dataset_context=context, plan=None)  # type: ignore

    def test_validation_reject_wrong_types(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="dataset_context must be a DatasetContext"):
            builder.build(dataset_context=object(), plan=plan)  # type: ignore
        with pytest.raises(PreprocessingPipelineBuilderError, match="plan must be an MLPlan"):
            builder.build(dataset_context=context, plan=object())  # type: ignore

    def test_validation_reject_duplicate_step_ids(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.preprocessing_steps = [
            PreprocessingStep(
                step_id="step_duplicate",
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["age"],
                reason="First",
            ),
            PreprocessingStep(
                step_id="step_duplicate",
                operation=PreprocessingOperation.STANDARD_SCALE,
                columns=["salary"],
                reason="Second",
            ),
        ]

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Duplicate preprocessing step_id detected"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_unknown_operation(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Create step with an unsupported operation
        plan.preprocessing_steps = [
            PreprocessingStep(
                step_id="step_mean",
                operation=PreprocessingOperation.IMPUTE_MEAN,  # Unplanned/unsupported
                columns=["age"],
                reason="Unused mean imputer",
            )
        ]

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Unsupported preprocessing operation"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_unknown_column(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.preprocessing_steps = [
            PreprocessingStep(
                step_id="step1",
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["missing_col"],
                reason="Missing",
            )
        ]

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Column 'missing_col' not found in dataset columns"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_target_column(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.preprocessing_steps = [
            PreprocessingStep(
                step_id="step1",
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["churn"],  # churn is target column
                reason="Impute target",
            )
        ]

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Target column 'churn' cannot appear in preprocessing steps"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_conflicting_imputations(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.preprocessing_steps = [
            PreprocessingStep(
                step_id="step1",
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["age"],
                reason="Impute median",
            ),
            PreprocessingStep(
                step_id="step2",
                operation=PreprocessingOperation.IMPUTE_MODE,
                columns=["age"],  # Duplicate imputation
                reason="Impute mode",
            ),
        ]

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Conflicting/duplicate imputation operation on column 'age'"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_conflicting_scalings(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.preprocessing_steps = [
            PreprocessingStep(
                step_id="step1",
                operation=PreprocessingOperation.STANDARD_SCALE,
                columns=["salary"],
                reason="Std scale",
            ),
            PreprocessingStep(
                step_id="step2",
                operation=PreprocessingOperation.ROBUST_SCALE,
                columns=["salary"],  # Duplicate scale
                reason="Robust scale",
            ),
        ]

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Conflicting/duplicate scaling operation on column 'salary'"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_conflicting_encodings(self):
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.preprocessing_steps = [
            PreprocessingStep(
                step_id="step1",
                operation=PreprocessingOperation.ONE_HOT_ENCODE,
                columns=["department"],
                reason="First OHE",
            ),
            PreprocessingStep(
                step_id="step2",
                operation=PreprocessingOperation.ONE_HOT_ENCODE,
                columns=["department"],  # Duplicate OHE
                reason="Second OHE",
            ),
        ]

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Conflicting/duplicate encoding operation on column 'department'"):
            builder.build(dataset_context=context, plan=plan)

    def test_non_mutation_rule(self):
        """Verify inputs are not mutated by builder."""
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        context_original = copy.deepcopy(context)
        plan_original = copy.deepcopy(plan)

        builder = PreprocessingPipelineBuilder()
        builder.build(dataset_context=context, plan=plan)

        assert context == context_original
        assert plan == plan_original

    def test_validation_reject_invalid_step_types(self):
        """Preprocessing step must be instance of PreprocessingStep."""
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        plan.preprocessing_steps = [object()]  # type: ignore

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Each preprocessing step must be a PreprocessingStep instance"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_empty_step_id(self):
        """Preprocessing step_id is empty."""
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        # Bypass pydantic validation using copy / update
        step = plan.preprocessing_steps[0]
        step.step_id = "   "

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="step_id cannot be empty"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_empty_columns_list(self):
        """Preprocessing step columns list is empty."""
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        step = plan.preprocessing_steps[0]
        step.columns = []

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="columns list cannot be empty"):
            builder.build(dataset_context=context, plan=plan)

    def test_validation_reject_empty_column_name(self):
        """Preprocessing step column name is empty."""
        context = _make_dataset_context()
        prob_def = _make_problem_definition()
        req = _make_user_request()
        caps = _make_compute_capabilities()
        plan = BaselineMLPlanner().create_plan(dataset_context=context, problem_definition=prob_def, user_request=req, compute_capabilities=caps)

        step = plan.preprocessing_steps[0]
        step.columns = ["   "]

        builder = PreprocessingPipelineBuilder()
        with pytest.raises(PreprocessingPipelineBuilderError, match="Column name cannot be empty"):
            builder.build(dataset_context=context, plan=plan)
