"""Unit tests for FeatureEngineeringExecutor."""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)
from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeTier,
)
from backend.app.ml_plan.schemas import (
    FeatureEngineeringOperation,
    FeatureEngineeringStep,
    MLPlan,
    ProblemType,
    FeatureSelectionPlan,
    DatasetSplitPlan,
    EvaluationPlan,
    ExecutionConstraints,
    SplitStrategy,
    FeatureSelectionMethod,
    ModelCandidate,
    ModelFamily,
    SearchStrategy,
)
from backend.app.ml_execution.feature_engineering_executor import (
    FeatureEngineeringExecutor,
    FeatureEngineeringExecutorError,
    FeatureEngineeringResult,
)


# ── Helper Builders ───────────────────────────────────────────────────

def _make_dataset_context(
    dataset_id: str = "ds_01",
    columns: list[ColumnContext] = None,
) -> DatasetContext:
    if columns is None:
        columns = [
            ColumnContext(
                name="A",
                dtype="float64",
                is_numeric=True,
                is_categorical=False,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=5,
                unique_percentage=0.5,
                sample_values=[1.0, 2.0],
            ),
            ColumnContext(
                name="B",
                dtype="float64",
                is_numeric=True,
                is_categorical=False,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=5,
                unique_percentage=0.5,
                sample_values=[3.0, 4.0],
            ),
            ColumnContext(
                name="target",
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
        row_count=10,
        column_count=len(columns),
        memory_usage_bytes=1000,
    )
    return DatasetContext(
        basic_info=basic_info,
        columns=columns,
        missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
    )



def _make_ml_plan(
    target_column: str = "target",
    feature_columns: list[str] = None,
    feature_engineering_steps: list[FeatureEngineeringStep] = None,
) -> MLPlan:
    if feature_columns is None:
        feature_columns = ["A", "B"]
    if feature_engineering_steps is None:
        feature_engineering_steps = []

    return MLPlan(
        plan_id="plan_01",
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        problem_type=ProblemType.CLASSIFICATION,
        target_column=target_column,
        feature_columns=feature_columns,
        preprocessing_steps=[],
        feature_engineering_steps=feature_engineering_steps,
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["A", "B"],
            max_features=2,
            parameters={},
            reason="No feature selection required.",
        ),
        split_plan=DatasetSplitPlan(
            strategy=SplitStrategy.RANDOM,
            test_size=0.2,
            validation_size=0.1,
            random_state=42,
            shuffle=True,
        ),
        model_candidates=[
            ModelCandidate(
                candidate_id="model_01",
                model_family=ModelFamily.RANDOM_FOREST,
                parameters={"n_estimators": 100},
                search_strategy=SearchStrategy.NONE,
                search_space={},
                reason="Baseline random forest.",
            )
        ],
        evaluation_plan=EvaluationPlan(
            primary_metric="accuracy",
            secondary_metrics=["f1"],
            cross_validation_folds=5,
        ),
        execution_constraints=ExecutionConstraints(
            parallel_workers=4,
            use_gpu_acceleration=False,
            accelerator_type=AcceleratorType.NONE,
            compute_tier=ComputeTier.STANDARD,
        ),
    )




def _make_valid_dataframe(rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [float(i) for i in range(rows)],
            "B": [float(i * 2) for i in range(rows)],
            "target": [i % 2 for i in range(rows)],
        }
    )


# ── Test Suite ──────────────────────────────────────────────────────────────

class TestFeatureEngineeringExecutor:
    """Tests covering FeatureEngineeringExecutor behavior and edge cases."""

    def test_successful_interaction(self):
        """Verify standard INTERACTION operation (A * B) works correctly."""
        df = _make_valid_dataframe(5)
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step_interaction",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["A_mult_B"],
            reason="Interaction between A and B",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert isinstance(res, FeatureEngineeringResult)
        assert "A_mult_B" in res.dataframe.columns
        assert res.engineered_columns == ["A_mult_B"]
        assert res.created_columns == ["A_mult_B"]
        
        # Verify math: A * B
        # A = [0, 1, 2, 3, 4]
        # B = [0, 2, 4, 6, 8]
        # Expected: [0, 2, 8, 18, 32]
        expected = [0.0, 2.0, 8.0, 18.0, 32.0]
        pd.testing.assert_series_equal(res.dataframe["A_mult_B"], pd.Series(expected, name="A_mult_B"))

    def test_successful_ratio(self):
        """Verify standard RATIO operation (A / B) works correctly."""
        df = pd.DataFrame({"A": [10.0, 20.0], "B": [2.0, 4.0], "target": [0, 1]})
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step_ratio",
            operation=FeatureEngineeringOperation.RATIO,
            input_columns=["A", "B"],
            output_columns=["A_ratio_B"],
            reason="Ratio of A to B",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert "A_ratio_B" in res.dataframe.columns
        expected = [5.0, 5.0]
        pd.testing.assert_series_equal(res.dataframe["A_ratio_B"], pd.Series(expected, name="A_ratio_B"))

    def test_ratio_division_by_zero(self):
        """Verify RATIO operation handles division by zero by outputting NaN."""
        df = pd.DataFrame({"A": [10.0, 20.0, 30.0], "B": [2.0, 0.0, -1.0], "target": [0, 1, 0]})
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step_ratio",
            operation=FeatureEngineeringOperation.RATIO,
            input_columns=["A", "B"],
            output_columns=["A_ratio_B"],
            reason="Ratio of A to B with zero denominator",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        # Row 1: 10 / 2 = 5
        # Row 2: 20 / 0 = NaN
        # Row 3: 30 / -1 = -30
        assert np.isnan(res.dataframe.loc[1, "A_ratio_B"])
        assert res.dataframe.loc[0, "A_ratio_B"] == 5.0
        assert res.dataframe.loc[2, "A_ratio_B"] == -30.0

    def test_successful_difference(self):
        """Verify DIFFERENCE operation (A - B) works correctly."""
        df = pd.DataFrame({"A": [10.0, 5.0], "B": [2.0, 8.0], "target": [0, 1]})
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step_diff",
            operation=FeatureEngineeringOperation.DIFFERENCE,
            input_columns=["A", "B"],
            output_columns=["A_diff_B"],
            reason="Difference between A and B",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert "A_diff_B" in res.dataframe.columns
        expected = [8.0, -3.0]
        pd.testing.assert_series_equal(res.dataframe["A_diff_B"], pd.Series(expected, name="A_diff_B"))

    def test_successful_log(self):
        """Verify LOG operation log1p(A) works correctly."""
        df = pd.DataFrame({"A": [0.0, 1.0, -1.0], "target": [0, 1, 0]})
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step_log",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["A"],
            output_columns=["log_A"],
            reason="Log transform of A",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert "log_A" in res.dataframe.columns
        expected = [np.log1p(0.0), np.log1p(1.0), np.log1p(-1.0)]
        pd.testing.assert_series_equal(res.dataframe["log_A"], pd.Series(expected, name="log_A"))

    def test_log_rejection_for_invalid_negative_values(self):
        """Verify LOG operation rejects values below -1."""
        df = pd.DataFrame({"A": [-1.01, 1.0], "target": [0, 1]})
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step_log",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["A"],
            output_columns=["log_A"],
            reason="Log transform of A",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="contains values below -1"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_successful_polynomial(self):
        """Verify POLYNOMIAL operation (degree=2) generates A^2 using PolynomialFeatures."""
        df = pd.DataFrame({"A": [2.0, 3.0, 4.0], "target": [0, 1, 0]})
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step_poly",
            operation=FeatureEngineeringOperation.POLYNOMIAL,
            input_columns=["A"],
            output_columns=["A_squared"],
            reason="Square of A",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert "A_squared" in res.dataframe.columns
        expected = [4.0, 9.0, 16.0]
        pd.testing.assert_series_equal(res.dataframe["A_squared"], pd.Series(expected, name="A_squared"))

    def test_sequential_execution_and_using_new_columns(self):
        """Verify that steps are executed in order and later steps can consume columns created by earlier ones."""
        df = pd.DataFrame({"A": [2.0, 3.0], "B": [5.0, 10.0], "target": [0, 1]})
        context = _make_dataset_context()
        
        # Step 1: Interaction A * B -> mult
        step1 = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["mult"],
            reason="interaction",
        )
        # Step 2: Difference mult - B -> diff
        step2 = FeatureEngineeringStep(
            step_id="step2",
            operation=FeatureEngineeringOperation.DIFFERENCE,
            input_columns=["mult", "B"],
            output_columns=["diff"],
            reason="difference using created column",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step1, step2])

        executor = FeatureEngineeringExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert "mult" in res.dataframe.columns
        assert "diff" in res.dataframe.columns
        
        # Step 1: 2*5 = 10, 3*10 = 30
        assert list(res.dataframe["mult"]) == [10.0, 30.0]
        # Step 2: 10 - 5 = 5, 30 - 10 = 20
        assert list(res.dataframe["diff"]) == [5.0, 20.0]
        assert res.engineered_columns == ["mult", "diff"]

    def test_validation_reject_none_inputs(self):
        """Verify that None inputs are rejected."""
        executor = FeatureEngineeringExecutor()
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()

        with pytest.raises(FeatureEngineeringExecutorError, match="dataframe cannot be None"):
            executor.execute(dataframe=None, dataset_context=context, plan=plan)

        with pytest.raises(FeatureEngineeringExecutorError, match="dataset_context cannot be None"):
            executor.execute(dataframe=df, dataset_context=None, plan=plan)

        with pytest.raises(FeatureEngineeringExecutorError, match="plan cannot be None"):
            executor.execute(dataframe=df, dataset_context=context, plan=None)

    def test_validation_reject_wrong_types(self):
        """Verify that wrong input types are rejected."""
        executor = FeatureEngineeringExecutor()
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()

        with pytest.raises(FeatureEngineeringExecutorError, match="dataframe must be a pandas.DataFrame"):
            executor.execute(dataframe="not-a-dataframe", dataset_context=context, plan=plan)

        with pytest.raises(FeatureEngineeringExecutorError, match="dataset_context must be a DatasetContext"):
            executor.execute(dataframe=df, dataset_context="not-a-context", plan=plan)

        with pytest.raises(FeatureEngineeringExecutorError, match="plan must be an MLPlan"):
            executor.execute(dataframe=df, dataset_context=context, plan="not-a-plan")

    def test_validation_reject_empty_dataframe(self):
        """Verify that empty dataframes are rejected."""
        executor = FeatureEngineeringExecutor()
        df = pd.DataFrame()
        context = _make_dataset_context()
        plan = _make_ml_plan()

        with pytest.raises(FeatureEngineeringExecutorError, match="dataframe cannot be empty"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_step_ids(self):
        """Verify that duplicate step IDs are rejected."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        step1 = FeatureEngineeringStep(
            step_id="step_1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["out1"],
            reason="step1",
        )
        step2 = FeatureEngineeringStep(
            step_id="step_2",
            operation=FeatureEngineeringOperation.DIFFERENCE,
            input_columns=["A", "B"],
            output_columns=["out2"],
            reason="step2",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step1, step2])
        from unittest.mock import patch
        with patch.object(step2, "step_id", "step_1"):
            executor = FeatureEngineeringExecutor()
            with pytest.raises(FeatureEngineeringExecutorError, match="Duplicate step_id found"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_empty_target_column(self):
        """Verify that empty target column name is rejected."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()
        from unittest.mock import patch
        with patch.object(plan, "target_column", ""):
            executor = FeatureEngineeringExecutor()
            with pytest.raises(FeatureEngineeringExecutorError, match="target_column cannot be empty"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_output_columns_within_step(self):
        """Verify that a single step cannot declare duplicate output columns."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["out1"],
            reason="step1",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])
        from unittest.mock import patch
        with patch.object(step, "output_columns", ["out1", "out1"]):
            executor = FeatureEngineeringExecutor()
            with pytest.raises(FeatureEngineeringExecutorError, match="Duplicate output columns found in step"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_output_names_across_steps(self):
        """Verify that output column names cannot be duplicated across steps."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        step1 = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["common_out"],
            reason="step1",
        )
        step2 = FeatureEngineeringStep(
            step_id="step2",
            operation=FeatureEngineeringOperation.DIFFERENCE,
            input_columns=["A", "B"],
            output_columns=["common_out"],
            reason="step2",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step1, step2])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="Duplicate output columns found across steps"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_existing_output_columns(self):
        """Verify that steps cannot output to columns that already exist in the input dataframe."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["A"],  # "A" already exists in dataframe
            reason="step1",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="already exists in the input dataframe"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_target_column_in_inputs(self):
        """Verify that target column cannot be used as an input column."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["target"],
            output_columns=["log_target"],
            reason="step1",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="cannot be used as an input column"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_target_column_in_outputs(self):
        """Verify that target column cannot be used as an output column."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["A"],
            output_columns=["target"],
            reason="step1",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="cannot be used as an output column"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_custom_operation(self):
        """Verify that CUSTOM operations are explicitly rejected."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.CUSTOM,
            input_columns=["A"],
            output_columns=["custom_out"],
            reason="step1",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="CUSTOM operation is rejected"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_unknown_operation(self):
        """Verify that unknown/unsupported operations are rejected."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        # Create a step with a non-supported operation (e.g. DATETIME_PARTS)
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.DATETIME_PARTS,
            input_columns=["A"],
            output_columns=["dt_out"],
            reason="step1",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="Unknown or unsupported operation"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_input_column(self):
        """Verify that missing input columns at execution time trigger a failure."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["non_existent_column"],
            output_columns=["log_out"],
            reason="step1",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="does not exist for step"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_wrong_argument_lengths(self):
        """Verify that operations reject incorrect numbers of input/output columns."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        executor = FeatureEngineeringExecutor()
        from unittest.mock import patch

        # Interaction with 1 input
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["out1"],
            reason="interaction",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])
        with patch.object(step, "input_columns", ["A"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 2 input columns"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

        # Interaction with 2 outputs
        with patch.object(step, "output_columns", ["out1", "out2"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 1 output column"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

        # Ratio with 3 inputs
        step_ratio = FeatureEngineeringStep(
            step_id="step2",
            operation=FeatureEngineeringOperation.RATIO,
            input_columns=["A", "B"],
            output_columns=["out1"],
            reason="ratio",
        )
        plan_ratio = _make_ml_plan(feature_engineering_steps=[step_ratio])
        with patch.object(step_ratio, "input_columns", ["A", "B", "A"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 2 input columns"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_ratio)

        # Ratio with 2 outputs
        with patch.object(step_ratio, "output_columns", ["out1", "out2"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 1 output column"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_ratio)

        # Difference with 1 input
        step_diff = FeatureEngineeringStep(
            step_id="step3",
            operation=FeatureEngineeringOperation.DIFFERENCE,
            input_columns=["A", "B"],
            output_columns=["out1"],
            reason="difference",
        )
        plan_diff = _make_ml_plan(feature_engineering_steps=[step_diff])
        with patch.object(step_diff, "input_columns", ["A"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 2 input columns"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_diff)

        # Difference with 2 outputs
        with patch.object(step_diff, "output_columns", ["out1", "out2"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 1 output column"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_diff)

        # Log with 2 inputs
        step_log = FeatureEngineeringStep(
            step_id="step4",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["A"],
            output_columns=["out1"],
            reason="log",
        )
        plan_log = _make_ml_plan(feature_engineering_steps=[step_log])
        with patch.object(step_log, "input_columns", ["A", "B"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 1 input column"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_log)

        # Log with 2 outputs
        with patch.object(step_log, "output_columns", ["out1", "out2"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 1 output column"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_log)

        # Polynomial with 2 inputs
        step_poly = FeatureEngineeringStep(
            step_id="step5",
            operation=FeatureEngineeringOperation.POLYNOMIAL,
            input_columns=["A"],
            output_columns=["out1"],
            reason="poly",
        )
        plan_poly = _make_ml_plan(feature_engineering_steps=[step_poly])
        with patch.object(step_poly, "input_columns", ["A", "B"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 1 input column"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_poly)

        # Polynomial with 2 outputs
        with patch.object(step_poly, "output_columns", ["out1", "out2"]):
            with pytest.raises(FeatureEngineeringExecutorError, match="requires exactly 1 output column"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_poly)


    def test_non_mutation(self):
        """Verify that the executor does not mutate inputs (dataframe, plan, context)."""
        df = _make_valid_dataframe(5)
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["out1"],
            reason="no mutation test",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        # Save copies of the original models to compare later
        df_orig = df.copy()
        context_orig = copy.deepcopy(context)
        plan_orig = copy.deepcopy(plan)

        executor = FeatureEngineeringExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        # Check results are valid
        assert isinstance(res, FeatureEngineeringResult)

        # Check original dataframe was not modified
        pd.testing.assert_frame_equal(df, df_orig)

        # Check dataset context and plan models were not modified
        assert context == context_orig
        assert plan == plan_orig

    def test_execution_failure_bubble_up(self):
        """Verify that underlying calculation errors raise a FeatureEngineeringExecutorError."""
        # DataFrame has string/object types that will raise TypeError on multiplication
        df = pd.DataFrame({"A": ["some-string"], "B": [2.0], "target": [0]})
        context = _make_dataset_context()
        step = FeatureEngineeringStep(
            step_id="step1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["A", "B"],
            output_columns=["out1"],
            reason="fail",
        )
        plan = _make_ml_plan(feature_engineering_steps=[step])

        executor = FeatureEngineeringExecutor()
        with pytest.raises(FeatureEngineeringExecutorError, match="Execution failed for step"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)
