"""Unit tests for FeatureSelectionExecutor."""

from __future__ import annotations

import copy
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from backend.app.compute_capabilities.schemas import AcceleratorType, ComputeTier
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    ModelCandidate,
    ModelFamily,
    ProblemType,
    SearchStrategy,
    SplitStrategy,
)
from backend.app.ml_execution.feature_selection_executor import (
    FeatureSelectionExecutor,
    FeatureSelectionExecutorError,
    FeatureSelectionResult,
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
    method: FeatureSelectionMethod = FeatureSelectionMethod.NONE,
    candidate_columns: list[str] = None,
    max_features: int | None = None,
    parameters: dict = None,
) -> MLPlan:
    if feature_columns is None:
        feature_columns = ["A", "B"]
    if candidate_columns is None:
        candidate_columns = ["A", "B"]
    if parameters is None:
        parameters = {}

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
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=method,
            candidate_columns=candidate_columns,
            max_features=max_features,
            parameters=parameters,
            reason="Feature selection logic choice.",
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


def _make_valid_dataframe(rows: int = 100) -> pd.DataFrame:
    # Target values
    target_values = [int(i % 2) for i in range(rows)]
    # A is perfectly correlated with target (identical to target)
    A_values = [float(t) for t in target_values]
    # B is noise
    np.random.seed(42)
    B_values = np.random.normal(0, 1, rows).tolist()
    return pd.DataFrame(
        {
            "A": A_values,
            "B": B_values,
            "target": target_values,
        }
    )



# ── Test Suite ──────────────────────────────────────────────────────────────

class TestFeatureSelectionExecutor:
    """Tests covering FeatureSelectionExecutor behavior and edge cases."""

    def test_method_none_success(self):
        """Verify NONE method returns dataframe with candidates + target unchanged."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        plan = _make_ml_plan(method=FeatureSelectionMethod.NONE)

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert isinstance(res, FeatureSelectionResult)
        assert res.selector_object is None
        assert res.selected_columns == ["A", "B"]
        assert res.removed_columns == []
        pd.testing.assert_frame_equal(res.selected_dataframe, df[["A", "B", "target"]])

    def test_method_variance_threshold_default(self):
        """Verify VARIANCE_THRESHOLD drops constant features by default (threshold=0.0)."""
        # Column B is constant, Column A has variance
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [5.0, 5.0, 5.0, 5.0],
                "target": [0, 1, 0, 1],
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(method=FeatureSelectionMethod.VARIANCE_THRESHOLD)

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert res.selected_columns == ["A"]
        assert res.removed_columns == ["B"]
        assert isinstance(res.selector_object, VarianceThreshold)
        assert list(res.selected_dataframe.columns) == ["A", "target"]

    def test_method_variance_threshold_custom(self):
        """Verify VARIANCE_THRESHOLD respects custom threshold from parameters."""
        # A variance = 1.66, B variance = 0.25 (since B is 1, 2, 1, 2, var is ~0.25)
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [1.0, 1.5, 1.0, 1.5],
                "target": [0, 1, 0, 1],
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(
            method=FeatureSelectionMethod.VARIANCE_THRESHOLD,
            parameters={"threshold": 0.5},
        )

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        # B variance is ~0.08, below 0.5. A variance is ~1.25, above 0.5.
        assert res.selected_columns == ["A"]
        assert res.removed_columns == ["B"]

    def test_method_correlation_filter_default(self):
        """Verify CORRELATION_FILTER drops highly correlated columns (> 0.95) deterministically."""
        # A and B are perfectly correlated (B = A * 2)
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [2.0, 4.0, 6.0, 8.0],
                "target": [0, 1, 0, 1],
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(method=FeatureSelectionMethod.CORRELATION_FILTER)

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        # A is evaluated first, B is correlated > 0.95, so B (second column) is removed.
        assert res.selected_columns == ["A"]
        assert res.removed_columns == ["B"]
        assert list(res.selected_dataframe.columns) == ["A", "target"]

    def test_method_correlation_filter_custom(self):
        """Verify CORRELATION_FILTER respects custom threshold override."""
        # A and B correlation is ~0.98
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [1.0, 2.0, 3.0, 5.0],
                "target": [0, 1, 0, 1],
            }
        )
        context = _make_dataset_context()
        
        # Default 0.95 threshold -> correlation (0.98) > 0.95 -> B is removed
        plan_default = _make_ml_plan(method=FeatureSelectionMethod.CORRELATION_FILTER)
        executor = FeatureSelectionExecutor()
        res_default = executor.execute(dataframe=df, dataset_context=context, plan=plan_default)
        assert res_default.selected_columns == ["A"]

        # Custom 0.99 threshold -> correlation (0.98) <= 0.99 -> keep both
        plan_custom = _make_ml_plan(
            method=FeatureSelectionMethod.CORRELATION_FILTER,
            parameters={"threshold": 0.99},
        )
        res_custom = executor.execute(dataframe=df, dataset_context=context, plan=plan_custom)
        assert res_custom.selected_columns == ["A", "B"]


    def test_method_mutual_information_classification(self):
        """Verify MUTUAL_INFORMATION classification ranks and selects correctly."""
        # A has high mutual info with target, B is noise.
        df = _make_valid_dataframe(100)
        context = _make_dataset_context()
        plan = _make_ml_plan(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            max_features=1,
        )

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert res.selected_columns == ["A"]
        assert res.removed_columns == ["B"]

    def test_method_mutual_information_regression(self):
        """Verify MUTUAL_INFORMATION regression ranks and selects correctly."""
        # Target column is continuous
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [-5.0, 10.0, 3.0, -8.0, 2.0],  # less correlated
                "target": [2.0, 4.0, 6.0, 8.0, 10.0],  # target = A * 2
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            max_features=1,
        )
        plan.problem_type = ProblemType.REGRESSION

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert res.selected_columns == ["A"]
        assert res.removed_columns == ["B"]

    def test_mutual_information_deterministic_tie_breaking(self):
        """Verify MUTUAL_INFORMATION tie-breaking sorts identical-scoring columns alphabetically."""
        # Create identical features
        df = pd.DataFrame(
            {
                "Z": [1.0, 2.0, 3.0, 4.0],
                "Y": [1.0, 2.0, 3.0, 4.0],
                "X": [1.0, 2.0, 3.0, 4.0],
                "target": [0, 1, 0, 1],
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            candidate_columns=["Z", "Y", "X"],
            feature_columns=["Z", "Y", "X"],
            max_features=2,
        )

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        # X, Y, Z will have identical MI scores.
        # Alphabetical sorting makes order: X, Y, Z. Keep max_features=2 -> Selected must be X and Y.
        assert res.selected_columns == ["X", "Y"]
        assert res.removed_columns == ["Z"]

    def test_method_model_based_classification(self):
        """Verify MODEL_BASED classification ranks using RandomForestClassifier importances."""
        df = _make_valid_dataframe(100)
        context = _make_dataset_context()
        plan = _make_ml_plan(
            method=FeatureSelectionMethod.MODEL_BASED,
            max_features=1,
        )

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert res.selected_columns == ["A"]
        assert res.removed_columns == ["B"]
        assert isinstance(res.selector_object, RandomForestClassifier)

    def test_method_model_based_regression(self):
        """Verify MODEL_BASED regression ranks using RandomForestRegressor importances."""
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
                "B": [0.0, 0.0, 0.0, 0.0, 0.0] * 10,
                "target": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(
            method=FeatureSelectionMethod.MODEL_BASED,
            max_features=1,
        )
        plan.problem_type = ProblemType.REGRESSION

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert res.selected_columns == ["A"]
        assert res.removed_columns == ["B"]
        assert isinstance(res.selector_object, RandomForestRegressor)

    def test_model_based_deterministic_tie_breaking(self):
        """Verify MODEL_BASED tie-breaking sorts identical-importance columns alphabetically."""
        # Const column features -> identical RF importances (0.0)
        df = pd.DataFrame(
            {
                "C": [1.0, 1.0, 1.0, 1.0],
                "B": [1.0, 1.0, 1.0, 1.0],
                "A": [1.0, 1.0, 1.0, 1.0],
                "target": [0, 1, 0, 1],
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(
            method=FeatureSelectionMethod.MODEL_BASED,
            candidate_columns=["C", "B", "A"],
            feature_columns=["C", "B", "A"],
            max_features=2,
        )

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        # Importances will be 0.0. Alphabetical sort: A, B, C. Keep 2 -> A and B.
        assert res.selected_columns == ["A", "B"]
        assert res.removed_columns == ["C"]

    def test_validation_reject_none_inputs(self):
        """Verify None inputs raise FeatureSelectionExecutorError."""
        executor = FeatureSelectionExecutor()
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()

        with pytest.raises(FeatureSelectionExecutorError, match="dataframe cannot be None"):
            executor.execute(dataframe=None, dataset_context=context, plan=plan)

        with pytest.raises(FeatureSelectionExecutorError, match="dataset_context cannot be None"):
            executor.execute(dataframe=df, dataset_context=None, plan=plan)

        with pytest.raises(FeatureSelectionExecutorError, match="plan cannot be None"):
            executor.execute(dataframe=df, dataset_context=context, plan=None)

    def test_validation_reject_wrong_types(self):
        """Verify wrong argument types raise FeatureSelectionExecutorError."""
        executor = FeatureSelectionExecutor()
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()

        with pytest.raises(FeatureSelectionExecutorError, match="dataframe must be a pandas.DataFrame"):
            executor.execute(dataframe="not-a-dataframe", dataset_context=context, plan=plan)

        with pytest.raises(FeatureSelectionExecutorError, match="dataset_context must be a DatasetContext"):
            executor.execute(dataframe=df, dataset_context="not-a-context", plan=plan)

        with pytest.raises(FeatureSelectionExecutorError, match="plan must be an MLPlan"):
            executor.execute(dataframe=df, dataset_context=context, plan="not-a-plan")

    def test_validation_reject_empty_dataframe(self):
        """Verify empty dataframe raises FeatureSelectionExecutorError."""
        df = pd.DataFrame()
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with pytest.raises(FeatureSelectionExecutorError, match="dataframe cannot be empty"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_feature_selection_plan(self):
        """Verify missing plan.feature_selection raises FeatureSelectionExecutorError."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with patch.object(plan, "feature_selection", None):
            with pytest.raises(FeatureSelectionExecutorError, match="plan.feature_selection cannot be None"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_empty_target_column(self):
        """Verify empty target column raises FeatureSelectionExecutorError."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with patch.object(plan, "target_column", ""):
            with pytest.raises(FeatureSelectionExecutorError, match="plan.target_column cannot be empty"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_target_column_missing_from_df(self):
        """Verify target column missing from dataframe raises FeatureSelectionExecutorError."""
        df = _make_valid_dataframe()
        # Drop target column
        df = df.drop(columns=["target"])
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with pytest.raises(FeatureSelectionExecutorError, match="Target column 'target' does not exist"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_feature_columns_in_df(self):
        """Verify duplicate columns in dataframe raise FeatureSelectionExecutorError."""
        df = _make_valid_dataframe()
        # Mock duplicate columns by joining columns
        df_dups = pd.concat([df, df["A"].rename("A")], axis=1)
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with pytest.raises(FeatureSelectionExecutorError, match="Duplicate column names detected"):
            executor.execute(dataframe=df_dups, dataset_context=context, plan=plan)

    def test_validation_reject_missing_plan_feature_column_in_df(self):
        """Verify that if plan feature_columns contains a missing column, it is rejected."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan(feature_columns=["A", "B", "missing_col"])
        executor = FeatureSelectionExecutor()
        with pytest.raises(FeatureSelectionExecutorError, match="Feature column 'missing_col' does not exist"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_target_in_candidates(self):
        """Verify that target column inside feature selection candidates is rejected."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with patch.object(plan.feature_selection, "candidate_columns", ["A", "target"]):
            with pytest.raises(FeatureSelectionExecutorError, match="cannot be included in feature selection candidate_columns"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_candidate_column(self):
        """Verify that if a candidate column does not exist in dataframe, it is rejected."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with patch.object(plan.feature_selection, "candidate_columns", ["A", "missing_candidate"]):
            with pytest.raises(FeatureSelectionExecutorError, match="Candidate column 'missing_candidate' does not exist"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_duplicate_candidate_columns(self):
        """Verify duplicate candidate columns in plan raise FeatureSelectionExecutorError."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with patch.object(plan.feature_selection, "candidate_columns", ["A", "A"]):
            with pytest.raises(FeatureSelectionExecutorError, match="Duplicate candidate columns detected"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_unknown_method(self):
        """Verify unknown feature selection method raises FeatureSelectionExecutorError."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        plan = _make_ml_plan()
        executor = FeatureSelectionExecutor()
        with patch.object(plan.feature_selection, "method", "unknown_selection_method"):
            with pytest.raises(FeatureSelectionExecutorError, match="Unknown or unsupported selection method"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_validation_reject_missing_max_features_for_ranking(self):
        """Verify MUTUAL_INFORMATION/MODEL_BASED reject None max_features."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        executor = FeatureSelectionExecutor()

        plan1 = _make_ml_plan(method=FeatureSelectionMethod.MUTUAL_INFORMATION)
        with pytest.raises(FeatureSelectionExecutorError, match="max_features must be specified for ranking"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan1)

        plan2 = _make_ml_plan(method=FeatureSelectionMethod.MODEL_BASED)
        with pytest.raises(FeatureSelectionExecutorError, match="max_features must be specified for ranking"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan2)

    def test_validation_reject_invalid_max_features_bounds(self):
        """Verify max_features <= 0 or > candidate columns count are rejected."""
        df = _make_valid_dataframe()
        context = _make_dataset_context()
        executor = FeatureSelectionExecutor()

        # max_features <= 0
        plan1 = _make_ml_plan(method=FeatureSelectionMethod.MUTUAL_INFORMATION)
        with patch.object(plan1.feature_selection, "max_features", 0):
            with pytest.raises(FeatureSelectionExecutorError, match="max_features must be strictly greater than 0"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan1)

        # max_features > candidates count (which is 2)
        plan2 = _make_ml_plan(method=FeatureSelectionMethod.MUTUAL_INFORMATION)
        with patch.object(plan2.feature_selection, "max_features", 5):
            with pytest.raises(FeatureSelectionExecutorError, match="cannot be larger than the number of candidate columns"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan2)

    def test_non_mutation(self):
        """Verify that feature selection does not mutate original inputs."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        plan = _make_ml_plan(method=FeatureSelectionMethod.NONE)

        df_orig = df.copy()
        context_orig = copy.deepcopy(context)
        plan_orig = copy.deepcopy(plan)

        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert isinstance(res, FeatureSelectionResult)
        pd.testing.assert_frame_equal(df, df_orig)
        assert context == context_orig
        assert plan == plan_orig

    def test_determinism(self):
        """Verify that running selection twice on identical data produces identical results."""
        df = _make_valid_dataframe(50)
        context = _make_dataset_context()
        
        # Test model-based classification determinism
        plan = _make_ml_plan(method=FeatureSelectionMethod.MODEL_BASED, max_features=1)
        
        executor = FeatureSelectionExecutor()
        res1 = executor.execute(dataframe=df, dataset_context=context, plan=plan)
        res2 = executor.execute(dataframe=df, dataset_context=context, plan=plan)

        assert res1.selected_columns == res2.selected_columns
        pd.testing.assert_frame_equal(res1.selected_dataframe, res2.selected_dataframe)

    def test_execution_failure_bubble_up(self):
        """Verify that sklearn internal fitting or correlation exceptions are caught and bubbled up."""
        # RandomForest will fail if dataframe contains string/object non-numeric values
        df = pd.DataFrame(
            {
                "A": ["cat", "dog", "mouse", "cat"],
                "B": [1.0, 2.0, 3.0, 4.0],
                "target": [0, 1, 0, 1],
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(method=FeatureSelectionMethod.MODEL_BASED, max_features=1)

        executor = FeatureSelectionExecutor()
        with pytest.raises(FeatureSelectionExecutorError, match="Execution failed during feature selection method"):
            executor.execute(dataframe=df, dataset_context=context, plan=plan)

    def test_unsupported_problem_types_bubble_up(self):
        """Verify that unsupported problem types raise FeatureSelectionExecutorError."""
        df = _make_valid_dataframe(10)
        context = _make_dataset_context()
        
        # Test Mutual Info with unknown problem type
        plan_mi = _make_ml_plan(method=FeatureSelectionMethod.MUTUAL_INFORMATION, max_features=1)
        with patch.object(plan_mi, "problem_type", "UNKNOWN_PROBLEM_TYPE"):
            executor = FeatureSelectionExecutor()
            with pytest.raises(FeatureSelectionExecutorError, match="Unsupported problem type for mutual information"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_mi)

        # Test Model Based with unknown problem type
        plan_mb = _make_ml_plan(method=FeatureSelectionMethod.MODEL_BASED, max_features=1)
        with patch.object(plan_mb, "problem_type", "UNKNOWN_PROBLEM_TYPE"):
            executor = FeatureSelectionExecutor()
            with pytest.raises(FeatureSelectionExecutorError, match="Unsupported problem type for model based selection"):
                executor.execute(dataframe=df, dataset_context=context, plan=plan_mb)

    def test_correlation_filter_j_already_removed(self):
        """Verify that correlation filtering skips columns already in the removed set."""
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [5.0, 10.0, 2.0, 8.0],
                "C": [2.0, 4.0, 6.0, 8.0],
                "target": [0, 1, 0, 1],
            }
        )
        context = _make_dataset_context()
        plan = _make_ml_plan(
            method=FeatureSelectionMethod.CORRELATION_FILTER,
            candidate_columns=["A", "B", "C"],
            feature_columns=["A", "B", "C"],
        )
        executor = FeatureSelectionExecutor()
        res = executor.execute(dataframe=df, dataset_context=context, plan=plan)
        assert res.selected_columns == ["A", "B"]
        assert res.removed_columns == ["C"]

