"""Unit tests for HyperparameterOptimizer."""

from __future__ import annotations

import copy
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from backend.app.compute_capabilities.schemas import AcceleratorType, ComputeTier
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
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ResolutionStatus,
    TargetSource,
)
from backend.app.ml_execution.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    HyperparameterOptimizerError,
    HyperparameterOptimizationResult,
)


# ── Helper Builders ───────────────────────────────────────────────────

def _make_problem_definition(
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    primary_metric: str = "accuracy",
) -> ProblemDefinition:
    return ProblemDefinition(
        definition_id="def_01",
        request_id="req_01",
        dataset_id="ds_01",
        goal="Optimize model parameters.",
        problem_type=problem_type,
        target_column="target",
        target_source=TargetSource.USER,
        feature_columns=["A", "B"],
        excluded_columns=[],
        primary_metric=primary_metric,
        status=ResolutionStatus.RESOLVED,
        warnings=[],
        confirmation_items=[],
    )


def _make_ml_plan(cv_folds: int = 3) -> MLPlan:
    return MLPlan(
        plan_id="plan_01",
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="def_01",
        compute_capability_id="cap_01",
        problem_type=ProblemType.CLASSIFICATION,
        target_column="target",
        feature_columns=["A", "B"],
        preprocessing_steps=[],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["A", "B"],
            max_features=2,
            parameters={},
            reason="No feature selection",
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
                candidate_id="model_rf",
                model_family=ModelFamily.RANDOM_FOREST,
                parameters={"n_estimators": 5},
                search_strategy=SearchStrategy.GRID,
                search_space={"n_estimators": [5, 10]},
                reason="Tuning RF.",
            )
        ],
        evaluation_plan=EvaluationPlan(
            primary_metric="accuracy",
            secondary_metrics=["f1"],
            cross_validation_folds=cv_folds,
        ),
        execution_constraints=ExecutionConstraints(
            parallel_workers=1,
            use_gpu_acceleration=False,
            accelerator_type=AcceleratorType.NONE,
            compute_tier=ComputeTier.STANDARD,
        ),
    )


def _make_candidate(
    family: ModelFamily = ModelFamily.RANDOM_FOREST,
    strategy: SearchStrategy = SearchStrategy.NONE,
    space: dict = None,
) -> ModelCandidate:
    if space is None:
        space = {}
    return ModelCandidate(
        candidate_id="model_01",
        model_family=family,
        parameters={"random_state": 42},
        search_strategy=strategy,
        search_space=space,
        reason="Model candidate decision.",
    )


# ── Test Suite ──────────────────────────────────────────────────────────────

class TestHyperparameterOptimizer:
    """Tests covering HyperparameterOptimizer functionality and edge cases."""

    def test_optimize_none_strategy(self):
        """Verify NONE search strategy returns unmodified estimator immediately without fitting."""
        estimator = RandomForestClassifier(n_estimators=5, random_state=42)
        candidate = _make_candidate(strategy=SearchStrategy.NONE)
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        y = pd.Series([0, 1])

        optimizer = HyperparameterOptimizer()
        res = optimizer.optimize(
            estimator=estimator,
            candidate=candidate,
            X_train=X,
            y_train=y,
            problem_definition=problem_def,
        )

        assert isinstance(res, HyperparameterOptimizationResult)
        assert res.optimized_estimator is estimator
        assert res.best_score is None
        assert res.search_strategy == SearchStrategy.NONE
        assert res.cv_results is None
        assert res.best_parameters == estimator.get_params()

    def test_optimize_grid_strategy(self):
        """Verify GRID strategy fits correctly and returns optimized results."""
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate(
            strategy=SearchStrategy.GRID,
            space={"n_estimators": [5, 10], "max_depth": [2, 4]},
        )
        problem_def = _make_problem_definition()
        plan = _make_ml_plan(cv_folds=2)

        # Simple classification dataset
        X = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "B": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        optimizer = HyperparameterOptimizer()
        res = optimizer.optimize(
            estimator=estimator,
            candidate=candidate,
            X_train=X,
            y_train=y,
            problem_definition=problem_def,
            plan=plan,
        )

        assert isinstance(res, HyperparameterOptimizationResult)
        assert res.search_strategy == SearchStrategy.GRID
        assert res.best_score is not None
        assert isinstance(res.best_parameters, dict)
        assert "n_estimators" in res.best_parameters
        assert "max_depth" in res.best_parameters
        assert isinstance(res.cv_results, dict)
        assert "mean_test_score" in res.cv_results
        assert res.optimized_estimator is not estimator
        assert isinstance(res.optimized_estimator, RandomForestClassifier)
        assert res.optimized_estimator.n_estimators in [5, 10]

    def test_optimize_random_strategy(self):
        """Verify RANDOM strategy fits correctly and returns optimized results."""
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate(
            strategy=SearchStrategy.RANDOM,
            space={"n_estimators": [5, 10], "max_depth": [2, 4]},
        )
        problem_def = _make_problem_definition()
        plan = _make_ml_plan(cv_folds=2)

        # Simple classification dataset
        X = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "B": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        optimizer = HyperparameterOptimizer()
        res = optimizer.optimize(
            estimator=estimator,
            candidate=candidate,
            X_train=X,
            y_train=y,
            problem_definition=problem_def,
            plan=plan,
        )

        assert isinstance(res, HyperparameterOptimizationResult)
        assert res.search_strategy == SearchStrategy.RANDOM
        assert res.best_score is not None
        assert isinstance(res.best_parameters, dict)
        assert isinstance(res.cv_results, dict)
        assert res.optimized_estimator is not estimator
        assert isinstance(res.optimized_estimator, RandomForestClassifier)

    def test_metric_and_folds_mapping(self):
        """Verify metric overrides (classification & regression mapping) and default folds work."""
        # 1. Regression r2 check
        estimator = Ridge(random_state=42)
        candidate = _make_candidate(
            family=ModelFamily.RIDGE,
            strategy=SearchStrategy.GRID,
            space={"alpha": [0.1, 1.0]},
        )
        problem_def = _make_problem_definition(
            problem_type=ProblemType.REGRESSION,
            primary_metric="r_squared",
        )

        X = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "B": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
        y = pd.Series([10.0, 20.0, 15.0, 30.0, 25.0, 40.0])

        optimizer = HyperparameterOptimizer()
        res = optimizer.optimize(
            estimator=estimator,
            candidate=candidate,
            X_train=X,
            y_train=y,
            problem_definition=problem_def,
            plan=None,  # Should default to 5 folds
        )
        assert res.execution_summary["scoring"] == "r2"
        assert res.execution_summary["folds"] == 5

        # 2. Regression mse check
        problem_def.primary_metric = "mse"
        res_mse = optimizer.optimize(
            estimator=estimator,
            candidate=candidate,
            X_train=X,
            y_train=y,
            problem_definition=problem_def,
        )
        assert res_mse.execution_summary["scoring"] == "neg_mean_squared_error"

    def test_validation_rejects_none_inputs(self):
        """Verify None inputs raise HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate()
        problem_def = _make_problem_definition()
        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])

        with pytest.raises(HyperparameterOptimizerError, match="estimator cannot be None"):
            optimizer.optimize(None, candidate, X, y, problem_def)

        with pytest.raises(HyperparameterOptimizerError, match="candidate cannot be None"):
            optimizer.optimize(estimator, None, X, y, problem_def)

        with pytest.raises(HyperparameterOptimizerError, match="X_train cannot be None"):
            optimizer.optimize(estimator, candidate, None, y, problem_def)

        with pytest.raises(HyperparameterOptimizerError, match="y_train cannot be None"):
            optimizer.optimize(estimator, candidate, X, None, problem_def)

        with pytest.raises(HyperparameterOptimizerError, match="problem_definition cannot be None"):
            optimizer.optimize(estimator, candidate, X, y, None)

    def test_validation_rejects_wrong_input_types(self):
        """Verify wrong input types raise HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate()
        problem_def = _make_problem_definition()
        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])

        with pytest.raises(HyperparameterOptimizerError, match="estimator must be an instance"):
            optimizer.optimize("not-estimator", candidate, X, y, problem_def)

        with pytest.raises(HyperparameterOptimizerError, match="candidate must be a ModelCandidate"):
            optimizer.optimize(estimator, "not-candidate", X, y, problem_def)

        with pytest.raises(HyperparameterOptimizerError, match="problem_definition must be a ProblemDefinition"):
            optimizer.optimize(estimator, candidate, X, y, "not-probdef")

        with pytest.raises(HyperparameterOptimizerError, match="X_train must be a pandas DataFrame"):
            optimizer.optimize(estimator, candidate, "not-df", y, problem_def)

        with pytest.raises(HyperparameterOptimizerError, match="y_train must be a pandas Series"):
            optimizer.optimize(estimator, candidate, X, "not-series", problem_def)

        with pytest.raises(HyperparameterOptimizerError, match="plan must be an MLPlan"):
            optimizer.optimize(estimator, candidate, X, y, problem_def, plan="not-plan")

    def test_validation_rejects_empty_datasets(self):
        """Verify empty datasets raise HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate()
        problem_def = _make_problem_definition()

        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype="int64")

        # Empty features
        with pytest.raises(HyperparameterOptimizerError, match="X_train dataset cannot be empty"):
            optimizer.optimize(estimator, candidate, empty_df, pd.Series([1]), problem_def)

        # Empty labels
        with pytest.raises(HyperparameterOptimizerError, match="y_train labels cannot be empty"):
            optimizer.optimize(estimator, candidate, pd.DataFrame({"A": [1.0]}), empty_series, problem_def)

    def test_validation_rejects_dataset_size_mismatches(self):
        """Verify length mismatch between X and y raises HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate()
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0, 2.0]})
        y = pd.Series([1])  # 2 vs 1 row

        with pytest.raises(HyperparameterOptimizerError, match="Dataset size mismatch"):
            optimizer.optimize(estimator, candidate, X, y, problem_def)

    def test_validation_rejects_candidate_mismatch(self):
        """Verify mismatched estimator class for candidate family raises HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        # RandomForest estimator with LinearRegression candidate family
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate(family=ModelFamily.LINEAR_REGRESSION)
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])

        with pytest.raises(HyperparameterOptimizerError, match="does not match candidate family"):
            optimizer.optimize(estimator, candidate, X, y, problem_def)

    def test_validation_rejects_unsupported_search_strategy(self):
        """Verify unsupported search strategies raise HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate()
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])

        # Patch search_strategy to an invalid strategy string
        with patch.object(candidate, "search_strategy", "GRID_RANDOM_HYBRID"):
            with pytest.raises(HyperparameterOptimizerError, match="Unsupported search strategy"):
                optimizer.optimize(estimator, candidate, X, y, problem_def)

    def test_validation_rejects_empty_search_space(self):
        """Verify empty search space for GRID or RANDOM raises HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate(strategy=SearchStrategy.NONE, space={})
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])

        with patch.object(candidate, "search_strategy", SearchStrategy.GRID):
            with pytest.raises(HyperparameterOptimizerError, match="search_space cannot be empty"):
                optimizer.optimize(estimator, candidate, X, y, problem_def)


    def test_validation_rejects_invalid_parameter_names(self):
        """Verify invalid parameter names in search space raise HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate(
            strategy=SearchStrategy.GRID,
            space={"invalid_estimator_parameter": [10, 20]},
        )
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])

        with pytest.raises(HyperparameterOptimizerError, match="Invalid parameter"):
            optimizer.optimize(estimator, candidate, X, y, problem_def)

    def test_validation_unsupported_problem_type_fallback(self):
        """Verify unsupported problem types in optimizer raise HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        candidate = _make_candidate(strategy=SearchStrategy.GRID, space={"n_estimators": [5]})
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        with patch.object(problem_def, "problem_type", "UNKNOWN_TYPE"):
            with pytest.raises(HyperparameterOptimizerError, match="Unsupported problem type"):
                optimizer.optimize(estimator, candidate, X, y, problem_def)

    def test_fitting_failures_bubble_up(self):
        """Verify fit exceptions from scikit-learn search methods raise HyperparameterOptimizerError."""
        optimizer = HyperparameterOptimizer()
        estimator = RandomForestClassifier(random_state=42)
        
        # Fit will fail if search space values are invalid (e.g. string for integer n_estimators)
        candidate = _make_candidate(
            strategy=SearchStrategy.GRID,
            space={"n_estimators": ["invalid_value_for_testing"]},
        )
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        with pytest.raises(HyperparameterOptimizerError, match="Failed during grid search fit"):
            optimizer.optimize(estimator, candidate, X, y, problem_def)

    def test_non_mutation(self):
        """Verify that optimize does not mutate input datasets, estimator, candidate or problem_definition."""
        estimator = RandomForestClassifier(n_estimators=5, random_state=42)
        candidate = _make_candidate(strategy=SearchStrategy.NONE)
        problem_def = _make_problem_definition()

        X = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        y = pd.Series([0, 1])
        plan = _make_ml_plan()

        X_orig = X.copy()
        y_orig = y.copy()
        estimator_orig = copy.deepcopy(estimator)
        candidate_orig = copy.deepcopy(candidate)
        problem_def_orig = copy.deepcopy(problem_def)
        plan_orig = copy.deepcopy(plan)

        optimizer = HyperparameterOptimizer()
        res = optimizer.optimize(
            estimator=estimator,
            candidate=candidate,
            X_train=X,
            y_train=y,
            problem_definition=problem_def,
            plan=plan,
        )

        assert isinstance(res, HyperparameterOptimizationResult)
        
        # Non-mutation checks
        pd.testing.assert_frame_equal(X, X_orig)
        pd.testing.assert_series_equal(y, y_orig)
        assert estimator.get_params() == estimator_orig.get_params()
        assert candidate == candidate_orig
        assert problem_def == problem_def_orig
        assert plan == plan_orig
