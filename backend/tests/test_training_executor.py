"""Unit tests for TrainingExecutor."""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from backend.app.ml_plan.schemas import ModelCandidate, ModelFamily, SearchStrategy
from backend.app.ml_execution.training_executor import (
    TrainingExecutor,
    TrainingExecutorError,
    TrainingResult,
)


# ── Helper Builders ───────────────────────────────────────────────────

def _make_candidate(
    candidate_id: str = "model_01",
    family: ModelFamily = ModelFamily.RANDOM_FOREST,
    parameters: dict = None,
) -> ModelCandidate:
    if parameters is None:
        parameters = {}
    return ModelCandidate(
        candidate_id=candidate_id,
        model_family=family,
        parameters=parameters,
        search_strategy=SearchStrategy.NONE,
        search_space={},
        reason="Model candidate configuration.",
    )


# ── Test Suite ──────────────────────────────────────────────────────────────

class TestTrainingExecutor:
    """Tests covering TrainingExecutor fitting, validators, and edge cases."""

    def test_train_classification_models(self):
        """Verify successful training on supported classification estimators."""
        models_to_test = [
            (RandomForestClassifier(n_estimators=3, random_state=42), ModelFamily.RANDOM_FOREST),
            (GradientBoostingClassifier(n_estimators=3, random_state=42), ModelFamily.GRADIENT_BOOSTING),
            (LogisticRegression(random_state=42), ModelFamily.LOGISTIC_REGRESSION),
        ]

        X = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]})
        y = pd.Series([0, 1, 0, 1])

        executor = TrainingExecutor()

        for idx, (estimator, family) in enumerate(models_to_test):
            candidate = _make_candidate(candidate_id=f"clf_{idx}", family=family)
            res = executor.train(estimator=estimator, X_train=X, y_train=y, candidate=candidate)

            assert isinstance(res, TrainingResult)
            assert res.trained_estimator is estimator
            assert res.candidate_id == f"clf_{idx}"
            assert res.model_family == family
            assert res.training_duration_seconds > 0.0
            assert res.number_of_training_rows == 4
            assert res.number_of_training_features == 2
            assert res.fitted_successfully is True
            
            # Verify the estimator is indeed fitted (can predict)
            preds = res.trained_estimator.predict(X)
            assert len(preds) == 4

    def test_train_regression_models(self):
        """Verify successful training on supported regression estimators."""
        models_to_test = [
            (RandomForestRegressor(n_estimators=3, random_state=42), ModelFamily.RANDOM_FOREST),
            (GradientBoostingRegressor(n_estimators=3, random_state=42), ModelFamily.GRADIENT_BOOSTING),
            (LinearRegression(), ModelFamily.LINEAR_REGRESSION),
        ]

        X = pd.DataFrame({"Feature1": [1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([10.0, 20.0, 30.0, 40.0])

        executor = TrainingExecutor()

        for idx, (estimator, family) in enumerate(models_to_test):
            candidate = _make_candidate(candidate_id=f"reg_{idx}", family=family)
            res = executor.train(estimator=estimator, X_train=X, y_train=y, candidate=candidate)

            assert isinstance(res, TrainingResult)
            assert res.trained_estimator is estimator
            assert res.candidate_id == f"reg_{idx}"
            assert res.model_family == family
            assert res.training_duration_seconds > 0.0
            assert res.number_of_training_rows == 4
            assert res.number_of_training_features == 1
            assert res.fitted_successfully is True
            
            preds = res.trained_estimator.predict(X)
            assert len(preds) == 4

    def test_validation_rejects_none_inputs(self):
        """Verify None arguments raise TrainingExecutorError."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])
        candidate = _make_candidate(family=ModelFamily.LOGISTIC_REGRESSION)

        with pytest.raises(TrainingExecutorError, match="estimator cannot be None"):
            executor.train(None, X, y, candidate)

        with pytest.raises(TrainingExecutorError, match="X_train cannot be None"):
            executor.train(estimator, None, y, candidate)

        with pytest.raises(TrainingExecutorError, match="y_train cannot be None"):
            executor.train(estimator, X, None, candidate)

        with pytest.raises(TrainingExecutorError, match="candidate cannot be None"):
            executor.train(estimator, X, y, None)

    def test_validation_rejects_wrong_input_types(self):
        """Verify incorrect argument types raise TrainingExecutorError."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])
        candidate = _make_candidate(family=ModelFamily.LOGISTIC_REGRESSION)

        with pytest.raises(TrainingExecutorError, match="estimator must be an instance"):
            executor.train("not-estimator", X, y, candidate)

        with pytest.raises(TrainingExecutorError, match="candidate must be a ModelCandidate"):
            executor.train(estimator, X, y, "not-candidate")

        with pytest.raises(TrainingExecutorError, match="X_train must be a pandas DataFrame"):
            executor.train(estimator, "not-df", y, candidate)

        with pytest.raises(TrainingExecutorError, match="y_train must be a pandas Series"):
            executor.train(estimator, X, "not-series", candidate)

    def test_validation_rejects_empty_datasets(self):
        """Verify empty datasets raise TrainingExecutorError."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        candidate = _make_candidate(family=ModelFamily.LOGISTIC_REGRESSION)

        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype="int64")

        with pytest.raises(TrainingExecutorError, match="X_train dataset cannot be empty"):
            executor.train(estimator, empty_df, pd.Series([1]), candidate)

        with pytest.raises(TrainingExecutorError, match="y_train labels cannot be empty"):
            executor.train(estimator, pd.DataFrame({"A": [1.0]}), empty_series, candidate)

    def test_validation_rejects_row_mismatch(self):
        """Verify length mismatch between X and y raises TrainingExecutorError."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        candidate = _make_candidate(family=ModelFamily.LOGISTIC_REGRESSION)

        X = pd.DataFrame({"A": [1.0, 2.0]})
        y = pd.Series([1])  # 2 rows vs 1 label

        with pytest.raises(TrainingExecutorError, match="Row count mismatch"):
            executor.train(estimator, X, y, candidate)

    def test_validation_rejects_zero_feature_columns(self):
        """Verify zero feature columns in X raises TrainingExecutorError."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        candidate = _make_candidate(family=ModelFamily.LOGISTIC_REGRESSION)

        # DataFrame with rows but 0 columns
        X = pd.DataFrame(index=[0, 1, 2])
        y = pd.Series([0, 1, 0])

        with pytest.raises(TrainingExecutorError, match="zero columns"):
            executor.train(estimator, X, y, candidate)

    def test_validation_rejects_zero_feature_columns_numpy(self):
        """Verify zero feature columns in a numpy array raises TrainingExecutorError."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        candidate = _make_candidate(family=ModelFamily.LOGISTIC_REGRESSION)

        # 2D numpy array with 0 columns
        X = np.empty((3, 0))
        y = np.array([0, 1, 0])

        with pytest.raises(TrainingExecutorError, match="zero columns"):
            executor.train(estimator, X, y, candidate)


    def test_validation_rejects_candidate_mismatch(self):
        """Verify mismatched estimator family raises TrainingExecutorError."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        # candidate specifies RANDOM_FOREST, but estimator is LogisticRegression
        candidate = _make_candidate(family=ModelFamily.RANDOM_FOREST)

        X = pd.DataFrame({"A": [1.0, 2.0]})
        y = pd.Series([0, 1])

        with pytest.raises(TrainingExecutorError, match="does not match candidate family"):
            executor.train(estimator, X, y, candidate)

    def test_fit_failures_wrapped_in_training_error(self):
        """Verify fit exceptions are wrapped in TrainingExecutorError with cause preserved."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        candidate = _make_candidate(family=ModelFamily.LOGISTIC_REGRESSION)

        # Fit will fail if data contains strings
        X = pd.DataFrame({"A": ["cat", "dog"]})
        y = pd.Series([0, 1])

        with pytest.raises(TrainingExecutorError, match="Model training fit failed") as exc_info:
            executor.train(estimator, X, y, candidate)

        # Check original cause is preserved
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_non_mutation(self):
        """Verify input datasets and candidates are not mutated during train execution."""
        executor = TrainingExecutor()
        estimator = LogisticRegression()
        candidate = _make_candidate(family=ModelFamily.LOGISTIC_REGRESSION)

        X = pd.DataFrame({"A": [1.0, 2.0]})
        y = pd.Series([0, 1])

        X_orig = X.copy()
        y_orig = y.copy()
        candidate_orig = copy.deepcopy(candidate)

        res = executor.train(estimator=estimator, X_train=X, y_train=y, candidate=candidate)

        assert isinstance(res, TrainingResult)
        
        pd.testing.assert_frame_equal(X, X_orig)
        pd.testing.assert_series_equal(y, y_orig)
        assert candidate == candidate_orig
