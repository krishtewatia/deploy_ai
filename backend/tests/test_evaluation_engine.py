"""Unit tests for EvaluationEngine."""

from __future__ import annotations

import copy
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError

from backend.app.ml_plan.schemas import ModelFamily
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ResolutionStatus,
    TargetSource,
    ProblemType,
    ProblemWarning,
)
from backend.app.ml_execution.training_executor import TrainingResult
from backend.app.ml_execution.evaluation_engine import (
    EvaluationEngine,
    EvaluationEngineError,
    EvaluationResult,
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
        goal="Evaluate model predictions.",
        problem_type=problem_type,
        target_column="target",
        target_source=TargetSource.USER,
        feature_columns=["A", "B"],
        excluded_columns=[],
        primary_metric=primary_metric,
        status=ResolutionStatus.RESOLVED,
        warnings=[ProblemWarning(code="WAR_01", message="A test warning.")],

        confirmation_items=[],
    )


def _make_training_result(
    estimator: Any,
    family: ModelFamily = ModelFamily.RANDOM_FOREST,
) -> TrainingResult:
    return TrainingResult(
        trained_estimator=estimator,
        candidate_id="model_01",
        model_family=family,
        training_duration_seconds=1.2,
        number_of_training_rows=10,
        number_of_training_features=2,
        fitted_successfully=True,
    )


# ── Test Suite ──────────────────────────────────────────────────────────────

class TestEvaluationEngine:
    """Tests covering EvaluationEngine metrics calculation, validations, and edge cases."""

    def test_evaluate_classification(self):
        """Verify evaluation works for classification (calculating accuracy, precision, recall, f1, roc-auc, report)."""
        # Fit a classification model
        model = LogisticRegression(random_state=42)
        X_train = pd.DataFrame({"A": [1.0, 2.0, 1.1, 2.1, 1.2, 2.2], "B": [2.0, 1.0, 2.1, 1.1, 2.2, 1.2]})
        y_train = pd.Series([0, 1, 0, 1, 0, 1])
        model.fit(X_train, y_train)

        X_test = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
        y_test = pd.Series([0, 1])

        problem_def = _make_problem_definition(problem_type=ProblemType.CLASSIFICATION, primary_metric="f1")
        train_res = _make_training_result(estimator=model, family=ModelFamily.LOGISTIC_REGRESSION)

        engine = EvaluationEngine()
        res = engine.evaluate(
            trained_estimator=model,
            X_test=X_test,
            y_test=y_test,
            problem_definition=problem_def,
            training_result=train_res,
        )

        assert isinstance(res, EvaluationResult)
        assert res.candidate_id == "model_01"
        assert res.model_family == ModelFamily.LOGISTIC_REGRESSION
        assert isinstance(res.predictions, list)
        assert len(res.predictions) == 2
        assert res.primary_metric == "f1"
        assert res.primary_metric_value == res.all_metrics["f1"]
        
        # Verify AI model critic fields
        assert res.train_score is None
        assert res.test_score == res.primary_metric_value
        assert res.prediction_count == 2
        assert res.training_duration == 1.2
        assert res.evaluation_duration > 0.0
        assert isinstance(res.model_parameters, dict)
        assert isinstance(res.warnings, list)

        
        # Classification metric validation
        metrics = res.all_metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        
        assert isinstance(res.confusion_matrix, list)
        assert isinstance(res.classification_report, dict)
        assert "accuracy" in res.classification_report
        
        # Linear coefficient mapping validation
        assert isinstance(res.feature_importance, dict)
        assert "A" in res.feature_importance
        assert "B" in res.feature_importance

    def test_evaluate_classification_multiclass(self):
        """Verify multiclass classification evaluations (accuracy, macro precision/recall/f1, and OvR roc-auc)."""
        model = LogisticRegression(random_state=42)
        # Multiclass data: classes 0, 1, 2
        X_train = pd.DataFrame({"A": [1.0, 2.0, 3.0, 1.1, 2.2, 3.3], "B": [3.0, 2.0, 1.0, 3.1, 2.1, 1.1]})
        y_train = pd.Series([0, 1, 2, 0, 1, 2])
        model.fit(X_train, y_train)

        X_test = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]})
        y_test = pd.Series([0, 1, 2])

        problem_def = _make_problem_definition(problem_type=ProblemType.CLASSIFICATION, primary_metric="accuracy")
        train_res = _make_training_result(estimator=model, family=ModelFamily.LOGISTIC_REGRESSION)

        engine = EvaluationEngine()
        res = engine.evaluate(
            trained_estimator=model,
            X_test=X_test,
            y_test=y_test,
            problem_definition=problem_def,
            training_result=train_res,
        )

        assert isinstance(res, EvaluationResult)
        assert res.all_metrics["roc_auc"] > 0.0
        assert len(res.confusion_matrix) == 3

    def test_evaluate_regression(self):
        """Verify regression evaluation metrics (MAE, MSE, RMSE, R2, feature importances)."""
        model = RandomForestRegressor(n_estimators=3, random_state=42)
        X_train = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]})
        y_train = pd.Series([10.0, 20.0, 15.0, 30.0])
        model.fit(X_train, y_train)

        X_test = pd.DataFrame({"A": [1.5, 2.5], "B": [5.5, 6.5]})
        y_test = pd.Series([12.0, 22.0])

        problem_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, primary_metric="rmse")
        train_res = _make_training_result(estimator=model, family=ModelFamily.RANDOM_FOREST)

        engine = EvaluationEngine()
        res = engine.evaluate(
            trained_estimator=model,
            X_test=X_test,
            y_test=y_test,
            problem_definition=problem_def,
            training_result=train_res,
        )

        assert isinstance(res, EvaluationResult)
        assert res.primary_metric == "rmse"
        assert res.primary_metric_value == res.all_metrics["rmse"]

        # Regression metrics validation
        metrics = res.all_metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        
        # Classification summaries must be None for Regression
        assert res.confusion_matrix is None
        assert res.classification_report is None

        # Feature importances validation
        assert isinstance(res.feature_importance, dict)
        assert "A" in res.feature_importance
        assert "B" in res.feature_importance

    def test_evaluate_numpy_and_no_importance(self):
        """Verify evaluation works for numpy arrays and model families with no feature importances."""
        model = LinearRegression()
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([2.0, 4.0, 6.0])
        model.fit(X_train, y_train)

        X_test = np.array([[1.5], [2.5]])
        y_test = np.array([3.0, 5.0])

        problem_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, primary_metric="mean_squared_error")
        train_res = _make_training_result(estimator=model, family=ModelFamily.LINEAR_REGRESSION)

        engine = EvaluationEngine()
        res = engine.evaluate(
            trained_estimator=model,
            X_test=X_test,
            y_test=y_test,
            problem_definition=problem_def,
            training_result=train_res,
        )

        assert isinstance(res, EvaluationResult)
        assert res.primary_metric_value == res.all_metrics["mse"]
        assert isinstance(res.feature_importance, dict)
        assert "feature_0" in res.feature_importance

    def test_cv_scores_extraction(self):
        """Verify CV scores are successfully extracted if cv_results_ exists on the estimator."""
        model = RandomForestClassifier(random_state=42)
        # Mock cv_results_ and best_score_ on model
        model.cv_results_ = {"mean_test_score": np.array([0.85, 0.90])}
        model.best_score_ = 0.92
        # Mock check_is_fitted check
        model.classes_ = np.array([0, 1])

        X_test = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        y_test = pd.Series([0, 1])

        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model, family=ModelFamily.RANDOM_FOREST)

        # Patch prediction to return static results
        with patch.object(model, "predict", return_value=np.array([0, 1])):
            engine = EvaluationEngine()
            res = engine.evaluate(
                trained_estimator=model,
                X_test=X_test,
                y_test=y_test,
                problem_definition=problem_def,
                training_result=train_res,
            )
            assert res.cross_validation_scores == [0.85, 0.90]
            assert res.train_score == 0.92
            assert res.warnings == ["A test warning."]


    def test_validation_rejects_none_inputs(self):
        """Verify None inputs raise EvaluationEngineError."""
        engine = EvaluationEngine()
        model = LogisticRegression()
        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])
        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model)

        with pytest.raises(EvaluationEngineError, match="trained_estimator cannot be None"):
            engine.evaluate(None, X, y, problem_def, train_res)

        with pytest.raises(EvaluationEngineError, match="X_test cannot be None"):
            engine.evaluate(model, None, y, problem_def, train_res)

        with pytest.raises(EvaluationEngineError, match="y_test cannot be None"):
            engine.evaluate(model, X, None, problem_def, train_res)

        with pytest.raises(EvaluationEngineError, match="problem_definition cannot be None"):
            engine.evaluate(model, X, y, None, train_res)

        with pytest.raises(EvaluationEngineError, match="training_result cannot be None"):
            engine.evaluate(model, X, y, problem_def, None)

    def test_validation_rejects_wrong_input_types(self):
        """Verify wrong input types raise EvaluationEngineError."""
        engine = EvaluationEngine()
        model = LogisticRegression()
        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])
        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model)

        with pytest.raises(EvaluationEngineError, match="trained_estimator must be a scikit-learn"):
            engine.evaluate("not-model", X, y, problem_def, train_res)

        with pytest.raises(EvaluationEngineError, match="problem_definition must be a ProblemDefinition"):
            engine.evaluate(model, X, y, "not-def", train_res)

        with pytest.raises(EvaluationEngineError, match="training_result must be a TrainingResult"):
            engine.evaluate(model, X, y, problem_def, "not-result")

        with pytest.raises(EvaluationEngineError, match="X_test must be a pandas DataFrame"):
            engine.evaluate(model, "not-df", y, problem_def, train_res)

        with pytest.raises(EvaluationEngineError, match="y_test must be a pandas Series"):
            engine.evaluate(model, X, "not-series", problem_def, train_res)

    def test_validation_rejects_empty_datasets(self):
        """Verify empty datasets raise EvaluationEngineError."""
        engine = EvaluationEngine()
        # Mock fitted state
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model)

        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype="int64")

        with pytest.raises(EvaluationEngineError, match="X_test dataset cannot be empty"):
            engine.evaluate(model, empty_df, pd.Series([1]), problem_def, train_res)

        with pytest.raises(EvaluationEngineError, match="y_test labels cannot be empty"):
            engine.evaluate(model, pd.DataFrame({"A": [1.0]}), empty_series, problem_def, train_res)

    def test_validation_rejects_row_mismatch(self):
        """Verify length mismatches raise EvaluationEngineError."""
        engine = EvaluationEngine()
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model)

        X = pd.DataFrame({"A": [1.0, 2.0]})
        y = pd.Series([1])

        with pytest.raises(EvaluationEngineError, match="Row mismatch"):
            engine.evaluate(model, X, y, problem_def, train_res)

    def test_validation_rejects_untrained_estimator(self):
        """Verify check_is_fitted raises error for untrained models."""
        engine = EvaluationEngine()
        model = LogisticRegression()  # Not fitted

        X = pd.DataFrame({"A": [1.0]})
        y = pd.Series([1])
        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model)

        with pytest.raises(EvaluationEngineError, match="trained_estimator is not fitted"):
            engine.evaluate(model, X, y, problem_def, train_res)

    def test_validation_rejects_missing_targets(self):
        """Verify missing/null values in y_test target are rejected."""
        engine = EvaluationEngine()
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model)

        # 1. Null in Pandas Series
        X_df = pd.DataFrame({"A": [1.0, 2.0]})
        y_series = pd.Series([1, None], dtype="float64")
        with pytest.raises(EvaluationEngineError, match="target contains missing/null"):
            engine.evaluate(model, X_df, y_series, problem_def, train_res)

        # 2. NaN in NumPy numeric array
        y_np_nan = np.array([1.0, np.nan])
        with pytest.raises(EvaluationEngineError, match="target contains missing/NaN"):
            engine.evaluate(model, X_df, y_np_nan, problem_def, train_res)

        # 3. None in NumPy object array
        y_np_obj = np.array([1, None], dtype=object)
        with pytest.raises(EvaluationEngineError, match="target contains missing/None"):
            engine.evaluate(model, X_df, y_np_obj, problem_def, train_res)

    def test_validation_unsupported_problem_type(self):
        """Verify unknown problem type raises EvaluationEngineError."""
        engine = EvaluationEngine()
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model)

        X = pd.DataFrame({"A": [1.0, 2.0]})
        y = pd.Series([0, 1])

        with patch.object(model, "predict", return_value=np.array([0, 1])):
            with patch.object(problem_def, "problem_type", "UNKNOWN_TYPE"):
                with pytest.raises(EvaluationEngineError, match="Unsupported problem type"):
                    engine.evaluate(model, X, y, problem_def, train_res)

    def test_prediction_execution_failure_wrapping(self):
        """Verify failures during predict calls are wrapped in EvaluationEngineError."""
        engine = EvaluationEngine()
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model)

        X = pd.DataFrame({"A": [1.0, 2.0]})
        y = pd.Series([0, 1])

        # Force predict to raise an exception
        with patch.object(model, "predict", side_effect=ValueError("Predict crash")):
            with pytest.raises(EvaluationEngineError, match="Prediction execution failed") as exc_info:
                engine.evaluate(model, X, y, problem_def, train_res)
            assert isinstance(exc_info.value.__cause__, ValueError)

    def test_non_mutation(self):
        """Verify that evaluate does not mutate inputs."""
        model = LogisticRegression(random_state=42)
        X_train = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        y_train = pd.Series([0, 1])
        model.fit(X_train, y_train)

        X_test = pd.DataFrame({"A": [1.5, 2.5], "B": [3.5, 4.5]})
        y_test = pd.Series([0, 1])

        problem_def = _make_problem_definition()
        train_res = _make_training_result(estimator=model, family=ModelFamily.LOGISTIC_REGRESSION)

        X_test_orig = X_test.copy()
        y_test_orig = y_test.copy()
        model_orig = copy.deepcopy(model)
        problem_def_orig = copy.deepcopy(problem_def)
        train_res_orig = copy.copy(train_res)

        engine = EvaluationEngine()
        res = engine.evaluate(
            trained_estimator=model,
            X_test=X_test,
            y_test=y_test,
            problem_definition=problem_def,
            training_result=train_res,
        )

        assert isinstance(res, EvaluationResult)
        
        # Verify inputs did not change
        pd.testing.assert_frame_equal(X_test, X_test_orig)
        pd.testing.assert_series_equal(y_test, y_test_orig)
        assert model.get_params() == model_orig.get_params()
        assert problem_def == problem_def_orig
        
        # Verify training_result attributes
        assert train_res.candidate_id == train_res_orig.candidate_id
        assert train_res.model_family == train_res_orig.model_family
        assert train_res.training_duration_seconds == train_res_orig.training_duration_seconds
        assert train_res.number_of_training_rows == train_res_orig.number_of_training_rows
        assert train_res.number_of_training_features == train_res_orig.number_of_training_features
        assert train_res.fitted_successfully == train_res_orig.fitted_successfully
        assert train_res.trained_estimator is train_res_orig.trained_estimator

    def test_evaluate_numpy_with_feature_importances(self):
        """Verify evaluation extracts feature importances for numpy arrays."""
        model = RandomForestRegressor(n_estimators=3, random_state=42)
        X_train = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])
        y_train = np.array([10.0, 20.0, 15.0, 30.0])
        model.fit(X_train, y_train)

        X_test = np.array([[1.5, 5.5], [2.5, 6.5]])
        y_test = np.array([12.0, 22.0])

        problem_def = _make_problem_definition(problem_type=ProblemType.REGRESSION, primary_metric="r_squared")
        train_res = _make_training_result(estimator=model, family=ModelFamily.RANDOM_FOREST)

        engine = EvaluationEngine()
        res = engine.evaluate(
            trained_estimator=model,
            X_test=X_test,
            y_test=y_test,
            problem_definition=problem_def,
            training_result=train_res,
        )

        assert isinstance(res, EvaluationResult)
        assert isinstance(res.feature_importance, dict)
        assert "feature_0" in res.feature_importance
        assert "feature_1" in res.feature_importance


