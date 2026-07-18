"""Model Evaluation Engine for DeployAI.

Stage 9I evaluates a trained estimator on test datasets, computing performance
metrics and generating detailed classification/regression evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from backend.app.ml_plan.schemas import ModelFamily, ProblemType
from backend.app.problem_definition.schemas import ProblemDefinition
from backend.app.ml_execution.training_executor import TrainingResult


class EvaluationEngineError(Exception):
    """Raised when evaluation fails due to validation errors, fit issues, or prediction failures."""

    pass


@dataclass
class EvaluationResult:
    """The structured result of the model evaluation execution."""

    candidate_id: str
    model_family: ModelFamily
    predictions: list[Any]
    primary_metric: str
    primary_metric_value: float
    all_metrics: dict[str, float]
    confusion_matrix: list[list[int]] | None
    classification_report: dict[str, Any] | None
    feature_importance: dict[str, float] | None
    cross_validation_scores: list[float] | None
    evaluation_duration_seconds: float
    train_score: float | None
    test_score: float
    prediction_count: int
    training_duration: float
    evaluation_duration: float
    model_parameters: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    execution_summary: dict[str, Any] = field(default_factory=dict)



class EvaluationEngine:
    """Evaluates trained scikit-learn estimators on test datasets."""

    def evaluate(
        self,
        trained_estimator: BaseEstimator,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        problem_definition: ProblemDefinition,
        training_result: TrainingResult,
    ) -> EvaluationResult:
        """Evaluate a trained estimator on held-out test data.

        Args:
            trained_estimator: The fitted scikit-learn estimator.
            X_test: Test features.
            y_test: Test target values.
            problem_definition: Resolved ML problem definition contract.
            training_result: Result output from the TrainingExecutor.

        Returns:
            EvaluationResult containing performance metrics, predictions, and metadata.

        Raises:
            EvaluationEngineError: For validation failures or fit issues.
        """
        # 1. Validation: Reject None
        if trained_estimator is None:
            raise EvaluationEngineError("trained_estimator cannot be None")
        if X_test is None:
            raise EvaluationEngineError("X_test cannot be None")
        if y_test is None:
            raise EvaluationEngineError("y_test cannot be None")
        if problem_definition is None:
            raise EvaluationEngineError("problem_definition cannot be None")
        if training_result is None:
            raise EvaluationEngineError("training_result cannot be None")

        # 2. Validation: Wrong types
        if not isinstance(trained_estimator, BaseEstimator):
            raise EvaluationEngineError("trained_estimator must be a scikit-learn BaseEstimator instance")
        if not isinstance(problem_definition, ProblemDefinition):
            raise EvaluationEngineError("problem_definition must be a ProblemDefinition instance")
        if not isinstance(training_result, TrainingResult):
            raise EvaluationEngineError("training_result must be a TrainingResult instance")
        if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
            raise EvaluationEngineError("X_test must be a pandas DataFrame or numpy ndarray")
        if not isinstance(y_test, (pd.Series, np.ndarray)):
            raise EvaluationEngineError("y_test must be a pandas Series or numpy ndarray")

        # 3. Validation: Empty datasets and size mismatch
        n_samples = X_test.shape[0] if isinstance(X_test, np.ndarray) else len(X_test)
        n_labels = y_test.shape[0] if isinstance(y_test, np.ndarray) else len(y_test)

        if n_samples == 0:
            raise EvaluationEngineError("X_test dataset cannot be empty")
        if n_labels == 0:
            raise EvaluationEngineError("y_test labels cannot be empty")
        if n_samples != n_labels:
            raise EvaluationEngineError(
                f"Row mismatch: X_test has {n_samples} rows, y_test has {n_labels} rows"
            )

        # 4. Validation: Untrained estimator (verify check_is_fitted)
        try:
            check_is_fitted(trained_estimator)
        except NotFittedError as e:
            raise EvaluationEngineError(f"trained_estimator is not fitted/trained: {e}") from e

        # 5. Validation: Missing target (check for NaN / Null values in y_test)
        if isinstance(y_test, pd.Series):
            if y_test.isnull().any():
                raise EvaluationEngineError("y_test target contains missing/null values")
        elif isinstance(y_test, np.ndarray):
            # check for NaN in float arrays, or None in object arrays
            if np.issubdtype(y_test.dtype, np.number):
                if np.isnan(y_test).any():
                    raise EvaluationEngineError("y_test target contains missing/NaN values")
            else:
                if any(val is None for val in y_test):
                    raise EvaluationEngineError("y_test target contains missing/None values")

        start_time = time.perf_counter()

        # 6. Execute predictions (wrap failures)
        try:
            preds = trained_estimator.predict(X_test)
        except Exception as e:
            raise EvaluationEngineError(f"Prediction execution failed: {e}") from e

        # 7. Compute metrics
        all_metrics: dict[str, float] = {}
        conf_matrix: list[list[int]] | None = None
        class_report: dict[str, Any] | None = None

        problem_type = problem_definition.problem_type
        if problem_type == ProblemType.CLASSIFICATION:
            # Classification metrics
            all_metrics["accuracy"] = float(accuracy_score(y_test, preds))

            unique_labels = np.unique(y_test)
            is_multiclass = len(unique_labels) > 2
            avg_mode = "macro" if is_multiclass else "binary"

            all_metrics["precision"] = float(precision_score(y_test, preds, average=avg_mode, zero_division=0))
            all_metrics["recall"] = float(recall_score(y_test, preds, average=avg_mode, zero_division=0))
            all_metrics["f1"] = float(f1_score(y_test, preds, average=avg_mode, zero_division=0))

            # ROC-AUC calculation (if estimator supports probabilities)
            if hasattr(trained_estimator, "predict_proba"):
                try:
                    probs = trained_estimator.predict_proba(X_test)
                    if is_multiclass:
                        all_metrics["roc_auc"] = float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
                    else:
                        # binary: extract probability of positive class (column 1) if shape allows, else use as-is
                        probs_for_auc = probs[:, 1] if (len(probs.shape) > 1 and probs.shape[1] == 2) else probs
                        all_metrics["roc_auc"] = float(roc_auc_score(y_test, probs_for_auc))
                except Exception:
                    # Fallback if roc_auc calculation fails due to class counts
                    all_metrics["roc_auc"] = 0.0

            # Confusion Matrix and Classification Report
            conf_matrix = confusion_matrix(y_test, preds).tolist()
            class_report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        elif problem_type == ProblemType.REGRESSION:
            # Regression metrics
            all_metrics["mae"] = float(mean_absolute_error(y_test, preds))
            all_metrics["mse"] = float(mean_squared_error(y_test, preds))
            all_metrics["rmse"] = float(np.sqrt(all_metrics["mse"]))
            all_metrics["r2"] = float(r2_score(y_test, preds))

        else:
            raise EvaluationEngineError(f"Unsupported problem type: {problem_type}")

        # 8. Feature Importance / Coefficients extraction
        feature_importance: dict[str, float] | None = None
        if hasattr(trained_estimator, "feature_importances_"):
            importances = getattr(trained_estimator, "feature_importances_")
            if isinstance(X_test, pd.DataFrame):
                feature_importance = dict(zip(X_test.columns, importances.tolist()))
            else:
                feature_importance = {f"feature_{i}": val for i, val in enumerate(importances.tolist())}
        elif hasattr(trained_estimator, "coef_"):
            coefs = getattr(trained_estimator, "coef_")
            if isinstance(coefs, np.ndarray):
                # Handle multi-class 2D coefs
                if len(coefs.shape) > 1 and coefs.shape[0] > 1:
                    importance_values = np.mean(np.abs(coefs), axis=0).tolist()
                else:
                    importance_values = coefs.ravel().tolist()

                if isinstance(X_test, pd.DataFrame):
                    feature_importance = dict(zip(X_test.columns, importance_values))
                else:
                    feature_importance = {f"feature_{i}": val for i, val in enumerate(importance_values)}

        # 9. Cross Validation Scores Extraction
        cv_scores: list[float] | None = None
        if hasattr(trained_estimator, "cv_results_"):
            cv_results = getattr(trained_estimator, "cv_results_")
            if cv_results and "mean_test_score" in cv_results:
                scores = cv_results["mean_test_score"]
                cv_scores = scores.tolist() if isinstance(scores, np.ndarray) else list(scores)

        # 10. Extract primary metric separately
        primary_metric_name = problem_definition.primary_metric
        metric_lower = primary_metric_name.lower().strip()
        metric_map = {
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
            "auc": "roc_auc",
            "mae": "mae",
            "mean_absolute_error": "mae",
            "mse": "mse",
            "mean_squared_error": "mse",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "r2",
            "r_squared": "r2",
        }
        mapped_key = metric_map.get(metric_lower, metric_lower)
        primary_metric_value = all_metrics.get(mapped_key, 0.0)

        end_time = time.perf_counter()
        duration = end_time - start_time

        execution_summary = {
            "candidate_id": training_result.candidate_id,
            "problem_type": problem_type.value,
            "samples": n_samples,
            "duration_seconds": duration,
        }

        # Convert predictions to a list
        predictions_list = preds.tolist() if isinstance(preds, np.ndarray) else list(preds)

        train_score = None
        if hasattr(trained_estimator, "best_score_"):
            train_score = float(getattr(trained_estimator, "best_score_"))

        model_params = trained_estimator.get_params()

        warnings_list = []
        if problem_definition.warnings:
            warnings_list = [w.message for w in problem_definition.warnings]

        return EvaluationResult(
            candidate_id=training_result.candidate_id,
            model_family=training_result.model_family,
            predictions=predictions_list,
            primary_metric=primary_metric_name,
            primary_metric_value=primary_metric_value,
            all_metrics=all_metrics,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            evaluation_duration_seconds=duration,
            train_score=train_score,
            test_score=primary_metric_value,
            prediction_count=len(predictions_list),
            training_duration=training_result.training_duration_seconds,
            evaluation_duration=duration,
            model_parameters=model_params,
            warnings=warnings_list,
            execution_summary=execution_summary,
        )
