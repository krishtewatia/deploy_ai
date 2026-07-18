"""Model Training Executor for DeployAI.

Stage 9H trains an instantiated model estimator on the prepared training dataset
and returns a TrainingResult containing the trained model and execution metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from backend.app.ml_plan.schemas import ModelCandidate, ModelFamily


class TrainingExecutorError(Exception):
    """Raised when model training or dataset validation fails."""

    pass


@dataclass
class TrainingResult:
    """The structured result of the model training execution."""

    trained_estimator: BaseEstimator
    candidate_id: str
    model_family: ModelFamily
    training_duration_seconds: float
    number_of_training_rows: int
    number_of_training_features: int
    fitted_successfully: bool
    training_summary: dict[str, Any] = field(default_factory=dict)


class TrainingExecutor:
    """Trains a scikit-learn estimator on prepared training data."""

    def train(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        candidate: ModelCandidate,
    ) -> TrainingResult:
        """Fit the provided estimator on training features and labels.

        Args:
            estimator: The scikit-learn estimator instance to train.
            X_train: Training feature inputs.
            y_train: Training target labels.
            candidate: ModelCandidate configuration metadata.

        Returns:
            TrainingResult containing the fitted estimator and metadata.

        Raises:
            TrainingExecutorError: For validation failures or fit execution errors.
        """
        # 1. Validation: Reject None
        if estimator is None:
            raise TrainingExecutorError("estimator cannot be None")
        if X_train is None:
            raise TrainingExecutorError("X_train cannot be None")
        if y_train is None:
            raise TrainingExecutorError("y_train cannot be None")
        if candidate is None:
            raise TrainingExecutorError("candidate cannot be None")

        # 2. Validation: Wrong input types
        if not hasattr(estimator, "fit") or not isinstance(estimator, BaseEstimator):
            raise TrainingExecutorError("estimator must be an instance of scikit-learn BaseEstimator")
        if not isinstance(candidate, ModelCandidate):
            raise TrainingExecutorError("candidate must be a ModelCandidate instance")
        if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
            raise TrainingExecutorError("X_train must be a pandas DataFrame or numpy ndarray")
        if not isinstance(y_train, (pd.Series, np.ndarray)):
            raise TrainingExecutorError("y_train must be a pandas Series or numpy ndarray")

        # 3. Validation: Empty datasets and size mismatch
        n_samples = X_train.shape[0] if isinstance(X_train, np.ndarray) else len(X_train)
        n_labels = y_train.shape[0] if isinstance(y_train, np.ndarray) else len(y_train)

        if n_samples == 0:
            raise TrainingExecutorError("X_train dataset cannot be empty")
        if n_labels == 0:
            raise TrainingExecutorError("y_train labels cannot be empty")
        if n_samples != n_labels:
            raise TrainingExecutorError(
                f"Row count mismatch: X_train has {n_samples} rows, y_train has {n_labels} rows"
            )

        # 4. Validation: Zero feature columns
        n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
        if isinstance(X_train, pd.DataFrame) and len(X_train.columns) == 0:
            raise TrainingExecutorError("X_train features dataset has zero columns")
        if n_features == 0:
            raise TrainingExecutorError("X_train features dataset has zero columns")

        # 5. Validation: Candidate mismatch
        family_classes = {
            ModelFamily.LINEAR_REGRESSION: (LinearRegression,),
            ModelFamily.LOGISTIC_REGRESSION: (LogisticRegression,),
            ModelFamily.RIDGE: (Ridge,),
            ModelFamily.LASSO: (Lasso,),
            ModelFamily.DECISION_TREE: (DecisionTreeClassifier, DecisionTreeRegressor),
            ModelFamily.RANDOM_FOREST: (RandomForestClassifier, RandomForestRegressor),
            ModelFamily.GRADIENT_BOOSTING: (GradientBoostingClassifier, GradientBoostingRegressor),
            ModelFamily.KNN: (KNeighborsClassifier, KNeighborsRegressor),
            ModelFamily.SVM: (SVC, SVR),
        }

        family = candidate.model_family
        allowed_classes = family_classes.get(family)
        if not allowed_classes or not isinstance(estimator, allowed_classes):
            estimator_class = estimator.__class__.__name__
            raise TrainingExecutorError(
                f"Estimator '{estimator_class}' does not match candidate family '{family.value}'"
            )

        # 6. Fit estimator and measure duration
        start_time = time.perf_counter()
        try:
            estimator.fit(X_train, y_train)
        except Exception as e:
            raise TrainingExecutorError(
                f"Model training fit failed for candidate '{candidate.candidate_id}': {e}"
            ) from e
        end_time = time.perf_counter()
        duration = end_time - start_time

        training_summary = {
            "candidate_id": candidate.candidate_id,
            "model_family": family.value,
            "samples": n_samples,
            "features": n_features,
            "duration_seconds": duration,
        }

        return TrainingResult(
            trained_estimator=estimator,
            candidate_id=candidate.candidate_id,
            model_family=family,
            training_duration_seconds=duration,
            number_of_training_rows=n_samples,
            number_of_training_features=n_features,
            fitted_successfully=True,
            training_summary=training_summary,
        )
