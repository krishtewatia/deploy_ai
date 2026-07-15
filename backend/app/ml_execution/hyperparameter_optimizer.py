"""Hyperparameter Optimization Engine for DeployAI.

Stage 9G performs hyperparameter tuning using Grid Search and Random Search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
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

from backend.app.ml_plan.schemas import MLPlan, ModelCandidate, ModelFamily, ProblemType, SearchStrategy
from backend.app.problem_definition.schemas import ProblemDefinition


class HyperparameterOptimizerError(Exception):
    """Raised when hyperparameter optimization fails due to validation errors or fit failures."""

    pass


@dataclass
class HyperparameterOptimizationResult:
    """The structured result of hyperparameter optimization."""

    optimized_estimator: BaseEstimator
    best_parameters: dict[str, Any]
    best_score: float | None
    search_strategy: SearchStrategy
    cv_results: dict[str, Any] | None
    execution_summary: dict[str, Any] = field(default_factory=dict)


class HyperparameterOptimizer:
    """Performs Grid and Random hyperparameter optimization on classical ML models."""

    def optimize(
        self,
        estimator: BaseEstimator,
        candidate: ModelCandidate,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        problem_definition: ProblemDefinition,
        plan: MLPlan | None = None,
    ) -> HyperparameterOptimizationResult:
        """Optimize model hyperparameters using GRID or RANDOM search strategies.

        Args:
            estimator: The instantiated scikit-learn estimator.
            candidate: The target ModelCandidate specifying search space and strategy.
            X_train: Training feature dataset.
            y_train: Training target labels.
            problem_definition: Resolved ML problem definition contract.
            plan: The optional full MLPlan containing evaluation cross-validation folds.

        Returns:
            HyperparameterOptimizationResult containing the best estimator, parameters, and scores.

        Raises:
            HyperparameterOptimizerError: If validation fails or search execution fails.
        """
        # 1. Validation: Reject None
        if estimator is None:
            raise HyperparameterOptimizerError("estimator cannot be None")
        if candidate is None:
            raise HyperparameterOptimizerError("candidate cannot be None")
        if X_train is None:
            raise HyperparameterOptimizerError("X_train cannot be None")
        if y_train is None:
            raise HyperparameterOptimizerError("y_train cannot be None")
        if problem_definition is None:
            raise HyperparameterOptimizerError("problem_definition cannot be None")

        # 2. Validation: Wrong types
        if not isinstance(estimator, BaseEstimator):
            raise HyperparameterOptimizerError("estimator must be an instance of scikit-learn BaseEstimator")
        if not isinstance(candidate, ModelCandidate):
            raise HyperparameterOptimizerError("candidate must be a ModelCandidate instance")
        if not isinstance(problem_definition, ProblemDefinition):
            raise HyperparameterOptimizerError("problem_definition must be a ProblemDefinition instance")
        if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
            raise HyperparameterOptimizerError("X_train must be a pandas DataFrame or numpy ndarray")
        if not isinstance(y_train, (pd.Series, np.ndarray)):
            raise HyperparameterOptimizerError("y_train must be a pandas Series or numpy ndarray")
        if plan is not None and not isinstance(plan, MLPlan):
            raise HyperparameterOptimizerError("plan must be an MLPlan instance")

        # 3. Validation: Empty datasets and size mismatch
        n_samples = X_train.shape[0] if isinstance(X_train, np.ndarray) else len(X_train)
        n_labels = y_train.shape[0] if isinstance(y_train, np.ndarray) else len(y_train)

        if n_samples == 0:
            raise HyperparameterOptimizerError("X_train dataset cannot be empty")
        if n_labels == 0:
            raise HyperparameterOptimizerError("y_train labels cannot be empty")
        if n_samples != n_labels:
            raise HyperparameterOptimizerError(
                f"Dataset size mismatch: X_train has {n_samples} rows, y_train has {n_labels} rows"
            )

        # 4. Validation: Candidate mismatch
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
            raise HyperparameterOptimizerError(
                f"Estimator '{estimator_class}' does not match candidate family '{family.value}'"
            )

        # 5. Handle NONE search strategy
        strategy = candidate.search_strategy
        strategy_str = strategy.value if hasattr(strategy, "value") else str(strategy)

        if strategy == SearchStrategy.NONE:
            # Return estimator unchanged. No fitting, no optimization.
            return HyperparameterOptimizationResult(
                optimized_estimator=estimator,
                best_parameters=estimator.get_params(),
                best_score=None,
                search_strategy=strategy,
                cv_results=None,
                execution_summary={
                    "strategy": strategy_str,
                    "status": "Skipped (Search strategy is NONE)",
                },
            )

        # 6. Validate strategy is supported
        if strategy not in (SearchStrategy.GRID, SearchStrategy.RANDOM):
            raise HyperparameterOptimizerError(f"Unsupported search strategy: {strategy_str}")

        # 7. Validation: Search Space
        space = candidate.search_space
        if not space:
            raise HyperparameterOptimizerError(
                f"search_space cannot be empty or None when search_strategy is '{strategy_str}'"
            )

        # Validation: check parameter names in search_space match estimator parameters
        valid_params = set(estimator.get_params().keys())
        for param_name in space.keys():
            if param_name not in valid_params:
                raise HyperparameterOptimizerError(
                    f"Invalid parameter '{param_name}' in search space for estimator class '{estimator.__class__.__name__}'"
                )

        # 8. Setup Cross Validation and Scoring
        folds = 5
        if plan is not None and plan.evaluation_plan is not None:
            folds = plan.evaluation_plan.cross_validation_folds

        problem_type = problem_definition.problem_type
        if problem_type == ProblemType.CLASSIFICATION:
            # Deterministic Stratified CV splitter
            cv_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        elif problem_type == ProblemType.REGRESSION:
            # Deterministic CV splitter
            cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            raise HyperparameterOptimizerError(f"Unsupported problem type: {problem_type}")

        # Map metric name to scikit-learn scoring string
        metric = problem_definition.primary_metric
        metric_lower = metric.lower().strip()
        if problem_type == ProblemType.CLASSIFICATION:
            metric_mapping = {
                "accuracy": "accuracy",
                "f1": "f1",
                "precision": "precision",
                "recall": "recall",
                "roc_auc": "roc_auc",
                "auc": "roc_auc",
            }
            scoring = metric_mapping.get(metric_lower, metric_lower)
        else:
            metric_mapping = {
                "r2": "r2",
                "r_squared": "r2",
                "mean_squared_error": "neg_mean_squared_error",
                "mse": "neg_mean_squared_error",
                "mean_absolute_error": "neg_mean_absolute_error",
                "mae": "neg_mean_absolute_error",
                "root_mean_squared_error": "neg_root_mean_squared_error",
                "rmse": "neg_root_mean_squared_error",
            }
            scoring = metric_mapping.get(metric_lower, metric_lower)

        # 9. Perform Grid / Random Search
        try:
            if strategy == SearchStrategy.GRID:
                search_object = GridSearchCV(
                    estimator=estimator,
                    param_grid=space,
                    scoring=scoring,
                    cv=cv_splitter,
                    n_jobs=1,  # Single-process for deterministic output and test isolation
                )
            else:
                search_object = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=space,
                    scoring=scoring,
                    cv=cv_splitter,
                    random_state=42,
                    n_jobs=1,
                )

            # Fit the hyperparameter search model
            search_object.fit(X_train, y_train)

        except Exception as e:
            raise HyperparameterOptimizerError(
                f"Failed during {strategy_str} search fit execution: {e}"
            ) from e

        execution_summary = {
            "strategy": strategy_str,
            "scoring": scoring,
            "folds": folds,
            "best_score": float(search_object.best_score_),
        }

        # Convert numpy types in cv_results_ if needed for json serialization
        raw_cv_results = search_object.cv_results_
        serializable_cv_results = {}
        for k, v in raw_cv_results.items():
            if isinstance(v, np.ndarray):
                serializable_cv_results[k] = v.tolist()
            else:
                serializable_cv_results[k] = v

        return HyperparameterOptimizationResult(
            optimized_estimator=search_object.best_estimator_,
            best_parameters=search_object.best_params_,
            best_score=float(search_object.best_score_),
            search_strategy=strategy,
            cv_results=serializable_cv_results,
            execution_summary=execution_summary,
        )
