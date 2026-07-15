"""Model Factory for DeployAI.

Stage 9F instantiates scikit-learn estimators from plan model candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
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

from backend.app.ml_plan.schemas import MLPlan, ModelFamily, ProblemType


class ModelFactoryError(Exception):
    """Raised when model instantiation fails due to invalid parameters or configuration."""

    pass


@dataclass
class ModelFactoryResult:
    """The structured result of building estimators from plan candidates."""

    models: dict[str, BaseEstimator]
    candidate_ids: list[str]
    model_families: list[ModelFamily]
    execution_summary: dict[str, Any] = field(default_factory=dict)


class ModelFactory:
    """Instantiates classical machine learning estimators defined in the MLPlan."""

    def build(self, plan: MLPlan) -> ModelFactoryResult:
        """Instantiate scikit-learn estimators for each model candidate in the plan.

        Args:
            plan: The planned execution MLPlan.

        Returns:
            ModelFactoryResult containing mapping of candidate IDs to instantiated estimators.

        Raises:
            ModelFactoryError: For validation or instantiation failures.
        """
        # 1. Reject None input
        if plan is None:
            raise ModelFactoryError("plan cannot be None")

        # 2. Reject wrong input types
        if not isinstance(plan, MLPlan):
            raise ModelFactoryError("plan must be an MLPlan instance")

        # 3. Reject empty candidate list
        candidates = plan.model_candidates
        if not candidates:
            raise ModelFactoryError("plan.model_candidates list cannot be empty")

        # 4. Maps for supported model families by problem type
        regression_map = {
            ModelFamily.LINEAR_REGRESSION: LinearRegression,
            ModelFamily.RIDGE: Ridge,
            ModelFamily.LASSO: Lasso,
            ModelFamily.DECISION_TREE: DecisionTreeRegressor,
            ModelFamily.RANDOM_FOREST: RandomForestRegressor,
            ModelFamily.GRADIENT_BOOSTING: GradientBoostingRegressor,
            ModelFamily.KNN: KNeighborsRegressor,
            ModelFamily.SVM: SVR,
        }

        classification_map = {
            ModelFamily.LOGISTIC_REGRESSION: LogisticRegression,
            ModelFamily.DECISION_TREE: DecisionTreeClassifier,
            ModelFamily.RANDOM_FOREST: RandomForestClassifier,
            ModelFamily.GRADIENT_BOOSTING: GradientBoostingClassifier,
            ModelFamily.KNN: KNeighborsClassifier,
            ModelFamily.SVM: SVC,
        }

        problem_type = plan.problem_type
        if problem_type == ProblemType.CLASSIFICATION:
            supported_map = classification_map
        elif problem_type == ProblemType.REGRESSION:
            supported_map = regression_map
        else:
            raise ModelFactoryError(f"Unsupported problem type: {problem_type}")

        # 5. Validation collections
        candidate_ids: list[str] = []
        model_families: list[ModelFamily] = []
        models: dict[str, BaseEstimator] = {}

        for candidate in candidates:
            cid = candidate.candidate_id
            family = candidate.model_family
            params = candidate.parameters

            # Reject empty candidate ID
            if not cid or not isinstance(cid, str) or not cid.strip():
                raise ModelFactoryError("candidate_id cannot be empty or whitespace-only")

            # Reject duplicate candidate ID
            if cid in candidate_ids:
                raise ModelFactoryError(f"Duplicate candidate_id '{cid}' found in model candidates")

            # Reject duplicate model family
            if family in model_families:
                raise ModelFactoryError(f"Duplicate model_family '{family.value}' found in model candidates")

            # Reject non-dictionary parameters
            if params is not None and not isinstance(params, dict):
                raise ModelFactoryError(f"Parameters for candidate '{cid}' must be a dictionary")

            # Reject unsupported model family for current problem type
            if family not in supported_map:
                raise ModelFactoryError(
                    f"Model family '{family.value}' is not supported for problem type '{problem_type.value}'"
                )

            # 6. Instantiate the scikit-learn estimator
            estimator_class = supported_map[family]
            ctor_params = params or {}

            try:
                # Forward parameters directly to estimator constructor
                estimator = estimator_class(**ctor_params)
            except Exception as e:
                raise ModelFactoryError(
                    f"Failed to construct model '{cid}' of family '{family.value}' with parameters {ctor_params}: {e}"
                ) from e

            # Append validated metadata and mapping
            candidate_ids.append(cid)
            model_families.append(family)
            models[cid] = estimator

        execution_summary = {
            "problem_type": problem_type.value,
            "total_models_built": len(models),
            "candidate_ids": candidate_ids,
            "model_families": [family.value for family in model_families],
        }

        return ModelFactoryResult(
            models=models,
            candidate_ids=candidate_ids,
            model_families=model_families,
            execution_summary=execution_summary,
        )
