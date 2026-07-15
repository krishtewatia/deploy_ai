"""Feature Selection Executor for DeployAI.

Stage 9E executes feature selection strategy defined in MLPlan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)

from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import (
    FeatureSelectionMethod,
    MLPlan,
    ProblemType,
)


class FeatureSelectionExecutorError(Exception):
    """Raised when feature selection execution fails due to invalid parameters or data issues."""

    pass


@dataclass
class FeatureSelectionResult:
    """The structured result of running feature selection."""

    selected_dataframe: pd.DataFrame
    selected_columns: list[str]
    removed_columns: list[str]
    selector_object: Any
    execution_summary: dict[str, Any] = field(default_factory=dict)


class FeatureSelectionExecutor:
    """Executes planned feature selection method deterministically on a dataset."""

    def execute(
        self,
        *,
        dataframe: pd.DataFrame,
        dataset_context: DatasetContext,
        plan: MLPlan,
    ) -> FeatureSelectionResult:
        """Partition or filter the input dataframe's columns according to the plan's feature selection.

        Args:
            dataframe: Input pandas DataFrame.
            dataset_context: Target dataset metadata profile.
            plan: The planned execution MLPlan.

        Returns:
            FeatureSelectionResult containing the filtered dataframe and column choices.

        Raises:
            FeatureSelectionExecutorError: For validation or execution failures.
        """
        # 1. Reject None inputs
        if dataframe is None:
            raise FeatureSelectionExecutorError("dataframe cannot be None")
        if dataset_context is None:
            raise FeatureSelectionExecutorError("dataset_context cannot be None")
        if plan is None:
            raise FeatureSelectionExecutorError("plan cannot be None")

        # 2. Reject wrong types
        if not isinstance(dataframe, pd.DataFrame):
            raise FeatureSelectionExecutorError("dataframe must be a pandas.DataFrame instance")
        if not isinstance(dataset_context, DatasetContext):
            raise FeatureSelectionExecutorError("dataset_context must be a DatasetContext instance")
        if not isinstance(plan, MLPlan):
            raise FeatureSelectionExecutorError("plan must be an MLPlan instance")

        # 3. Reject empty dataframe
        if dataframe.empty:
            raise FeatureSelectionExecutorError("dataframe cannot be empty")

        fs_plan = plan.feature_selection
        if fs_plan is None:
            raise FeatureSelectionExecutorError("plan.feature_selection cannot be None")

        method = fs_plan.method
        candidate_columns = fs_plan.candidate_columns or []
        max_features = fs_plan.max_features
        target = plan.target_column

        # 4. Target column check
        if not target:
            raise FeatureSelectionExecutorError("plan.target_column cannot be empty")
        if target not in dataframe.columns:
            raise FeatureSelectionExecutorError(f"Target column '{target}' does not exist in dataframe")

        # 5. Duplicate columns in dataframe check
        if len(dataframe.columns) != len(set(dataframe.columns)):
            raise FeatureSelectionExecutorError("Duplicate column names detected in input dataframe")

        # 6. Validate plan feature_columns existence
        for col in plan.feature_columns or []:
            if col not in dataframe.columns:
                raise FeatureSelectionExecutorError(f"Feature column '{col}' does not exist in dataframe")

        # 7. Validate candidate_columns existence and target constraints
        if target in candidate_columns:
            raise FeatureSelectionExecutorError(
                f"Target column '{target}' cannot be included in feature selection candidate_columns"
            )

        for col in candidate_columns:
            if col not in dataframe.columns:
                raise FeatureSelectionExecutorError(
                    f"Candidate column '{col}' does not exist in dataframe"
                )

        # 8. Check duplicate candidate columns
        if len(candidate_columns) != len(set(candidate_columns)):
            raise FeatureSelectionExecutorError("Duplicate candidate columns detected in plan")

        # 9. Supported methods check
        supported_methods = {
            FeatureSelectionMethod.NONE,
            FeatureSelectionMethod.VARIANCE_THRESHOLD,
            FeatureSelectionMethod.CORRELATION_FILTER,
            FeatureSelectionMethod.MUTUAL_INFORMATION,
            FeatureSelectionMethod.MODEL_BASED,
        }
        if method not in supported_methods:
            raise FeatureSelectionExecutorError(f"Unknown or unsupported selection method: {method}")

        # 10. Validate max_features if ranking method is used
        if method in (FeatureSelectionMethod.MUTUAL_INFORMATION, FeatureSelectionMethod.MODEL_BASED):
            if max_features is None:
                raise FeatureSelectionExecutorError(
                    f"max_features must be specified for ranking selection method: {method}"
                )

        if max_features is not None:
            if max_features <= 0:
                raise FeatureSelectionExecutorError(
                    f"max_features must be strictly greater than 0, got {max_features}"
                )
            if max_features > len(candidate_columns):
                raise FeatureSelectionExecutorError(
                    f"max_features ({max_features}) cannot be larger than the number of candidate columns ({len(candidate_columns)})"
                )

        # 11. Execute selection method
        try:
            selected_columns: list[str] = []
            removed_columns: list[str] = []
            selector_object: Any = None

            if method == FeatureSelectionMethod.NONE:
                # Return the dataframe copy unchanged (keeping target column as well)
                selected_columns = list(candidate_columns)
                removed_columns = []
                selector_object = None

            elif method == FeatureSelectionMethod.VARIANCE_THRESHOLD:
                threshold = fs_plan.parameters.get("threshold", 0.0)
                selector = VarianceThreshold(threshold=threshold)
                # Fit only on the candidate columns
                selector.fit(dataframe[candidate_columns])
                support = selector.get_support()

                selected_columns = [col for col, keep in zip(candidate_columns, support) if keep]
                removed_columns = [col for col, keep in zip(candidate_columns, support) if not keep]
                selector_object = selector

            elif method == FeatureSelectionMethod.CORRELATION_FILTER:
                threshold = fs_plan.parameters.get("threshold", 0.95)
                # Pearson correlation matrix (absolute value)
                corr_matrix = dataframe[candidate_columns].corr(method="pearson").abs()

                removed_set: set[str] = set()
                # Run deterministic correlation filtering in order of candidate columns
                for i in range(len(candidate_columns)):
                    col_i = candidate_columns[i]
                    if col_i in removed_set:
                        continue
                    for j in range(i + 1, len(candidate_columns)):
                        col_j = candidate_columns[j]
                        if col_j in removed_set:
                            continue
                        
                        val = corr_matrix.loc[col_i, col_j]
                        # Exceed the threshold means strictly greater
                        if val > threshold:
                            removed_set.add(col_j)

                selected_columns = [c for c in candidate_columns if c not in removed_set]
                removed_columns = [c for c in candidate_columns if c in removed_set]
                selector_object = None

            elif method == FeatureSelectionMethod.MUTUAL_INFORMATION:
                X = dataframe[candidate_columns]
                y = dataframe[target]

                if plan.problem_type == ProblemType.CLASSIFICATION:
                    scores = mutual_info_classif(X, y, random_state=42)
                elif plan.problem_type == ProblemType.REGRESSION:
                    scores = mutual_info_regression(X, y, random_state=42)
                else:
                    raise FeatureSelectionExecutorError(
                        f"Unsupported problem type for mutual information: {plan.problem_type}"
                    )

                feature_scores = list(zip(candidate_columns, scores))
                # Sort: primary descending by score, secondary ascending alphabetically by column name
                sorted_features = sorted(feature_scores, key=lambda x: (-x[1], x[0]))

                # Keep max_features highest ranked
                selected_columns = [x[0] for x in sorted_features[:max_features]]
                removed_columns = [c for c in candidate_columns if c not in selected_columns]
                selector_object = None

            elif method == FeatureSelectionMethod.MODEL_BASED:
                X = dataframe[candidate_columns]
                y = dataframe[target]

                if plan.problem_type == ProblemType.CLASSIFICATION:
                    rf = RandomForestClassifier(random_state=42)
                elif plan.problem_type == ProblemType.REGRESSION:
                    rf = RandomForestRegressor(random_state=42)
                else:
                    raise FeatureSelectionExecutorError(
                        f"Unsupported problem type for model based selection: {plan.problem_type}"
                    )

                rf.fit(X, y)
                importances = rf.feature_importances_

                feature_importances = list(zip(candidate_columns, importances))
                # Sort: primary descending by importance, secondary ascending alphabetically by column name
                sorted_features = sorted(feature_importances, key=lambda x: (-x[1], x[0]))

                selected_columns = [x[0] for x in sorted_features[:max_features]]
                removed_columns = [c for c in candidate_columns if c not in selected_columns]
                selector_object = rf

            # Build final selected dataframe copying the columns + target column
            keep_cols = selected_columns + [target]
            selected_dataframe = dataframe[keep_cols].copy()


            execution_summary = {
                "method": method.value,
                "initial_candidate_count": len(candidate_columns),
                "selected_count": len(selected_columns),
                "removed_count": len(removed_columns),
                "selected_columns": selected_columns,
                "removed_columns": removed_columns,
            }

            return FeatureSelectionResult(
                selected_dataframe=selected_dataframe,
                selected_columns=selected_columns,
                removed_columns=removed_columns,
                selector_object=selector_object,
                execution_summary=execution_summary,
            )

        except FeatureSelectionExecutorError:
            # Re-raise explicit executor errors
            raise
        except Exception as e:
            raise FeatureSelectionExecutorError(
                f"Execution failed during feature selection method '{method.value}': {e}"
            ) from e
