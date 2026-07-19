"""Deterministic ProblemResolver implementation.

Resolves DatasetContext and UserMLRequest into a structured ProblemDefinition.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_request.schemas import ProblemTypePreference, UserMLRequest
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ProblemType,
    ResolutionStatus,
    TargetSource,
)

logger = logging.getLogger(__name__)


class ProblemResolverError(Exception):
    """Raised when problem resolution fails due to invalid settings or mismatching context."""


class ProblemResolver:
    """Deterministic resolver to adapt user request and dataset context into a ProblemDefinition."""

    def resolve(
        self,
        *,
        dataset_context: DatasetContext,
        user_request: UserMLRequest,
    ) -> ProblemDefinition:
        """Resolve a ProblemDefinition from a DatasetContext and a UserMLRequest.

        Parameters
        ----------
        dataset_context:
            The structured context of the analyzed dataset.
        user_request:
            The user ML request containing goals and intent.

        Returns
        -------
        ProblemDefinition
            The resolved machine learning problem definition.

        Raises
        ------
        ProblemResolverError
            If target_column is missing, target or excluded columns do not exist in the dataset,
            no features remain, or problem type cannot be inferred.
        """
        logger.info(
            "Resolving problem definition for request_id=%r, dataset_id=%r.",
            user_request.request_id,
            dataset_context.basic_info.dataset_id,
        )

        # 1. Validate target column (infer last column if not explicitly provided)
        target_name = user_request.target_column
        if target_name is None:
            if not dataset_context.columns:
                raise ProblemResolverError("Dataset context contains no columns.")
            target_name = dataset_context.columns[-1].name
            logger.info("Auto-selected target column: %r", target_name)

        # Map dataset columns for O(1) existence and metadata lookups
        col_map = {col.name: col for col in dataset_context.columns}

        # 2. Validate target column exists in dataset
        if target_name not in col_map:
            raise ProblemResolverError(
                f"Target column '{target_name}' does not exist in the dataset."
            )

        target_col = col_map[target_name]

        # 3. Validate excluded columns exist in dataset
        invalid_excluded = [
            col for col in user_request.excluded_columns if col not in col_map
        ]
        if invalid_excluded:
            raise ProblemResolverError(
                f"Excluded column(s) do not exist in the dataset: {sorted(invalid_excluded)}"
            )

        # 4. Determine Problem Type
        resolved_type: ProblemType
        if user_request.problem_type == ProblemTypePreference.CLASSIFICATION:
            resolved_type = ProblemType.CLASSIFICATION
        elif user_request.problem_type == ProblemTypePreference.REGRESSION:
            resolved_type = ProblemType.REGRESSION
        elif user_request.problem_type == ProblemTypePreference.AUTO:
            # Deterministic inference logic
            if target_col.is_categorical or target_col.unique_count == 2:
                resolved_type = ProblemType.CLASSIFICATION
            elif target_col.is_numeric:
                # low cardinality check: unique_count <= 20 and unique_percentage <= 5.0
                if (
                    target_col.unique_count <= 20
                    and target_col.unique_percentage <= 5.0
                ):
                    resolved_type = ProblemType.CLASSIFICATION
                else:
                    resolved_type = ProblemType.REGRESSION
            else:
                raise ProblemResolverError(
                    f"Problem type cannot be inferred for target column '{target_name}' "
                    f"with dtype '{target_col.dtype}'."
                )
        else:
            raise ProblemResolverError(
                f"Unsupported problem type preference: {user_request.problem_type}"
            )

        # 5. Resolve Excluded Columns (preserving request order)
        # Deep copy or slice copy to prevent mutation reference issues
        resolved_exclusions = list(user_request.excluded_columns)

        # 6. Resolve Feature Columns (preserving DatasetContext columns order)
        excluded_set = set(resolved_exclusions)
        resolved_features = [
            col.name
            for col in dataset_context.columns
            if col.name != target_name and col.name not in excluded_set
        ]

        if not resolved_features:
            raise ProblemResolverError(
                "No feature columns remain after removing the target and excluded columns."
            )

        # 7. Resolve Primary Metric
        resolved_metric: str
        if user_request.primary_metric is not None:
            resolved_metric = user_request.primary_metric
        else:
            if resolved_type == ProblemType.CLASSIFICATION:
                resolved_metric = "f1"
            else:
                resolved_metric = "rmse"

        # 8. Goal
        resolved_goal = user_request.goal

        # 9. Generate definition_id
        def_id = f"definition_{uuid.uuid4()}"

        return ProblemDefinition(
            definition_id=def_id,
            request_id=user_request.request_id,
            dataset_id=dataset_context.basic_info.dataset_id,
            goal=resolved_goal,
            problem_type=resolved_type,
            target_column=target_name,
            target_source=TargetSource.USER,
            feature_columns=resolved_features,
            excluded_columns=resolved_exclusions,
            primary_metric=resolved_metric,
            status=ResolutionStatus.RESOLVED,
            warnings=[],
            confirmation_items=[],
        )
