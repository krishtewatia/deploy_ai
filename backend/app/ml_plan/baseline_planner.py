"""Deterministic baseline ML planner.

Generates a safe baseline MLPlan from DatasetContext, ProblemDefinition,
UserMLRequest, and ComputeCapabilities.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional

from backend.app.compute_capabilities.schemas import ComputeCapabilities, ComputeTier
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import (
    DatasetSplitPlan,
    EvaluationPlan,
    ExecutionConstraints,
    FeatureSelectionMethod,
    FeatureSelectionPlan,
    MLPlan,
    MLPlanStatus,
    MLPlanWarning,
    ModelCandidate,
    ModelFamily,
    PreprocessingOperation,
    PreprocessingStep,
    SearchStrategy,
    SplitStrategy,
)
from backend.app.ml_plan.validator import MLPlanValidator
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import ProblemDefinition, ProblemType, ResolutionStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------


class BaselineMLPlannerError(Exception):
    """Raised when baseline ML planning fails or inputs are invalid."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_scale_sensitive(family: ModelFamily) -> bool:
    """Determine if a model family is scale-sensitive."""
    return family in (
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.LINEAR_REGRESSION,
        ModelFamily.RIDGE,
        ModelFamily.LASSO,
        ModelFamily.KNN,
        ModelFamily.SVM,
    )


# ---------------------------------------------------------------------------
# BaselineMLPlanner Class
# ---------------------------------------------------------------------------


class BaselineMLPlanner:
    """Deterministic ML Planner creating robust baseline execution plans."""

    def __init__(self, validator: Optional[MLPlanValidator] = None) -> None:
        """Initialize the baseline planner with a validator.

        Args:
            validator: Optional custom MLPlanValidator instance.
        """
        if validator is not None:
            if not isinstance(validator, MLPlanValidator):
                raise BaselineMLPlannerError("validator must be an instance of MLPlanValidator")
            self._validator = validator
        else:
            self._validator = MLPlanValidator()

    def create_plan(
        self,
        *,
        dataset_context: DatasetContext,
        problem_definition: ProblemDefinition,
        user_request: UserMLRequest,
        compute_capabilities: ComputeCapabilities,
    ) -> MLPlan:
        """Create a deterministic baseline MLPlan from inputs.

        Args:
            dataset_context: Target dataset metadata profile.
            problem_definition: Authoritative resolved problem statement.
            user_request: User requests/preferences.
            compute_capabilities: System resources constraints.

        Returns:
            A validated MLPlan.

        Raises:
            BaselineMLPlannerError: If inputs are invalid or planning fails.
        """
        # 1. Input Type Validation
        if not isinstance(dataset_context, DatasetContext):
            raise BaselineMLPlannerError("dataset_context must be an instance of DatasetContext")
        if not isinstance(problem_definition, ProblemDefinition):
            raise BaselineMLPlannerError("problem_definition must be an instance of ProblemDefinition")
        if not isinstance(user_request, UserMLRequest):
            raise BaselineMLPlannerError("user_request must be an instance of UserMLRequest")
        if not isinstance(compute_capabilities, ComputeCapabilities):
            raise BaselineMLPlannerError("compute_capabilities must be an instance of ComputeCapabilities")

        # 2. Upstream State Requirements
        if problem_definition.status != ResolutionStatus.RESOLVED:
            raise BaselineMLPlannerError(
                f"Cannot plan for unresolved problem definition. Current status: '{problem_definition.status.value}'"
            )

        # 3. Input Consistency Checks
        if dataset_context.basic_info.dataset_id != problem_definition.dataset_id:
            raise BaselineMLPlannerError("dataset_context ID mismatch with problem_definition dataset_id")

        if user_request.request_id != problem_definition.request_id:
            raise BaselineMLPlannerError("user_request ID mismatch with problem_definition request_id")

        col_lookup = {col.name: col for col in dataset_context.columns}
        if problem_definition.target_column not in col_lookup:
            raise BaselineMLPlannerError(
                f"Target column '{problem_definition.target_column}' does not exist in DatasetContext"
            )

        for col in problem_definition.feature_columns:
            if col not in col_lookup:
                raise BaselineMLPlannerError(
                    f"Feature column '{col}' does not exist in DatasetContext"
                )

        # 4. Initialize Plan Identity
        plan_id = f"mlplan_{uuid.uuid4().hex}"

        # -------------------------------------------------------------------
        # 5. Preprocessing Steps Generation
        # -------------------------------------------------------------------
        preprocessing_steps: list[PreprocessingStep] = []
        step_counter = 1

        def next_step_id() -> str:
            nonlocal step_counter
            step_id = f"preprocess_{step_counter:03d}"
            step_counter += 1
            return step_id

        # Generate per-column baseline steps
        for col_name in problem_definition.feature_columns:
            col_ctx = col_lookup[col_name]

            if col_ctx.is_numeric:
                if col_ctx.missing_count > 0:
                    preprocessing_steps.append(
                        PreprocessingStep(
                            step_id=next_step_id(),
                            operation=PreprocessingOperation.IMPUTE_MEDIAN,
                            columns=[col_name],
                            parameters={},
                            reason=f"Impute missing numeric values in '{col_name}' using median.",
                        )
                    )
            elif col_ctx.is_categorical:
                if col_ctx.missing_count > 0:
                    preprocessing_steps.append(
                        PreprocessingStep(
                            step_id=next_step_id(),
                            operation=PreprocessingOperation.IMPUTE_MODE,
                            columns=[col_name],
                            parameters={},
                            reason=f"Impute missing categorical values in '{col_name}' using mode.",
                        )
                    )
                # categorical gets encoded
                preprocessing_steps.append(
                    PreprocessingStep(
                        step_id=next_step_id(),
                        operation=PreprocessingOperation.ONE_HOT_ENCODE,
                        columns=[col_name],
                        parameters={},
                        reason=f"Encode categorical column '{col_name}' to one-hot representation.",
                    )
                )
            elif col_ctx.is_datetime:
                preprocessing_steps.append(
                    PreprocessingStep(
                        step_id=next_step_id(),
                        operation=PreprocessingOperation.DATETIME_EXTRACT,
                        columns=[col_name],
                        parameters={},
                        reason=f"Extract date/time components from '{col_name}' for model training.",
                    )
                )
            else:
                # Ambiguous column type gets passthrough if needed
                preprocessing_steps.append(
                    PreprocessingStep(
                        step_id=next_step_id(),
                        operation=PreprocessingOperation.PASSTHROUGH,
                        columns=[col_name],
                        parameters={},
                        reason=f"Passthrough unclassified feature column '{col_name}'.",
                    )
                )

        # -------------------------------------------------------------------
        # 6. Model Candidates Portfolio Selection
        # -------------------------------------------------------------------
        model_candidates: list[ModelCandidate] = []
        is_classification = problem_definition.problem_type == ProblemType.CLASSIFICATION

        if is_classification:
            # Classification Pool
            class_candidates = [
                ModelCandidate(
                    candidate_id="model_001",
                    model_family=ModelFamily.LOGISTIC_REGRESSION,
                    parameters={"random_state": 42},
                    search_strategy=SearchStrategy.NONE,
                    search_space={},
                    reason="Provides a simple and efficient classification baseline.",
                ),
                ModelCandidate(
                    candidate_id="model_002",
                    model_family=ModelFamily.RANDOM_FOREST,
                    parameters={"random_state": 42},
                    search_strategy=SearchStrategy.NONE,
                    search_space={},
                    reason="Provides a robust non-linear tree-based baseline.",
                ),
                ModelCandidate(
                    candidate_id="model_003",
                    model_family=ModelFamily.GRADIENT_BOOSTING,
                    parameters={"random_state": 42},
                    search_strategy=SearchStrategy.NONE,
                    search_space={},
                    reason="Provides a strong ensemble baseline for complex relationships.",
                ),
            ]
            # Adapt to ComputeTier
            if compute_capabilities.compute_tier == ComputeTier.MINIMAL:
                model_candidates = class_candidates[:2]
            else:
                model_candidates = class_candidates[:3]
        else:
            # Regression Pool
            reg_candidates = [
                ModelCandidate(
                    candidate_id="model_001",
                    model_family=ModelFamily.LINEAR_REGRESSION,
                    parameters={},
                    search_strategy=SearchStrategy.NONE,
                    search_space={},
                    reason="Provides a simple and efficient regression baseline.",
                ),
                ModelCandidate(
                    candidate_id="model_002",
                    model_family=ModelFamily.RANDOM_FOREST,
                    parameters={"random_state": 42},
                    search_strategy=SearchStrategy.NONE,
                    search_space={},
                    reason="Provides a robust non-linear tree-based baseline.",
                ),
                ModelCandidate(
                    candidate_id="model_003",
                    model_family=ModelFamily.GRADIENT_BOOSTING,
                    parameters={"random_state": 42},
                    search_strategy=SearchStrategy.NONE,
                    search_space={},
                    reason="Provides a strong ensemble baseline for complex relationships.",
                ),
            ]
            # Adapt to ComputeTier
            if compute_capabilities.compute_tier == ComputeTier.MINIMAL:
                model_candidates = reg_candidates[:2]
            else:
                model_candidates = reg_candidates[:3]

        # -------------------------------------------------------------------
        # 7. Add Numeric Scaling if scale-sensitive models exist
        # -------------------------------------------------------------------
        has_scale_sensitive = any(_is_scale_sensitive(c.model_family) for c in model_candidates)
        if has_scale_sensitive:
            numeric_cols = [
                col_name for col_name in problem_definition.feature_columns
                if (col_ctx := col_lookup.get(col_name)) and col_ctx.is_numeric
            ]
            if numeric_cols:
                preprocessing_steps.append(
                    PreprocessingStep(
                        step_id=next_step_id(),
                        operation=PreprocessingOperation.STANDARD_SCALE,
                        columns=numeric_cols,
                        parameters={},
                        reason="Scale numeric features to standard normal distribution.",
                    )
                )

        # -------------------------------------------------------------------
        # 8. Feature Selection Plan
        # -------------------------------------------------------------------
        feature_selection = FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=problem_definition.feature_columns,
            reason="Feature selection is disabled for baseline planning.",
        )

        # -------------------------------------------------------------------
        # 9. Dataset Split Plan
        # -------------------------------------------------------------------
        if is_classification:
            split_plan = DatasetSplitPlan(
                strategy=SplitStrategy.STRATIFIED,
                test_size=0.2,
                validation_size=0.0,
                random_state=42,
                shuffle=True,
                stratify_column=problem_definition.target_column,
                time_column=None,
            )
        else:
            split_plan = DatasetSplitPlan(
                strategy=SplitStrategy.RANDOM,
                test_size=0.2,
                validation_size=0.0,
                random_state=42,
                shuffle=True,
                stratify_column=None,
                time_column=None,
            )

        # -------------------------------------------------------------------
        # 10. Evaluation Plan
        # -------------------------------------------------------------------
        primary_metric = problem_definition.primary_metric
        primary_norm = primary_metric.strip().lower()

        if is_classification:
            sec_pool = ["accuracy", "precision", "recall", "f1"]
            secondary_metrics = [m for m in sec_pool if m != primary_norm]
        else:
            sec_pool = ["mae", "rmse", "r2"]
            secondary_metrics = [m for m in sec_pool if m != primary_norm]

        cv_folds = 3 if compute_capabilities.compute_tier == ComputeTier.MINIMAL else 5
        evaluation_plan = EvaluationPlan(
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            cross_validation_folds=cv_folds,
        )

        # -------------------------------------------------------------------
        # 11. Execution Constraints
        # -------------------------------------------------------------------
        execution_constraints = ExecutionConstraints(
            parallel_workers=compute_capabilities.safe_parallel_workers,
            use_gpu_acceleration=compute_capabilities.gpu_acceleration_available,
            accelerator_type=compute_capabilities.accelerator_type,
            compute_tier=compute_capabilities.compute_tier,
        )

        # -------------------------------------------------------------------
        # 12. Warnings Propagation
        # -------------------------------------------------------------------
        warnings = [
            MLPlanWarning(code=w.code, message=w.message)
            for w in compute_capabilities.warnings
        ]

        # -------------------------------------------------------------------
        # 13. MLPlan Instantiation
        # -------------------------------------------------------------------
        plan = MLPlan(
            plan_id=plan_id,
            dataset_id=dataset_context.basic_info.dataset_id,
            request_id=user_request.request_id,
            problem_definition_id=problem_definition.definition_id,
            compute_capability_id=compute_capabilities.capability_id,
            problem_type=problem_definition.problem_type,
            target_column=problem_definition.target_column,
            feature_columns=problem_definition.feature_columns,
            preprocessing_steps=preprocessing_steps,
            feature_engineering_steps=[],
            feature_selection=feature_selection,
            split_plan=split_plan,
            model_candidates=model_candidates,
            evaluation_plan=evaluation_plan,
            execution_constraints=execution_constraints,
            status=MLPlanStatus.READY,
            warnings=warnings,
            confirmation_items=[],
        )

        # 14. Internal Validation Gate
        validation_result = self._validator.validate(
            plan=plan,
            dataset_context=dataset_context,
            problem_definition=problem_definition,
            compute_capabilities=compute_capabilities,
        )

        if not validation_result.is_valid:
            error_codes = [err.code for err in validation_result.errors]
            raise BaselineMLPlannerError(
                f"Generated baseline plan failed internal validation check. Error codes: {error_codes}"
            )

        return plan
