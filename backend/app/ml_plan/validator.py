"""Deterministic cross-artifact MLPlanValidator implementation.

Verifies plan consistency with DatasetContext, ProblemDefinition, and ComputeCapabilities.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.compute_capabilities.schemas import AcceleratorType, ComputeCapabilities
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import (
    MLPlan,
    MLPlanStatus,
    ModelFamily,
    PreprocessingOperation,
)
from backend.app.problem_definition.schemas import ProblemDefinition, ProblemType, ResolutionStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Validation Models
# ---------------------------------------------------------------------------


class ValidationSeverity(str, Enum):
    """Severity of a plan validation issue."""

    ERROR = "error"
    WARNING = "warning"


class MLPlanValidationIssue(BaseModel):
    """Structured description of an inconsistency or warning found in an MLPlan."""

    code: str = Field(
        ...,
        description="Stable machine-readable code identifying the issue type.",
    )
    message: str = Field(
        ...,
        description="Human-readable warning or error explanation.",
    )
    severity: ValidationSeverity = Field(
        ...,
        description="Severity indicating whether the plan is blocked or runs with warnings.",
    )
    location: Optional[str] = Field(
        default=None,
        description="Logical location of the field or step containing the issue.",
    )

    @field_validator("code", "message", mode="before")
    @classmethod
    def _strip_and_validate_required_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped

    @field_validator("location", mode="before")
    @classmethod
    def _strip_and_validate_location(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("location must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("location cannot be empty or whitespace-only")
        return stripped


class MLPlanValidationResult(BaseModel):
    """The result of deterministic cross-artifact plan validation checks."""

    plan_id: str = Field(
        ...,
        description="ID of the validated plan.",
    )
    is_valid: bool = Field(
        ...,
        description="True if the plan contains zero validation errors.",
    )
    errors: list[MLPlanValidationIssue] = Field(
        default_factory=list,
        description="Deterministic list of validation errors (blocking execution).",
    )
    warnings: list[MLPlanValidationIssue] = Field(
        default_factory=list,
        description="Deterministic list of warnings (non-blocking concerns).",
    )

    @field_validator("plan_id", mode="before")
    @classmethod
    def _strip_and_validate_plan_id(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("plan_id must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("plan_id cannot be empty or whitespace-only")
        return stripped

    @model_validator(mode="after")
    def _validate_result_consistency(self) -> MLPlanValidationResult:
        # Check severity of errors
        for idx, err in enumerate(self.errors):
            if err.severity != ValidationSeverity.ERROR:
                raise ValueError(f"Issue at errors[{idx}] must have severity 'error'")

        # Check severity of warnings
        for idx, warn in enumerate(self.warnings):
            if warn.severity != ValidationSeverity.WARNING:
                raise ValueError(f"Issue at warnings[{idx}] must have severity 'warning'")

        # Check is_valid consistency
        expected_is_valid = len(self.errors) == 0
        if self.is_valid != expected_is_valid:
            raise ValueError(f"is_valid must be {expected_is_valid} since there are {len(self.errors)} errors")

        return self


class MLPlanValidationError(Exception):
    """Raised when there is an API misuse or incorrect input types passed to the validator."""


# ---------------------------------------------------------------------------
# Validator Class
# ---------------------------------------------------------------------------


class MLPlanValidator:
    """Validator class to run cross-artifact checks on a planned MLPlan."""

    def validate(
        self,
        plan: MLPlan,
        dataset_context: DatasetContext,
        problem_definition: ProblemDefinition,
        compute_capabilities: ComputeCapabilities,
    ) -> MLPlanValidationResult:
        """Run deterministic cross-artifact semantic checks on the plan.

        Args:
            plan: The planned execution MLPlan.
            dataset_context: Target dataset metadata profile.
            problem_definition: Authoritative resolved problem statement.
            compute_capabilities: Interpreted system resources constraints.

        Returns:
            A structured MLPlanValidationResult containing errors and warnings.

        Raises:
            MLPlanValidationError: If any of the inputs are of incorrect types.
        """
        # 1. Input Type Validation
        if not isinstance(plan, MLPlan):
            raise MLPlanValidationError("plan must be an instance of MLPlan")
        if not isinstance(dataset_context, DatasetContext):
            raise MLPlanValidationError("dataset_context must be an instance of DatasetContext")
        if not isinstance(problem_definition, ProblemDefinition):
            raise MLPlanValidationError("problem_definition must be an instance of ProblemDefinition")
        if not isinstance(compute_capabilities, ComputeCapabilities):
            raise MLPlanValidationError("compute_capabilities must be an instance of ComputeCapabilities")

        errors: list[MLPlanValidationIssue] = []
        warnings: list[MLPlanValidationIssue] = []

        # Create quick column lookup from DatasetContext
        col_lookup = {col.name: col for col in dataset_context.columns}

        # -------------------------------------------------------------------
        # Category 1: Artifact Identity Consistency
        # -------------------------------------------------------------------
        if plan.dataset_id != dataset_context.basic_info.dataset_id:
            errors.append(
                MLPlanValidationIssue(
                    code="DATASET_ID_MISMATCH",
                    message="plan.dataset_id does not match dataset_context dataset_id.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if plan.dataset_id != problem_definition.dataset_id:
            errors.append(
                MLPlanValidationIssue(
                    code="PROBLEM_DATASET_ID_MISMATCH",
                    message="plan.dataset_id does not match problem_definition dataset_id.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if plan.request_id != problem_definition.request_id:
            errors.append(
                MLPlanValidationIssue(
                    code="REQUEST_ID_MISMATCH",
                    message="plan.request_id does not match problem_definition request_id.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if plan.problem_definition_id != problem_definition.definition_id:
            errors.append(
                MLPlanValidationIssue(
                    code="PROBLEM_DEFINITION_ID_MISMATCH",
                    message="plan.problem_definition_id does not match problem_definition definition_id.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if plan.compute_capability_id != compute_capabilities.capability_id:
            errors.append(
                MLPlanValidationIssue(
                    code="COMPUTE_CAPABILITY_ID_MISMATCH",
                    message="plan.compute_capability_id does not match compute_capabilities capability_id.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # -------------------------------------------------------------------
        # Category 2: Problem Definition Consistency
        # -------------------------------------------------------------------
        if plan.problem_type != problem_definition.problem_type:
            errors.append(
                MLPlanValidationIssue(
                    code="PROBLEM_TYPE_MISMATCH",
                    message="plan.problem_type does not match problem_definition problem_type.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if plan.target_column != problem_definition.target_column:
            errors.append(
                MLPlanValidationIssue(
                    code="TARGET_COLUMN_MISMATCH",
                    message="plan.target_column does not match problem_definition target_column.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if plan.feature_columns != problem_definition.feature_columns:
            errors.append(
                MLPlanValidationIssue(
                    code="FEATURE_COLUMNS_MISMATCH",
                    message="plan.feature_columns list or order does not match problem_definition feature_columns.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if problem_definition.status == ResolutionStatus.BLOCKED:
            errors.append(
                MLPlanValidationIssue(
                    code="PROBLEM_DEFINITION_BLOCKED",
                    message="The underlying problem definition is currently blocked.",
                    severity=ValidationSeverity.ERROR,
                )
            )
        elif problem_definition.status == ResolutionStatus.NEEDS_CONFIRMATION:
            errors.append(
                MLPlanValidationIssue(
                    code="PROBLEM_DEFINITION_NEEDS_CONFIRMATION",
                    message="The underlying problem definition requires confirmation.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # -------------------------------------------------------------------
        # Category 3: Dataset Column Consistency
        # -------------------------------------------------------------------
        if plan.target_column not in col_lookup:
            errors.append(
                MLPlanValidationIssue(
                    code="TARGET_COLUMN_NOT_FOUND",
                    message=f"Target column '{plan.target_column}' does not exist in dataset columns.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        for col in plan.feature_columns:
            if col not in col_lookup:
                errors.append(
                    MLPlanValidationIssue(
                        code="FEATURE_COLUMN_NOT_FOUND",
                        message=f"Feature column '{col}' does not exist in dataset columns.",
                        severity=ValidationSeverity.ERROR,
                    )
                )

        # -------------------------------------------------------------------
        # Category 4: Preprocessing Step Consistency
        # -------------------------------------------------------------------
        for step_idx, step in enumerate(plan.preprocessing_steps):
            loc = f"preprocessing_steps[{step_idx}].columns"
            for col in step.columns:
                # Target column in preprocessing
                if col == plan.target_column:
                    errors.append(
                        MLPlanValidationIssue(
                            code="TARGET_IN_PREPROCESSING_STEP",
                            message=f"Target column '{col}' cannot appear in preprocessing steps.",
                            severity=ValidationSeverity.ERROR,
                            location=loc,
                        )
                    )

                # Existence check
                col_ctx = col_lookup.get(col)
                if col_ctx is None:
                    errors.append(
                        MLPlanValidationIssue(
                            code="PREPROCESSING_COLUMN_NOT_FOUND",
                            message=f"Column '{col}' referenced in preprocessing step {step.step_id} does not exist in dataset.",
                            severity=ValidationSeverity.ERROR,
                            location=loc,
                        )
                    )
                    continue

                # Preprocessing Type Compatibility
                if step.operation in (
                    PreprocessingOperation.IMPUTE_MEAN,
                    PreprocessingOperation.IMPUTE_MEDIAN,
                    PreprocessingOperation.STANDARD_SCALE,
                    PreprocessingOperation.MINMAX_SCALE,
                    PreprocessingOperation.ROBUST_SCALE,
                ):
                    if not col_ctx.is_numeric:
                        errors.append(
                            MLPlanValidationIssue(
                                code="PREPROCESSING_REQUIRES_NUMERIC",
                                message=f"Operation '{step.operation.value}' on column '{col}' requires numeric data.",
                                severity=ValidationSeverity.ERROR,
                                location=loc,
                            )
                        )
                elif step.operation in (
                    PreprocessingOperation.IMPUTE_MODE,
                    PreprocessingOperation.ONE_HOT_ENCODE,
                    PreprocessingOperation.ORDINAL_ENCODE,
                ):
                    if not col_ctx.is_categorical:
                        errors.append(
                            MLPlanValidationIssue(
                                code="PREPROCESSING_REQUIRES_CATEGORICAL",
                                message=f"Operation '{step.operation.value}' on column '{col}' requires categorical data.",
                                severity=ValidationSeverity.ERROR,
                                location=loc,
                            )
                        )
                elif step.operation == PreprocessingOperation.DATETIME_EXTRACT:
                    if not col_ctx.is_datetime:
                        errors.append(
                            MLPlanValidationIssue(
                                code="PREPROCESSING_REQUIRES_DATETIME",
                                message=f"Operation '{step.operation.value}' on column '{col}' requires datetime data.",
                                severity=ValidationSeverity.ERROR,
                                location=loc,
                            )
                        )

        # -------------------------------------------------------------------
        # Category 5: Feature Engineering Consistency
        # -------------------------------------------------------------------
        available_columns = set(col_lookup.keys())
        for step_idx, step in enumerate(plan.feature_engineering_steps):
            input_loc = f"feature_engineering_steps[{step_idx}].input_columns"
            output_loc = f"feature_engineering_steps[{step_idx}].output_columns"

            # Validate input columns
            for col in step.input_columns:
                if col == plan.target_column:
                    errors.append(
                        MLPlanValidationIssue(
                            code="TARGET_IN_FEATURE_ENGINEERING_INPUT",
                            message=f"Target column '{col}' cannot be used as a feature engineering input.",
                            severity=ValidationSeverity.ERROR,
                            location=input_loc,
                        )
                    )
                if col not in available_columns:
                    errors.append(
                        MLPlanValidationIssue(
                            code="FEATURE_ENGINEERING_INPUT_NOT_FOUND",
                            message=f"Input column '{col}' for feature engineering step {step.step_id} does not exist.",
                            severity=ValidationSeverity.ERROR,
                            location=input_loc,
                        )
                    )

            # Validate output columns
            for col in step.output_columns:
                if col == plan.target_column:
                    errors.append(
                        MLPlanValidationIssue(
                            code="TARGET_IN_FEATURE_ENGINEERING_OUTPUT",
                            message=f"Target column '{col}' cannot be created as a feature engineering output.",
                            severity=ValidationSeverity.ERROR,
                            location=output_loc,
                        )
                    )
                if col in available_columns:
                    errors.append(
                        MLPlanValidationIssue(
                            code="FEATURE_ENGINEERING_OUTPUT_COLLISION",
                            message=f"Output column '{col}' of step {step.step_id} collides with an existing column.",
                            severity=ValidationSeverity.ERROR,
                            location=output_loc,
                        )
                    )
                else:
                    if col != plan.target_column:
                        available_columns.add(col)

        # -------------------------------------------------------------------
        # Category 6: Feature Selection Consistency
        # -------------------------------------------------------------------
        fe_outputs = set()
        for step in plan.feature_engineering_steps:
            for col in step.output_columns:
                if col != plan.target_column and col not in col_lookup:
                    fe_outputs.add(col)

        candidate_universe = set(plan.feature_columns) | fe_outputs
        sel_loc = "feature_selection.candidate_columns"

        for col in plan.feature_selection.candidate_columns:
            if col == plan.target_column:
                errors.append(
                    MLPlanValidationIssue(
                        code="TARGET_IN_FEATURE_SELECTION",
                        message=f"Target column '{col}' cannot be a feature selection candidate.",
                        severity=ValidationSeverity.ERROR,
                        location=sel_loc,
                    )
                )
            if col not in candidate_universe:
                errors.append(
                    MLPlanValidationIssue(
                        code="FEATURE_SELECTION_COLUMN_NOT_AVAILABLE",
                        message=f"Feature selection candidate column '{col}' is not available.",
                        severity=ValidationSeverity.ERROR,
                        location=sel_loc,
                    )
                )

        # -------------------------------------------------------------------
        # Category 7: Split Plan Consistency
        # -------------------------------------------------------------------
        split_plan = plan.split_plan
        if split_plan.strategy == "stratified":
            if plan.problem_type == ProblemType.CLASSIFICATION:
                if split_plan.stratify_column != plan.target_column:
                    errors.append(
                        MLPlanValidationIssue(
                            code="INVALID_STRATIFY_COLUMN",
                            message="split_plan.stratify_column must match target_column.",
                            severity=ValidationSeverity.ERROR,
                            location="split_plan.stratify_column",
                        )
                    )
            elif plan.problem_type == ProblemType.REGRESSION:
                errors.append(
                    MLPlanValidationIssue(
                        code="STRATIFIED_SPLIT_FOR_REGRESSION",
                        message="Stratified splits are not allowed for regression tasks.",
                        severity=ValidationSeverity.ERROR,
                        location="split_plan.strategy",
                    )
                )
        elif split_plan.strategy == "time_based":
            time_col = split_plan.time_column
            time_ctx = col_lookup.get(time_col) if time_col else None
            if time_col and time_ctx is None:
                errors.append(
                    MLPlanValidationIssue(
                        code="TIME_COLUMN_NOT_FOUND",
                        message=f"split time_column '{time_col}' does not exist in dataset.",
                        severity=ValidationSeverity.ERROR,
                        location="split_plan.time_column",
                    )
                )
            elif time_col and time_ctx is not None:
                if not time_ctx.is_datetime:
                    errors.append(
                        MLPlanValidationIssue(
                            code="TIME_COLUMN_NOT_DATETIME",
                            message=f"split time_column '{time_col}' must be a datetime column.",
                            severity=ValidationSeverity.ERROR,
                            location="split_plan.time_column",
                        )
                    )

        # -------------------------------------------------------------------
        # Category 8: Model Family Compatibility
        # -------------------------------------------------------------------
        classification_families = {
            ModelFamily.LOGISTIC_REGRESSION,
            ModelFamily.DECISION_TREE,
            ModelFamily.RANDOM_FOREST,
            ModelFamily.GRADIENT_BOOSTING,
            ModelFamily.EXTRA_TREES,
            ModelFamily.KNN,
            ModelFamily.SVM,
        }
        regression_families = {
            ModelFamily.LINEAR_REGRESSION,
            ModelFamily.RIDGE,
            ModelFamily.LASSO,
            ModelFamily.DECISION_TREE,
            ModelFamily.RANDOM_FOREST,
            ModelFamily.GRADIENT_BOOSTING,
            ModelFamily.EXTRA_TREES,
            ModelFamily.KNN,
            ModelFamily.SVM,
        }

        for idx, cand in enumerate(plan.model_candidates):
            loc = f"model_candidates[{idx}]"
            is_compatible = True
            if plan.problem_type == ProblemType.CLASSIFICATION:
                if cand.model_family not in classification_families:
                    is_compatible = False
            elif plan.problem_type == ProblemType.REGRESSION:
                if cand.model_family not in regression_families:
                    is_compatible = False

            if not is_compatible:
                errors.append(
                    MLPlanValidationIssue(
                        code="MODEL_FAMILY_INCOMPATIBLE",
                        message=f"Model family '{cand.model_family.value}' is incompatible with problem type '{plan.problem_type.value}'.",
                        severity=ValidationSeverity.ERROR,
                        location=loc,
                    )
                )

        # -------------------------------------------------------------------
        # Category 9: Evaluation Metric Compatibility
        # -------------------------------------------------------------------
        classification_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        regression_metrics = {"mae", "mse", "rmse", "r2"}

        eval_plan = plan.evaluation_plan

        # Primary metric
        primary_norm = eval_plan.primary_metric.strip().lower()
        is_primary_compatible = True
        if plan.problem_type == ProblemType.CLASSIFICATION:
            if primary_norm not in classification_metrics:
                is_primary_compatible = False
        elif plan.problem_type == ProblemType.REGRESSION:
            if primary_norm not in regression_metrics:
                is_primary_compatible = False

        if not is_primary_compatible:
            errors.append(
                MLPlanValidationIssue(
                    code="METRIC_INCOMPATIBLE",
                    message=f"Primary metric '{eval_plan.primary_metric}' is incompatible with problem type '{plan.problem_type.value}'.",
                    severity=ValidationSeverity.ERROR,
                    location="evaluation_plan.primary_metric",
                )
            )

        # Secondary metrics
        for s_idx, metric in enumerate(eval_plan.secondary_metrics):
            metric_norm = metric.strip().lower()
            is_sec_compatible = True
            if plan.problem_type == ProblemType.CLASSIFICATION:
                if metric_norm not in classification_metrics:
                    is_sec_compatible = False
            elif plan.problem_type == ProblemType.REGRESSION:
                if metric_norm not in regression_metrics:
                    is_sec_compatible = False

            if not is_sec_compatible:
                errors.append(
                    MLPlanValidationIssue(
                        code="METRIC_INCOMPATIBLE",
                        message=f"Secondary metric '{metric}' is incompatible with problem type '{plan.problem_type.value}'.",
                        severity=ValidationSeverity.ERROR,
                        location=f"evaluation_plan.secondary_metrics[{s_idx}]",
                    )
                )

        # -------------------------------------------------------------------
        # Category 10: Execution Constraint Consistency
        # -------------------------------------------------------------------
        exec_const = plan.execution_constraints
        if exec_const.compute_tier != compute_capabilities.compute_tier:
            errors.append(
                MLPlanValidationIssue(
                    code="COMPUTE_TIER_MISMATCH",
                    message=f"Constraint compute tier '{exec_const.compute_tier.value}' does not match capability tier '{compute_capabilities.compute_tier.value}'.",
                    severity=ValidationSeverity.ERROR,
                    location="execution_constraints.compute_tier",
                )
            )

        if exec_const.parallel_workers > compute_capabilities.safe_parallel_workers:
            errors.append(
                MLPlanValidationIssue(
                    code="PARALLEL_WORKERS_EXCEED_SAFE_LIMIT",
                    message=f"Requested parallel workers ({exec_const.parallel_workers}) exceeds capability safe limit ({compute_capabilities.safe_parallel_workers}).",
                    severity=ValidationSeverity.ERROR,
                    location="execution_constraints.parallel_workers",
                )
            )

        if exec_const.use_gpu_acceleration:
            if not compute_capabilities.gpu_acceleration_available:
                errors.append(
                    MLPlanValidationIssue(
                        code="GPU_ACCELERATION_UNAVAILABLE",
                        message="GPU acceleration is requested but not available in compute capabilities.",
                        severity=ValidationSeverity.ERROR,
                        location="execution_constraints.use_gpu_acceleration",
                    )
                )
            elif exec_const.accelerator_type != compute_capabilities.accelerator_type:
                errors.append(
                    MLPlanValidationIssue(
                        code="ACCELERATOR_TYPE_MISMATCH",
                        message=f"Constraint accelerator '{exec_const.accelerator_type.value}' does not match capability '{compute_capabilities.accelerator_type.value}'.",
                        severity=ValidationSeverity.ERROR,
                        location="execution_constraints.accelerator_type",
                    )
                )

        # -------------------------------------------------------------------
        # Category 11: Plan Status Consistency
        # -------------------------------------------------------------------
        if plan.status == MLPlanStatus.BLOCKED:
            warnings.append(
                MLPlanValidationIssue(
                    code="PLAN_STATUS_BLOCKED",
                    message="Execution is blocked because the ML plan is in blocked status.",
                    severity=ValidationSeverity.WARNING,
                    location="status",
                )
            )
        elif plan.status == MLPlanStatus.NEEDS_CONFIRMATION:
            warnings.append(
                MLPlanValidationIssue(
                    code="PLAN_REQUIRES_CONFIRMATION",
                    message="ML plan requires manual user confirmation before it can be executed.",
                    severity=ValidationSeverity.WARNING,
                    location="status",
                )
            )
        elif plan.status == MLPlanStatus.DRAFT:
            warnings.append(
                MLPlanValidationIssue(
                    code="PLAN_STATUS_DRAFT",
                    message="The ML plan is in draft status and not yet ready for execution.",
                    severity=ValidationSeverity.WARNING,
                    location="status",
                )
            )

        is_valid = len(errors) == 0
        return MLPlanValidationResult(
            plan_id=plan.plan_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )
