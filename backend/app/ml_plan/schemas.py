"""Pydantic v2 schemas and enums representing structured ML execution plans.

These models serve as data contracts specifying exactly what operations the local
execution engine should execute (preprocessing, splitting, candidates, training).
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.problem_definition.schemas import ProblemType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MLPlanStatus(str, Enum):
    """Execution readiness status of the plan."""

    DRAFT = "draft"
    READY = "ready"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"


class PreprocessingOperation(str, Enum):
    """Structured data preparation and imputation operations."""

    DROP_COLUMN = "drop_column"
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    IMPUTE_CONSTANT = "impute_constant"
    ONE_HOT_ENCODE = "one_hot_encode"
    ORDINAL_ENCODE = "ordinal_encode"
    STANDARD_SCALE = "standard_scale"
    MINMAX_SCALE = "minmax_scale"
    ROBUST_SCALE = "robust_scale"
    DATETIME_EXTRACT = "datetime_extract"
    PASSTHROUGH = "passthrough"


class FeatureEngineeringOperation(str, Enum):
    """Feature derivation and non-linear transformation operations."""

    INTERACTION = "interaction"
    POLYNOMIAL = "polynomial"
    RATIO = "ratio"
    DIFFERENCE = "difference"
    DATETIME_PARTS = "datetime_parts"
    LOG_TRANSFORM = "log_transform"
    CUSTOM = "custom"


class FeatureSelectionMethod(str, Enum):
    """Feature selection filters and wrappers."""

    NONE = "none"
    VARIANCE_THRESHOLD = "variance_threshold"
    CORRELATION_FILTER = "correlation_filter"
    MUTUAL_INFORMATION = "mutual_information"
    MODEL_BASED = "model_based"


class SplitStrategy(str, Enum):
    """Dataset splitting strategy for training and validation."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    TIME_BASED = "time_based"


class ModelFamily(str, Enum):
    """Supported candidate algorithms for classical machine learning."""

    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    EXTRA_TREES = "extra_trees"
    KNN = "knn"
    SVM = "svm"


class SearchStrategy(str, Enum):
    """Hyperparameter search algorithm strategy."""

    NONE = "none"
    GRID = "grid"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Helper Validators
# ---------------------------------------------------------------------------


def _check_json_serializable(name: str, val: Any) -> None:
    """Validate that the provided value can be correctly dumped to JSON."""
    try:
        json.dumps(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc


def _validate_columns_list(name: str, v: Any) -> list[str]:
    """Validate list elements are stripped, non-empty, and unique."""
    if not isinstance(v, list):
        raise ValueError(f"{name} must be a list of strings")
    if len(v) == 0:
        raise ValueError(f"{name} list must contain at least one column")
    processed = []
    seen = set()
    for idx, item in enumerate(v):
        if not isinstance(item, str):
            raise ValueError(f"{name} at index {idx} must be a string")
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{name} at index {idx} cannot be empty or whitespace-only")
        if stripped in seen:
            raise ValueError(f"Duplicate column name detected in {name}: '{stripped}'")
        seen.add(stripped)
        processed.append(stripped)
    return processed


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PreprocessingStep(BaseModel):
    """A single data transformation or imputation step applied to set of columns."""

    step_id: str = Field(
        ...,
        description="Unique identifier for this preprocessing step.",
    )
    operation: PreprocessingOperation = Field(
        ...,
        description="Preprocessing operation type.",
    )
    columns: list[str] = Field(
        ...,
        description="Dataset column names to apply this step to.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Operation hyperparameters and configuration options.",
    )
    reason: str = Field(
        ...,
        description="AI or planner rationale explaining why this step is planned.",
    )

    @field_validator("step_id", "reason", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped

    @field_validator("columns", mode="before")
    @classmethod
    def _validate_columns(cls, v: Any) -> list[str]:
        return _validate_columns_list("columns", v)

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: Any) -> Any:
        _check_json_serializable("parameters", v)
        return v


class FeatureEngineeringStep(BaseModel):
    """Definition of planned feature creation or non-linear mapping step."""

    step_id: str = Field(
        ...,
        description="Unique step identifier.",
    )
    operation: FeatureEngineeringOperation = Field(
        ...,
        description="Feature engineering transformation operation.",
    )
    input_columns: list[str] = Field(
        ...,
        description="Columns consumed as input fields.",
    )
    output_columns: list[str] = Field(
        ...,
        description="Newly generated output columns.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the feature engineer.",
    )
    reason: str = Field(
        ...,
        description="Planning decision rationale.",
    )

    @field_validator("step_id", "reason", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped

    @field_validator("input_columns", mode="before")
    @classmethod
    def _validate_input_columns(cls, v: Any) -> list[str]:
        return _validate_columns_list("input_columns", v)

    @field_validator("output_columns", mode="before")
    @classmethod
    def _validate_output_columns(cls, v: Any) -> list[str]:
        return _validate_columns_list("output_columns", v)

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: Any) -> Any:
        _check_json_serializable("parameters", v)
        return v


class FeatureSelectionPlan(BaseModel):
    """Algorithm configuration for dimensionality reduction or filtering."""

    method: FeatureSelectionMethod = Field(
        default=FeatureSelectionMethod.NONE,
        description="Feature selection algorithm.",
    )
    candidate_columns: list[str] = Field(
        ...,
        description="Columns eligible for filtering.",
    )
    max_features: Optional[int] = Field(
        default=None,
        description="Maximum number of columns to select.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Feature selector specific inputs.",
    )
    reason: str = Field(
        ...,
        description="Explanation behind the selection choice.",
    )

    @field_validator("reason", mode="before")
    @classmethod
    def _strip_and_validate_reason(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("reason must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("reason cannot be empty or whitespace-only")
        return stripped

    @field_validator("candidate_columns", mode="before")
    @classmethod
    def _validate_candidates(cls, v: Any) -> list[str]:
        return _validate_columns_list("candidate_columns", v)

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: Any) -> Any:
        _check_json_serializable("parameters", v)
        return v

    @model_validator(mode="after")
    def _validate_feature_selection(self) -> FeatureSelectionPlan:
        if self.max_features is not None:
            if self.max_features < 1:
                raise ValueError("max_features must be >= 1")
            if self.max_features > len(self.candidate_columns):
                raise ValueError("max_features cannot exceed the number of candidate columns")
        return self


class DatasetSplitPlan(BaseModel):
    """Validation partitioning configuration rules (train/test/validation split ratios)."""

    strategy: SplitStrategy = Field(
        ...,
        description="Partitioning style (random, stratified, or time-based).",
    )
    test_size: float = Field(
        ...,
        description="Fraction of dataset to allocate for test eval.",
    )
    validation_size: float = Field(
        default=0.0,
        description="Fraction to allocate for validation tuning.",
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Reproducibility random seed.",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle before split partition.",
    )
    stratify_column: Optional[str] = Field(
        default=None,
        description="Column used for stratified split distribution.",
    )
    time_column: Optional[str] = Field(
        default=None,
        description="Column used for temporal sequencing ordering.",
    )

    @field_validator("stratify_column", "time_column", mode="before")
    @classmethod
    def _strip_optional_columns(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Column name must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Column name cannot be empty or whitespace-only")
        return stripped

    @model_validator(mode="after")
    def _validate_split_plan_consistency(self) -> DatasetSplitPlan:
        # Sizes range validation
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be strictly between 0.0 and 1.0")
        if not (0.0 <= self.validation_size < 1.0):
            raise ValueError("validation_size must be between 0.0 and 1.0 (inclusive of 0.0)")
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("Sum of test_size and validation_size must be less than 1.0")

        # Strategy validations
        if self.strategy == SplitStrategy.RANDOM:
            if self.stratify_column is not None:
                raise ValueError("stratify_column must be None for random split strategy")
            if self.time_column is not None:
                raise ValueError("time_column must be None for random split strategy")
        elif self.strategy == SplitStrategy.STRATIFIED:
            if self.stratify_column is None:
                raise ValueError("stratify_column is required for stratified split strategy")
            if self.time_column is not None:
                raise ValueError("time_column must be None for stratified split strategy")
        elif self.strategy == SplitStrategy.TIME_BASED:
            if self.time_column is None:
                raise ValueError("time_column is required for time_based split strategy")
            if self.stratify_column is not None:
                raise ValueError("stratify_column must be None for time_based split strategy")
            if self.shuffle:
                raise ValueError("shuffle must be False for time_based split strategy")

        return self


class ModelCandidate(BaseModel):
    """Details representing a candidate model algorithm configuration to train."""

    candidate_id: str = Field(
        ...,
        description="Unique algorithm candidate tag.",
    )
    model_family: ModelFamily = Field(
        ...,
        description="Classical learning algorithm family.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic fixed training hyperparameters.",
    )
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.NONE,
        description="Hyperparameter optimization search algorithm.",
    )
    search_space: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameter boundaries grid for tuning search.",
    )
    reason: str = Field(
        ...,
        description="Decision rationale.",
    )

    @field_validator("candidate_id", "reason", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: Any) -> Any:
        _check_json_serializable("parameters", v)
        return v

    @field_validator("search_space")
    @classmethod
    def _validate_search_space(cls, v: Any) -> Any:
        _check_json_serializable("search_space", v)
        return v

    @model_validator(mode="after")
    def _validate_tuning_consistency(self) -> ModelCandidate:
        if self.search_strategy == SearchStrategy.NONE:
            if len(self.search_space) > 0:
                raise ValueError("search_space must be empty when search_strategy is 'none'")
        else:
            if len(self.search_space) == 0:
                raise ValueError(
                    f"search_space cannot be empty when search_strategy is '{self.search_strategy.value}'"
                )
        return self


class EvaluationPlan(BaseModel):
    """Metric configuration for measuring fit, training errors, and validation performance."""

    primary_metric: str = Field(
        ...,
        description="Metric used to select the overall champion model.",
    )
    secondary_metrics: list[str] = Field(
        default_factory=list,
        description="List of additional secondary metrics recorded.",
    )
    cross_validation_folds: int = Field(
        default=5,
        ge=2,
        description="Number of folds to use for K-Fold CV validation evaluation.",
    )

    @field_validator("primary_metric", mode="before")
    @classmethod
    def _strip_and_validate_primary(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("primary_metric must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("primary_metric cannot be empty or whitespace-only")
        return stripped

    @field_validator("secondary_metrics", mode="before")
    @classmethod
    def _validate_secondary(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("secondary_metrics must be a list of strings")
        processed = []
        seen = set()
        for idx, item in enumerate(v):
            if not isinstance(item, str):
                raise ValueError(f"Secondary metric at index {idx} must be a string")
            stripped = item.strip()
            if not stripped:
                raise ValueError(f"Secondary metric at index {idx} cannot be empty or whitespace-only")
            if stripped in seen:
                raise ValueError(f"Duplicate secondary metric detected: '{stripped}'")
            seen.add(stripped)
            processed.append(stripped)
        return processed

    @model_validator(mode="after")
    def _validate_metrics_consistency(self) -> EvaluationPlan:
        if self.primary_metric in self.secondary_metrics:
            raise ValueError(f"primary_metric '{self.primary_metric}' cannot appear in secondary_metrics")
        return self


class ExecutionConstraints(BaseModel):
    """Physical hardware allocations selected for this plan's pipeline execution."""

    parallel_workers: int = Field(
        ...,
        ge=1,
        description="Number of concurrent worker threads or processes to deploy.",
    )
    use_gpu_acceleration: bool = Field(
        ...,
        description="Whether to run GPU training code.",
    )
    accelerator_type: AcceleratorType = Field(
        ...,
        description="Assigned hardware accelerator device vendor.",
    )
    compute_tier: ComputeTier = Field(
        ...,
        description="General hardware tier constraint tag.",
    )

    @model_validator(mode="after")
    def _validate_constraints_consistency(self) -> ExecutionConstraints:
        if not self.use_gpu_acceleration:
            if self.accelerator_type != AcceleratorType.NONE:
                raise ValueError("accelerator_type must be 'none' when use_gpu_acceleration is False")
        else:
            if self.accelerator_type == AcceleratorType.NONE:
                raise ValueError("accelerator_type cannot be 'none' when use_gpu_acceleration is True")
        return self


class MLPlanWarning(BaseModel):
    """Resource constraints warnings captured during planning checks."""

    code: str = Field(
        ...,
        description="Classification warning code.",
    )
    message: str = Field(
        ...,
        description="Explaining message detail.",
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Whether this warning requires manual confirm review by the user.",
    )

    @field_validator("code", "message", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped


class MLPlanConfirmationItem(BaseModel):
    """Structured question item prompting the user before launching code execution."""

    key: str = Field(
        ...,
        description="Configuration key tag.",
    )
    question: str = Field(
        ...,
        description="Prompt message question for confirmation UI.",
    )
    reason: str = Field(
        ...,
        description="Why confirmation is requested.",
    )

    @field_validator("key", "question", "reason", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped


# ---------------------------------------------------------------------------
# Main MLPlan Schema
# ---------------------------------------------------------------------------


class MLPlan(BaseModel):
    """Top-level machine learning training and preparation execution plan."""

    plan_id: str = Field(
        ...,
        description="Unique execution plan ID.",
    )
    dataset_id: str = Field(
        ...,
        description="Linkage ID to DatasetContext.",
    )
    request_id: str = Field(
        ...,
        description="Linkage ID to UserMLRequest.",
    )
    problem_definition_id: str = Field(
        ...,
        description="Linkage ID to ProblemDefinition.",
    )
    compute_capability_id: str = Field(
        ...,
        description="Linkage ID to ComputeCapabilities.",
    )
    problem_type: ProblemType = Field(
        ...,
        description="Target ML workload type.",
    )
    target_column: str = Field(
        ...,
        description="Dependent variable label column.",
    )
    feature_columns: list[str] = Field(
        ...,
        description="List of independent variables to include in training.",
    )
    preprocessing_steps: list[PreprocessingStep] = Field(
        default_factory=list,
        description="Sequential list of preprocessing operations to run.",
    )
    feature_engineering_steps: list[FeatureEngineeringStep] = Field(
        default_factory=list,
        description="Feature engineering mappings list.",
    )
    feature_selection: FeatureSelectionPlan = Field(
        ...,
        description="Feature selection rules.",
    )
    split_plan: DatasetSplitPlan = Field(
        ...,
        description="Train/test partition validation split config.",
    )
    model_candidates: list[ModelCandidate] = Field(
        ...,
        description="Algorithms list to train and evaluate.",
    )
    evaluation_plan: EvaluationPlan = Field(
        ...,
        description="Evaluator metrics and cross-validation config.",
    )
    execution_constraints: ExecutionConstraints = Field(
        ...,
        description="Hardware resources assigned to this plan run.",
    )
    status: MLPlanStatus = Field(
        default=MLPlanStatus.DRAFT,
        description="Readiness status.",
    )
    warnings: list[MLPlanWarning] = Field(
        default_factory=list,
        description="Planner resource warnings.",
    )
    confirmation_items: list[MLPlanConfirmationItem] = Field(
        default_factory=list,
        description="Planner items needing user approval confirm.",
    )

    @field_validator(
        "plan_id",
        "dataset_id",
        "request_id",
        "problem_definition_id",
        "compute_capability_id",
        "target_column",
        mode="before",
    )
    @classmethod
    def _strip_and_validate_ids(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _validate_feature_columns(cls, v: Any) -> list[str]:
        return _validate_columns_list("feature_columns", v)

    @model_validator(mode="after")
    def _validate_ml_plan_consistency(self) -> MLPlan:
        # 1. Target not in features
        if self.target_column in self.feature_columns:
            raise ValueError(f"target_column '{self.target_column}' cannot appear in feature_columns")

        # 2. Model candidates unique and non-empty
        if len(self.model_candidates) == 0:
            raise ValueError("model_candidates must contain at least one candidate")
        candidate_ids = set()
        for idx, cand in enumerate(self.model_candidates):
            if cand.candidate_id in candidate_ids:
                raise ValueError(f"Duplicate model candidate_id detected: '{cand.candidate_id}'")
            candidate_ids.add(cand.candidate_id)

        # 3. Unique preprocessing step IDs
        preprocessing_ids = set()
        for idx, step in enumerate(self.preprocessing_steps):
            if step.step_id in preprocessing_ids:
                raise ValueError(f"Duplicate preprocessing step_id detected: '{step.step_id}'")
            preprocessing_ids.add(step.step_id)

        # 4. Unique feature engineering step IDs
        feature_engineering_ids = set()
        for idx, step in enumerate(self.feature_engineering_steps):
            if step.step_id in feature_engineering_ids:
                raise ValueError(f"Duplicate feature engineering step_id detected: '{step.step_id}'")
            feature_engineering_ids.add(step.step_id)

        # 5. Status consistency
        if self.status == MLPlanStatus.READY:
            if len(self.confirmation_items) > 0:
                raise ValueError("confirmation_items must be empty when status is 'ready'")
        elif self.status == MLPlanStatus.NEEDS_CONFIRMATION:
            if len(self.confirmation_items) == 0:
                raise ValueError("confirmation_items cannot be empty when status is 'needs_confirmation'")

        return self
