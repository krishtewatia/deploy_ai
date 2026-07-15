"""Preprocessing pipeline builder for DeployAI.

Stage 9C implements conversion of MLPlan preprocessing steps into an executable
scikit-learn preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any, List, Set
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import MLPlan, PreprocessingOperation, PreprocessingStep


class PreprocessingPipelineBuilderError(Exception):
    """Raised when pipeline building fails due to invalid parameters or configuration."""

    pass


class PreprocessingPipelineBuilder:
    """Converts the preprocessing steps of an MLPlan into an executable sklearn Pipeline."""

    def build(
        self,
        *,
        dataset_context: DatasetContext,
        plan: MLPlan,
    ) -> Pipeline:
        """Construct an executable scikit-learn preprocessing Pipeline.

        Args:
            dataset_context: Target dataset metadata profile.
            plan: The planned execution MLPlan.

        Returns:
            An unfitted sklearn.pipeline.Pipeline.

        Raises:
            PreprocessingPipelineBuilderError: For validation or building failures.
        """
        # 1. Reject None inputs
        if dataset_context is None:
            raise PreprocessingPipelineBuilderError("dataset_context cannot be None")
        if plan is None:
            raise PreprocessingPipelineBuilderError("plan cannot be None")

        # 2. Reject wrong types
        if not isinstance(dataset_context, DatasetContext):
            raise PreprocessingPipelineBuilderError("dataset_context must be a DatasetContext instance")
        if not isinstance(plan, MLPlan):
            raise PreprocessingPipelineBuilderError("plan must be an MLPlan instance")

        # 3. Quick lookups & validation collections
        dataset_columns = {col.name for col in dataset_context.columns}
        target = plan.target_column

        seen_step_ids: Set[str] = set()
        imputed_cols: Set[str] = set()
        scaled_cols: Set[str] = set()
        encoded_cols: Set[str] = set()

        supported_operations = {
            PreprocessingOperation.IMPUTE_MEDIAN,
            PreprocessingOperation.IMPUTE_MODE,
            PreprocessingOperation.ONE_HOT_ENCODE,
            PreprocessingOperation.STANDARD_SCALE,
            PreprocessingOperation.ROBUST_SCALE,
            PreprocessingOperation.PASSTHROUGH,
        }

        # 4. Validate preprocessing steps
        for step in plan.preprocessing_steps:
            if not isinstance(step, PreprocessingStep):
                raise PreprocessingPipelineBuilderError("Each preprocessing step must be a PreprocessingStep instance")

            step_id = step.step_id
            if not step_id or not step_id.strip():
                raise PreprocessingPipelineBuilderError("Preprocessing step_id cannot be empty or whitespace-only")

            if step_id in seen_step_ids:
                raise PreprocessingPipelineBuilderError(f"Duplicate preprocessing step_id detected: '{step_id}'")
            seen_step_ids.add(step_id)

            op = step.operation
            if op not in supported_operations:
                raise PreprocessingPipelineBuilderError(f"Unsupported preprocessing operation: {op}")

            if not step.columns:
                raise PreprocessingPipelineBuilderError(f"Preprocessing step '{step_id}' columns list cannot be empty")

            for col in step.columns:
                if not col or not col.strip():
                    raise PreprocessingPipelineBuilderError("Column name cannot be empty or whitespace-only")
                
                # Verify column exists inside DatasetContext
                if col not in dataset_columns:
                    raise PreprocessingPipelineBuilderError(f"Column '{col}' not found in dataset columns")

                # Reject target column inside preprocessing
                if col == target:
                    raise PreprocessingPipelineBuilderError(f"Target column '{col}' cannot appear in preprocessing steps")

                # Conflict checking
                if op in (PreprocessingOperation.IMPUTE_MEDIAN, PreprocessingOperation.IMPUTE_MODE):
                    if col in imputed_cols:
                        raise PreprocessingPipelineBuilderError(
                            f"Conflicting/duplicate imputation operation on column '{col}'"
                        )
                    imputed_cols.add(col)
                elif op in (PreprocessingOperation.STANDARD_SCALE, PreprocessingOperation.ROBUST_SCALE):
                    if col in scaled_cols:
                        raise PreprocessingPipelineBuilderError(
                            f"Conflicting/duplicate scaling operation on column '{col}'"
                        )
                    scaled_cols.add(col)
                elif op == PreprocessingOperation.ONE_HOT_ENCODE:
                    if col in encoded_cols:
                        raise PreprocessingPipelineBuilderError(
                            f"Conflicting/duplicate encoding operation on column '{col}'"
                        )
                    encoded_cols.add(col)

        # 5. Build Pipeline Steps
        pipeline_steps = []

        for step in plan.preprocessing_steps:
            op = step.operation

            # Map operation to sklearn transformer
            if op == PreprocessingOperation.IMPUTE_MEDIAN:
                transformer = SimpleImputer(strategy="median")
            elif op == PreprocessingOperation.IMPUTE_MODE:
                transformer = SimpleImputer(strategy="most_frequent")
            elif op == PreprocessingOperation.ONE_HOT_ENCODE:
                transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            elif op == PreprocessingOperation.STANDARD_SCALE:
                transformer = StandardScaler()
            elif op == PreprocessingOperation.ROBUST_SCALE:
                transformer = RobustScaler()
            else:
                # Must be PASSTHROUGH since operations are pre-validated
                transformer = "passthrough"

            # Create ColumnTransformer representing this step
            ct = ColumnTransformer(
                transformers=[
                    (step.step_id, transformer, list(step.columns)),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            )
            pipeline_steps.append((step.step_id, ct))

        # Handle empty preprocessing steps case by returning a passthrough pipeline
        if not pipeline_steps:
            pipeline = Pipeline(steps=[("passthrough", "passthrough")])
        else:
            pipeline = Pipeline(steps=pipeline_steps)

        # Set output format to pandas so it maintains column names and indices seamlessly
        pipeline.set_output(transform="pandas")
        return pipeline
