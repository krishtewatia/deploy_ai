"""Feature Engineering Executor for DeployAI.

Stage 9D executes feature engineering transformation steps on datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import (
    FeatureEngineeringOperation,
    FeatureEngineeringStep,
    MLPlan,
)


class FeatureEngineeringExecutorError(Exception):
    """Raised when feature engineering execution fails due to invalid parameters or data issues."""

    pass


@dataclass
class FeatureEngineeringResult:
    """The structured result of running feature engineering execution."""

    dataframe: pd.DataFrame
    engineered_columns: list[str]
    created_columns: list[str]
    execution_summary: dict[str, Any] = field(default_factory=dict)


class FeatureEngineeringExecutor:
    """Executes planned feature engineering steps deterministically on a dataset."""

    def execute(
        self,
        *,
        dataframe: pd.DataFrame,
        dataset_context: DatasetContext,
        plan: MLPlan,
    ) -> FeatureEngineeringResult:
        """Execute feature engineering steps sequentially on a copy of the input dataframe.

        Args:
            dataframe: Input pandas DataFrame to transform.
            dataset_context: Target dataset metadata profile.
            plan: The planned execution MLPlan.

        Returns:
            FeatureEngineeringResult containing the modified dataframe and lists of created columns.

        Raises:
            FeatureEngineeringExecutorError: For validation or execution failures.
        """
        # 1. Reject None inputs
        if dataframe is None:
            raise FeatureEngineeringExecutorError("dataframe cannot be None")
        if dataset_context is None:
            raise FeatureEngineeringExecutorError("dataset_context cannot be None")
        if plan is None:
            raise FeatureEngineeringExecutorError("plan cannot be None")

        # 2. Reject wrong types
        if not isinstance(dataframe, pd.DataFrame):
            raise FeatureEngineeringExecutorError("dataframe must be a pandas.DataFrame instance")
        if not isinstance(dataset_context, DatasetContext):
            raise FeatureEngineeringExecutorError("dataset_context must be a DatasetContext instance")
        if not isinstance(plan, MLPlan):
            raise FeatureEngineeringExecutorError("plan must be an MLPlan instance")

        # 3. Reject empty dataframe
        if dataframe.empty:
            raise FeatureEngineeringExecutorError("dataframe cannot be empty")

        steps = plan.feature_engineering_steps or []

        # 4. Check duplicate step IDs
        step_ids = [s.step_id for s in steps]
        if len(step_ids) != len(set(step_ids)):
            raise FeatureEngineeringExecutorError("Duplicate step_id found in feature engineering steps")

        # 5. Check target column constraints
        target = plan.target_column
        if not target:
            raise FeatureEngineeringExecutorError("target_column cannot be empty")

        # 6. Gather all output columns and validate uniqueness
        all_output_cols: list[str] = []
        for step in steps:
            # Check for duplicate output names inside a single step
            if len(step.output_columns) != len(set(step.output_columns)):
                raise FeatureEngineeringExecutorError(
                    f"Duplicate output columns found in step {step.step_id}: {step.output_columns}"
                )
            
            # Check target in input/output columns
            if target in step.input_columns:
                raise FeatureEngineeringExecutorError(
                    f"Target column '{target}' cannot be used as an input column in step {step.step_id}"
                )
            if target in step.output_columns:
                raise FeatureEngineeringExecutorError(
                    f"Target column '{target}' cannot be used as an output column in step {step.step_id}"
                )

            all_output_cols.extend(step.output_columns)

        # Check duplicate output names across steps
        if len(all_output_cols) != len(set(all_output_cols)):
            raise FeatureEngineeringExecutorError(
                f"Duplicate output columns found across steps: {all_output_cols}"
            )

        # Check if any output column name already exists in the original dataframe
        original_cols = set(dataframe.columns)
        for col in all_output_cols:
            if col in original_cols:
                raise FeatureEngineeringExecutorError(
                    f"Output column '{col}' already exists in the input dataframe"
                )

        # 7. Start execution on a copy of the dataframe
        df_copy = dataframe.copy()
        created_columns: list[str] = []
        steps_summary: list[dict[str, Any]] = []

        for step in steps:
            op = step.operation
            inputs = step.input_columns
            outputs = step.output_columns

            # Verify input columns exist in current dataframe (allows using previously created columns)
            dataset_cols = {col.name for col in dataset_context.columns}
            skip_step = False
            for in_col in inputs:
                if in_col not in df_copy.columns:
                    if in_col in dataset_cols:
                        skip_step = True
                        break
                    else:
                        raise FeatureEngineeringExecutorError(
                            f"Input column '{in_col}' does not exist for step {step.step_id}"
                        )
            if skip_step:
                continue

            try:
                if op == FeatureEngineeringOperation.CUSTOM:
                    raise FeatureEngineeringExecutorError(
                        f"CUSTOM operation is rejected in step {step.step_id}."
                    )

                elif op == FeatureEngineeringOperation.INTERACTION:
                    if len(inputs) != 2:
                        raise FeatureEngineeringExecutorError(
                            f"INTERACTION requires exactly 2 input columns, got {len(inputs)}"
                        )
                    if len(outputs) != 1:
                        raise FeatureEngineeringExecutorError(
                            f"INTERACTION requires exactly 1 output column, got {len(outputs)}"
                        )
                    A, B = inputs[0], inputs[1]
                    out = outputs[0]
                    # Enforce numeric conversion or type safety during calculation
                    df_copy[out] = df_copy[A] * df_copy[B]

                elif op == FeatureEngineeringOperation.RATIO:
                    if len(inputs) != 2:
                        raise FeatureEngineeringExecutorError(
                            f"RATIO requires exactly 2 input columns, got {len(inputs)}"
                        )
                    if len(outputs) != 1:
                        raise FeatureEngineeringExecutorError(
                            f"RATIO requires exactly 1 output column, got {len(outputs)}"
                        )
                    A, B = inputs[0], inputs[1]
                    out = outputs[0]

                    # Perform ratio calculation while protecting against division by zero
                    denom = df_copy[B].values
                    num = df_copy[A].values
                    # numpy.where is used to avoid RuntimeWarning when dividing by zero
                    res = np.where(denom != 0, num / np.where(denom != 0, denom, 1.0), np.nan)
                    df_copy[out] = res

                elif op == FeatureEngineeringOperation.DIFFERENCE:
                    if len(inputs) != 2:
                        raise FeatureEngineeringExecutorError(
                            f"DIFFERENCE requires exactly 2 input columns, got {len(inputs)}"
                        )
                    if len(outputs) != 1:
                        raise FeatureEngineeringExecutorError(
                            f"DIFFERENCE requires exactly 1 output column, got {len(outputs)}"
                        )
                    A, B = inputs[0], inputs[1]
                    out = outputs[0]
                    df_copy[out] = df_copy[A] - df_copy[B]

                elif op == FeatureEngineeringOperation.LOG_TRANSFORM:
                    if len(inputs) != 1:
                        raise FeatureEngineeringExecutorError(
                            f"LOG requires exactly 1 input column, got {len(inputs)}"
                        )
                    if len(outputs) != 1:
                        raise FeatureEngineeringExecutorError(
                            f"LOG requires exactly 1 output column, got {len(outputs)}"
                        )
                    A = inputs[0]
                    out = outputs[0]

                    # Reject negative values below -1
                    if (df_copy[A] < -1).any():
                        raise FeatureEngineeringExecutorError(
                            f"LOG transform fails: column '{A}' contains values below -1."
                        )
                    df_copy[out] = np.log1p(df_copy[A])

                elif op == FeatureEngineeringOperation.POLYNOMIAL:
                    if len(inputs) != 1:
                        raise FeatureEngineeringExecutorError(
                            f"POLYNOMIAL requires exactly 1 input column, got {len(inputs)}"
                        )
                    if len(outputs) != 1:
                        raise FeatureEngineeringExecutorError(
                            f"POLYNOMIAL requires exactly 1 output column, got {len(outputs)}"
                        )
                    A = inputs[0]
                    out = outputs[0]

                    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                    # Exclude NaN handling issues if sklearn fails internally (caught by try/except)
                    transformed = poly.fit_transform(df_copy[[A]])
                    # Index 1 of the output columns corresponds to A^2
                    df_copy[out] = transformed[:, 1]

                else:
                    raise FeatureEngineeringExecutorError(
                        f"Unknown or unsupported operation '{op}' in step {step.step_id}."
                    )


                created_columns.extend(outputs)
                steps_summary.append({
                    "step_id": step.step_id,
                    "operation": op.value,
                    "inputs": inputs,
                    "outputs": outputs,
                    "status": "success",
                })

            except FeatureEngineeringExecutorError:
                # Re-raise explicit validation/exec errors
                raise
            except Exception as e:
                raise FeatureEngineeringExecutorError(
                    f"Execution failed for step '{step.step_id}' ({op.value}): {e}"
                ) from e

        execution_summary = {
            "total_steps": len(steps),
            "executed_steps": steps_summary,
            "created_columns": created_columns,
        }

        return FeatureEngineeringResult(
            dataframe=df_copy,
            engineered_columns=created_columns,
            created_columns=created_columns,
            execution_summary=execution_summary,
        )
