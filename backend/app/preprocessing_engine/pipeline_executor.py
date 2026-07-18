"""Pipeline execution engine for applying preprocessing plans to DataFrames."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backend.app.preprocessing_engine.duplicate_handler import (
    DuplicateHandler,
)
from backend.app.preprocessing_engine.encoding_handler import (
    EncodingHandler,
)
from backend.app.preprocessing_engine.missing_handler import (
    MissingValueHandler,
)
from backend.app.preprocessing_engine.scaling_handler import (
    ScalingHandler,
)
from backend.app.preprocessing_engine.schemas import ExecutionPlan

logger = logging.getLogger(__name__)


class PipelineExecutionError(Exception):
    """Custom exception raised when pipeline execution fails."""


class PipelineExecutor:
    """Executes a structured preprocessing ExecutionPlan against a pandas DataFrame.

    Applies the preprocessing stages in the following strict order:
    1. Missing value handling
    2. Duplicate handling
    3. Categorical encoding
    4. Numerical scaling

    The input DataFrame is not mutated; a new DataFrame is returned.
    """

    def __init__(self) -> None:
        """Initialize handlers for each preprocessing stage."""
        self._missing_value_handler = MissingValueHandler()
        self._duplicate_handler = DuplicateHandler()
        self._encoding_handler = EncodingHandler()
        self._scaling_handler = ScalingHandler()

    def execute(
        self,
        df: pd.DataFrame,
        plan: ExecutionPlan,
    ) -> pd.DataFrame:
        """Execute the preprocessing plan against the given DataFrame.

        Args:
            df: The pandas DataFrame to preprocess.
            plan: The ExecutionPlan containing strategies for missing values,
                duplicates, encoding, and scaling.

        Returns:
            A new pandas DataFrame with all plan transformations applied.

        Raises:
            PipelineExecutionError: If any of the underlying handlers fail.
        """
        logger.info("Starting preprocessing pipeline execution.")

        if df.empty:
            logger.info("Input DataFrame is empty. Skipping preprocessing stages.")
            return df.copy(deep=True)

        current_df = df.copy(deep=True)

        # 1. Missing Value Handling
        logger.info("Stage 1/4: Applying missing value handling actions.")
        try:
            missing_actions = {
                col: action.model_dump()
                for col, action in plan.missing_values.items()
            }
            current_df = self._missing_value_handler.apply(current_df, missing_actions)
        except Exception as exc:
            logger.exception("Pipeline execution failed during missing value handling stage.")
            raise PipelineExecutionError(f"Missing value handling stage failed: {exc}") from exc
        logger.info("Stage 1/4: Completed missing value handling.")

        # 2. Duplicate Handling
        logger.info("Stage 2/4: Applying duplicate-row handling actions.")
        try:
            strategy_val = (
                plan.duplicates_action.value
                if hasattr(plan.duplicates_action, "value")
                else str(plan.duplicates_action)
            )
            current_df = self._duplicate_handler.apply(current_df, strategy_val)
        except Exception as exc:
            logger.exception("Pipeline execution failed during duplicate handling stage.")
            raise PipelineExecutionError(f"Duplicate handling stage failed: {exc}") from exc
        logger.info("Stage 2/4: Completed duplicate handling.")

        # 3. Categorical Encoding
        logger.info("Stage 3/4: Applying categorical encoding actions.")
        try:
            encoding_actions = {
                col: action.model_dump()
                for col, action in plan.encoding.items()
            }
            current_df = self._encoding_handler.apply(current_df, encoding_actions)
        except Exception as exc:
            logger.exception("Pipeline execution failed during categorical encoding stage.")
            raise PipelineExecutionError(f"Categorical encoding stage failed: {exc}") from exc
        logger.info("Stage 3/4: Completed categorical encoding.")

        # 4. Numerical Scaling
        logger.info("Stage 4/4: Applying numerical scaling actions.")
        try:
            scaling_actions = {
                col: action.model_dump()
                for col, action in plan.scaling.items()
            }
            current_df = self._scaling_handler.apply(current_df, scaling_actions)
        except Exception as exc:
            logger.exception("Pipeline execution failed during numerical scaling stage.")
            raise PipelineExecutionError(f"Numerical scaling stage failed: {exc}") from exc
        logger.info("Stage 4/4: Completed numerical scaling.")

        logger.info("Preprocessing pipeline execution completed successfully.")
        return current_df
