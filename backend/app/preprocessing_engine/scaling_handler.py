"""Numerical scaling preprocessing execution utilities."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from backend.app.preprocessing_engine.schemas import ScalingStrategy

logger = logging.getLogger(__name__)


class ScalingHandlerError(RuntimeError):
    """Raised when numerical scaling preprocessing cannot be completed."""


class ScalingHandler:
    """Apply numerical scaling actions to a pandas DataFrame."""

    def apply(
        self,
        df: pd.DataFrame,
        actions: dict[str, Mapping[str, Any]],
    ) -> pd.DataFrame:
        """Apply scaling actions and return a new DataFrame.

        Args:
            df: Input DataFrame to transform.
            actions: Mapping of column names to action payloads containing a
                ``strategy`` key.

        Returns:
            A transformed copy of ``df``. The original DataFrame is never
            mutated.

        Raises:
            ScalingHandlerError: If an unsupported strategy is requested or a
                scaling action fails unexpectedly.
        """
        logger.info("Applying numerical scaling actions.")
        result = df.copy(deep=True)

        if result.empty or not actions:
            logger.info("No numerical scaling actions applied.")
            return result

        for column_name, action in actions.items():
            if column_name not in result.columns:
                logger.warning("Skipping scaling action for unknown column: %s", column_name)
                continue

            if not pd.api.types.is_numeric_dtype(result[column_name]):
                logger.warning("Skipping scaling action for non-numeric column: %s", column_name)
                continue

            strategy = str(action.get("strategy", ""))
            try:
                self._apply_column_action(result, column_name, strategy)
            except Exception as exc:
                logger.exception(
                    "Failed to apply scaling strategy '%s' to column '%s'.",
                    strategy,
                    column_name,
                )
                raise ScalingHandlerError(
                    f"Failed to apply scaling strategy '{strategy}' "
                    f"to column '{column_name}'."
                ) from exc

        logger.info("Numerical scaling completed successfully.")
        return result

    @staticmethod
    def _apply_column_action(
        df: pd.DataFrame,
        column_name: str,
        strategy: str,
    ) -> None:
        """Apply one scaling strategy to one column in ``df``."""
        if strategy == ScalingStrategy.STANDARD_SCALING.value:
            mean = df[column_name].mean()
            std = df[column_name].std()
            if std == 0 or pd.isna(std):
                logger.warning("Skipping standard scaling for zero-variance column: %s", column_name)
                return
            df[column_name] = (df[column_name] - mean) / std
            return

        if strategy == ScalingStrategy.MINMAX_SCALING.value:
            minimum = df[column_name].min()
            maximum = df[column_name].max()
            denominator = maximum - minimum
            if denominator == 0 or pd.isna(denominator):
                logger.warning("Skipping min-max scaling for zero-variance column: %s", column_name)
                return
            df[column_name] = (df[column_name] - minimum) / denominator
            return

        if strategy == ScalingStrategy.NO_SCALING.value:
            return

        raise ValueError(f"Unsupported scaling strategy: {strategy}")
