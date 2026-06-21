"""Missing-value preprocessing execution utilities."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from backend.app.preprocessing_engine.schemas import MissingValueStrategy

logger = logging.getLogger(__name__)


class MissingValueHandlerError(RuntimeError):
    """Raised when missing-value preprocessing cannot be completed."""


class MissingValueHandler:
    """Apply missing-value preprocessing actions to a pandas DataFrame."""

    def apply(
        self,
        df: pd.DataFrame,
        actions: dict[str, Mapping[str, Any]],
    ) -> pd.DataFrame:
        """Apply missing-value actions and return a new DataFrame.

        Args:
            df: Input DataFrame to transform.
            actions: Mapping of column names to action payloads containing a
                ``strategy`` key.

        Returns:
            A transformed copy of ``df``. The original DataFrame is never
            mutated.

        Raises:
            MissingValueHandlerError: If an action fails unexpectedly.
        """
        logger.info("Applying missing-value preprocessing actions.")
        result = df.copy(deep=True)

        if result.empty or not actions:
            logger.info("No missing-value preprocessing actions applied.")
            return result

        for column_name, action in actions.items():
            if column_name not in result.columns:
                logger.warning("Skipping missing-value action for unknown column: %s", column_name)
                continue

            strategy = str(action.get("strategy", ""))
            try:
                self._apply_column_action(result, column_name, strategy)
            except Exception as exc:
                logger.exception(
                    "Failed to apply missing-value strategy '%s' to column '%s'.",
                    strategy,
                    column_name,
                )
                raise MissingValueHandlerError(
                    f"Failed to apply missing-value strategy '{strategy}' "
                    f"to column '{column_name}'."
                ) from exc

        logger.info("Missing-value preprocessing completed successfully.")
        return result

    @staticmethod
    def _apply_column_action(
        df: pd.DataFrame,
        column_name: str,
        strategy: str,
    ) -> None:
        """Apply one missing-value strategy to one column in ``df``."""
        if strategy == MissingValueStrategy.MEAN_IMPUTATION.value:
            df[column_name] = df[column_name].fillna(df[column_name].mean())
            return

        if strategy == MissingValueStrategy.MEDIAN_IMPUTATION.value:
            df[column_name] = df[column_name].fillna(df[column_name].median())
            return

        if strategy == MissingValueStrategy.MODE_IMPUTATION.value:
            mode = df[column_name].mode(dropna=True)
            if not mode.empty:
                df[column_name] = df[column_name].fillna(mode.iloc[0])
            return

        if strategy == MissingValueStrategy.DROP_COLUMN.value:
            df.drop(columns=[column_name], inplace=True)
            return

        raise ValueError(f"Unsupported missing-value strategy: {strategy}")
