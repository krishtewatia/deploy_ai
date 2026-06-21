"""Categorical encoding preprocessing execution utilities."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from backend.app.preprocessing_engine.schemas import EncodingStrategy

logger = logging.getLogger(__name__)


class EncodingHandlerError(RuntimeError):
    """Raised when categorical encoding preprocessing cannot be completed."""


class EncodingHandler:
    """Apply categorical encoding actions to a pandas DataFrame."""

    def apply(
        self,
        df: pd.DataFrame,
        actions: dict[str, Mapping[str, Any]],
    ) -> pd.DataFrame:
        """Apply encoding actions and return a new DataFrame.

        Args:
            df: Input DataFrame to transform.
            actions: Mapping of column names to action payloads containing a
                ``strategy`` key.

        Returns:
            A transformed copy of ``df``. The original DataFrame is never
            mutated.

        Raises:
            EncodingHandlerError: If an unsupported strategy is requested or
                an encoding action fails unexpectedly.
        """
        logger.info("Applying categorical encoding actions.")
        result = df.copy(deep=True)

        if result.empty or not actions:
            logger.info("No categorical encoding actions applied.")
            return result

        for column_name, action in actions.items():
            if column_name not in result.columns:
                logger.warning("Skipping encoding action for unknown column: %s", column_name)
                continue

            strategy = str(action.get("strategy", ""))
            try:
                result = self._apply_column_action(result, column_name, strategy)
            except Exception as exc:
                logger.exception(
                    "Failed to apply encoding strategy '%s' to column '%s'.",
                    strategy,
                    column_name,
                )
                raise EncodingHandlerError(
                    f"Failed to apply encoding strategy '{strategy}' "
                    f"to column '{column_name}'."
                ) from exc

        logger.info("Categorical encoding completed successfully.")
        return result

    @staticmethod
    def _apply_column_action(
        df: pd.DataFrame,
        column_name: str,
        strategy: str,
    ) -> pd.DataFrame:
        """Apply one encoding strategy to one column in ``df``."""
        if strategy == EncodingStrategy.ONE_HOT_ENCODE.value:
            return pd.get_dummies(df, columns=[column_name], prefix=column_name, dtype=int)

        if strategy == EncodingStrategy.LABEL_ENCODE.value:
            result = df.copy(deep=True)
            non_null_values = result[column_name].dropna().unique()
            labels = {
                value: index
                for index, value in enumerate(sorted(non_null_values, key=str))
            }
            result[column_name] = result[column_name].map(labels)
            return result

        raise ValueError(f"Unsupported encoding strategy: {strategy}")
