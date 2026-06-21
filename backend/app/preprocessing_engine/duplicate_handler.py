"""Duplicate-row preprocessing execution utilities."""

from __future__ import annotations

import logging

import pandas as pd

from backend.app.preprocessing_engine.schemas import DuplicateStrategy

logger = logging.getLogger(__name__)


class DuplicateHandlerError(RuntimeError):
    """Raised when duplicate-row preprocessing cannot be completed."""


class DuplicateHandler:
    """Apply duplicate-row preprocessing strategies to a pandas DataFrame."""

    def apply(
        self,
        df: pd.DataFrame,
        strategy: str,
    ) -> pd.DataFrame:
        """Apply a duplicate-row strategy and return a new DataFrame.

        Args:
            df: Input DataFrame to transform.
            strategy: Duplicate-row strategy identifier.

        Returns:
            A transformed copy of ``df``. The original DataFrame is never
            mutated.

        Raises:
            DuplicateHandlerError: If ``strategy`` is unsupported.
        """
        logger.info("Applying duplicate-row preprocessing strategy: %s", strategy)
        result = df.copy(deep=True)

        if strategy == DuplicateStrategy.REMOVE_DUPLICATES.value:
            deduplicated = result.drop_duplicates()
            logger.info(
                "Duplicate-row preprocessing removed %d rows.",
                len(result) - len(deduplicated),
            )
            return deduplicated

        if strategy == DuplicateStrategy.KEEP_DUPLICATES.value:
            logger.info("Duplicate rows retained by strategy.")
            return result

        logger.error("Unsupported duplicate-row strategy: %s", strategy)
        raise DuplicateHandlerError(f"Unsupported duplicate-row strategy: {strategy}")
