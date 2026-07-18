"""Validation utilities for uploaded datasets."""

import logging

import pandas as pd


logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Base exception raised when dataset validation fails."""


class EmptyDataFrameError(DatasetValidationError):
    """Raised when a dataset DataFrame is empty."""


class MissingColumnsError(DatasetValidationError):
    """Raised when a dataset DataFrame does not contain any columns."""


class DuplicateColumnNamesError(DatasetValidationError):
    """Raised when a dataset DataFrame contains duplicate column names."""


class DatasetValidator:
    """Validate uploaded dataset DataFrames before downstream processing."""

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate a DataFrame against required dataset integrity rules.

        Args:
            df: DataFrame to validate.

        Raises:
            EmptyDataFrameError: If the DataFrame contains no data.
            MissingColumnsError: If the DataFrame has no columns.
            DuplicateColumnNamesError: If duplicate column names are present.
        """
        logger.debug("Validating uploaded dataset DataFrame.")

        self._validate_not_empty(df)
        self._validate_has_columns(df)
        self._validate_unique_columns(df)

        logger.info("Dataset DataFrame validation completed successfully.")

    def _validate_not_empty(self, df: pd.DataFrame) -> None:
        """Validate that the DataFrame is not empty."""
        if len(df.index) == 0:
            logger.error("Dataset validation failed: DataFrame is empty.")
            raise EmptyDataFrameError("DataFrame must not be empty.")

    def _validate_has_columns(self, df: pd.DataFrame) -> None:
        """Validate that the DataFrame contains at least one column."""
        if len(df.columns) == 0:
            logger.error("Dataset validation failed: DataFrame has no columns.")
            raise MissingColumnsError("DataFrame must contain at least one column.")

    def _validate_unique_columns(self, df: pd.DataFrame) -> None:
        """Validate that all DataFrame column names are unique."""
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()

        if duplicate_columns:
            logger.error(
                "Dataset validation failed: duplicate column names detected: %s",
                duplicate_columns,
            )
            raise DuplicateColumnNamesError(
                f"Duplicate column names detected: {duplicate_columns}"
            )
