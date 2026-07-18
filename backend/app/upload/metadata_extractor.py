"""Metadata extraction utilities for uploaded datasets."""

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from backend.app.upload.schemas import DatasetMetadata


class MetadataExtractor:
    """Extract structured metadata from uploaded dataset DataFrames."""

    def extract(self, df: pd.DataFrame, file_name: str) -> DatasetMetadata:
        """Extract dataset metadata from a DataFrame.

        Args:
            df: DataFrame to inspect.
            file_name: Original name of the uploaded dataset file.

        Returns:
            DatasetMetadata object containing shape and inferred column categories.
        """
        numerical_columns = self._get_numerical_columns(df)
        datetime_columns = self._get_datetime_columns(df)
        categorical_columns = self._get_categorical_columns(df)

        return DatasetMetadata(
            file_name=file_name,
            rows=df.shape[0],
            columns=df.shape[1],
            column_names=df.columns.tolist(),
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            target_column=None,
        )

    def _get_numerical_columns(self, df: pd.DataFrame) -> list[str]:
        """Return names of columns with numeric dtypes."""
        return [column for column in df.columns if is_numeric_dtype(df[column])]

    def _get_datetime_columns(self, df: pd.DataFrame) -> list[str]:
        """Return names of columns with datetime dtypes."""
        return [column for column in df.columns if is_datetime64_any_dtype(df[column])]

    def _get_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        """Return names of columns with categorical, object, or string dtypes."""
        return [
            column
            for column in df.columns
            if isinstance(df[column].dtype, pd.CategoricalDtype)
            or is_object_dtype(df[column])
            or is_string_dtype(df[column])
        ]

