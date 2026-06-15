"""Pydantic schemas for dataset upload metadata."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata describing an uploaded dataset and its inferred column types."""

    file_name: str = Field(..., description="Original name of the uploaded dataset file.")
    rows: int = Field(..., ge=0, description="Number of rows in the dataset.")
    columns: int = Field(..., ge=0, description="Number of columns in the dataset.")
    column_names: List[str] = Field(..., description="Names of all columns in the dataset.")
    numerical_columns: List[str] = Field(
        ...,
        description="Names of columns detected as numerical.",
    )
    categorical_columns: List[str] = Field(
        ...,
        description="Names of columns detected as categorical.",
    )
    datetime_columns: List[str] = Field(
        ...,
        description="Names of columns detected as datetime values.",
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Optional target column selected for model training.",
    )
