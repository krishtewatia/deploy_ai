"""Pydantic v2 schemas for dataset intelligence.

These models represent compact, machine-readable dataset metadata
designed to be:

- consumed by deterministic ML planning logic,
- serialized to JSON for AI provider consumption,
- compact enough for local AI models with limited context windows.

This module defines data contracts only.  It does NOT contain any
analysis execution logic, DataFrame processing, or AI integration.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 1. DatasetBasicInfo
# ---------------------------------------------------------------------------


class DatasetBasicInfo(BaseModel):
    """Basic dataset-level metadata."""

    dataset_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the dataset.",
    )
    file_name: str = Field(
        ...,
        min_length=1,
        description="Original file name of the uploaded dataset.",
    )
    row_count: int = Field(
        ...,
        ge=0,
        description="Total number of rows in the dataset.",
    )
    column_count: int = Field(
        ...,
        ge=0,
        description="Total number of columns in the dataset.",
    )
    memory_usage_bytes: int = Field(
        ...,
        ge=0,
        description="Approximate in-memory size in bytes.",
    )


# ---------------------------------------------------------------------------
# 2. ColumnStatistics
# ---------------------------------------------------------------------------


class ColumnStatistics(BaseModel):
    """Optional compact numerical statistics for a column.

    All fields default to ``None`` so the model can represent columns
    where statistics are unavailable or inapplicable.
    """

    mean: Optional[float] = Field(
        default=None, description="Arithmetic mean."
    )
    median: Optional[float] = Field(
        default=None, description="Median (50th percentile)."
    )
    std: Optional[float] = Field(
        default=None, description="Standard deviation."
    )
    min: Optional[float] = Field(
        default=None, description="Minimum value."
    )
    max: Optional[float] = Field(
        default=None, description="Maximum value."
    )


# ---------------------------------------------------------------------------
# 3. ColumnContext
# ---------------------------------------------------------------------------


class ColumnContext(BaseModel):
    """Machine-readable profile for a single dataset column."""

    name: str = Field(
        ...,
        min_length=1,
        description="Column name.",
    )
    dtype: str = Field(
        ...,
        min_length=1,
        description="String representation of the column data type.",
    )

    is_numeric: bool = Field(
        ..., description="Whether the column holds numeric data."
    )
    is_categorical: bool = Field(
        ..., description="Whether the column holds categorical data."
    )
    is_datetime: bool = Field(
        ..., description="Whether the column holds datetime data."
    )

    missing_count: int = Field(
        ...,
        ge=0,
        description="Number of missing (null/NaN) values.",
    )
    missing_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of missing values (0-100).",
    )

    unique_count: int = Field(
        ...,
        ge=0,
        description="Number of unique non-null values.",
    )
    unique_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of unique values relative to total rows (0-100).",
    )

    sample_values: list[Any] = Field(
        ...,
        description="A small list of representative sample values (JSON serializable).",
    )

    statistics: Optional[ColumnStatistics] = Field(
        default=None,
        description="Compact numerical statistics (only for numeric columns).",
    )

    @model_validator(mode="after")
    def _validate_sample_values_serializable(self) -> ColumnContext:
        """Ensure every element in ``sample_values`` is JSON serializable."""
        try:
            json.dumps(self.sample_values)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sample_values must be JSON serializable."
            ) from exc
        return self


# ---------------------------------------------------------------------------
# 4. MissingDataSummary
# ---------------------------------------------------------------------------


class MissingDataSummary(BaseModel):
    """Dataset-wide summary of missing data."""

    total_missing_cells: int = Field(
        ...,
        ge=0,
        description="Total number of missing cells across all columns.",
    )
    columns_with_missing: list[str] = Field(
        ...,
        description="List of column names that contain at least one missing value.",
    )


# ---------------------------------------------------------------------------
# 5. DuplicateSummary
# ---------------------------------------------------------------------------


class DuplicateSummary(BaseModel):
    """Summary of duplicate rows in the dataset."""

    duplicate_rows: int = Field(
        ...,
        ge=0,
        description="Total number of duplicate rows.",
    )
    duplicate_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of duplicate rows (0-100).",
    )


# ---------------------------------------------------------------------------
# 6. TargetCandidateSummary
# ---------------------------------------------------------------------------


class TargetCandidateSummary(BaseModel):
    """Deterministic target-variable candidate metadata.

    This does NOT confirm the column as the ML target.  It only
    records that deterministic heuristics flagged it as a candidate.
    """

    column_name: str = Field(
        ...,
        min_length=1,
        description="Name of the candidate target column.",
    )
    unique_count: int = Field(
        ...,
        ge=0,
        description="Number of unique values in the candidate column.",
    )
    unique_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Unique-value percentage relative to total rows (0-100).",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Short deterministic rationale for candidate status.",
    )


# ---------------------------------------------------------------------------
# 7. DatasetContext
# ---------------------------------------------------------------------------


class DatasetContext(BaseModel):
    """Top-level dataset intelligence container.

    Aggregates all compact, machine-readable metadata required for
    downstream ML planning and execution.
    """

    schema_version: str = Field(
        default="1.0",
        description="Semantic version of this schema.",
    )
    basic_info: DatasetBasicInfo = Field(
        ..., description="High-level dataset metadata."
    )
    columns: list[ColumnContext] = Field(
        ...,
        min_length=1,
        description="Per-column context profiles (must contain at least one column).",
    )
    missing_data: MissingDataSummary = Field(
        ..., description="Dataset-wide missing-data summary."
    )
    duplicates: DuplicateSummary = Field(
        ..., description="Duplicate-row summary."
    )
    target_candidates: list[TargetCandidateSummary] = Field(
        default_factory=list,
        description="Deterministic target-variable candidates (may be empty).",
    )

    @model_validator(mode="after")
    def _reject_duplicate_column_names(self) -> DatasetContext:
        """Ensure no two columns share the same name."""
        names = [col.name for col in self.columns]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate column names are not allowed: {sorted(duplicates)}"
            )
        return self
