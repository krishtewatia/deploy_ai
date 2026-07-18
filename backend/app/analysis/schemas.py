"""Pydantic schemas for dataset analysis reports."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class MissingValueReport(BaseModel):
    """Report detailing missing values across dataset columns."""

    total_missing: int = Field(
        ...,
        ge=0,
        description="Total number of missing cells across all columns."
    )
    missing_by_column: Dict[str, int] = Field(
        ...,
        description="Mapping of column names to the count of their missing values."
    )
    missing_percentage: Dict[str, float] = Field(
        ...,
        description="Mapping of column names to the percentage of their missing values."
    )


class DuplicateReport(BaseModel):
    """Report detailing duplicate rows in the dataset."""

    duplicate_rows: int = Field(
        ...,
        ge=0,
        description="Total number of duplicate rows found in the dataset."
    )
    duplicate_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of the dataset rows that are duplicates."
    )


class StatisticsReport(BaseModel):
    """Report summarizing descriptive statistics for numerical columns."""

    numerical_summary: Dict[str, Dict[str, float]] = Field(
        ...,
        description=(
            "Summary statistics mapping numerical column names to descriptive statistical metrics "
            "(e.g., mean, median, std, min, max)."
        )
    )


class ImbalanceReport(BaseModel):
    """Report detailing label/class imbalance for categorical target variables."""

    imbalanced: bool = Field(
        ...,
        description="Flag indicating if class imbalance is detected based on threshold."
    )
    distribution: Dict[str, int] = Field(
        ...,
        description="Value counts mapping category labels to their occurrence frequencies."
    )


class DatasetAnalysisReport(BaseModel):
    """Aggregated analysis report summarizing the quality and statistics of a dataset."""

    missing_values: MissingValueReport = Field(
        ...,
        description="Summary of missing values across all columns."
    )
    duplicates: DuplicateReport = Field(
        ...,
        description="Summary of duplicate rows found in the dataset."
    )
    statistics: StatisticsReport = Field(
        ...,
        description="Descriptive statistics summary of the numerical columns."
    )
    imbalance: Optional[ImbalanceReport] = Field(
        default=None,
        description="Optional class imbalance report (only generated when a target variable is analyzed)."
    )
    column_profiles: dict[str, dict] = Field(
        ...,
        description="Detailed profiling information for each column in the dataset."
    )
