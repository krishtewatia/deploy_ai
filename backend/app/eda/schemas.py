"""Pydantic v2 schemas for the Exploratory Data Analysis (EDA) module.

This module defines the data contracts used to represent EDA results
including dataset summaries, descriptive and diagnostic analytics,
feature engineering suggestions, and visualization recommendations.
All enums use ``str`` bases for JSON-friendly serialisation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────────────────────


class Severity(str, Enum):
    """Severity level for an analytical insight."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Priority(str, Enum):
    """Priority level for a feature engineering suggestion."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChartType(str, Enum):
    """Supported visualization chart types."""

    HISTOGRAM = "histogram"
    BOXPLOT = "boxplot"
    SCATTERPLOT = "scatterplot"
    CORRELATION_HEATMAP = "correlation_heatmap"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    LINE_CHART = "line_chart"
    MISSING_VALUE_HEATMAP = "missing_value_heatmap"


# ── Component models ───────────────────────────────────────────────────────


class DatasetSummary(BaseModel):
    """High-level statistical summary of the dataset."""

    model_config = {"frozen": False, "populate_by_name": True}

    rows: int = Field(
        ...,
        ge=0,
        description="Total number of rows in the dataset.",
    )
    columns: int = Field(
        ...,
        ge=0,
        description="Total number of columns in the dataset.",
    )
    numerical_columns: list[str] = Field(
        default_factory=list,
        description="Names of columns with numerical dtypes.",
    )
    categorical_columns: list[str] = Field(
        default_factory=list,
        description="Names of columns with categorical or object dtypes.",
    )
    datetime_columns: list[str] = Field(
        default_factory=list,
        description="Names of columns with datetime dtypes.",
    )
    missing_cells: int = Field(
        ...,
        ge=0,
        description="Total number of missing cells across the entire dataset.",
    )
    duplicate_rows: int = Field(
        ...,
        ge=0,
        description="Number of duplicate rows detected in the dataset.",
    )


class Insight(BaseModel):
    """A single analytical insight produced during EDA.

    Insights are used in both descriptive and diagnostic analytics to
    communicate findings to the user with an associated severity level.
    """

    model_config = {"frozen": False, "populate_by_name": True, "use_enum_values": True}

    title: str = Field(
        ...,
        min_length=1,
        description="Short title summarising the insight.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Detailed explanation of the insight.",
    )
    severity: Severity = Field(
        ...,
        description="Severity level: 'info', 'warning', or 'critical'.",
    )


class VisualizationRecommendation(BaseModel):
    """A recommended visualization for exploring the dataset."""

    model_config = {"frozen": False, "populate_by_name": True, "use_enum_values": True}

    chart_type: ChartType = Field(
        ...,
        description=(
            "Type of chart recommended (e.g. 'histogram', 'boxplot', "
            "'scatterplot')."
        ),
    )
    column_names: list[str] = Field(
        ...,
        min_length=1,
        description="Column names involved in this visualization.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Human-readable rationale for recommending this chart.",
    )


class FeatureSuggestion(BaseModel):
    """A suggested engineered feature derived from existing columns."""

    model_config = {"frozen": False, "populate_by_name": True, "use_enum_values": True}

    feature_name: str = Field(
        ...,
        min_length=1,
        description="Proposed name for the new feature.",
    )
    source_columns: list[str] = Field(
        ...,
        min_length=1,
        description="Existing columns from which the feature is derived.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Explanation of how the feature should be constructed.",
    )
    priority: Priority = Field(
        ...,
        description="Priority level: 'low', 'medium', or 'high'.",
    )


# ── Composite analytics models ─────────────────────────────────────────────


class DescriptiveAnalytics(BaseModel):
    """Container for descriptive analytics produced during EDA.

    Includes the high-level dataset summary and any key findings
    surfaced from the descriptive pass.
    """

    model_config = {"frozen": False, "populate_by_name": True}

    dataset_summary: DatasetSummary = Field(
        ...,
        description="Statistical summary of the dataset.",
    )
    key_findings: list[Insight] = Field(
        default_factory=list,
        description="Key insights discovered during descriptive analysis.",
    )


class DiagnosticAnalytics(BaseModel):
    """Container for diagnostic analytics produced during EDA.

    Separates correlation-related findings from anomaly-related ones
    so that consumers can present them in distinct UI sections.
    """

    model_config = {"frozen": False, "populate_by_name": True}

    correlation_findings: list[Insight] = Field(
        default_factory=list,
        description="Insights relating to feature correlations.",
    )
    anomaly_findings: list[Insight] = Field(
        default_factory=list,
        description="Insights relating to detected anomalies or outliers.",
    )


# ── Top-level report ───────────────────────────────────────────────────────


class EDAReport(BaseModel):
    """Top-level EDA report aggregating all analytics artefacts.

    This is the main return type of the EDA service, bundling
    descriptive analytics, diagnostic analytics, feature suggestions,
    visualization recommendations, and an overall AI-generated summary.
    """

    model_config = {"frozen": False, "populate_by_name": True}

    descriptive: DescriptiveAnalytics = Field(
        ...,
        description="Descriptive analytics section of the report.",
    )
    diagnostic: DiagnosticAnalytics = Field(
        ...,
        description="Diagnostic analytics section of the report.",
    )
    feature_suggestions: list[FeatureSuggestion] = Field(
        default_factory=list,
        description="AI-generated feature engineering suggestions.",
    )
    visualization_recommendations: list[VisualizationRecommendation] = Field(
        default_factory=list,
        description="Recommended visualizations for exploring the dataset.",
    )
    overall_summary: str = Field(
        ...,
        min_length=1,
        description="AI-generated natural-language summary of the EDA findings.",
    )
