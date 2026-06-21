"""Pydantic v2 schemas for the EDA Orchestration Service.

This module defines the consolidated report returned by the EDA orchestration
service, representing the complete output of the Exploratory Data Analysis
pipeline.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from backend.app.eda.insight_schemas import InsightReport
from backend.app.eda.schemas import (
    DescriptiveAnalytics,
    DiagnosticAnalytics,
    VisualizationRecommendation,
)


class EDAServiceReport(BaseModel):
    """Aggregated EDA report bundling all pipeline stages.

    This schema consolidates the output of descriptive profiling,
    diagnostic analytics, recommended visualizations, generated chart
    filepaths, and AI-powered natural language insights for a specific
    dataset.
    """

    model_config = {"frozen": False, "populate_by_name": True}

    dataset_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the dataset this report belongs to.",
    )
    descriptive: DescriptiveAnalytics = Field(
        ...,
        description="Descriptive statistics and key data quality findings.",
    )
    diagnostic: DiagnosticAnalytics = Field(
        ...,
        description="Correlation and anomaly insights.",
    )
    visualizations: list[VisualizationRecommendation] = Field(
        default_factory=list,
        description="Recommended visualizations for exploring the dataset.",
    )
    generated_charts: list[str] = Field(
        default_factory=list,
        description="Absolute file paths of the generated chart PNG images.",
    )
    insights: InsightReport = Field(
        ...,
        description="AI-generated insights and recommendations.",
    )
    generated_at: datetime = Field(
        ...,
        description="UTC timestamp indicating when this consolidated report was generated.",
    )
