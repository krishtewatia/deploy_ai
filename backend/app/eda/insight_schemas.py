"""Pydantic v2 schemas for the AI-powered Insight Generation module.

This module defines the data contract for insight reports produced by
the :class:`~backend.app.eda.insight_generator.InsightGenerator`.  Each
report captures four categories of AI-generated analysis:

* **Descriptive insights** — patterns, data quality, distributions.
* **Diagnostic insights** — root causes, correlations, anomalies.
* **Predictive observations** — risks, trends, modelling concerns.
* **Prescriptive recommendations** — cleaning, engineering, next steps.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class InsightReport(BaseModel):
    """AI-generated insight report for a specific dataset.

    The report is produced by sending deterministic EDA findings to an
    LLM (via Groq) and structuring the response into four insight
    categories.  The ``generated_at`` timestamp and ``dataset_id``
    anchor the report in time and to its source dataset.
    """

    model_config = {"frozen": False, "populate_by_name": True}

    descriptive_insights: list[str] = Field(
        default_factory=list,
        description="AI-generated descriptive insights about patterns, data quality, and distributions.",
    )
    diagnostic_insights: list[str] = Field(
        default_factory=list,
        description="AI-generated diagnostic insights about causes, correlations, and anomalies.",
    )
    predictive_observations: list[str] = Field(
        default_factory=list,
        description="AI-generated predictive observations about risks, trends, and modelling concerns.",
    )
    prescriptive_recommendations: list[str] = Field(
        default_factory=list,
        description="AI-generated prescriptive recommendations for cleaning, engineering, and next steps.",
    )
    generated_at: datetime = Field(
        ...,
        description="UTC timestamp when the insight report was generated.",
    )
    dataset_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the dataset this report belongs to.",
    )
