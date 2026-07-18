"""Pydantic v2 schemas for the AI Recommendation Engine.

This module defines the data contracts used to communicate between the
dataset analysis layer and the AI recommendation engine.  Every schema
uses strict Pydantic v2 conventions: ``model_config``, ``Field``
descriptions, and full type annotations.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class ColumnRecommendation(BaseModel):
    """Recommendation for a single column transformation.

    Encapsulates *what* strategy should be applied to a column and *why*
    the engine chose it.

    Example::

        {
            "strategy": "median_imputation",
            "reason": "Column contains outliers and is right-skewed."
        }
    """

    model_config = {"frozen": False, "populate_by_name": True}

    strategy: str = Field(
        ...,
        min_length=1,
        description=(
            "Machine-readable identifier for the recommended preprocessing "
            "strategy (e.g. 'median_imputation', 'one_hot_encoding', "
            "'standard_scaling')."
        ),
        examples=["median_imputation", "one_hot_encoding", "min_max_scaling"],
    )
    reason: str = Field(
        ...,
        min_length=1,
        description=(
            "Human-readable explanation of why this strategy was selected "
            "for the column."
        ),
        examples=["Column contains outliers and is right-skewed."],
    )


class CleaningPlan(BaseModel):
    """Full cleaning / preprocessing plan returned by the AI engine.

    Groups recommendations by transformation category so the downstream
    pipeline can apply them in the correct order.
    """

    model_config = {"frozen": False, "populate_by_name": True}

    missing_values: Dict[str, ColumnRecommendation] = Field(
        default_factory=dict,
        description=(
            "Per-column strategies for handling missing values.  Keys are "
            "column names; values describe the imputation strategy and "
            "reasoning."
        ),
    )
    duplicates_action: str = Field(
        ...,
        min_length=1,
        description=(
            "Global action to take on duplicate rows "
            "(e.g. 'drop', 'keep_first', 'keep_last', 'flag')."
        ),
        examples=["drop", "keep_first", "flag"],
    )
    encoding: Dict[str, ColumnRecommendation] = Field(
        default_factory=dict,
        description=(
            "Per-column encoding strategies for categorical features.  "
            "Keys are column names; values describe the encoding method "
            "and reasoning."
        ),
    )
    scaling: Dict[str, ColumnRecommendation] = Field(
        default_factory=dict,
        description=(
            "Per-column scaling/normalisation strategies for numerical "
            "features.  Keys are column names; values describe the scaling "
            "method and reasoning."
        ),
    )


class RecommendationResponse(BaseModel):
    """Top-level response envelope returned by the recommendation endpoint.

    Wraps the :class:`CleaningPlan` together with a free-text summary so
    that consumers receive both structured actions *and* a natural-language
    rationale.
    """

    model_config = {"frozen": False, "populate_by_name": True}

    cleaning_plan: CleaningPlan = Field(
        ...,
        description="The full cleaning and preprocessing plan.",
    )
    overall_reasoning: str = Field(
        ...,
        min_length=1,
        description=(
            "High-level natural-language summary explaining the overall "
            "preprocessing strategy chosen by the AI engine."
        ),
        examples=[
            "The dataset is moderately clean with 5% missing values "
            "concentrated in two columns.  Median imputation is preferred "
            "due to skewed distributions."
        ],
    )
