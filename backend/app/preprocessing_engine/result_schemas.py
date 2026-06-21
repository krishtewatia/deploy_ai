"""Result schemas for the preprocessing orchestration service.

Defines the :class:`PreprocessingResult` model returned by
:meth:`PreprocessingService.process`.  It bundles every artefact produced
during the pipeline — analysis report, AI recommendation, execution plan,
the processed preview, and metadata — into a single, serialisable object.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.app.ai_engine.schemas import RecommendationResponse
from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.preprocessing_engine.schemas import ExecutionPlan


class PreprocessingResult(BaseModel):
    """Immutable result container returned after preprocessing completes.

    Every field is populated by the orchestration service so that callers
    have full visibility into the original data shape, the transformations
    that were applied, and a preview of the processed output.
    """

    model_config = {
        "frozen": False,
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }

    original_shape: tuple[int, int] = Field(
        ...,
        description="Shape ``(rows, columns)`` of the input DataFrame before processing.",
    )
    processed_shape: tuple[int, int] = Field(
        ...,
        description="Shape ``(rows, columns)`` of the output DataFrame after processing.",
    )
    analysis_report: DatasetAnalysisReport = Field(
        ...,
        description="Full dataset analysis report produced during the analysis stage.",
    )
    recommendation: RecommendationResponse = Field(
        ...,
        description="AI-generated recommendation response including the cleaning plan.",
    )
    execution_plan: ExecutionPlan = Field(
        ...,
        description="Validated execution plan derived from the recommendation.",
    )
    transformations_applied: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable list of transformations that were applied, "
            "e.g. ``['median_imputation: salary', 'remove_duplicates']``."
        ),
    )
    preview: list[dict[str, Any]] = Field(
        default_factory=list,
        description="First rows of the processed DataFrame as a list of dictionaries.",
    )
