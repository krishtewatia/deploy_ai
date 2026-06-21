"""Orchestration service for end-to-end DataFrame preprocessing.

This module provides the single entry-point that wires together the
analysis layer, AI recommendation engine, and preprocessing pipeline into
one cohesive workflow:

.. code-block:: text

    DataFrame
      ↓  AnalysisService
    DatasetAnalysisReport
      ↓  RecommendationService
    RecommendationResponse
      ↓  convert to ExecutionPlan
    ExecutionPlan
      ↓  PipelineExecutor
    Processed DataFrame
      ↓  wrap
    PreprocessingResult

Usage::

    from backend.app.preprocessing_engine.preprocessing_service import (
        PreprocessingService,
    )

    service = PreprocessingService()
    result  = service.process(df)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from backend.app.ai_engine.recommendation_service import RecommendationService
from backend.app.ai_engine.schemas import CleaningPlan, RecommendationResponse
from backend.app.analysis.analysis_service import AnalysisService
from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.preprocessing_engine.pipeline_executor import PipelineExecutor
from backend.app.preprocessing_engine.result_schemas import PreprocessingResult
from backend.app.preprocessing_engine.schemas import (
    ColumnAction,
    ExecutionPlan,
)

logger = logging.getLogger(__name__)


# ── Custom exception ────────────────────────────────────────────────────────


class PreprocessingServiceError(Exception):
    """Raised when any stage of the preprocessing orchestration fails."""


# ── Service ─────────────────────────────────────────────────────────────────


class PreprocessingService:
    """End-to-end orchestrator that analyses, recommends, and executes
    preprocessing transformations on a pandas DataFrame.

    All heavyweight dependencies are injected via the constructor so that
    each component can be replaced or mocked independently in tests.

    Parameters
    ----------
    analysis_service:
        Service that produces a :class:`DatasetAnalysisReport`.
    recommendation_service:
        Service that produces a :class:`RecommendationResponse` from an
        analysis report.
    pipeline_executor:
        Executor that applies an :class:`ExecutionPlan` to a DataFrame.
    """

    def __init__(
        self,
        analysis_service: Optional[AnalysisService] = None,
        recommendation_service: Optional[RecommendationService] = None,
        pipeline_executor: Optional[PipelineExecutor] = None,
    ) -> None:
        """Initialise the service with optional dependency injection."""
        self._analysis_service = analysis_service or AnalysisService()
        self._recommendation_service = (
            recommendation_service or RecommendationService()
        )
        self._pipeline_executor = pipeline_executor or PipelineExecutor()
        logger.info("PreprocessingService initialised.")

    # ── public API ──────────────────────────────────────────────────────

    def process(
        self,
        df: pd.DataFrame,
    ) -> PreprocessingResult:
        """Run the full preprocessing pipeline on *df*.

        Steps
        -----
        1. Generate a :class:`DatasetAnalysisReport`.
        2. Generate a :class:`RecommendationResponse`.
        3. Convert the cleaning plan into an :class:`ExecutionPlan`.
        4. Execute the plan via :class:`PipelineExecutor`.
        5. Build the ``transformations_applied`` list.
        6. Generate a preview from the processed DataFrame.
        7. Return a :class:`PreprocessingResult`.

        Parameters
        ----------
        df:
            The raw pandas DataFrame to preprocess.

        Returns
        -------
        PreprocessingResult
            A result object containing every artefact produced during
            the pipeline.

        Raises
        ------
        PreprocessingServiceError
            Wraps any exception raised by the analysis, recommendation,
            or pipeline execution stages.
        """
        logger.info("Starting preprocessing service pipeline.")
        original_shape: tuple[int, int] = (df.shape[0], df.shape[1])

        # Step 1 — Analysis
        analysis_report = self._run_analysis(df)

        # Step 2 — Recommendation
        recommendation = self._run_recommendation(analysis_report)

        # Step 3 — Convert to ExecutionPlan
        execution_plan = self._build_execution_plan(recommendation.cleaning_plan)

        # Step 4 — Execute pipeline
        processed_df = self._run_pipeline(df, execution_plan)

        # Step 5 — Build transformation list
        transformations_applied = self._build_transformations_applied(
            execution_plan,
        )

        # Step 6 — Preview
        preview = self._build_preview(processed_df)

        processed_shape: tuple[int, int] = (
            processed_df.shape[0],
            processed_df.shape[1],
        )

        logger.info("Preprocessing service pipeline completed successfully.")
        return PreprocessingResult(
            original_shape=original_shape,
            processed_shape=processed_shape,
            analysis_report=analysis_report,
            recommendation=recommendation,
            execution_plan=execution_plan,
            transformations_applied=transformations_applied,
            preview=preview,
        )

    # ── private helpers ─────────────────────────────────────────────────

    def _run_analysis(
        self,
        df: pd.DataFrame,
    ) -> DatasetAnalysisReport:
        """Execute the analysis stage, wrapping failures."""
        try:
            logger.info("Step 1/4: Generating dataset analysis report.")
            report = self._analysis_service.analyze(df)
            logger.info("Step 1/4: Analysis report generated successfully.")
            return report
        except Exception as exc:
            logger.exception("Preprocessing pipeline failed during analysis stage.")
            raise PreprocessingServiceError(
                f"Analysis stage failed: {exc}"
            ) from exc

    def _run_recommendation(
        self,
        analysis_report: DatasetAnalysisReport,
    ) -> RecommendationResponse:
        """Execute the recommendation stage, wrapping failures."""
        try:
            logger.info("Step 2/4: Generating AI recommendations.")
            recommendation = self._recommendation_service.generate_recommendations(
                analysis_report,
            )
            logger.info("Step 2/4: Recommendations generated successfully.")
            return recommendation
        except Exception as exc:
            logger.exception(
                "Preprocessing pipeline failed during recommendation stage."
            )
            raise PreprocessingServiceError(
                f"Recommendation stage failed: {exc}"
            ) from exc

    @staticmethod
    def _build_execution_plan(cleaning_plan: CleaningPlan) -> ExecutionPlan:
        """Convert a :class:`CleaningPlan` into an :class:`ExecutionPlan`.

        The two schemas share the same structural shape; the conversion
        maps each :class:`ColumnRecommendation` to a :class:`ColumnAction`.
        """
        logger.info("Step 3/4: Converting cleaning plan to execution plan.")

        missing_values: dict[str, ColumnAction] = {
            col: ColumnAction(strategy=rec.strategy, reason=rec.reason)
            for col, rec in cleaning_plan.missing_values.items()
        }
        encoding: dict[str, ColumnAction] = {
            col: ColumnAction(strategy=rec.strategy, reason=rec.reason)
            for col, rec in cleaning_plan.encoding.items()
        }
        scaling: dict[str, ColumnAction] = {
            col: ColumnAction(strategy=rec.strategy, reason=rec.reason)
            for col, rec in cleaning_plan.scaling.items()
        }

        plan = ExecutionPlan(
            missing_values=missing_values,
            duplicates_action=cleaning_plan.duplicates_action,
            encoding=encoding,
            scaling=scaling,
        )
        logger.info("Step 3/4: Execution plan built successfully.")
        return plan

    def _run_pipeline(
        self,
        df: pd.DataFrame,
        execution_plan: ExecutionPlan,
    ) -> pd.DataFrame:
        """Execute the preprocessing pipeline, wrapping failures."""
        try:
            logger.info("Step 4/4: Executing preprocessing pipeline.")
            processed_df = self._pipeline_executor.execute(df, execution_plan)
            logger.info("Step 4/4: Pipeline executed successfully.")
            return processed_df
        except Exception as exc:
            logger.exception(
                "Preprocessing pipeline failed during execution stage."
            )
            raise PreprocessingServiceError(
                f"Pipeline execution stage failed: {exc}"
            ) from exc

    @staticmethod
    def _build_transformations_applied(
        plan: ExecutionPlan,
    ) -> list[str]:
        """Build a human-readable list of transformations from *plan*.

        Examples::

            [
                "median_imputation: salary",
                "remove_duplicates",
                "one_hot_encode: department",
                "standard_scaling: salary",
            ]
        """
        transformations: list[str] = []

        for col, action in plan.missing_values.items():
            transformations.append(f"{action.strategy}: {col}")

        duplicates_val = (
            plan.duplicates_action.value
            if hasattr(plan.duplicates_action, "value")
            else str(plan.duplicates_action)
        )
        transformations.append(duplicates_val)

        for col, action in plan.encoding.items():
            transformations.append(f"{action.strategy}: {col}")

        for col, action in plan.scaling.items():
            transformations.append(f"{action.strategy}: {col}")

        return transformations

    @staticmethod
    def _build_preview(
        df: pd.DataFrame,
        n_rows: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the first *n_rows* rows of *df* as a list of dicts."""
        return df.head(n_rows).to_dict(orient="records")  # type: ignore[return-value]
