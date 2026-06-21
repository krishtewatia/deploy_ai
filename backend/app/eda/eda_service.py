"""Orchestration service for conducting full Exploratory Data Analysis (EDA)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.eda.chart_generator import ChartGenerator
from backend.app.eda.descriptive_analyzer import DescriptiveAnalyzer
from backend.app.eda.diagnostic_analyzer import DiagnosticAnalyzer
from backend.app.eda.eda_service_schemas import EDAServiceReport
from backend.app.eda.insight_generator import InsightGenerator
from backend.app.eda.visualization_recommender import VisualizationRecommender

logger = logging.getLogger(__name__)


# ── Custom Exception ───────────────────────────────────────────────────────


class EDAServiceError(Exception):
    """Raised when any stage of the EDA orchestration pipeline fails."""


# ── Orchestration Service ──────────────────────────────────────────────────


class EDAService:
    """Orchestrator for the complete Exploratory Data Analysis pipeline.

    Coordinates descriptive profiling, diagnostic analysis, visualization
    recommendation, chart rendering, and AI-powered insight generation.
    """

    def __init__(
        self,
        descriptive_analyzer: DescriptiveAnalyzer,
        diagnostic_analyzer: DiagnosticAnalyzer,
        visualization_recommender: VisualizationRecommender,
        chart_generator: ChartGenerator,
        insight_generator: InsightGenerator,
    ) -> None:
        """Initialize the EDAService with its analytical components.

        Parameters
        ----------
        descriptive_analyzer : DescriptiveAnalyzer
            Component to compute descriptive profiles and basic findings.
        diagnostic_analyzer : DiagnosticAnalyzer
            Component to identify correlations and data anomalies.
        visualization_recommender : VisualizationRecommender
            Component to recommend charts for exploring the dataset.
        chart_generator : ChartGenerator
            Component to render and persist visualization images.
        insight_generator : InsightGenerator
            Component to produce AI-generated natural-language reports.
        """
        self.descriptive_analyzer = descriptive_analyzer
        self.diagnostic_analyzer = diagnostic_analyzer
        self.visualization_recommender = visualization_recommender
        self.chart_generator = chart_generator
        self.insight_generator = insight_generator

    def run(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        analysis_report: DatasetAnalysisReport,
    ) -> EDAServiceReport:
        """Execute the complete EDA pipeline sequentially.

        Parameters
        ----------
        dataset_id : str
            Unique identifier of the dataset.
        df : pd.DataFrame
            The pandas DataFrame containing the raw dataset.
        analysis_report : DatasetAnalysisReport
            A validated dataset profiling report from the analysis service.

        Returns
        -------
        EDAServiceReport
            A unified, serializable report containing results from all stages.

        Raises
        ------
        EDAServiceError
            If any input validation or pipeline step fails.
        """
        logger.info("EDA pipeline started")

        # ── 1. Validate inputs
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            logger.error("EDA pipeline failed: Invalid dataset_id.")
            raise EDAServiceError("Invalid dataset_id: must be a non-empty string.")

        if not isinstance(df, pd.DataFrame):
            logger.error("EDA pipeline failed: Input 'df' is not a pandas DataFrame.")
            raise EDAServiceError("Invalid dataframe: must be a pandas DataFrame.")

        if df.empty:
            logger.error("EDA pipeline failed: DataFrame is empty.")
            raise EDAServiceError("Empty dataframe: cannot process empty data.")

        if not isinstance(analysis_report, DatasetAnalysisReport):
            logger.error("EDA pipeline failed: Invalid analysis_report type.")
            raise EDAServiceError("Invalid analysis_report: must be a DatasetAnalysisReport.")

        # ── 2. Configure ChartGenerator path dynamically if it is a real instance
        if hasattr(self.chart_generator, "base_output_dir") and hasattr(self.chart_generator, "dataset_id"):
            try:
                self.chart_generator.dataset_id = dataset_id
                self.chart_generator._dataset_chart_dir = self.chart_generator.base_output_dir / dataset_id
                self.chart_generator.output_dir = self.chart_generator._dataset_chart_dir
                self.chart_generator._dataset_chart_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.exception("Failed to configure chart generator output directory.")
                raise EDAServiceError(f"Failed to configure chart generator output directory: {exc}") from exc

        # ── 3. Step-by-step pipeline execution

        # Step 1: Descriptive Analyzer
        try:
            descriptive = self.descriptive_analyzer.analyze(analysis_report)
            logger.info("Descriptive complete")
        except Exception as exc:
            logger.exception("Descriptive analyzer stage failed.")
            raise EDAServiceError(f"Descriptive analyzer failure: {exc}") from exc

        # Step 2: Diagnostic Analyzer
        try:
            diagnostic = self.diagnostic_analyzer.analyze(df, analysis_report)
            logger.info("Diagnostic complete")
        except Exception as exc:
            logger.exception("Diagnostic analyzer stage failed.")
            raise EDAServiceError(f"Diagnostic analyzer failure: {exc}") from exc

        # Step 3: Visualization Recommender
        try:
            visualizations = self.visualization_recommender.recommend(analysis_report)
            logger.info("Visualization recommendations complete")
        except Exception as exc:
            logger.exception("Visualization recommender stage failed.")
            raise EDAServiceError(f"Visualization recommender failure: {exc}") from exc

        # Step 4: Chart Generator
        try:
            generated_charts = self.chart_generator.generate_charts(df, visualizations)
            logger.info("Charts generated")
        except Exception as exc:
            logger.exception("Chart generation stage failed.")
            raise EDAServiceError(f"Chart generation failure: {exc}") from exc

        # Step 5: Insight Generator
        try:
            insights = self.insight_generator.generate_insights(
                dataset_id=dataset_id,
                descriptive=descriptive,
                diagnostic=diagnostic,
                visualizations=visualizations,
            )
            logger.info("Insights generated")
        except Exception as exc:
            logger.exception("Insight generation stage failed.")
            raise EDAServiceError(f"Insight generation failure: {exc}") from exc

        logger.info("EDA pipeline complete")

        return EDAServiceReport(
            dataset_id=dataset_id,
            descriptive=descriptive,
            diagnostic=diagnostic,
            visualizations=visualizations,
            generated_charts=generated_charts,
            insights=insights,
            generated_at=datetime.now(timezone.utc),
        )
