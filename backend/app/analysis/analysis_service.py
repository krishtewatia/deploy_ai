"""Orchestration service for conducting full dataset analysis."""

import logging
from typing import Optional
import pandas as pd

from backend.app.analysis.duplicates import DuplicateAnalyzer
from backend.app.analysis.imbalance import ImbalanceAnalyzer
from backend.app.analysis.missing_values import MissingValueAnalyzer
from backend.app.analysis.report_generator import ReportGenerator
from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.analysis.statistics import StatisticsAnalyzer

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service to coordinate all dataset analysis tools and generate a consolidated report."""

    def __init__(
        self,
        missing_value_analyzer: Optional[MissingValueAnalyzer] = None,
        duplicate_analyzer: Optional[DuplicateAnalyzer] = None,
        statistics_analyzer: Optional[StatisticsAnalyzer] = None,
        imbalance_analyzer: Optional[ImbalanceAnalyzer] = None,
        report_generator: Optional[ReportGenerator] = None,
    ) -> None:
        """Initialize AnalysisService with dependency injection.

        Args:
            missing_value_analyzer: Analyzer for missing values.
            duplicate_analyzer: Analyzer for duplicates.
            statistics_analyzer: Analyzer for numerical statistics.
            imbalance_analyzer: Analyzer for target class imbalance.
            report_generator: Generator to compile reports.
        """
        self.missing_value_analyzer = missing_value_analyzer or MissingValueAnalyzer()
        self.duplicate_analyzer = duplicate_analyzer or DuplicateAnalyzer()
        self.statistics_analyzer = statistics_analyzer or StatisticsAnalyzer()
        self.imbalance_analyzer = imbalance_analyzer or ImbalanceAnalyzer()
        self.report_generator = report_generator or ReportGenerator()

    def analyze(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> DatasetAnalysisReport:
        """Run all data profiling analyses and return a consolidated report.

        Args:
            df: The pandas DataFrame to analyze.
            target_column: The optional target column name for imbalance analysis.

        Returns:
            A DatasetAnalysisReport containing statistics and checks.

        Raises:
            ValueError: If the DataFrame is empty or if validation fails.
        """
        logger.info("Starting dataset analysis pipeline.")

        if df.empty:
            logger.error("Analysis pipeline failed: DataFrame is empty.")
            raise ValueError("DataFrame is empty. Cannot perform analysis.")

        # 1. Run MissingValueAnalyzer
        missing_values_report = self.missing_value_analyzer.analyze(df)

        # 2. Run DuplicateAnalyzer
        duplicates_report = self.duplicate_analyzer.analyze(df)

        # 3. Run StatisticsAnalyzer
        statistics_report = self.statistics_analyzer.analyze(df)

        # 4. Run ImbalanceAnalyzer if target_column is provided
        imbalance_report = None
        if target_column is not None:
            imbalance_report = self.imbalance_analyzer.analyze(df, target_column)

        # 5. Generate final report using ReportGenerator
        final_report = self.report_generator.generate(
            missing_values=missing_values_report,
            duplicates=duplicates_report,
            statistics=statistics_report,
            imbalance=imbalance_report,
        )

        logger.info("Dataset analysis pipeline completed successfully.")
        return final_report
