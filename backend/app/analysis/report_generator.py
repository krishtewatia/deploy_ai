"""Module for combining individual data reports into a consolidated analysis report."""

import logging
from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    ImbalanceReport,
    MissingValueReport,
    StatisticsReport,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Orchestrate and combine multiple data analyzer outputs into a single analysis report."""

    def generate(
        self,
        missing_values: MissingValueReport,
        duplicates: DuplicateReport,
        statistics: StatisticsReport,
        imbalance: ImbalanceReport | None = None,
    ) -> DatasetAnalysisReport:
        """Combine specific reports into a unified DatasetAnalysisReport.

        Args:
            missing_values: Report describing missing values.
            duplicates: Report describing duplicate rows.
            statistics: Report describing descriptive statistics.
            imbalance: Optional report detailing target class imbalance.

        Returns:
            A consolidated DatasetAnalysisReport.
        """
        logger.info("Combining analysis components into a consolidated report.")

        report = DatasetAnalysisReport(
            missing_values=missing_values,
            duplicates=duplicates,
            statistics=statistics,
            imbalance=imbalance,
        )

        logger.info("Consolidated dataset analysis report generated successfully.")
        return report
