"""Module for analyzing duplicate rows in pandas DataFrames."""

import logging
import pandas as pd
from backend.app.analysis.schemas import DuplicateReport

logger = logging.getLogger(__name__)


class DuplicateAnalyzer:
    """Analyze duplicate rows within a pandas DataFrame."""

    def analyze(self, df: pd.DataFrame) -> DuplicateReport:
        """Analyze duplicate rows in the provided DataFrame.

        Args:
            df: The pandas DataFrame to analyze.

        Returns:
            A DuplicateReport containing duplicate row statistics.
        """
        logger.info("Starting duplicate row analysis.")

        # Explicitly handle empty DataFrame cases
        if df.empty:
            logger.info("DataFrame is empty. Returning zero duplicate report.")
            return DuplicateReport(
                duplicate_rows=0,
                duplicate_percentage=0.0
            )

        total_rows = len(df)
        duplicate_count = int(df.duplicated().sum())

        # Calculate percentage
        percentage = (duplicate_count / total_rows) * 100.0
        duplicate_percentage = round(percentage, 2)

        logger.info(
            "Duplicate row analysis completed. Duplicate rows: %d (%s%%)",
            duplicate_count,
            duplicate_percentage
        )
        return DuplicateReport(
            duplicate_rows=duplicate_count,
            duplicate_percentage=duplicate_percentage
        )
