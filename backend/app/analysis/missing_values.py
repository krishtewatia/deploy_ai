"""Module for analyzing missing values in pandas DataFrames."""

import logging
import pandas as pd
from backend.app.analysis.schemas import MissingValueReport

logger = logging.getLogger(__name__)


class MissingValueAnalyzer:
    """Analyze missing values within a pandas DataFrame."""

    def analyze(self, df: pd.DataFrame) -> MissingValueReport:
        """Analyze missing values in the provided DataFrame.

        Args:
            df: The pandas DataFrame to analyze.

        Returns:
            A MissingValueReport containing detailed missing value statistics.
        """
        logger.info("Starting missing value analysis.")

        # Explicitly handle empty DataFrame or empty rows/columns cases
        if df.empty or len(df.columns) == 0:
            logger.info("DataFrame is empty or has no columns. Returning empty report.")
            return MissingValueReport(
                total_missing=0,
                missing_by_column={},
                missing_percentage={}
            )

        total_rows = len(df)
        null_counts = df.isnull().sum()
        total_missing = int(null_counts.sum())

        missing_by_column = {}
        missing_percentage = {}

        for col, count in null_counts.items():
            col_name = str(col)
            count_val = int(count)
            if count_val > 0:
                missing_by_column[col_name] = count_val
                # Calculate percentage based on total rows
                percentage = (count_val / total_rows) * 100.0
                missing_percentage[col_name] = round(percentage, 2)

        logger.info(
            "Missing value analysis completed. Total missing values: %d",
            total_missing
        )
        return MissingValueReport(
            total_missing=total_missing,
            missing_by_column=missing_by_column,
            missing_percentage=missing_percentage
        )
