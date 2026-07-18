"""Module for analyzing class imbalance in classification datasets."""

import logging
import pandas as pd
from backend.app.analysis.schemas import ImbalanceReport

logger = logging.getLogger(__name__)


class ImbalanceAnalyzer:
    """Analyze target column distributions to detect class imbalance."""

    def analyze(self, df: pd.DataFrame, target_column: str) -> ImbalanceReport:
        """Analyze class distributions and detect if any class falls below 20%.

        Args:
            df: The pandas DataFrame to analyze.
            target_column: The target column to compute class imbalance on.

        Returns:
            An ImbalanceReport containing classification imbalance details.

        Raises:
            ValueError: If the DataFrame is empty, if the target column does not exist,
                        or if target column contains no non-null values.
        """
        logger.info("Starting class imbalance analysis on target column: %s", target_column)

        if df.empty:
            logger.error("Imbalance analysis failed: DataFrame is empty.")
            raise ValueError("DataFrame is empty. Cannot perform class imbalance analysis.")

        if target_column not in df.columns:
            logger.error("Imbalance analysis failed: target column '%s' not found.", target_column)
            raise ValueError(f"Target column '{target_column}' does not exist in DataFrame.")

        # Compute target distribution ignoring NaNs
        target_series = df[target_column].dropna()
        total_valid_rows = len(target_series)

        if total_valid_rows == 0:
            logger.error("Imbalance analysis failed: target column has no valid data.")
            raise ValueError(f"Target column '{target_column}' has no valid non-null rows.")

        # Get counts
        value_counts = target_series.value_counts()
        distribution = {str(k): int(v) for k, v in value_counts.items()}

        # Detect imbalance based on the 20% rule
        imbalanced = False
        for count in value_counts:
            percentage = (count / total_valid_rows) * 100.0
            if percentage < 20.0:
                imbalanced = True
                break

        logger.info(
            "Imbalance analysis completed. Target class distribution: %s. Imbalanced: %s",
            distribution,
            imbalanced
        )
        return ImbalanceReport(
            imbalanced=imbalanced,
            distribution=distribution
        )
