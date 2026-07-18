"""Module for generating statistical summaries of numerical columns in DataFrames."""

import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
from backend.app.analysis.schemas import StatisticsReport

logger = logging.getLogger(__name__)


class StatisticsAnalyzer:
    """Analyze numerical descriptive statistics within a pandas DataFrame."""

    def analyze(self, df: pd.DataFrame) -> StatisticsReport:
        """Calculate mean, median, standard deviation, min, and max for numerical columns.

        Args:
            df: The pandas DataFrame to analyze.

        Returns:
            A StatisticsReport containing descriptive statistics.
        """
        logger.info("Starting numerical statistics analysis.")

        # Explicitly handle empty DataFrame
        if df.empty or len(df.columns) == 0:
            logger.info("DataFrame is empty or has no columns. Returning empty report.")
            return StatisticsReport(numerical_summary={})

        numerical_summary = {}

        for col in df.columns:
            series = df[col]
            if is_numeric_dtype(series):
                col_name = str(col)
                # Compute descriptive statistics, ignoring NaN values by default
                mean_val = series.mean()
                median_val = series.median()
                std_val = series.std()
                min_val = series.min()
                max_val = series.max()

                # Safely format statistics, defaulting to 0.0 if calculations yield NaN
                summary = {
                    "mean": round(float(mean_val), 2) if not pd.isna(mean_val) else 0.0,
                    "median": round(float(median_val), 2) if not pd.isna(median_val) else 0.0,
                    "std": round(float(std_val), 2) if not pd.isna(std_val) else 0.0,
                    "min": round(float(min_val), 2) if not pd.isna(min_val) else 0.0,
                    "max": round(float(max_val), 2) if not pd.isna(max_val) else 0.0,
                }
                numerical_summary[col_name] = summary

        logger.info("Numerical statistics analysis completed successfully.")
        return StatisticsReport(numerical_summary=numerical_summary)
