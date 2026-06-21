"""Chart generation engine for the EDA module.

Generates actual chart image files (PNG) from a pandas DataFrame and a
VisualizationRecommendation. All generated charts are saved deterministically
and cleaned up to avoid memory leaks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import uuid

# Use non-interactive Agg backend to avoid GUI window popup and thread issues
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.app.eda.schemas import ChartType, VisualizationRecommendation

logger = logging.getLogger(__name__)


# ── Custom Exception ───────────────────────────────────────────────────────


class ChartGenerationError(Exception):
    """Raised when chart generation fails due to invalid inputs, missing columns, or plotting errors."""


# ── Chart Generator ─────────────────────────────────────────────────────────


class ChartGenerator:
    """Generates charts from a pandas DataFrame and VisualizationRecommendation."""

    def __init__(
        self,
        output_dir: str = "reports/charts",
        dataset_id: str | None = None,
    ) -> None:
        """Initialize the ChartGenerator and create the dataset-scoped output directory.

        Parameters
        ----------
        output_dir : str
            The base directory path where generated charts will be saved.
        dataset_id : str | None
            The unique identifier for the dataset. If not provided, a random 8-character ID is generated.
        """
        self.base_output_dir = Path(output_dir).resolve()
        if dataset_id is None:
            dataset_id = uuid.uuid4().hex[:8]
        self.dataset_id = dataset_id
        self._dataset_chart_dir = self.base_output_dir / self.dataset_id
        self.output_dir = self._dataset_chart_dir  # for backward compatibility in internal plotting helpers
        try:
            self._dataset_chart_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created dataset chart directory: %s", self._dataset_chart_dir)
        except Exception as exc:
            raise ChartGenerationError(
                f"Failed to create dataset directory '{self._dataset_chart_dir}': {exc}"
            ) from exc

    @property
    def dataset_chart_dir(self) -> Path:
        """Get the directory path where this dataset's charts are saved."""
        return self._dataset_chart_dir

    def generate_chart(
        self,
        df: pd.DataFrame,
        recommendation: VisualizationRecommendation,
    ) -> str:
        """Generate a chart from the DataFrame and recommendation.

        Parameters
        ----------
        df : pd.DataFrame
            The input pandas DataFrame.
        recommendation : VisualizationRecommendation
            The visualization recommendation specifying columns and chart type.

        Returns
        -------
        str
            The absolute path of the generated PNG file.

        Raises
        ------
        ChartGenerationError
            If validation fails or chart generation runs into an error.
        """
        # 1. Validation checks
        if not isinstance(df, pd.DataFrame):
            raise ChartGenerationError("Input 'df' must be a pandas DataFrame.")

        if df.empty:
            raise ChartGenerationError("Cannot generate chart from an empty DataFrame.")

        if not isinstance(recommendation, VisualizationRecommendation):
            raise ChartGenerationError("Input 'recommendation' must be a VisualizationRecommendation.")

        logger.info(
            "Generating chart for type '%s' with columns %s",
            recommendation.chart_type,
            recommendation.column_names,
        )

        chart_type_val = (
            recommendation.chart_type.value
            if hasattr(recommendation.chart_type, "value")
            else recommendation.chart_type
        )

        supported_types = {
            ChartType.HISTOGRAM.value,
            ChartType.BOXPLOT.value,
            ChartType.CORRELATION_HEATMAP.value,
            ChartType.BAR_CHART.value,
            ChartType.MISSING_VALUE_HEATMAP.value,
            ChartType.LINE_CHART.value,
        }

        if chart_type_val not in supported_types:
            raise ChartGenerationError(f"Unsupported chart type: {chart_type_val}")

        # Ensure all columns in recommendation exist in df
        for col in recommendation.column_names:
            if col not in df.columns:
                raise ChartGenerationError(
                    f"Column '{col}' specified in recommendation not found in DataFrame."
                )

        # 2. Dispatch to specific generator
        fig = None
        try:
            if chart_type_val == ChartType.HISTOGRAM.value:
                fig, filepath = self._generate_histogram(df, recommendation)
            elif chart_type_val == ChartType.BOXPLOT.value:
                fig, filepath = self._generate_boxplot(df, recommendation)
            elif chart_type_val == ChartType.CORRELATION_HEATMAP.value:
                fig, filepath = self._generate_correlation_heatmap(df, recommendation)
            elif chart_type_val == ChartType.BAR_CHART.value:
                fig, filepath = self._generate_bar_chart(df, recommendation)
            elif chart_type_val == ChartType.MISSING_VALUE_HEATMAP.value:
                fig, filepath = self._generate_missing_value_heatmap(df, recommendation)
            else:  # ChartType.LINE_CHART
                fig, filepath = self._generate_line_chart(df, recommendation)

            # Apply tight layout and save
            fig.tight_layout()
            fig.savefig(filepath, format="png")
            logger.info("Successfully saved chart to %s", filepath)
            return str(filepath.resolve())

        except ChartGenerationError:
            raise
        except Exception as exc:
            logger.exception("An error occurred during chart generation.")
            raise ChartGenerationError(
                f"Failed to generate chart for type '{chart_type_val}': {exc}"
            ) from exc
        finally:
            if fig is not None:
                plt.close(fig)

    def generate_charts(
        self,
        df: pd.DataFrame,
        recommendations: list[VisualizationRecommendation],
    ) -> list[str]:
        """Generate multiple charts for a dataset and return all absolute file paths.

        Parameters
        ----------
        df : pd.DataFrame
            The input pandas DataFrame.
        recommendations : list[VisualizationRecommendation]
            The list of visualization recommendations.

        Returns
        -------
        list[str]
            A list of absolute file paths to the generated charts.

        Raises
        ------
        ChartGenerationError
            If validation fails or chart generation runs into an error.
        """
        if not isinstance(recommendations, list):
            raise ChartGenerationError("Input 'recommendations' must be a list.")

        logger.info("Batch generating %d charts.", len(recommendations))
        paths = []
        for rec in recommendations:
            paths.append(self.generate_chart(df, rec))
        return paths

    # ── Plotting Helpers ──────────────────────────────────────────────────────

    def _generate_histogram(
        self,
        df: pd.DataFrame,
        recommendation: VisualizationRecommendation,
    ) -> tuple[plt.Figure, Path]:
        """Generate a histogram for a single numerical column."""
        col = recommendation.column_names[0]
        filepath = self.output_dir / f"{col}_histogram.png"

        # Check if the column has any non-null numeric values
        non_null_data = df[col].dropna()
        if non_null_data.empty:
            raise ChartGenerationError(f"Column '{col}' has no valid numerical data for a histogram.")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(non_null_data, bins=20, edgecolor="black", color="#3498db", alpha=0.8)
        ax.set_title(f"Distribution of {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)

        return fig, filepath

    def _generate_boxplot(
        self,
        df: pd.DataFrame,
        recommendation: VisualizationRecommendation,
    ) -> tuple[plt.Figure, Path]:
        """Generate a boxplot for a single numerical column."""
        col = recommendation.column_names[0]
        filepath = self.output_dir / f"{col}_boxplot.png"

        # Check if the column has any non-null numeric values
        non_null_data = df[col].dropna()
        if non_null_data.empty:
            raise ChartGenerationError(f"Column '{col}' has no valid numerical data for a boxplot.")

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(y=non_null_data, ax=ax, color="#2ecc71")
        ax.set_title(f"Boxplot of {col}", fontsize=14, fontweight="bold")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.5)

        return fig, filepath

    def _generate_correlation_heatmap(
        self,
        df: pd.DataFrame,
        recommendation: VisualizationRecommendation,
    ) -> tuple[plt.Figure, Path]:
        """Generate a correlation heatmap for specified numerical columns."""
        cols = recommendation.column_names
        filepath = self.output_dir / "correlation_heatmap.png"

        # Filter cols to make sure they are numeric
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            raise ChartGenerationError(
                "Correlation heatmap requires at least 2 numerical columns."
            )

        corr_df = df[numeric_cols].corr()
        if corr_df.empty or corr_df.isna().all().all():
            raise ChartGenerationError(
                "Cannot compute correlation matrix from the specified columns."
            )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_df,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax,
            vmin=-1,
            vmax=1,
            cbar=True,
            square=True,
        )
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")

        return fig, filepath

    def _generate_bar_chart(
        self,
        df: pd.DataFrame,
        recommendation: VisualizationRecommendation,
    ) -> tuple[plt.Figure, Path]:
        """Generate a bar chart for a categorical column."""
        col = recommendation.column_names[0]
        filepath = self.output_dir / f"{col}_bar_chart.png"

        counts = df[col].value_counts(dropna=True)
        if counts.empty:
            raise ChartGenerationError(f"Column '{col}' has no data to plot a bar chart.")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, palette="viridis", legend=False)
        ax.set_title(f"Bar Chart of {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")

        return fig, filepath

    def _generate_missing_value_heatmap(
        self,
        df: pd.DataFrame,
        recommendation: VisualizationRecommendation,
    ) -> tuple[plt.Figure, Path]:
        """Generate a missing value heatmap using df.isnull()."""
        filepath = self.output_dir / "missing_value_heatmap.png"

        fig, ax = plt.subplots(figsize=(10, 6))
        # Use seaborn.heatmap(df.isnull())
        sns.heatmap(
            df.isnull(),
            cbar=True,
            yticklabels=False,
            cmap="viridis",
            ax=ax,
        )
        ax.set_title("Missing Value Heatmap", fontsize=14, fontweight="bold")

        return fig, filepath

    def _generate_line_chart(
        self,
        df: pd.DataFrame,
        recommendation: VisualizationRecommendation,
    ) -> tuple[plt.Figure, Path]:
        """Generate a line chart for a temporal column."""
        col = recommendation.column_names[0]
        filepath = self.output_dir / f"{col}_line_chart.png"

        # Safely convert to datetime
        try:
            series = pd.to_datetime(df[col]).dropna()
        except Exception as exc:
            raise ChartGenerationError(
                f"Column '{col}' could not be parsed as datetime: {exc}"
            ) from exc

        if series.empty:
            raise ChartGenerationError(f"No valid temporal data in column '{col}'.")

        counts = series.value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(counts.index, counts.values, marker="o", linestyle="-", color="#e74c3c")
        ax.set_title(f"Event Trend Over Time ({col})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.autofmt_xdate()

        return fig, filepath
