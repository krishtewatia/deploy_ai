"""Visualization recommendation engine for the EDA module.

Produces a list of :class:`VisualizationRecommendation` objects from a
:class:`DatasetAnalysisReport`.  All recommendations are deterministic
and rule-based — no AI / LLM calls are involved.

Rules
-----
1. **Histogram** — one per numerical column.
2. **Boxplot** — one per numerical column (outlier-prone).
3. **Correlation Heatmap** — when ≥ 2 numerical columns exist.
4. **Bar Chart** — one per categorical column.
5. **Missing Value Heatmap** — when missing values exist.
6. **Line Chart** — one per datetime column.
"""

from __future__ import annotations

import logging

from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.eda.schemas import (
    ChartType,
    VisualizationRecommendation,
)

logger = logging.getLogger(__name__)


# ── Custom exception ───────────────────────────────────────────────────────


class VisualizationRecommenderError(Exception):
    """Raised when visualization recommendation fails."""


# ── Recommender ─────────────────────────────────────────────────────────────


class VisualizationRecommender:
    """Recommend visualizations from a :class:`DatasetAnalysisReport`.

    Every recommendation is generated through deterministic rules.
    Duplicate recommendations (same chart type + same column set) are
    automatically suppressed.
    """

    def recommend(
        self,
        analysis_report: DatasetAnalysisReport,
    ) -> list[VisualizationRecommendation]:
        """Generate visualization recommendations.

        Parameters
        ----------
        analysis_report:
            A validated :class:`DatasetAnalysisReport`.

        Returns
        -------
        list[VisualizationRecommendation]
            De-duplicated list of recommended charts.

        Raises
        ------
        VisualizationRecommenderError
            If the report is invalid or recommendation fails.
        """
        logger.info("Starting visualization recommendation.")
        try:
            self._validate_input(analysis_report)

            recommendations: list[VisualizationRecommendation] = []

            numerical = self._columns_by_flag(analysis_report, "is_numeric")
            categorical = self._columns_by_flag(analysis_report, "is_categorical")
            datetime_cols = self._columns_by_flag(analysis_report, "is_datetime")

            # 1 — Histograms
            recommendations.extend(self._recommend_histograms(numerical))

            # 2 — Boxplots
            recommendations.extend(self._recommend_boxplots(numerical))

            # 3 — Correlation heatmap
            recommendations.extend(self._recommend_correlation_heatmap(numerical))

            # 4 — Bar charts
            recommendations.extend(self._recommend_bar_charts(categorical))

            # 5 — Missing-value heatmap
            recommendations.extend(
                self._recommend_missing_heatmap(analysis_report),
            )

            # 6 — Line charts
            recommendations.extend(self._recommend_line_charts(datetime_cols))

            deduplicated = self._deduplicate(recommendations)

        except VisualizationRecommenderError:
            raise
        except Exception as exc:
            logger.exception("Visualization recommendation failed unexpectedly.")
            raise VisualizationRecommenderError(
                f"Failed to generate visualization recommendations: {exc}"
            ) from exc

        logger.info(
            "Visualization recommendation complete — %d recommendation(s).",
            len(deduplicated),
        )
        return deduplicated

    # ── Input validation ────────────────────────────────────────────────

    @staticmethod
    def _validate_input(analysis_report: DatasetAnalysisReport) -> None:
        """Raise on invalid input."""
        if not isinstance(analysis_report, DatasetAnalysisReport):
            raise VisualizationRecommenderError(
                f"Expected a DatasetAnalysisReport, got {type(analysis_report).__name__}."
            )

    # ── Column helpers ──────────────────────────────────────────────────

    @staticmethod
    def _columns_by_flag(
        report: DatasetAnalysisReport,
        flag: str,
    ) -> list[str]:
        """Return column names where *flag* is truthy."""
        return [
            col
            for col, profile in (report.column_profiles or {}).items()
            if profile.get(flag, False)
        ]

    # ── Rule implementations ────────────────────────────────────────────

    @staticmethod
    def _recommend_histograms(
        numerical: list[str],
    ) -> list[VisualizationRecommendation]:
        """Rule 1 — one histogram per numerical column."""
        return [
            VisualizationRecommendation(
                chart_type=ChartType.HISTOGRAM,
                column_names=[col],
                reason="Visualize distribution and skewness.",
            )
            for col in numerical
        ]

    @staticmethod
    def _recommend_boxplots(
        numerical: list[str],
    ) -> list[VisualizationRecommendation]:
        """Rule 2 — one boxplot per numerical column."""
        return [
            VisualizationRecommendation(
                chart_type=ChartType.BOXPLOT,
                column_names=[col],
                reason="Detect outliers and spread.",
            )
            for col in numerical
        ]

    @staticmethod
    def _recommend_correlation_heatmap(
        numerical: list[str],
    ) -> list[VisualizationRecommendation]:
        """Rule 3 — correlation heatmap when ≥ 2 numerical columns."""
        if len(numerical) < 2:
            return []
        return [
            VisualizationRecommendation(
                chart_type=ChartType.CORRELATION_HEATMAP,
                column_names=sorted(numerical),
                reason="Analyze feature relationships.",
            )
        ]

    @staticmethod
    def _recommend_bar_charts(
        categorical: list[str],
    ) -> list[VisualizationRecommendation]:
        """Rule 4 — one bar chart per categorical column."""
        return [
            VisualizationRecommendation(
                chart_type=ChartType.BAR_CHART,
                column_names=[col],
                reason="Compare category frequencies.",
            )
            for col in categorical
        ]

    @staticmethod
    def _recommend_missing_heatmap(
        report: DatasetAnalysisReport,
    ) -> list[VisualizationRecommendation]:
        """Rule 5 — missing-value heatmap when missing data exists."""
        if report.missing_values.total_missing <= 0:
            return []

        cols_with_missing = sorted(report.missing_values.missing_by_column.keys())
        if not cols_with_missing:
            return []

        return [
            VisualizationRecommendation(
                chart_type=ChartType.MISSING_VALUE_HEATMAP,
                column_names=cols_with_missing,
                reason="Visualize missing-data patterns.",
            )
        ]

    @staticmethod
    def _recommend_line_charts(
        datetime_cols: list[str],
    ) -> list[VisualizationRecommendation]:
        """Rule 6 — one line chart per datetime column."""
        return [
            VisualizationRecommendation(
                chart_type=ChartType.LINE_CHART,
                column_names=[col],
                reason="Visualize temporal trends.",
            )
            for col in datetime_cols
        ]

    # ── Deduplication ───────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(
        recommendations: list[VisualizationRecommendation],
    ) -> list[VisualizationRecommendation]:
        """Remove duplicate recommendations (same chart_type + columns)."""
        seen: set[tuple[str, tuple[str, ...]]] = set()
        unique: list[VisualizationRecommendation] = []

        for rec in recommendations:
            key = (rec.chart_type, tuple(sorted(rec.column_names)))
            if key not in seen:
                seen.add(key)
                unique.append(rec)

        return unique
