"""Descriptive analytics generator for the EDA module.

Produces a :class:`DescriptiveAnalytics` result directly from an existing
:class:`DatasetAnalysisReport` — no AI calls are involved.  All insights
are derived deterministically from the report's numerical fields.
"""

from __future__ import annotations

import logging

from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.eda.schemas import (
    DatasetSummary,
    DescriptiveAnalytics,
    Insight,
    Severity,
)

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────

_LARGE_DATASET_ROW_THRESHOLD: int = 100_000
_HIGH_MISSINGNESS_PERCENTAGE_THRESHOLD: float = 30.0


# ── Custom exception ───────────────────────────────────────────────────────


class DescriptiveAnalyzerError(Exception):
    """Raised when descriptive analysis cannot be completed."""


# ── Analyzer ────────────────────────────────────────────────────────────────


class DescriptiveAnalyzer:
    """Generate :class:`DescriptiveAnalytics` from a :class:`DatasetAnalysisReport`.

    The analyzer extracts a :class:`DatasetSummary` and produces a list of
    rule-based :class:`Insight` objects without making any AI / LLM calls.
    """

    def analyze(
        self,
        analysis_report: DatasetAnalysisReport,
    ) -> DescriptiveAnalytics:
        """Produce descriptive analytics from *analysis_report*.

        Parameters
        ----------
        analysis_report:
            A validated :class:`DatasetAnalysisReport` produced by the
            analysis service.

        Returns
        -------
        DescriptiveAnalytics
            A container holding the dataset summary and key findings.

        Raises
        ------
        DescriptiveAnalyzerError
            If the report is malformed or missing required attributes.
        """
        logger.info("Starting descriptive analysis.")
        try:
            summary = self._build_dataset_summary(analysis_report)
            findings = self._generate_key_findings(analysis_report, summary)
        except DescriptiveAnalyzerError:
            raise
        except Exception as exc:
            logger.exception("Descriptive analysis failed due to an unexpected error.")
            raise DescriptiveAnalyzerError(
                f"Failed to generate descriptive analytics: {exc}"
            ) from exc

        logger.info(
            "Descriptive analysis complete — %d insight(s) generated.",
            len(findings),
        )
        return DescriptiveAnalytics(
            dataset_summary=summary,
            key_findings=findings,
        )

    # ── Dataset summary construction ────────────────────────────────────

    def _build_dataset_summary(
        self,
        report: DatasetAnalysisReport,
    ) -> DatasetSummary:
        """Extract a :class:`DatasetSummary` from the analysis report."""
        try:
            column_profiles = report.column_profiles or {}

            numerical_columns = self._columns_by_flag(column_profiles, "is_numeric")
            categorical_columns = self._columns_by_flag(column_profiles, "is_categorical")
            datetime_columns = self._columns_by_flag(column_profiles, "is_datetime")

            rows = self._infer_row_count(report)
            columns = len(column_profiles)

            return DatasetSummary(
                rows=rows,
                columns=columns,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                datetime_columns=datetime_columns,
                missing_cells=report.missing_values.total_missing,
                duplicate_rows=report.duplicates.duplicate_rows,
            )
        except Exception as exc:
            logger.exception("Failed to build dataset summary.")
            raise DescriptiveAnalyzerError(
                f"Failed to build dataset summary: {exc}"
            ) from exc

    # ── Key findings generation ─────────────────────────────────────────

    @staticmethod
    def _generate_key_findings(
        report: DatasetAnalysisReport,
        summary: DatasetSummary,
    ) -> list[Insight]:
        """Build a list of rule-based :class:`Insight` objects."""
        findings: list[Insight] = []

        # Missing values
        if summary.missing_cells > 0:
            findings.append(
                Insight(
                    title="Missing Values Detected",
                    description=(
                        f"Dataset contains {summary.missing_cells} missing "
                        f"value(s) across {len(report.missing_values.missing_by_column)} "
                        f"column(s)."
                    ),
                    severity=Severity.WARNING,
                )
            )

        # High missingness per column
        for col, pct in report.missing_values.missing_percentage.items():
            if pct > _HIGH_MISSINGNESS_PERCENTAGE_THRESHOLD:
                findings.append(
                    Insight(
                        title="High Missingness",
                        description=(
                            f"Column '{col}' has {pct:.1f}% missing values, "
                            f"exceeding the {_HIGH_MISSINGNESS_PERCENTAGE_THRESHOLD}% threshold."
                        ),
                        severity=Severity.CRITICAL,
                    )
                )

        # Duplicates
        if summary.duplicate_rows > 0:
            findings.append(
                Insight(
                    title="Duplicate Records Found",
                    description=(
                        f"Dataset contains {summary.duplicate_rows} duplicate "
                        f"row(s) ({report.duplicates.duplicate_percentage:.1f}%)."
                    ),
                    severity=Severity.WARNING,
                )
            )

        # Large dataset
        if summary.rows > _LARGE_DATASET_ROW_THRESHOLD:
            findings.append(
                Insight(
                    title="Large Dataset",
                    description=(
                        f"Dataset contains {summary.rows:,} rows. "
                        f"Consider sampling or incremental processing."
                    ),
                    severity=Severity.INFO,
                )
            )

        # Numerical columns available
        if summary.numerical_columns:
            findings.append(
                Insight(
                    title="Numerical Features Available",
                    description=(
                        f"{len(summary.numerical_columns)} numerical column(s) "
                        f"detected: {', '.join(summary.numerical_columns)}."
                    ),
                    severity=Severity.INFO,
                )
            )

        # Categorical columns available
        if summary.categorical_columns:
            findings.append(
                Insight(
                    title="Categorical Features Available",
                    description=(
                        f"{len(summary.categorical_columns)} categorical column(s) "
                        f"detected: {', '.join(summary.categorical_columns)}."
                    ),
                    severity=Severity.INFO,
                )
            )

        return findings

    # ── Utility helpers ─────────────────────────────────────────────────

    @staticmethod
    def _columns_by_flag(
        column_profiles: dict[str, dict],
        flag: str,
    ) -> list[str]:
        """Return column names where *flag* is truthy in their profile."""
        return [
            col
            for col, profile in column_profiles.items()
            if profile.get(flag, False)
        ]

    @staticmethod
    def _infer_row_count(report: DatasetAnalysisReport) -> int:
        """Infer the total row count from the analysis report.

        Strategy:
        1. Derive from ``duplicate_rows`` and ``duplicate_percentage`` if
           the percentage is positive.
        2. Fall back to deriving from any column's ``missing_count`` and
           ``missing_percentage`` pair.
        3. Return ``0`` if neither approach yields a result.
        """
        dup = report.duplicates
        if dup.duplicate_percentage > 0 and dup.duplicate_rows > 0:
            return round(dup.duplicate_rows * 100 / dup.duplicate_percentage)

        # Fall back to column-level missing data
        for col, count in report.missing_values.missing_by_column.items():
            pct = report.missing_values.missing_percentage.get(col, 0.0)
            if count > 0 and pct > 0:
                return round(count * 100 / pct)

        # Try column_profiles unique_values / unique_percentage
        for profile in (report.column_profiles or {}).values():
            unique_count = profile.get("unique_values")
            unique_pct = profile.get("unique_percentage")
            if (
                isinstance(unique_count, (int, float))
                and isinstance(unique_pct, (int, float))
                and unique_count > 0
                and unique_pct > 0
            ):
                return round(unique_count * 100 / unique_pct)

        return 0
