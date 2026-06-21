"""Diagnostic analytics generator for the EDA module.

Produces a :class:`DiagnosticAnalytics` result from a pandas DataFrame and
its accompanying :class:`DatasetAnalysisReport`.  Every finding is generated
deterministically through statistical rules — no AI / LLM calls are made.

Diagnostic categories
---------------------
**Correlation findings** (stored in ``correlation_findings``):
    * Strong correlations   (|r| ≥ 0.80)
    * Moderate correlations (0.50 ≤ |r| < 0.80)

**Anomaly findings** (stored in ``anomaly_findings``):
    * Potential identifier columns
    * Low-variance features
    * Outliers (IQR method)
    * Potential target columns
    * Potential data-leakage columns
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.eda.schemas import (
    DiagnosticAnalytics,
    Insight,
    Severity,
)

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────

_STRONG_CORR_THRESHOLD: float = 0.80
_MODERATE_CORR_LOWER: float = 0.50
_MODERATE_CORR_UPPER: float = 0.80

_IDENTIFIER_UNIQUE_PCT: float = 95.0
_LOW_VARIANCE_DOMINANT_PCT: float = 95.0

_IQR_MULTIPLIER: float = 1.5

_IDENTIFIER_NAME_PATTERNS: re.Pattern[str] = re.compile(
    r"(?i)^(id|identifier|uuid|guid)$|_id$|^id_",
)
_TARGET_NAME_PATTERNS: re.Pattern[str] = re.compile(
    r"(target|label|class|outcome)", re.IGNORECASE,
)
_LEAKAGE_NAME_PATTERNS: re.Pattern[str] = re.compile(
    r"(^id$|identifier|uuid)", re.IGNORECASE,
)


# ── Custom exception ───────────────────────────────────────────────────────


class DiagnosticAnalyzerError(Exception):
    """Raised when diagnostic analysis cannot be completed."""


# ── Analyzer ────────────────────────────────────────────────────────────────


class DiagnosticAnalyzer:
    """Generate :class:`DiagnosticAnalytics` from a DataFrame and its report.

    All findings are rule-based and deterministic.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        analysis_report: DatasetAnalysisReport,
    ) -> DiagnosticAnalytics:
        """Produce diagnostic analytics.

        Parameters
        ----------
        df:
            The pandas DataFrame to diagnose.
        analysis_report:
            A validated :class:`DatasetAnalysisReport` produced by the
            analysis service.

        Returns
        -------
        DiagnosticAnalytics
            Container with ``correlation_findings`` and ``anomaly_findings``.

        Raises
        ------
        DiagnosticAnalyzerError
            If the inputs are invalid or analysis fails unexpectedly.
        """
        logger.info("Starting diagnostic analysis.")
        try:
            self._validate_inputs(df, analysis_report)

            correlation_findings = self._analyze_correlations(df)
            anomaly_findings: list[Insight] = []

            anomaly_findings.extend(
                self._detect_identifiers(analysis_report),
            )
            anomaly_findings.extend(
                self._detect_low_variance(df, analysis_report),
            )
            anomaly_findings.extend(
                self._detect_outliers(df, analysis_report),
            )
            anomaly_findings.extend(
                self._detect_target_candidates(df, analysis_report),
            )
            anomaly_findings.extend(
                self._detect_leakage_candidates(analysis_report),
            )
        except DiagnosticAnalyzerError:
            raise
        except Exception as exc:
            logger.exception("Diagnostic analysis failed unexpectedly.")
            raise DiagnosticAnalyzerError(
                f"Failed to generate diagnostic analytics: {exc}"
            ) from exc

        logger.info(
            "Diagnostic analysis complete — %d correlation finding(s), "
            "%d anomaly finding(s).",
            len(correlation_findings),
            len(anomaly_findings),
        )
        return DiagnosticAnalytics(
            correlation_findings=correlation_findings,
            anomaly_findings=anomaly_findings,
        )

    # ── Input validation ────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(
        df: pd.DataFrame,
        analysis_report: DatasetAnalysisReport,
    ) -> None:
        """Raise :class:`DiagnosticAnalyzerError` on invalid inputs."""
        if not isinstance(df, pd.DataFrame):
            raise DiagnosticAnalyzerError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if not isinstance(analysis_report, DatasetAnalysisReport):
            raise DiagnosticAnalyzerError(
                f"Expected a DatasetAnalysisReport, got {type(analysis_report).__name__}."
            )

    # ── Correlation analysis ────────────────────────────────────────────

    @staticmethod
    def _analyze_correlations(df: pd.DataFrame) -> list[Insight]:
        """Detect strong and moderate pairwise correlations."""
        findings: list[Insight] = []

        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            logger.info("Fewer than 2 numerical columns — skipping correlation analysis.")
            return findings

        corr_matrix = numeric_df.corr(numeric_only=True)

        seen: set[tuple[str, str]] = set()
        for col_a in corr_matrix.columns:
            for col_b in corr_matrix.columns:
                if col_a == col_b:
                    continue
                pair = tuple(sorted((col_a, col_b)))
                if pair in seen:
                    continue
                seen.add(pair)

                r = corr_matrix.loc[col_a, col_b]
                if pd.isna(r):
                    continue
                abs_r = abs(r)

                if abs_r >= _STRONG_CORR_THRESHOLD:
                    findings.append(
                        Insight(
                            title="Strong Correlation Detected",
                            description=(
                                f"{col_a} and {col_b} have a strong "
                                f"correlation ({r:.2f})."
                            ),
                            severity=Severity.INFO,
                        )
                    )
                elif abs_r >= _MODERATE_CORR_LOWER:
                    findings.append(
                        Insight(
                            title="Moderate Correlation Detected",
                            description=(
                                f"{col_a} and {col_b} have a moderate "
                                f"correlation ({r:.2f})."
                            ),
                            severity=Severity.INFO,
                        )
                    )

        return findings

    # ── Identifier detection ────────────────────────────────────────────

    @staticmethod
    def _detect_identifiers(
        report: DatasetAnalysisReport,
    ) -> list[Insight]:
        """Flag columns whose names match identifier patterns AND have ≥ 95 % unique values.

        Both conditions must be satisfied to avoid false positives on
        high-cardinality feature columns such as ``salary`` or ``age``.
        """
        findings: list[Insight] = []
        for col, profile in (report.column_profiles or {}).items():
            if profile.get("is_datetime", False):
                continue
            if not _IDENTIFIER_NAME_PATTERNS.search(col):
                continue
            unique_pct = profile.get("unique_percentage", 0.0)
            if isinstance(unique_pct, (int, float)) and unique_pct >= _IDENTIFIER_UNIQUE_PCT:
                findings.append(
                    Insight(
                        title="Potential Identifier Column",
                        description=(
                            f"'{col}' appears to be an identifier "
                            f"({unique_pct:.1f}% unique values)."
                        ),
                        severity=Severity.WARNING,
                    )
                )
        return findings

    # ── Low-variance detection ──────────────────────────────────────────

    @staticmethod
    def _detect_low_variance(
        df: pd.DataFrame,
        report: DatasetAnalysisReport,
    ) -> list[Insight]:
        """Flag categorical columns where one value dominates ≥ 95 %."""
        findings: list[Insight] = []
        for col, profile in (report.column_profiles or {}).items():
            if not profile.get("is_categorical", False):
                continue
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            most_common_pct = series.value_counts(normalize=True).iloc[0] * 100
            if most_common_pct >= _LOW_VARIANCE_DOMINANT_PCT:
                findings.append(
                    Insight(
                        title="Low Variance Feature",
                        description=(
                            f"Column '{col}' has low variance — "
                            f"one value occupies {most_common_pct:.1f}% of rows."
                        ),
                        severity=Severity.WARNING,
                    )
                )
        return findings

    # ── Outlier detection (IQR) ─────────────────────────────────────────

    @staticmethod
    def _detect_outliers(
        df: pd.DataFrame,
        report: DatasetAnalysisReport,
    ) -> list[Insight]:
        """Detect outliers using the IQR method on numerical columns."""
        findings: list[Insight] = []
        for col, profile in (report.column_profiles or {}).items():
            if not profile.get("is_numeric", False):
                continue
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - _IQR_MULTIPLIER * iqr
            upper = q3 + _IQR_MULTIPLIER * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())

            if outlier_count > 0:
                findings.append(
                    Insight(
                        title="Outliers Detected",
                        description=(
                            f"Column '{col}' contains {outlier_count} "
                            f"outlier(s) based on the IQR method."
                        ),
                        severity=Severity.WARNING,
                    )
                )
        return findings

    # ── Target candidate detection ──────────────────────────────────────

    @staticmethod
    def _detect_target_candidates(
        df: pd.DataFrame,
        report: DatasetAnalysisReport,
    ) -> list[Insight]:
        """Identify columns that may represent a prediction target."""
        findings: list[Insight] = []
        flagged: set[str] = set()

        # Rule 1 — name-based
        for col in (report.column_profiles or {}):
            if _TARGET_NAME_PATTERNS.search(col):
                findings.append(
                    Insight(
                        title="Potential Target Column",
                        description=(
                            f"Column '{col}' may be a target variable "
                            f"based on its name."
                        ),
                        severity=Severity.INFO,
                    )
                )
                flagged.add(col)

        # Rule 2 — binary columns
        for col, profile in (report.column_profiles or {}).items():
            if col in flagged:
                continue
            if col not in df.columns:
                continue
            unique_count = profile.get("unique_values", 0)
            if isinstance(unique_count, int) and unique_count == 2:
                findings.append(
                    Insight(
                        title="Potential Target Column",
                        description=(
                            f"Column '{col}' is binary (2 unique values) "
                            f"and may be a target variable."
                        ),
                        severity=Severity.INFO,
                    )
                )
        return findings

    # ── Leakage candidate detection ─────────────────────────────────────

    @staticmethod
    def _detect_leakage_candidates(
        report: DatasetAnalysisReport,
    ) -> list[Insight]:
        """Flag columns whose names suggest data leakage risk."""
        findings: list[Insight] = []
        for col in (report.column_profiles or {}):
            if _LEAKAGE_NAME_PATTERNS.search(col):
                findings.append(
                    Insight(
                        title="Potential Data Leakage Risk",
                        description=(
                            f"Column '{col}' may cause data leakage "
                            f"based on its name."
                        ),
                        severity=Severity.WARNING,
                    )
                )
        return findings
