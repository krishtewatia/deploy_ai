"""DatasetContextBuilder — converts analysis outputs into DatasetContext.

This module provides a deterministic builder that transforms existing
:class:`~backend.app.analysis.schemas.DatasetAnalysisReport` data into
the compact :class:`~backend.app.dataset_intelligence.schemas.DatasetContext`
schema.

The builder is a pure adapter/assembler.  It does NOT:

- access DataFrames,
- calculate statistics,
- make AI calls,
- detect targets, identifiers, or leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from backend.app.analysis.schemas import DatasetAnalysisReport
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    ColumnStatistics,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)

logger = logging.getLogger(__name__)

# Keys that every column profile dict is expected to contain.
_REQUIRED_PROFILE_KEYS = frozenset({
    "dtype",
    "unique_values",
    "unique_percentage",
    "missing_count",
    "missing_percentage",
    "sample_values",
    "is_numeric",
    "is_categorical",
    "is_datetime",
})


class DatasetContextBuilderError(Exception):
    """Raised when the builder encounters an unrecoverable conversion error."""


class DatasetContextBuilder:
    """Builds a :class:`DatasetContext` from existing analysis outputs.

    Usage::

        builder = DatasetContextBuilder()
        context = builder.build(
            dataset_id="ds-001",
            file_name="sales.csv",
            row_count=1000,
            column_count=10,
            memory_usage_bytes=80_000,
            analysis_report=report,
        )
    """

    def build(
        self,
        *,
        dataset_id: str,
        file_name: str,
        row_count: int,
        column_count: int,
        memory_usage_bytes: int,
        analysis_report: DatasetAnalysisReport,
    ) -> DatasetContext:
        """Convert a :class:`DatasetAnalysisReport` into a :class:`DatasetContext`.

        Parameters
        ----------
        dataset_id:
            Unique identifier for the dataset.
        file_name:
            Original file name of the uploaded dataset.
        row_count:
            Total number of rows.
        column_count:
            Total number of columns.
        memory_usage_bytes:
            Approximate in-memory size in bytes.
        analysis_report:
            The completed analysis report to convert.

        Returns
        -------
        DatasetContext
            A fully populated, validated dataset context.

        Raises
        ------
        DatasetContextBuilderError
            If required analysis data is missing or malformed.
        """
        logger.info(
            "Building DatasetContext for dataset_id=%r, file=%r.",
            dataset_id,
            file_name,
        )

        if not analysis_report.column_profiles:
            raise DatasetContextBuilderError(
                "column_profiles is empty — cannot build DatasetContext "
                "without at least one column profile."
            )

        basic_info = DatasetBasicInfo(
            dataset_id=dataset_id,
            file_name=file_name,
            row_count=row_count,
            column_count=column_count,
            memory_usage_bytes=memory_usage_bytes,
        )

        # Numerical statistics lookup: column_name -> {mean, median, ...}
        numerical_stats = analysis_report.statistics.numerical_summary

        # Build ColumnContext list preserving column_profiles iteration order.
        columns: list[ColumnContext] = []
        for col_name, profile in analysis_report.column_profiles.items():
            columns.append(
                self._build_column_context(col_name, profile, numerical_stats)
            )

        missing_data = self._build_missing_data_summary(
            analysis_report, analysis_report.column_profiles
        )

        duplicates = self._build_duplicate_summary(analysis_report)

        context = DatasetContext(
            basic_info=basic_info,
            columns=columns,
            missing_data=missing_data,
            duplicates=duplicates,
            target_candidates=[],  # Stage 2: always empty
        )

        logger.info(
            "DatasetContext built successfully: %d columns, schema_version=%s.",
            len(context.columns),
            context.schema_version,
        )
        return context

    # ── private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_column_context(
        col_name: str,
        profile: Dict[str, Any],
        numerical_stats: Dict[str, Dict[str, float]],
    ) -> ColumnContext:
        """Map a single column profile dict to a :class:`ColumnContext`."""
        missing_keys = _REQUIRED_PROFILE_KEYS - profile.keys()
        if missing_keys:
            raise DatasetContextBuilderError(
                f"Column profile for {col_name!r} is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        # Build ColumnStatistics if numerical stats exist for this column.
        statistics: Optional[ColumnStatistics] = None
        col_stats = numerical_stats.get(col_name)
        if col_stats is not None:
            statistics = ColumnStatistics(
                mean=col_stats.get("mean"),
                median=col_stats.get("median"),
                std=col_stats.get("std"),
                min=col_stats.get("min"),
                max=col_stats.get("max"),
            )

        return ColumnContext(
            name=col_name,
            dtype=profile["dtype"],
            is_numeric=profile["is_numeric"],
            is_categorical=profile["is_categorical"],
            is_datetime=profile["is_datetime"],
            missing_count=profile["missing_count"],
            missing_percentage=profile["missing_percentage"],
            # Map the existing "unique_values" (count) to "unique_count".
            unique_count=profile["unique_values"],
            unique_percentage=profile["unique_percentage"],
            sample_values=profile["sample_values"],
            statistics=statistics,
        )

    @staticmethod
    def _build_missing_data_summary(
        report: DatasetAnalysisReport,
        column_profiles: Dict[str, Dict[str, Any]],
    ) -> MissingDataSummary:
        """Map :class:`MissingValueReport` to :class:`MissingDataSummary`.

        ``columns_with_missing`` preserves the iteration order of
        *column_profiles* (original column order) rather than the
        arbitrary key order of ``missing_by_column``.
        """
        missing_by_column = report.missing_values.missing_by_column

        # Preserve original column order from column_profiles.
        columns_with_missing = [
            col_name
            for col_name in column_profiles
            if missing_by_column.get(col_name, 0) > 0
        ]

        return MissingDataSummary(
            total_missing_cells=report.missing_values.total_missing,
            columns_with_missing=columns_with_missing,
        )

    @staticmethod
    def _build_duplicate_summary(
        report: DatasetAnalysisReport,
    ) -> DuplicateSummary:
        """Map :class:`DuplicateReport` to :class:`DuplicateSummary`."""
        return DuplicateSummary(
            duplicate_rows=report.duplicates.duplicate_rows,
            duplicate_percentage=report.duplicates.duplicate_percentage,
        )
