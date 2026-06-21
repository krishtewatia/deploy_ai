"""Tests for backend.app.eda.descriptive_analyzer — targeting 100% coverage.

Covers:
    1.  Empty dataset report.
    2.  Missing values insight.
    3.  High missingness insight.
    4.  Duplicate insight.
    5.  Large dataset insight.
    6.  Numerical columns insight.
    7.  Categorical columns insight.
    8.  Multiple insights generated.
    9.  Valid DatasetSummary generation.
    10. Exception handling.
    11. Serialization compatibility.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    MissingValueReport,
    StatisticsReport,
)
from backend.app.eda.descriptive_analyzer import (
    DescriptiveAnalyzer,
    DescriptiveAnalyzerError,
)
from backend.app.eda.schemas import (
    DatasetSummary,
    DescriptiveAnalytics,
    Insight,
    Severity,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_report(
    *,
    total_missing: int = 0,
    missing_by_column: dict[str, int] | None = None,
    missing_percentage: dict[str, float] | None = None,
    duplicate_rows: int = 0,
    duplicate_percentage: float = 0.0,
    numerical_summary: dict | None = None,
    column_profiles: dict[str, dict] | None = None,
) -> DatasetAnalysisReport:
    """Build a DatasetAnalysisReport with sensible defaults."""
    return DatasetAnalysisReport(
        missing_values=MissingValueReport(
            total_missing=total_missing,
            missing_by_column=missing_by_column or {},
            missing_percentage=missing_percentage or {},
        ),
        duplicates=DuplicateReport(
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
        ),
        statistics=StatisticsReport(
            numerical_summary=numerical_summary or {},
        ),
        imbalance=None,
        column_profiles=column_profiles or {},
    )


@pytest.fixture()
def analyzer() -> DescriptiveAnalyzer:
    """Return a DescriptiveAnalyzer instance."""
    return DescriptiveAnalyzer()


# ── 1. Empty dataset report ────────────────────────────────────────────────


def test_empty_dataset_report(analyzer: DescriptiveAnalyzer) -> None:
    """An empty report should produce a zero-summary and no findings."""
    report = _build_report()
    result = analyzer.analyze(report)

    assert isinstance(result, DescriptiveAnalytics)
    assert result.dataset_summary.rows == 0
    assert result.dataset_summary.columns == 0
    assert result.dataset_summary.missing_cells == 0
    assert result.dataset_summary.duplicate_rows == 0
    assert result.dataset_summary.numerical_columns == []
    assert result.dataset_summary.categorical_columns == []
    assert result.dataset_summary.datetime_columns == []
    assert result.key_findings == []


# ── 2. Missing values insight ──────────────────────────────────────────────


def test_missing_values_insight(analyzer: DescriptiveAnalyzer) -> None:
    """Missing values > 0 should produce a 'Missing Values Detected' insight."""
    report = _build_report(
        total_missing=5,
        missing_by_column={"age": 3, "salary": 2},
        missing_percentage={"age": 10.0, "salary": 6.7},
        column_profiles={
            "age": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                    "unique_values": 27, "unique_percentage": 90.0},
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 28, "unique_percentage": 93.33},
        },
    )

    result = analyzer.analyze(report)

    titles = [f.title for f in result.key_findings]
    assert "Missing Values Detected" in titles

    missing_insight = next(f for f in result.key_findings if f.title == "Missing Values Detected")
    assert missing_insight.severity == Severity.WARNING.value
    assert "5" in missing_insight.description


# ── 3. High missingness insight ────────────────────────────────────────────


def test_high_missingness_insight(analyzer: DescriptiveAnalyzer) -> None:
    """Columns with > 30% missing should produce a 'High Missingness' insight."""
    report = _build_report(
        total_missing=40,
        missing_by_column={"notes": 40},
        missing_percentage={"notes": 40.0},
        column_profiles={
            "notes": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                      "unique_values": 60, "unique_percentage": 60.0},
        },
    )

    result = analyzer.analyze(report)

    titles = [f.title for f in result.key_findings]
    assert "High Missingness" in titles

    high_miss = next(f for f in result.key_findings if f.title == "High Missingness")
    assert high_miss.severity == Severity.CRITICAL.value
    assert "notes" in high_miss.description
    assert "40.0%" in high_miss.description


def test_no_high_missingness_at_threshold(analyzer: DescriptiveAnalyzer) -> None:
    """Columns with exactly 30% should NOT trigger 'High Missingness'."""
    report = _build_report(
        total_missing=30,
        missing_by_column={"col": 30},
        missing_percentage={"col": 30.0},
        column_profiles={
            "col": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                    "unique_values": 70, "unique_percentage": 70.0},
        },
    )

    result = analyzer.analyze(report)

    titles = [f.title for f in result.key_findings]
    assert "High Missingness" not in titles


# ── 4. Duplicate insight ──────────────────────────────────────────────────


def test_duplicate_insight(analyzer: DescriptiveAnalyzer) -> None:
    """Duplicate rows > 0 should produce a 'Duplicate Records Found' insight."""
    report = _build_report(
        duplicate_rows=10,
        duplicate_percentage=5.0,
        column_profiles={
            "id": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                   "unique_values": 190, "unique_percentage": 95.0},
        },
    )

    result = analyzer.analyze(report)

    titles = [f.title for f in result.key_findings]
    assert "Duplicate Records Found" in titles

    dup_insight = next(f for f in result.key_findings if f.title == "Duplicate Records Found")
    assert dup_insight.severity == Severity.WARNING.value
    assert "10" in dup_insight.description


# ── 5. Large dataset insight ──────────────────────────────────────────────


def test_large_dataset_insight(analyzer: DescriptiveAnalyzer) -> None:
    """Datasets with > 100,000 rows should produce a 'Large Dataset' insight."""
    report = _build_report(
        duplicate_rows=10,
        duplicate_percentage=0.005,  # 10 / 200_000 * 100 = 0.005
        column_profiles={
            "id": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                   "unique_values": 199_990, "unique_percentage": 99.995},
        },
    )

    result = analyzer.analyze(report)

    assert result.dataset_summary.rows > 100_000

    titles = [f.title for f in result.key_findings]
    assert "Large Dataset" in titles

    large_insight = next(f for f in result.key_findings if f.title == "Large Dataset")
    assert large_insight.severity == Severity.INFO.value


def test_no_large_dataset_at_threshold(analyzer: DescriptiveAnalyzer) -> None:
    """Datasets with exactly 100,000 rows should NOT trigger 'Large Dataset'."""
    # 50 duplicates at 0.05% → 50 * 100 / 0.05 = 100_000
    report = _build_report(
        duplicate_rows=50,
        duplicate_percentage=0.05,
        column_profiles={
            "id": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                   "unique_values": 99_950, "unique_percentage": 99.95},
        },
    )

    result = analyzer.analyze(report)

    assert result.dataset_summary.rows == 100_000
    titles = [f.title for f in result.key_findings]
    assert "Large Dataset" not in titles


# ── 6. Numerical columns insight ─────────────────────────────────────────


def test_numerical_columns_insight(analyzer: DescriptiveAnalyzer) -> None:
    """Numerical columns should produce a 'Numerical Features Available' insight."""
    report = _build_report(
        column_profiles={
            "age": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                    "unique_values": 50, "unique_percentage": 50.0},
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 80, "unique_percentage": 80.0},
        },
    )

    result = analyzer.analyze(report)

    titles = [f.title for f in result.key_findings]
    assert "Numerical Features Available" in titles

    num_insight = next(f for f in result.key_findings if f.title == "Numerical Features Available")
    assert num_insight.severity == Severity.INFO.value
    assert "age" in num_insight.description
    assert "salary" in num_insight.description


# ── 7. Categorical columns insight ───────────────────────────────────────


def test_categorical_columns_insight(analyzer: DescriptiveAnalyzer) -> None:
    """Categorical columns should produce a 'Categorical Features Available' insight."""
    report = _build_report(
        column_profiles={
            "department": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                           "unique_values": 5, "unique_percentage": 5.0},
        },
    )

    result = analyzer.analyze(report)

    titles = [f.title for f in result.key_findings]
    assert "Categorical Features Available" in titles

    cat_insight = next(f for f in result.key_findings if f.title == "Categorical Features Available")
    assert cat_insight.severity == Severity.INFO.value
    assert "department" in cat_insight.description


# ── 8. Multiple insights generated ───────────────────────────────────────


def test_multiple_insights_generated(analyzer: DescriptiveAnalyzer) -> None:
    """A rich report should produce multiple insights simultaneously."""
    report = _build_report(
        total_missing=50,
        missing_by_column={"notes": 40, "salary": 10},
        missing_percentage={"notes": 40.0, "salary": 10.0},
        duplicate_rows=5,
        duplicate_percentage=5.0,
        column_profiles={
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 85, "unique_percentage": 85.0},
            "notes": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                      "unique_values": 50, "unique_percentage": 50.0},
        },
    )

    result = analyzer.analyze(report)

    titles = [f.title for f in result.key_findings]
    assert "Missing Values Detected" in titles
    assert "High Missingness" in titles
    assert "Duplicate Records Found" in titles
    assert "Numerical Features Available" in titles
    assert "Categorical Features Available" in titles
    assert len(result.key_findings) >= 5


# ── 9. Valid DatasetSummary generation ────────────────────────────────────


def test_valid_dataset_summary_generation(analyzer: DescriptiveAnalyzer) -> None:
    """The DatasetSummary should accurately reflect the report contents."""
    report = _build_report(
        total_missing=3,
        missing_by_column={"age": 3},
        missing_percentage={"age": 10.0},
        duplicate_rows=2,
        duplicate_percentage=6.67,
        column_profiles={
            "age": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                    "unique_values": 25, "unique_percentage": 83.33},
            "department": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                           "unique_values": 3, "unique_percentage": 10.0},
            "hire_date": {"is_numeric": False, "is_categorical": False, "is_datetime": True,
                          "unique_values": 28, "unique_percentage": 93.33},
        },
    )

    result = analyzer.analyze(report)
    summary = result.dataset_summary

    assert isinstance(summary, DatasetSummary)
    assert summary.columns == 3
    assert summary.rows == 30  # 2 * 100 / 6.67 ≈ 30
    assert summary.numerical_columns == ["age"]
    assert summary.categorical_columns == ["department"]
    assert summary.datetime_columns == ["hire_date"]
    assert summary.missing_cells == 3
    assert summary.duplicate_rows == 2


def test_row_count_inferred_from_missing_data(analyzer: DescriptiveAnalyzer) -> None:
    """When there are no duplicates, rows should be inferred from missing data."""
    report = _build_report(
        total_missing=10,
        missing_by_column={"age": 10},
        missing_percentage={"age": 20.0},
        duplicate_rows=0,
        duplicate_percentage=0.0,
        column_profiles={
            "age": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                    "unique_values": 40, "unique_percentage": 80.0},
        },
    )

    result = analyzer.analyze(report)

    assert result.dataset_summary.rows == 50  # 10 * 100 / 20.0


def test_row_count_inferred_from_column_profiles(analyzer: DescriptiveAnalyzer) -> None:
    """When neither duplicates nor missing data exist, infer from unique values."""
    report = _build_report(
        column_profiles={
            "id": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                   "unique_values": 100, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(report)

    assert result.dataset_summary.rows == 100  # 100 * 100 / 100.0


def test_row_count_zero_when_no_data_available(analyzer: DescriptiveAnalyzer) -> None:
    """When no inference source is available, rows should be 0."""
    report = _build_report(
        column_profiles={
            "empty_col": {"is_numeric": False, "is_categorical": False, "is_datetime": False},
        },
    )

    result = analyzer.analyze(report)

    assert result.dataset_summary.rows == 0


# ── 10. Exception handling ────────────────────────────────────────────────


def test_malformed_report_raises_error(analyzer: DescriptiveAnalyzer) -> None:
    """A report with broken attributes should raise DescriptiveAnalyzerError."""
    broken_report = MagicMock(spec=DatasetAnalysisReport)
    broken_report.column_profiles = None
    broken_report.missing_values = MagicMock()
    broken_report.missing_values.total_missing = "not_an_int"  # Will fail pydantic
    broken_report.duplicates = MagicMock()
    broken_report.duplicates.duplicate_rows = None
    broken_report.duplicates.duplicate_percentage = None

    with pytest.raises(DescriptiveAnalyzerError):
        analyzer.analyze(broken_report)


def test_exception_wraps_cause(analyzer: DescriptiveAnalyzer) -> None:
    """DescriptiveAnalyzerError should preserve the original exception chain."""
    broken_report = MagicMock(spec=DatasetAnalysisReport)
    type(broken_report).column_profiles = PropertyMock(side_effect=AttributeError("boom"))

    with pytest.raises(DescriptiveAnalyzerError) as exc_info:
        analyzer.analyze(broken_report)

    assert exc_info.value.__cause__ is not None


def test_descriptive_analyzer_error_reraise(analyzer: DescriptiveAnalyzer) -> None:
    """A DescriptiveAnalyzerError raised in _build_dataset_summary is re-raised directly."""
    with patch.object(
        DescriptiveAnalyzer,
        "_build_dataset_summary",
        side_effect=DescriptiveAnalyzerError("summary failed"),
    ):
        report = _build_report()
        with pytest.raises(DescriptiveAnalyzerError, match="summary failed"):
            analyzer.analyze(report)


def test_unexpected_error_wrapped(analyzer: DescriptiveAnalyzer) -> None:
    """A non-DescriptiveAnalyzerError in the pipeline is wrapped."""
    with patch.object(
        DescriptiveAnalyzer,
        "_generate_key_findings",
        side_effect=RuntimeError("unexpected boom"),
    ):
        report = _build_report(
            column_profiles={
                "col": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                        "unique_values": 10, "unique_percentage": 100.0},
            },
        )
        with pytest.raises(DescriptiveAnalyzerError, match="Failed to generate descriptive analytics"):
            analyzer.analyze(report)


# ── 11. Serialization compatibility ──────────────────────────────────────


def test_serialization_compatibility(analyzer: DescriptiveAnalyzer) -> None:
    """DescriptiveAnalytics result should serialise and deserialise cleanly."""
    report = _build_report(
        total_missing=2,
        missing_by_column={"col_a": 2},
        missing_percentage={"col_a": 10.0},
        duplicate_rows=1,
        duplicate_percentage=5.0,
        column_profiles={
            "col_a": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                      "unique_values": 18, "unique_percentage": 90.0},
        },
    )

    result = analyzer.analyze(report)

    # model_dump roundtrip
    data = result.model_dump()
    reconstructed = DescriptiveAnalytics.model_validate(data)
    assert reconstructed.dataset_summary.missing_cells == result.dataset_summary.missing_cells
    assert len(reconstructed.key_findings) == len(result.key_findings)

    # JSON roundtrip
    json_str = result.model_dump_json()
    from_json = DescriptiveAnalytics.model_validate_json(json_str)
    assert from_json == result
