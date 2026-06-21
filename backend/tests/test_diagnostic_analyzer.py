"""Tests for backend.app.eda.diagnostic_analyzer — targeting 100% coverage.

Covers:
    1.  Strong correlation detection.
    2.  Moderate correlation detection.
    3.  Identifier detection.
    4.  Low variance detection.
    5.  Outlier detection.
    6.  Target candidate detection.
    7.  Leakage detection.
    8.  Multiple findings.
    9.  Empty DataFrame.
    10. Invalid input.
    11. Serialization compatibility.
    12. No findings scenario.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    MissingValueReport,
    StatisticsReport,
)
from backend.app.eda.diagnostic_analyzer import (
    DiagnosticAnalyzer,
    DiagnosticAnalyzerError,
)
from backend.app.eda.schemas import (
    DiagnosticAnalytics,
    Insight,
    Severity,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_report(
    *,
    column_profiles: dict[str, dict] | None = None,
    total_missing: int = 0,
    missing_by_column: dict[str, int] | None = None,
    missing_percentage: dict[str, float] | None = None,
    duplicate_rows: int = 0,
    duplicate_percentage: float = 0.0,
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
        statistics=StatisticsReport(numerical_summary={}),
        imbalance=None,
        column_profiles=column_profiles or {},
    )


@pytest.fixture()
def analyzer() -> DiagnosticAnalyzer:
    """Return a DiagnosticAnalyzer instance."""
    return DiagnosticAnalyzer()


# ── 1. Strong correlation detection ───────────────────────────────────────


def test_strong_correlation_detection(analyzer: DiagnosticAnalyzer) -> None:
    """Pairs with |r| >= 0.80 should produce a 'Strong Correlation' insight."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],  # perfectly correlated with a
    })
    report = _build_report(
        column_profiles={
            "a": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 5, "unique_percentage": 100.0},
            "b": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 5, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.correlation_findings]
    assert "Strong Correlation Detected" in titles

    strong = next(f for f in result.correlation_findings if f.title == "Strong Correlation Detected")
    assert strong.severity == Severity.INFO.value
    assert "a" in strong.description
    assert "b" in strong.description


# ── 2. Moderate correlation detection ─────────────────────────────────────


def test_moderate_correlation_detection(analyzer: DiagnosticAnalyzer) -> None:
    """Pairs with 0.50 <= |r| < 0.80 should produce 'Moderate Correlation'."""
    np.random.seed(42)
    x = np.arange(50, dtype=float)
    noise = np.random.normal(0, 15, size=50)
    y = x + noise

    df = pd.DataFrame({"x": x, "y": y})
    corr_val = abs(df["x"].corr(df["y"]))
    assert 0.50 <= corr_val < 0.80, f"Correlation is {corr_val}, adjust noise."

    report = _build_report(
        column_profiles={
            "x": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 50, "unique_percentage": 100.0},
            "y": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 50, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.correlation_findings]
    assert "Moderate Correlation Detected" in titles


def test_no_correlation_findings_below_threshold(analyzer: DiagnosticAnalyzer) -> None:
    """Pairs with |r| < 0.50 should not produce correlation findings."""
    np.random.seed(123)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
    })

    report = _build_report(
        column_profiles={
            "a": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 100, "unique_percentage": 100.0},
            "b": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 100, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)
    assert result.correlation_findings == []


def test_single_numeric_column_skips_correlation(analyzer: DiagnosticAnalyzer) -> None:
    """Fewer than 2 numerical columns should skip correlation analysis."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    report = _build_report(
        column_profiles={
            "a": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 3, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)
    assert result.correlation_findings == []


# ── 3. Identifier detection ──────────────────────────────────────────────


def test_identifier_detection(analyzer: DiagnosticAnalyzer) -> None:
    """Columns with id-like names AND >= 95% unique values should be flagged."""
    df = pd.DataFrame({"customer_id": range(100), "salary": range(100)})
    report = _build_report(
        column_profiles={
            "customer_id": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                            "unique_values": 100, "unique_percentage": 100.0},
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 100, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    ident_findings = [f for f in result.anomaly_findings if f.title == "Potential Identifier Column"]
    ident_cols = [f.description for f in ident_findings]

    assert any("customer_id" in d for d in ident_cols)
    assert not any("salary" in d for d in ident_cols)


def test_identifier_various_patterns(analyzer: DiagnosticAnalyzer) -> None:
    """All supported id-like naming patterns should be detected."""
    df = pd.DataFrame({
        "id": range(100),
        "uuid": range(100),
        "guid": range(100),
        "employee_id": range(100),
        "id_order": range(100),
        "identifier": range(100),
        "user_id": range(100),
        "account_id": range(100),
    })
    report = _build_report(
        column_profiles={
            col: {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 100, "unique_percentage": 100.0}
            for col in df.columns
        },
    )

    result = analyzer.analyze(df, report)

    ident_findings = [f for f in result.anomaly_findings if f.title == "Potential Identifier Column"]
    flagged_cols = " ".join(f.description for f in ident_findings)

    for col in ["id", "uuid", "guid", "employee_id", "id_order", "identifier", "user_id", "account_id"]:
        assert col in flagged_cols, f"{col} should be flagged as identifier"


def test_non_id_columns_not_flagged(analyzer: DiagnosticAnalyzer) -> None:
    """Feature columns like salary, experience, age should NOT be flagged."""
    df = pd.DataFrame({
        "salary": range(100),
        "experience": range(100),
        "age": range(100),
    })
    report = _build_report(
        column_profiles={
            col: {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 100, "unique_percentage": 100.0}
            for col in df.columns
        },
    )

    result = analyzer.analyze(df, report)

    ident_findings = [f for f in result.anomaly_findings if f.title == "Potential Identifier Column"]
    assert ident_findings == []


def test_id_name_low_uniqueness_not_flagged(analyzer: DiagnosticAnalyzer) -> None:
    """An id-like column with < 95% uniqueness should NOT be flagged."""
    df = pd.DataFrame({"customer_id": [1, 1, 2, 2, 3]})
    report = _build_report(
        column_profiles={
            "customer_id": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                            "unique_values": 3, "unique_percentage": 60.0},
        },
    )

    result = analyzer.analyze(df, report)

    ident_findings = [f for f in result.anomaly_findings if f.title == "Potential Identifier Column"]
    assert ident_findings == []


def test_datetime_column_excluded_from_identifiers(analyzer: DiagnosticAnalyzer) -> None:
    """Datetime columns should NOT be flagged as identifiers even if unique."""
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=100)})
    report = _build_report(
        column_profiles={
            "ts": {"is_numeric": False, "is_categorical": False, "is_datetime": True,
                   "unique_values": 100, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.anomaly_findings]
    assert "Potential Identifier Column" not in titles


# ── 4. Low variance detection ────────────────────────────────────────────


def test_low_variance_detection(analyzer: DiagnosticAnalyzer) -> None:
    """Categorical column with one value >= 95% should trigger low variance."""
    values = ["A"] * 96 + ["B"] * 4
    df = pd.DataFrame({"status": values})
    report = _build_report(
        column_profiles={
            "status": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                       "unique_values": 2, "unique_percentage": 2.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.anomaly_findings]
    assert "Low Variance Feature" in titles

    lv = next(f for f in result.anomaly_findings if f.title == "Low Variance Feature")
    assert "status" in lv.description
    assert lv.severity == Severity.WARNING.value


def test_no_low_variance_below_threshold(analyzer: DiagnosticAnalyzer) -> None:
    """Columns with balanced values should NOT trigger low variance."""
    df = pd.DataFrame({"status": ["A"] * 50 + ["B"] * 50})
    report = _build_report(
        column_profiles={
            "status": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                       "unique_values": 2, "unique_percentage": 2.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.anomaly_findings]
    assert "Low Variance Feature" not in titles


def test_low_variance_skips_empty_series(analyzer: DiagnosticAnalyzer) -> None:
    """All-NaN categorical columns should be safely skipped."""
    df = pd.DataFrame({"cat": [None, None, None]})
    report = _build_report(
        column_profiles={
            "cat": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                    "unique_values": 0, "unique_percentage": 0.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.anomaly_findings]
    assert "Low Variance Feature" not in titles


# ── 5. Outlier detection ─────────────────────────────────────────────────


def test_outlier_detection(analyzer: DiagnosticAnalyzer) -> None:
    """Numerical columns with IQR-based outliers should be flagged."""
    data = list(range(1, 101)) + [1000]
    df = pd.DataFrame({"value": data})
    report = _build_report(
        column_profiles={
            "value": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                      "unique_values": 101, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.anomaly_findings]
    assert "Outliers Detected" in titles

    outlier_insight = next(f for f in result.anomaly_findings if f.title == "Outliers Detected")
    assert "value" in outlier_insight.description
    assert outlier_insight.severity == Severity.WARNING.value


def test_no_outliers_for_uniform_data(analyzer: DiagnosticAnalyzer) -> None:
    """Uniform data should NOT trigger outlier detection."""
    df = pd.DataFrame({"value": list(range(1, 101))})
    report = _build_report(
        column_profiles={
            "value": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                      "unique_values": 100, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.anomaly_findings]
    assert "Outliers Detected" not in titles


def test_no_outliers_for_constant_column(analyzer: DiagnosticAnalyzer) -> None:
    """A constant numerical column (IQR = 0) should NOT trigger outliers."""
    df = pd.DataFrame({"const": [5] * 100})
    report = _build_report(
        column_profiles={
            "const": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                      "unique_values": 1, "unique_percentage": 1.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.anomaly_findings]
    assert "Outliers Detected" not in titles


def test_outlier_skips_empty_numeric(analyzer: DiagnosticAnalyzer) -> None:
    """All-NaN numerical column should be safely skipped."""
    df = pd.DataFrame({"num": [np.nan, np.nan, np.nan]})
    report = _build_report(
        column_profiles={
            "num": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                    "unique_values": 0, "unique_percentage": 0.0},
        },
    )

    result = analyzer.analyze(df, report)

    titles = [f.title for f in result.anomaly_findings]
    assert "Outliers Detected" not in titles


# ── 6. Target candidate detection ───────────────────────────────────────


def test_target_candidate_by_name(analyzer: DiagnosticAnalyzer) -> None:
    """Columns named 'target', 'label', 'class', 'outcome' should be flagged."""
    df = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "feature": [10, 20, 30, 40],
    })
    report = _build_report(
        column_profiles={
            "target": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 2, "unique_percentage": 50.0},
            "feature": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                        "unique_values": 4, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    target_findings = [
        f for f in result.anomaly_findings if f.title == "Potential Target Column"
    ]
    assert any("target" in f.description for f in target_findings)


def test_target_candidate_by_binary(analyzer: DiagnosticAnalyzer) -> None:
    """Binary columns (2 unique values) should be flagged as potential targets."""
    df = pd.DataFrame({
        "is_active": [0, 1, 0, 1],
        "score": [10, 20, 30, 40],
    })
    report = _build_report(
        column_profiles={
            "is_active": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                          "unique_values": 2, "unique_percentage": 50.0},
            "score": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                      "unique_values": 4, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    target_findings = [
        f for f in result.anomaly_findings if f.title == "Potential Target Column"
    ]
    assert any("is_active" in f.description for f in target_findings)


def test_target_name_not_duplicated_with_binary(analyzer: DiagnosticAnalyzer) -> None:
    """A column matching by name AND being binary should only appear once for name."""
    df = pd.DataFrame({"target": [0, 1, 0, 1]})
    report = _build_report(
        column_profiles={
            "target": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 2, "unique_percentage": 50.0},
        },
    )

    result = analyzer.analyze(df, report)

    target_findings = [
        f for f in result.anomaly_findings if f.title == "Potential Target Column"
    ]
    # Should get exactly 1: name-matched (binary rule skips already flagged)
    assert len(target_findings) == 1
    assert "name" in target_findings[0].description


# ── 7. Leakage detection ────────────────────────────────────────────────


def test_leakage_detection(analyzer: DiagnosticAnalyzer) -> None:
    """Columns named 'id', 'identifier', 'uuid' should be flagged."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "uuid": ["a-b-c", "d-e-f", "g-h-i"],
        "value": [10, 20, 30],
    })
    report = _build_report(
        column_profiles={
            "id": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                   "unique_values": 3, "unique_percentage": 100.0},
            "uuid": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                     "unique_values": 3, "unique_percentage": 100.0},
            "value": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                      "unique_values": 3, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    leakage = [f for f in result.anomaly_findings if f.title == "Potential Data Leakage Risk"]
    leakage_cols = [f.description for f in leakage]
    assert any("id" in d for d in leakage_cols)
    assert any("uuid" in d for d in leakage_cols)


def test_no_leakage_for_safe_names(analyzer: DiagnosticAnalyzer) -> None:
    """Columns without leakage patterns should not be flagged."""
    df = pd.DataFrame({"salary": [50000, 60000], "age": [30, 40]})
    report = _build_report(
        column_profiles={
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 2, "unique_percentage": 100.0},
            "age": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                    "unique_values": 2, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    leakage = [f for f in result.anomaly_findings if f.title == "Potential Data Leakage Risk"]
    assert leakage == []


# ── 8. Multiple findings ────────────────────────────────────────────────


def test_multiple_findings(analyzer: DiagnosticAnalyzer) -> None:
    """A rich dataset should produce findings across multiple categories."""
    data = {
        "customer_id": range(100),
        "salary": list(range(1, 100)) + [10000],  # outlier
        "experience": list(range(1, 100)) + [9999],  # correlated + outlier
        "status": ["A"] * 97 + ["B"] * 3,
        "target": [0, 1] * 50,
    }
    df = pd.DataFrame(data)
    report = _build_report(
        column_profiles={
            "customer_id": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                            "unique_values": 100, "unique_percentage": 100.0},
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 100, "unique_percentage": 100.0},
            "experience": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                           "unique_values": 100, "unique_percentage": 100.0},
            "status": {"is_numeric": False, "is_categorical": True, "is_datetime": False,
                       "unique_values": 2, "unique_percentage": 2.0},
            "target": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                       "unique_values": 2, "unique_percentage": 2.0},
        },
    )

    result = analyzer.analyze(df, report)

    all_titles = (
        [f.title for f in result.correlation_findings]
        + [f.title for f in result.anomaly_findings]
    )
    assert len(all_titles) >= 4
    assert "Potential Identifier Column" in all_titles  # customer_id
    assert "Outliers Detected" in all_titles
    assert "Low Variance Feature" in all_titles
    assert "Potential Target Column" in all_titles


# ── 9. Empty DataFrame ──────────────────────────────────────────────────


def test_empty_dataframe(analyzer: DiagnosticAnalyzer) -> None:
    """An empty DataFrame should produce no findings."""
    df = pd.DataFrame()
    report = _build_report()

    result = analyzer.analyze(df, report)

    assert isinstance(result, DiagnosticAnalytics)
    assert result.correlation_findings == []
    assert result.anomaly_findings == []


# ── 10. Invalid input ───────────────────────────────────────────────────


def test_invalid_dataframe_type(analyzer: DiagnosticAnalyzer) -> None:
    """Passing a non-DataFrame should raise DiagnosticAnalyzerError."""
    report = _build_report()

    with pytest.raises(DiagnosticAnalyzerError, match="Expected a pandas DataFrame"):
        analyzer.analyze("not_a_dataframe", report)  # type: ignore[arg-type]


def test_invalid_report_type(analyzer: DiagnosticAnalyzer) -> None:
    """Passing a non-DatasetAnalysisReport should raise DiagnosticAnalyzerError."""
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(DiagnosticAnalyzerError, match="Expected a DatasetAnalysisReport"):
        analyzer.analyze(df, "not_a_report")  # type: ignore[arg-type]


def test_unexpected_error_wrapped(analyzer: DiagnosticAnalyzer) -> None:
    """A non-DiagnosticAnalyzerError should be wrapped."""
    with patch.object(
        DiagnosticAnalyzer,
        "_analyze_correlations",
        side_effect=RuntimeError("unexpected boom"),
    ):
        df = pd.DataFrame({"a": [1]})
        report = _build_report(
            column_profiles={
                "a": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                      "unique_values": 1, "unique_percentage": 100.0},
            },
        )
        with pytest.raises(DiagnosticAnalyzerError, match="Failed to generate diagnostic analytics"):
            analyzer.analyze(df, report)


def test_diagnostic_analyzer_error_reraise(analyzer: DiagnosticAnalyzer) -> None:
    """A DiagnosticAnalyzerError raised internally is re-raised directly."""
    with patch.object(
        DiagnosticAnalyzer,
        "_validate_inputs",
        side_effect=DiagnosticAnalyzerError("validation failed"),
    ):
        df = pd.DataFrame({"a": [1]})
        report = _build_report()
        with pytest.raises(DiagnosticAnalyzerError, match="validation failed"):
            analyzer.analyze(df, report)


# ── 11. Serialization compatibility ─────────────────────────────────────


def test_serialization_compatibility(analyzer: DiagnosticAnalyzer) -> None:
    """DiagnosticAnalytics result should serialise and deserialise cleanly."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],
    })
    report = _build_report(
        column_profiles={
            "a": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 5, "unique_percentage": 100.0},
            "b": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 5, "unique_percentage": 100.0},
        },
    )

    result = analyzer.analyze(df, report)

    data = result.model_dump()
    reconstructed = DiagnosticAnalytics.model_validate(data)
    assert len(reconstructed.correlation_findings) == len(result.correlation_findings)

    json_str = result.model_dump_json()
    from_json = DiagnosticAnalytics.model_validate_json(json_str)
    assert from_json == result


# ── 12. No findings scenario ───────────────────────────────────────────


def test_no_findings_scenario(analyzer: DiagnosticAnalyzer) -> None:
    """A clean dataset should produce zero findings."""
    df = pd.DataFrame({
        "score": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    })
    report = _build_report(
        column_profiles={
            "score": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                      "unique_values": 10, "unique_percentage": 50.0},
        },
    )

    result = analyzer.analyze(df, report)

    assert result.correlation_findings == []
    assert result.anomaly_findings == []


# ── Extra: column not in DataFrame is safely skipped ────────────────────


def test_column_not_in_dataframe_skipped(analyzer: DiagnosticAnalyzer) -> None:
    """Columns in report but missing from DataFrame should not crash."""
    df = pd.DataFrame({"existing": [1, 2, 3]})
    report = _build_report(
        column_profiles={
            "existing": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                         "unique_values": 3, "unique_percentage": 100.0},
            "missing_col": {"is_numeric": True, "is_categorical": True, "is_datetime": False,
                            "unique_values": 5, "unique_percentage": 50.0},
        },
    )

    result = analyzer.analyze(df, report)

    assert isinstance(result, DiagnosticAnalytics)


# ── Extra: NaN correlation values are safely handled ────────────────────


def test_nan_correlation_skipped(analyzer: DiagnosticAnalyzer) -> None:
    """Columns producing NaN correlations should not crash or produce findings."""
    df = pd.DataFrame({
        "a": [1, 1, 1, 1, 1],
        "b": [2, 2, 2, 2, 2],  # zero std → NaN correlation
    })
    report = _build_report(
        column_profiles={
            "a": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 1, "unique_percentage": 20.0},
            "b": {"is_numeric": True, "is_categorical": False, "is_datetime": False,
                  "unique_values": 1, "unique_percentage": 20.0},
        },
    )

    result = analyzer.analyze(df, report)

    assert result.correlation_findings == []
