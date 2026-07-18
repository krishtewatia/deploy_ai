import numpy as np
import pandas as pd
import pytest
from backend.app.analysis.missing_values import MissingValueAnalyzer
from backend.app.analysis.schemas import MissingValueReport


def test_missing_values_analyzer_with_missing_values() -> None:
    """Test analyzer with a dataset containing missing values."""
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5],
        "B": ["a", "b", "c", None, "e"],
        "C": [1.1, 2.2, 3.3, 4.4, 5.5]
    })

    analyzer = MissingValueAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, MissingValueReport)
    assert report.total_missing == 2
    assert report.missing_by_column == {"A": 1, "B": 1}
    assert report.missing_percentage == {"A": 20.0, "B": 20.0}


def test_missing_values_analyzer_without_missing_values() -> None:
    """Test analyzer with a dataset that has no missing values."""
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["a", "b", "c"]
    })

    analyzer = MissingValueAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, MissingValueReport)
    assert report.total_missing == 0
    assert report.missing_by_column == {}
    assert report.missing_percentage == {}


def test_missing_values_analyzer_empty_dataframe() -> None:
    """Test analyzer with different types of empty DataFrames."""
    df_empty = pd.DataFrame()
    df_no_rows = pd.DataFrame(columns=["A", "B"])
    df_no_cols = pd.DataFrame(index=[0, 1])

    analyzer = MissingValueAnalyzer()

    # Empty DataFrame
    report_empty = analyzer.analyze(df_empty)
    assert report_empty.total_missing == 0
    assert report_empty.missing_by_column == {}
    assert report_empty.missing_percentage == {}

    # No rows
    report_no_rows = analyzer.analyze(df_no_rows)
    assert report_no_rows.total_missing == 0
    assert report_no_rows.missing_by_column == {}
    assert report_no_rows.missing_percentage == {}

    # No columns
    report_no_cols = analyzer.analyze(df_no_cols)
    assert report_no_cols.total_missing == 0
    assert report_no_cols.missing_by_column == {}
    assert report_no_cols.missing_percentage == {}


def test_missing_values_analyzer_multiple_columns() -> None:
    """Test analyzer with multiple columns having missing values and check percentages."""
    df = pd.DataFrame({
        "A": [1, np.nan, 3, np.nan, 5, np.nan, 7, 8],
        "B": ["a", "b", "c", "d", None, "f", "g", "h"],
        "C": [10, 20, 30, 40, 50, 60, 70, 80]
    })

    analyzer = MissingValueAnalyzer()
    report = analyzer.analyze(df)

    assert report.total_missing == 4
    assert report.missing_by_column == {"A": 3, "B": 1}
    assert report.missing_percentage == {"A": 37.5, "B": 12.5}
