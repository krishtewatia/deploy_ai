import numpy as np
import pandas as pd
import pytest
from backend.app.analysis.statistics import StatisticsAnalyzer
from backend.app.analysis.schemas import StatisticsReport


def test_statistics_analyzer_single_numerical_column() -> None:
    """Test analyzer with a single numerical column."""
    df = pd.DataFrame({
        "Age": [20, 21, 22],
        "Name": ["A", "B", "C"]
    })

    analyzer = StatisticsAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, StatisticsReport)
    assert "Age" in report.numerical_summary
    assert "Name" not in report.numerical_summary

    summary = report.numerical_summary["Age"]
    assert summary["mean"] == 21.0
    assert summary["median"] == 21.0
    assert summary["std"] == 1.0
    assert summary["min"] == 20.0
    assert summary["max"] == 22.0


def test_statistics_analyzer_multiple_numerical_columns() -> None:
    """Test analyzer with multiple numerical columns."""
    df = pd.DataFrame({
        "Age": [10, 20, 30],
        "Score": [1.5, 2.5, 3.5]
    })

    analyzer = StatisticsAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, StatisticsReport)
    assert "Age" in report.numerical_summary
    assert "Score" in report.numerical_summary

    age_summary = report.numerical_summary["Age"]
    assert age_summary["mean"] == 20.0
    assert age_summary["median"] == 20.0
    assert age_summary["std"] == 10.0

    score_summary = report.numerical_summary["Score"]
    assert score_summary["mean"] == 2.5
    assert score_summary["median"] == 2.5
    assert score_summary["std"] == 1.0


def test_statistics_analyzer_no_numerical_columns() -> None:
    """Test analyzer with a dataset that has no numerical columns."""
    df = pd.DataFrame({
        "Name": ["A", "B"],
        "City": ["X", "Y"]
    })

    analyzer = StatisticsAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, StatisticsReport)
    assert report.numerical_summary == {}


def test_statistics_analyzer_empty_dataframe() -> None:
    """Test analyzer with an empty DataFrame."""
    df_empty = pd.DataFrame()
    df_no_cols = pd.DataFrame(index=[0, 1])

    analyzer = StatisticsAnalyzer()

    report_empty = analyzer.analyze(df_empty)
    assert report_empty.numerical_summary == {}

    report_no_cols = analyzer.analyze(df_no_cols)
    assert report_no_cols.numerical_summary == {}


def test_statistics_analyzer_with_missing_values() -> None:
    """Test analyzer with a dataset containing missing values (NaNs)."""
    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": [np.nan, np.nan, np.nan]
    })

    analyzer = StatisticsAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, StatisticsReport)
    assert "A" in report.numerical_summary
    assert "B" in report.numerical_summary

    a_summary = report.numerical_summary["A"]
    assert a_summary["mean"] == 2.0
    assert a_summary["median"] == 2.0
    assert a_summary["std"] == 1.41
    assert a_summary["min"] == 1.0
    assert a_summary["max"] == 3.0

    b_summary = report.numerical_summary["B"]
    assert b_summary["mean"] == 0.0
    assert b_summary["median"] == 0.0
    assert b_summary["std"] == 0.0
    assert b_summary["min"] == 0.0
    assert b_summary["max"] == 0.0
