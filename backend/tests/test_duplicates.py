import pandas as pd
import pytest
from backend.app.analysis.duplicates import DuplicateAnalyzer
from backend.app.analysis.schemas import DuplicateReport


def test_duplicate_analyzer_with_duplicates() -> None:
    """Test duplicate analyzer with duplicate rows in DataFrame."""
    df = pd.DataFrame({
        "A": [1, 2, 1, 3, 2],
        "B": ["a", "b", "a", "c", "b"]
    })

    # Total rows: 5
    # Duplicate rows: [1, "a"] at index 2, and [2, "b"] at index 4 (total 2 duplicates)
    # Percentage: 2 / 5 * 100 = 40.0%
    analyzer = DuplicateAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, DuplicateReport)
    assert report.duplicate_rows == 2
    assert report.duplicate_percentage == 40.0


def test_duplicate_analyzer_without_duplicates() -> None:
    """Test duplicate analyzer with unique rows in DataFrame."""
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["a", "b", "c"]
    })

    analyzer = DuplicateAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, DuplicateReport)
    assert report.duplicate_rows == 0
    assert report.duplicate_percentage == 0.0


def test_duplicate_analyzer_empty_dataframe() -> None:
    """Test duplicate analyzer with an empty DataFrame."""
    df = pd.DataFrame()

    analyzer = DuplicateAnalyzer()
    report = analyzer.analyze(df)

    assert isinstance(report, DuplicateReport)
    assert report.duplicate_rows == 0
    assert report.duplicate_percentage == 0.0


def test_duplicate_analyzer_all_except_one() -> None:
    """Test duplicate analyzer with N identical rows and 1 unique row."""
    # Total rows: 4
    # Three [1, "a"] rows (2 duplicates) and one unique [2, "b"] row
    # Percentage: 2 / 4 * 100 = 50.0%
    df = pd.DataFrame({
        "A": [1, 1, 1, 2],
        "B": ["a", "a", "a", "b"]
    })

    analyzer = DuplicateAnalyzer()
    report = analyzer.analyze(df)

    assert report.duplicate_rows == 2
    assert report.duplicate_percentage == 50.0
