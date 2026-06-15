import pandas as pd
import pytest
from backend.app.analysis.imbalance import ImbalanceAnalyzer
from backend.app.analysis.schemas import ImbalanceReport


def test_imbalance_analyzer_balanced_dataset() -> None:
    """Test analyzer with a balanced target distribution."""
    df = pd.DataFrame({
        "target": ["A", "B", "A", "B", "A", "B", "A", "B"]
    })

    analyzer = ImbalanceAnalyzer()
    report = analyzer.analyze(df, "target")

    assert isinstance(report, ImbalanceReport)
    assert report.imbalanced is False
    assert report.distribution == {"A": 4, "B": 4}


def test_imbalance_analyzer_imbalanced_dataset() -> None:
    """Test analyzer with an imbalanced target distribution."""
    df = pd.DataFrame({
        "target": ["A"] * 9 + ["B"]
    })

    analyzer = ImbalanceAnalyzer()
    report = analyzer.analyze(df, "target")

    assert isinstance(report, ImbalanceReport)
    assert report.imbalanced is True
    assert report.distribution == {"A": 9, "B": 1}


def test_imbalance_analyzer_missing_target_column() -> None:
    """Test analyzer raises ValueError when target column is missing."""
    df = pd.DataFrame({
        "feature": [1, 2, 3]
    })

    analyzer = ImbalanceAnalyzer()
    with pytest.raises(ValueError, match="Target column 'target' does not exist"):
        analyzer.analyze(df, "target")


def test_imbalance_analyzer_single_class_dataset() -> None:
    """Test analyzer with a target containing only a single class."""
    df = pd.DataFrame({
        "target": ["A"] * 5
    })

    analyzer = ImbalanceAnalyzer()
    report = analyzer.analyze(df, "target")

    assert isinstance(report, ImbalanceReport)
    assert report.imbalanced is False
    assert report.distribution == {"A": 5}


def test_imbalance_analyzer_empty_dataframe() -> None:
    """Test analyzer raises ValueError with an empty DataFrame."""
    df = pd.DataFrame()

    analyzer = ImbalanceAnalyzer()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        analyzer.analyze(df, "target")


def test_imbalance_analyzer_all_missing_values() -> None:
    """Test analyzer raises ValueError if target column contains only NaNs."""
    df = pd.DataFrame({
        "target": [None, None, None]
    })

    analyzer = ImbalanceAnalyzer()
    with pytest.raises(ValueError, match="has no valid non-null rows"):
        analyzer.analyze(df, "target")
