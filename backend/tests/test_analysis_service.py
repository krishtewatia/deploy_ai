import pandas as pd
import pytest
from backend.app.analysis.analysis_service import AnalysisService
from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    ImbalanceReport,
    MissingValueReport,
    StatisticsReport,
)


def test_analysis_service_without_target_column() -> None:
    """Test analysis service processes DataFrame without a target column."""
    df = pd.DataFrame({
        "Age": [20, 21, 22],
        "Salary": [50000, 60000, 55000]
    })

    service = AnalysisService()
    report = service.analyze(df)

    assert isinstance(report, DatasetAnalysisReport)
    assert report.missing_values.total_missing == 0
    assert report.duplicates.duplicate_rows == 0
    assert "Age" in report.statistics.numerical_summary
    assert report.imbalance is None


def test_analysis_service_with_target_column() -> None:
    """Test analysis service processes DataFrame with a target column."""
    df = pd.DataFrame({
        "Age": [20, 21, 22, 23, 24],
        "Target": [0, 0, 0, 0, 1]
    })

    service = AnalysisService()
    report = service.analyze(df, target_column="Target")

    assert isinstance(report, DatasetAnalysisReport)
    assert report.missing_values.total_missing == 0
    assert report.duplicates.duplicate_rows == 0
    assert report.imbalance is not None
    assert report.imbalance.imbalanced is False
    assert report.imbalance.distribution == {"0": 4, "1": 1}


def test_analysis_service_empty_dataframe() -> None:
    """Test analysis service raises ValueError when DataFrame is empty."""
    df = pd.DataFrame()

    service = AnalysisService()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        service.analyze(df)


def test_analysis_service_dependency_injection() -> None:
    """Test analysis service with custom mock analyzers injected."""

    class MockMissingAnalyzer:
        def __init__(self):
            self.called = False
        def analyze(self, df):
            self.called = True
            return MissingValueReport(total_missing=9, missing_by_column={}, missing_percentage={})

    class MockDuplicateAnalyzer:
        def __init__(self):
            self.called = False
        def analyze(self, df):
            self.called = True
            return DuplicateReport(duplicate_rows=7, duplicate_percentage=70.0)

    class MockStatisticsAnalyzer:
        def __init__(self):
            self.called = False
        def analyze(self, df):
            self.called = True
            return StatisticsReport(numerical_summary={"col": {"mean": 1.0, "median": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}})

    class MockImbalanceAnalyzer:
        def __init__(self):
            self.called = False
        def analyze(self, df, target_column):
            assert target_column == "Target"
            self.called = True
            return ImbalanceReport(imbalanced=True, distribution={"0": 1})

    class MockReportGenerator:
        def __init__(self):
            self.called = False
        def generate(self, missing_values, duplicates, statistics, imbalance, column_profiles=None):
            self.called = True
            return DatasetAnalysisReport(
                missing_values=missing_values,
                duplicates=duplicates,
                statistics=statistics,
                imbalance=imbalance,
                column_profiles=column_profiles if column_profiles is not None else {},
            )

    missing = MockMissingAnalyzer()
    duplicates = MockDuplicateAnalyzer()
    stats = MockStatisticsAnalyzer()
    imbalance = MockImbalanceAnalyzer()
    generator = MockReportGenerator()

    service = AnalysisService(
        missing_value_analyzer=missing,
        duplicate_analyzer=duplicates,
        statistics_analyzer=stats,
        imbalance_analyzer=imbalance,
        report_generator=generator,
    )

    df = pd.DataFrame({"Target": [1]})
    report = service.analyze(df, target_column="Target")

    assert missing.called is True
    assert duplicates.called is True
    assert stats.called is True
    assert imbalance.called is True
    assert generator.called is True

    assert report.missing_values.total_missing == 9
    assert report.duplicates.duplicate_rows == 7
    assert report.imbalance.imbalanced is True


def test_analysis_service_column_profiler_injection() -> None:
    """Test that custom ColumnProfiler can be injected into AnalysisService."""
    class MockColumnProfiler:
        def __init__(self):
            self.called = False
        def profile(self, df):
            self.called = True
            return {"mocked_col": {"dtype": "int", "unique_values": 1, "sample_values": [1]}}

    mock_profiler = MockColumnProfiler()
    service = AnalysisService(column_profiler=mock_profiler)
    df = pd.DataFrame({"col": [1]})
    report = service.analyze(df)

    assert mock_profiler.called is True
    assert "mocked_col" in report.column_profiles


def test_analysis_service_integration_column_profiles() -> None:
    """Integration test verifying column_profiles in DatasetAnalysisReport."""
    df = pd.DataFrame({
        "age": [20, 21, None],
        "department": ["AI", "ML", "AI"]
    })

    service = AnalysisService()
    report = service.analyze(df)

    assert isinstance(report, DatasetAnalysisReport)
    assert report.column_profiles is not None
    assert "age" in report.column_profiles
    assert "department" in report.column_profiles

    # Check that at least one column profile contains: dtype, unique_values, sample_values
    age_profile = report.column_profiles["age"]
    assert "dtype" in age_profile
    assert "unique_values" in age_profile
    assert "sample_values" in age_profile

    assert age_profile["unique_values"] == 2
    assert age_profile["sample_values"] == [20.0, 21.0]
