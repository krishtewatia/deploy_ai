import pytest
from backend.app.analysis.report_generator import ReportGenerator
from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    ImbalanceReport,
    MissingValueReport,
    StatisticsReport,
)


@pytest.fixture
def mock_missing_values() -> MissingValueReport:
    return MissingValueReport(
        total_missing=5,
        missing_by_column={"col1": 5},
        missing_percentage={"col1": 50.0}
    )


@pytest.fixture
def mock_duplicates() -> DuplicateReport:
    return DuplicateReport(
        duplicate_rows=2,
        duplicate_percentage=20.0
    )


@pytest.fixture
def mock_statistics() -> StatisticsReport:
    return StatisticsReport(
        numerical_summary={
            "col1": {
                "mean": 10.0,
                "median": 10.0,
                "std": 1.0,
                "min": 9.0,
                "max": 11.0
            }
        }
    )


@pytest.fixture
def mock_imbalance() -> ImbalanceReport:
    return ImbalanceReport(
        imbalanced=True,
        distribution={"0": 9, "1": 1}
    )


def test_report_generator_with_imbalance(
    mock_missing_values: MissingValueReport,
    mock_duplicates: DuplicateReport,
    mock_statistics: StatisticsReport,
    mock_imbalance: ImbalanceReport,
) -> None:
    """Test generating a consolidated report including target class imbalance."""
    generator = ReportGenerator()
    report = generator.generate(
        missing_values=mock_missing_values,
        duplicates=mock_duplicates,
        statistics=mock_statistics,
        imbalance=mock_imbalance,
    )

    assert isinstance(report, DatasetAnalysisReport)
    # Verify all sections are preserved correctly
    assert report.missing_values == mock_missing_values
    assert report.duplicates == mock_duplicates
    assert report.statistics == mock_statistics
    assert report.imbalance == mock_imbalance


def test_report_generator_without_imbalance(
    mock_missing_values: MissingValueReport,
    mock_duplicates: DuplicateReport,
    mock_statistics: StatisticsReport,
) -> None:
    """Test generating a consolidated report without class imbalance."""
    generator = ReportGenerator()
    report = generator.generate(
        missing_values=mock_missing_values,
        duplicates=mock_duplicates,
        statistics=mock_statistics,
        imbalance=None,
    )

    assert isinstance(report, DatasetAnalysisReport)
    # Verify all sections are preserved correctly
    assert report.missing_values == mock_missing_values
    assert report.duplicates == mock_duplicates
    assert report.statistics == mock_statistics
    assert report.imbalance is None


def test_report_generator_with_column_profiles(
    mock_missing_values: MissingValueReport,
    mock_duplicates: DuplicateReport,
    mock_statistics: StatisticsReport,
) -> None:
    """Test generating a consolidated report with column profiles passed explicitly."""
    generator = ReportGenerator()
    column_profiles = {"col1": {"dtype": "int64", "unique_values": 2, "sample_values": [1, 2]}}
    report = generator.generate(
        missing_values=mock_missing_values,
        duplicates=mock_duplicates,
        statistics=mock_statistics,
        column_profiles=column_profiles,
    )

    assert isinstance(report, DatasetAnalysisReport)
    assert report.column_profiles == column_profiles
