"""Tests for backend.app.eda.eda_service — targeting 100% coverage.

Covers:
    1.  Successful full pipeline execution.
    2.  Empty dataframe validation.
    3.  Invalid dataframe validation.
    4.  Descriptive analyzer failure wrapping.
    5.  Diagnostic analyzer failure wrapping.
    6.  Visualization recommender failure wrapping.
    7.  Chart generation failure wrapping.
    8.  Insight generation failure wrapping.
    9.  Correct output structure and fields.
    10. Serialization and deserialization compatibility.
    11. Invalid / empty dataset_id.
    12. Invalid analysis_report.
    13. Dynamic output path config for real ChartGenerator.
    14. Output path configuration failure on OSError.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    MissingValueReport,
    StatisticsReport,
)
from backend.app.eda.chart_generator import ChartGenerator
from backend.app.eda.descriptive_analyzer import DescriptiveAnalyzer
from backend.app.eda.diagnostic_analyzer import DiagnosticAnalyzer
from backend.app.eda.eda_service import EDAService, EDAServiceError
from backend.app.eda.eda_service_schemas import EDAServiceReport
from backend.app.eda.insight_generator import InsightGenerator
from backend.app.eda.insight_schemas import InsightReport
from backend.app.eda.schemas import (
    ChartType,
    DatasetSummary,
    DescriptiveAnalytics,
    DiagnosticAnalytics,
    Insight,
    Severity,
    VisualizationRecommendation,
)
from backend.app.eda.visualization_recommender import VisualizationRecommender


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_analysis_report() -> DatasetAnalysisReport:
    """Build a dummy DatasetAnalysisReport for testing."""
    return DatasetAnalysisReport(
        missing_values=MissingValueReport(
            total_missing=0,
            missing_by_column={},
            missing_percentage={},
        ),
        duplicates=DuplicateReport(
            duplicate_rows=0,
            duplicate_percentage=0.0,
        ),
        statistics=StatisticsReport(
            numerical_summary={},
        ),
        column_profiles={},
    )


def _build_descriptive() -> DescriptiveAnalytics:
    """Build DescriptiveAnalytics."""
    return DescriptiveAnalytics(
        dataset_summary=DatasetSummary(
            rows=100,
            columns=2,
            numerical_columns=["age"],
            categorical_columns=["name"],
            datetime_columns=[],
            missing_cells=0,
            duplicate_rows=0,
        ),
        key_findings=[
            Insight(
                title="Descriptive Key",
                description="Descriptive insight details.",
                severity=Severity.INFO,
            ),
        ],
    )


def _build_diagnostic() -> DiagnosticAnalytics:
    """Build DiagnosticAnalytics."""
    return DiagnosticAnalytics(
        correlation_findings=[
            Insight(
                title="Corr Title",
                description="Correlation details.",
                severity=Severity.INFO,
            )
        ],
        anomaly_findings=[],
    )


def _build_visualizations() -> list[VisualizationRecommendation]:
    """Build recommended visualizations list."""
    return [
        VisualizationRecommendation(
            chart_type=ChartType.HISTOGRAM,
            column_names=["age"],
            reason="Check distribution.",
        )
    ]


def _build_insight_report(dataset_id: str) -> InsightReport:
    """Build InsightReport."""
    return InsightReport(
        descriptive_insights=["Descriptive LLM text."],
        diagnostic_insights=["Diagnostic LLM text."],
        predictive_observations=["Predictive LLM text."],
        prescriptive_recommendations=["Prescriptive LLM text."],
        generated_at=datetime.now(timezone.utc),
        dataset_id=dataset_id,
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture for a simple valid pandas DataFrame."""
    return pd.DataFrame({"age": [20, 25, 30], "name": ["Alice", "Bob", "Charlie"]})


@pytest.fixture
def mock_components() -> tuple[
    MagicMock, MagicMock, MagicMock, MagicMock, MagicMock
]:
    """Fixture returning mocks for all EDAService components."""
    descriptive = MagicMock(spec=DescriptiveAnalyzer)
    diagnostic = MagicMock(spec=DiagnosticAnalyzer)
    recommender = MagicMock(spec=VisualizationRecommender)
    charts = MagicMock(spec=ChartGenerator)
    insights = MagicMock(spec=InsightGenerator)
    return descriptive, diagnostic, recommender, charts, insights


# ── Tests ───────────────────────────────────────────────────────────────────


def test_eda_service_successful_execution(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Test successful full pipeline execution returning correct structure."""
    m_descriptive, m_diagnostic, m_recommender, m_charts, m_insights = mock_components

    dataset_id = "ds_123"
    report_in = _build_analysis_report()

    # Stub the mock responses
    descriptive_out = _build_descriptive()
    diagnostic_out = _build_diagnostic()
    viz_out = _build_visualizations()
    charts_out = ["/path/to/chart1.png"]
    insights_out = _build_insight_report(dataset_id)

    m_descriptive.analyze.return_value = descriptive_out
    m_diagnostic.analyze.return_value = diagnostic_out
    m_recommender.recommend.return_value = viz_out
    m_charts.generate_charts.return_value = charts_out
    m_insights.generate_insights.return_value = insights_out

    service = EDAService(*mock_components)
    report = service.run(dataset_id, sample_df, report_in)

    # Asserts
    assert isinstance(report, EDAServiceReport)
    assert report.dataset_id == dataset_id
    assert report.descriptive == descriptive_out
    assert report.diagnostic == diagnostic_out
    assert report.visualizations == viz_out
    assert report.generated_charts == charts_out
    assert report.insights == insights_out
    assert report.generated_at is not None

    m_descriptive.analyze.assert_called_once_with(report_in)
    m_diagnostic.analyze.assert_called_once_with(sample_df, report_in)
    m_recommender.recommend.assert_called_once_with(report_in)
    m_charts.generate_charts.assert_called_once_with(sample_df, viz_out)
    m_insights.generate_insights.assert_called_once_with(
        dataset_id=dataset_id,
        descriptive=descriptive_out,
        diagnostic=diagnostic_out,
        visualizations=viz_out,
    )


def test_eda_service_empty_dataframe(mock_components: tuple) -> None:
    """Empty dataframe should raise EDAServiceError."""
    service = EDAService(*mock_components)
    empty_df = pd.DataFrame()
    report_in = _build_analysis_report()

    with pytest.raises(EDAServiceError, match="Empty dataframe"):
        service.run("ds_id", empty_df, report_in)


def test_eda_service_invalid_dataframe(mock_components: tuple) -> None:
    """Passing a non-pandas DataFrame should raise EDAServiceError."""
    service = EDAService(*mock_components)
    report_in = _build_analysis_report()

    with pytest.raises(EDAServiceError, match="Invalid dataframe"):
        service.run("ds_id", "not a dataframe", report_in)  # type: ignore


def test_eda_service_invalid_dataset_id(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Invalid or empty dataset_id should raise EDAServiceError."""
    service = EDAService(*mock_components)
    report_in = _build_analysis_report()

    with pytest.raises(EDAServiceError, match="Invalid dataset_id"):
        service.run("", sample_df, report_in)

    with pytest.raises(EDAServiceError, match="Invalid dataset_id"):
        service.run("  ", sample_df, report_in)

    with pytest.raises(EDAServiceError, match="Invalid dataset_id"):
        service.run(None, sample_df, report_in)  # type: ignore


def test_eda_service_invalid_analysis_report(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Passing an invalid analysis_report should raise EDAServiceError."""
    service = EDAService(*mock_components)

    with pytest.raises(EDAServiceError, match="Invalid analysis_report"):
        service.run("ds_id", sample_df, "not a report")  # type: ignore


def test_eda_service_descriptive_analyzer_failure(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Descriptive analyzer exception should be wrapped in EDAServiceError."""
    m_descriptive, _, _, _, _ = mock_components
    m_descriptive.analyze.side_effect = RuntimeError("Descriptive crash")

    service = EDAService(*mock_components)
    report_in = _build_analysis_report()

    with pytest.raises(EDAServiceError, match="Descriptive analyzer failure"):
        service.run("ds_id", sample_df, report_in)


def test_eda_service_diagnostic_analyzer_failure(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Diagnostic analyzer exception should be wrapped in EDAServiceError."""
    m_descriptive, m_diagnostic, _, _, _ = mock_components
    m_descriptive.analyze.return_value = _build_descriptive()
    m_diagnostic.analyze.side_effect = ValueError("Diagnostic error")

    service = EDAService(*mock_components)
    report_in = _build_analysis_report()

    with pytest.raises(EDAServiceError, match="Diagnostic analyzer failure"):
        service.run("ds_id", sample_df, report_in)


def test_eda_service_visualization_recommender_failure(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Visualization recommender exception should be wrapped in EDAServiceError."""
    m_descriptive, m_diagnostic, m_recommender, _, _ = mock_components
    m_descriptive.analyze.return_value = _build_descriptive()
    m_diagnostic.analyze.return_value = _build_diagnostic()
    m_recommender.recommend.side_effect = TypeError("Recommender failed")

    service = EDAService(*mock_components)
    report_in = _build_analysis_report()

    with pytest.raises(EDAServiceError, match="Visualization recommender failure"):
        service.run("ds_id", sample_df, report_in)


def test_eda_service_chart_generation_failure(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Chart generator exception should be wrapped in EDAServiceError."""
    m_descriptive, m_diagnostic, m_recommender, m_charts, _ = mock_components
    m_descriptive.analyze.return_value = _build_descriptive()
    m_diagnostic.analyze.return_value = _build_diagnostic()
    m_recommender.recommend.return_value = _build_visualizations()
    m_charts.generate_charts.side_effect = IOError("FileSystem error")

    service = EDAService(*mock_components)
    report_in = _build_analysis_report()

    with pytest.raises(EDAServiceError, match="Chart generation failure"):
        service.run("ds_id", sample_df, report_in)


def test_eda_service_insight_generation_failure(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Insight generator exception should be wrapped in EDAServiceError."""
    m_descriptive, m_diagnostic, m_recommender, m_charts, m_insights = mock_components
    m_descriptive.analyze.return_value = _build_descriptive()
    m_diagnostic.analyze.return_value = _build_diagnostic()
    m_recommender.recommend.return_value = _build_visualizations()
    m_charts.generate_charts.return_value = ["/path.png"]
    m_insights.generate_insights.side_effect = ValueError("Groq API rate limit")

    service = EDAService(*mock_components)
    report_in = _build_analysis_report()

    with pytest.raises(EDAServiceError, match="Insight generation failure"):
        service.run("ds_id", sample_df, report_in)


def test_eda_service_serialization_compatibility(
    sample_df: pd.DataFrame, mock_components: tuple
) -> None:
    """Test serialization/deserialization compatibility of the EDAServiceReport."""
    m_descriptive, m_diagnostic, m_recommender, m_charts, m_insights = mock_components

    dataset_id = "serial_ds"
    report_in = _build_analysis_report()

    descriptive_out = _build_descriptive()
    diagnostic_out = _build_diagnostic()
    viz_out = _build_visualizations()
    charts_out = ["/path/to/chart1.png"]
    insights_out = _build_insight_report(dataset_id)

    m_descriptive.analyze.return_value = descriptive_out
    m_diagnostic.analyze.return_value = diagnostic_out
    m_recommender.recommend.return_value = viz_out
    m_charts.generate_charts.return_value = charts_out
    m_insights.generate_insights.return_value = insights_out

    service = EDAService(*mock_components)
    report = service.run(dataset_id, sample_df, report_in)

    # model_dump round-trip
    dumped = report.model_dump()
    loaded = EDAServiceReport.model_validate(dumped)
    assert loaded.dataset_id == dataset_id
    assert loaded.descriptive.dataset_summary.rows == 100
    assert loaded.diagnostic.correlation_findings[0].title == "Corr Title"

    # model_dump_json round-trip
    json_str = report.model_dump_json()
    loaded_json = EDAServiceReport.model_validate_json(json_str)
    assert loaded_json.dataset_id == dataset_id
    assert loaded_json.descriptive.dataset_summary.rows == 100
    assert loaded_json.diagnostic.correlation_findings[0].title == "Corr Title"


def test_eda_service_dynamic_chart_generator_config(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Ensure the service updates a real ChartGenerator dataset_id and creates the folder."""
    descriptive = MagicMock(spec=DescriptiveAnalyzer)
    diagnostic = MagicMock(spec=DiagnosticAnalyzer)
    recommender = MagicMock(spec=VisualizationRecommender)
    insights = MagicMock(spec=InsightGenerator)

    descriptive.analyze.return_value = _build_descriptive()
    diagnostic.analyze.return_value = _build_diagnostic()
    recommender.recommend.return_value = _build_visualizations()
    insights.generate_insights.return_value = _build_insight_report("my_custom_ds")

    # Use a real ChartGenerator instance
    charts = ChartGenerator(output_dir=str(tmp_path), dataset_id="original_id")
    # Stub generate_charts so it doesn't actually plot anything during execution
    charts.generate_charts = MagicMock(return_value=["/dummy_path.png"])

    service = EDAService(descriptive, diagnostic, recommender, charts, insights)
    service.run("my_custom_ds", sample_df, _build_analysis_report())

    assert charts.dataset_id == "my_custom_ds"
    assert charts.dataset_chart_dir == tmp_path / "my_custom_ds"
    assert (tmp_path / "my_custom_ds").exists()
    assert (tmp_path / "my_custom_ds").is_dir()


def test_eda_service_dynamic_chart_generator_config_failure(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    """OSError during dynamic directory creation should raise EDAServiceError."""
    descriptive = MagicMock(spec=DescriptiveAnalyzer)
    diagnostic = MagicMock(spec=DiagnosticAnalyzer)
    recommender = MagicMock(spec=VisualizationRecommender)
    insights = MagicMock(spec=InsightGenerator)

    charts = ChartGenerator(output_dir=str(tmp_path), dataset_id="some_id")

    service = EDAService(descriptive, diagnostic, recommender, charts, insights)
    report_in = _build_analysis_report()

    with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(EDAServiceError, match="Failed to configure chart generator output directory"):
            service.run("new_ds_id", sample_df, report_in)
