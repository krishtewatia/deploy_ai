"""Tests for backend.app.eda.schemas — targeting 100% coverage.

Covers:
    1.  DatasetSummary creation.
    2.  Insight creation.
    3.  VisualizationRecommendation creation.
    4.  FeatureSuggestion creation.
    5.  DescriptiveAnalytics creation.
    6.  DiagnosticAnalytics creation.
    7.  EDAReport creation.
    8.  Invalid severity.
    9.  Invalid priority.
    10. Invalid chart type.
    11. Serialization.
    12. Deserialization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.eda.schemas import (
    ChartType,
    DatasetSummary,
    DescriptiveAnalytics,
    DiagnosticAnalytics,
    EDAReport,
    FeatureSuggestion,
    Insight,
    Priority,
    Severity,
    VisualizationRecommendation,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def dataset_summary() -> DatasetSummary:
    """Return a valid DatasetSummary."""
    return DatasetSummary(
        rows=1000,
        columns=12,
        numerical_columns=["age", "salary"],
        categorical_columns=["department", "gender"],
        datetime_columns=["hire_date"],
        missing_cells=45,
        duplicate_rows=10,
    )


@pytest.fixture()
def insight() -> Insight:
    """Return a valid Insight."""
    return Insight(
        title="High Missing Rate",
        description="Column 'salary' has 20% missing values.",
        severity=Severity.WARNING,
    )


@pytest.fixture()
def visualization_recommendation() -> VisualizationRecommendation:
    """Return a valid VisualizationRecommendation."""
    return VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["salary"],
        reason="Salary distribution is right-skewed.",
    )


@pytest.fixture()
def feature_suggestion() -> FeatureSuggestion:
    """Return a valid FeatureSuggestion."""
    return FeatureSuggestion(
        feature_name="salary_log",
        source_columns=["salary"],
        description="Log-transform salary to reduce skewness.",
        priority=Priority.HIGH,
    )


@pytest.fixture()
def descriptive_analytics(
    dataset_summary: DatasetSummary,
    insight: Insight,
) -> DescriptiveAnalytics:
    """Return a valid DescriptiveAnalytics."""
    return DescriptiveAnalytics(
        dataset_summary=dataset_summary,
        key_findings=[insight],
    )


@pytest.fixture()
def diagnostic_analytics(insight: Insight) -> DiagnosticAnalytics:
    """Return a valid DiagnosticAnalytics."""
    return DiagnosticAnalytics(
        correlation_findings=[insight],
        anomaly_findings=[
            Insight(
                title="Outlier detected",
                description="Column 'age' has values beyond 3 std.",
                severity=Severity.CRITICAL,
            )
        ],
    )


@pytest.fixture()
def eda_report(
    descriptive_analytics: DescriptiveAnalytics,
    diagnostic_analytics: DiagnosticAnalytics,
    feature_suggestion: FeatureSuggestion,
    visualization_recommendation: VisualizationRecommendation,
) -> EDAReport:
    """Return a valid EDAReport."""
    return EDAReport(
        descriptive=descriptive_analytics,
        diagnostic=diagnostic_analytics,
        feature_suggestions=[feature_suggestion],
        visualization_recommendations=[visualization_recommendation],
        overall_summary="The dataset is moderately clean with actionable insights.",
    )


# ── 1. DatasetSummary creation ─────────────────────────────────────────────


class TestDatasetSummary:
    """Validate DatasetSummary construction and field constraints."""

    def test_creation(self, dataset_summary: DatasetSummary) -> None:
        assert dataset_summary.rows == 1000
        assert dataset_summary.columns == 12
        assert dataset_summary.numerical_columns == ["age", "salary"]
        assert dataset_summary.categorical_columns == ["department", "gender"]
        assert dataset_summary.datetime_columns == ["hire_date"]
        assert dataset_summary.missing_cells == 45
        assert dataset_summary.duplicate_rows == 10

    def test_defaults_for_column_lists(self) -> None:
        summary = DatasetSummary(
            rows=0,
            columns=0,
            missing_cells=0,
            duplicate_rows=0,
        )
        assert summary.numerical_columns == []
        assert summary.categorical_columns == []
        assert summary.datetime_columns == []

    def test_negative_rows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetSummary(
                rows=-1,
                columns=5,
                missing_cells=0,
                duplicate_rows=0,
            )

    def test_negative_missing_cells_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetSummary(
                rows=10,
                columns=5,
                missing_cells=-1,
                duplicate_rows=0,
            )


# ── 2. Insight creation ───────────────────────────────────────────────────


class TestInsight:
    """Validate Insight construction and severity constraint."""

    def test_creation(self, insight: Insight) -> None:
        assert insight.title == "High Missing Rate"
        assert insight.description == "Column 'salary' has 20% missing values."
        assert insight.severity == Severity.WARNING.value

    def test_all_severities(self) -> None:
        for severity in Severity:
            i = Insight(
                title="Test",
                description="Test description.",
                severity=severity,
            )
            assert i.severity == severity.value

    def test_severity_from_string(self) -> None:
        i = Insight(
            title="Test",
            description="Test description.",
            severity="info",
        )
        assert i.severity == "info"


# ── 3. VisualizationRecommendation creation ────────────────────────────────


class TestVisualizationRecommendation:
    """Validate VisualizationRecommendation construction and chart_type."""

    def test_creation(
        self, visualization_recommendation: VisualizationRecommendation
    ) -> None:
        assert visualization_recommendation.chart_type == ChartType.HISTOGRAM.value
        assert visualization_recommendation.column_names == ["salary"]
        assert visualization_recommendation.reason == "Salary distribution is right-skewed."

    def test_all_chart_types(self) -> None:
        for chart in ChartType:
            rec = VisualizationRecommendation(
                chart_type=chart,
                column_names=["col_a"],
                reason="Valid chart type.",
            )
            assert rec.chart_type == chart.value

    def test_multiple_columns(self) -> None:
        rec = VisualizationRecommendation(
            chart_type=ChartType.SCATTERPLOT,
            column_names=["age", "salary"],
            reason="Explore bivariate relationship.",
        )
        assert rec.column_names == ["age", "salary"]


# ── 4. FeatureSuggestion creation ──────────────────────────────────────────


class TestFeatureSuggestion:
    """Validate FeatureSuggestion construction and priority constraint."""

    def test_creation(self, feature_suggestion: FeatureSuggestion) -> None:
        assert feature_suggestion.feature_name == "salary_log"
        assert feature_suggestion.source_columns == ["salary"]
        assert feature_suggestion.description == "Log-transform salary to reduce skewness."
        assert feature_suggestion.priority == Priority.HIGH.value

    def test_all_priorities(self) -> None:
        for priority in Priority:
            fs = FeatureSuggestion(
                feature_name="feat",
                source_columns=["col"],
                description="A feature.",
                priority=priority,
            )
            assert fs.priority == priority.value

    def test_multiple_source_columns(self) -> None:
        fs = FeatureSuggestion(
            feature_name="interaction",
            source_columns=["age", "salary"],
            description="Interaction between age and salary.",
            priority=Priority.MEDIUM,
        )
        assert fs.source_columns == ["age", "salary"]


# ── 5. DescriptiveAnalytics creation ──────────────────────────────────────


class TestDescriptiveAnalytics:
    """Validate DescriptiveAnalytics construction."""

    def test_creation(
        self,
        descriptive_analytics: DescriptiveAnalytics,
        dataset_summary: DatasetSummary,
    ) -> None:
        assert descriptive_analytics.dataset_summary == dataset_summary
        assert len(descriptive_analytics.key_findings) == 1

    def test_empty_findings(self, dataset_summary: DatasetSummary) -> None:
        da = DescriptiveAnalytics(dataset_summary=dataset_summary)
        assert da.key_findings == []


# ── 6. DiagnosticAnalytics creation ───────────────────────────────────────


class TestDiagnosticAnalytics:
    """Validate DiagnosticAnalytics construction."""

    def test_creation(self, diagnostic_analytics: DiagnosticAnalytics) -> None:
        assert len(diagnostic_analytics.correlation_findings) == 1
        assert len(diagnostic_analytics.anomaly_findings) == 1

    def test_empty_findings(self) -> None:
        da = DiagnosticAnalytics()
        assert da.correlation_findings == []
        assert da.anomaly_findings == []


# ── 7. EDAReport creation ────────────────────────────────────────────────


class TestEDAReport:
    """Validate EDAReport construction."""

    def test_creation(self, eda_report: EDAReport) -> None:
        assert eda_report.descriptive is not None
        assert eda_report.diagnostic is not None
        assert len(eda_report.feature_suggestions) == 1
        assert len(eda_report.visualization_recommendations) == 1
        assert eda_report.overall_summary == (
            "The dataset is moderately clean with actionable insights."
        )

    def test_empty_lists(
        self,
        descriptive_analytics: DescriptiveAnalytics,
        diagnostic_analytics: DiagnosticAnalytics,
    ) -> None:
        report = EDAReport(
            descriptive=descriptive_analytics,
            diagnostic=diagnostic_analytics,
            overall_summary="Minimal report.",
        )
        assert report.feature_suggestions == []
        assert report.visualization_recommendations == []


# ── 8. Invalid severity ──────────────────────────────────────────────────


class TestInvalidSeverity:
    """Rejected severity values should raise ValidationError."""

    def test_invalid_severity_string(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Insight(
                title="Bad severity",
                description="This should fail.",
                severity="catastrophic",
            )
        assert "severity" in str(exc_info.value).lower()

    def test_empty_severity(self) -> None:
        with pytest.raises(ValidationError):
            Insight(
                title="Empty severity",
                description="This should fail.",
                severity="",
            )


# ── 9. Invalid priority ─────────────────────────────────────────────────


class TestInvalidPriority:
    """Rejected priority values should raise ValidationError."""

    def test_invalid_priority_string(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FeatureSuggestion(
                feature_name="feat",
                source_columns=["col"],
                description="A feature.",
                priority="urgent",
            )
        assert "priority" in str(exc_info.value).lower()

    def test_empty_priority(self) -> None:
        with pytest.raises(ValidationError):
            FeatureSuggestion(
                feature_name="feat",
                source_columns=["col"],
                description="A feature.",
                priority="",
            )


# ── 10. Invalid chart type ──────────────────────────────────────────────


class TestInvalidChartType:
    """Rejected chart_type values should raise ValidationError."""

    def test_invalid_chart_type_string(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            VisualizationRecommendation(
                chart_type="treemap",
                column_names=["col"],
                reason="Not supported.",
            )
        assert "chart_type" in str(exc_info.value).lower()

    def test_empty_chart_type(self) -> None:
        with pytest.raises(ValidationError):
            VisualizationRecommendation(
                chart_type="",
                column_names=["col"],
                reason="Not supported.",
            )


# ── 11. Serialization ──────────────────────────────────────────────────


class TestSerialization:
    """model_dump and model_dump_json should produce valid output."""

    def test_eda_report_model_dump(self, eda_report: EDAReport) -> None:
        data = eda_report.model_dump()
        assert isinstance(data, dict)
        assert "descriptive" in data
        assert "diagnostic" in data
        assert "feature_suggestions" in data
        assert "visualization_recommendations" in data
        assert "overall_summary" in data

    def test_eda_report_model_dump_json(self, eda_report: EDAReport) -> None:
        json_str = eda_report.model_dump_json()
        assert isinstance(json_str, str)
        assert "descriptive" in json_str

    def test_severity_serialises_as_string(self, insight: Insight) -> None:
        data = insight.model_dump()
        assert data["severity"] == "warning"

    def test_chart_type_serialises_as_string(
        self, visualization_recommendation: VisualizationRecommendation
    ) -> None:
        data = visualization_recommendation.model_dump()
        assert data["chart_type"] == "histogram"

    def test_priority_serialises_as_string(
        self, feature_suggestion: FeatureSuggestion
    ) -> None:
        data = feature_suggestion.model_dump()
        assert data["priority"] == "high"

    def test_dataset_summary_model_dump(
        self, dataset_summary: DatasetSummary
    ) -> None:
        data = dataset_summary.model_dump()
        assert data["rows"] == 1000
        assert data["columns"] == 12


# ── 12. Deserialization ─────────────────────────────────────────────────


class TestDeserialization:
    """model_validate should reconstruct models from plain dicts."""

    def test_insight_from_dict(self) -> None:
        data = {
            "title": "Test insight",
            "description": "Reconstructed from dict.",
            "severity": "info",
        }
        insight = Insight.model_validate(data)
        assert insight.title == "Test insight"
        assert insight.severity == "info"

    def test_visualization_recommendation_from_dict(self) -> None:
        data = {
            "chart_type": "boxplot",
            "column_names": ["age"],
            "reason": "Detect outliers.",
        }
        rec = VisualizationRecommendation.model_validate(data)
        assert rec.chart_type == "boxplot"

    def test_feature_suggestion_from_dict(self) -> None:
        data = {
            "feature_name": "age_bucket",
            "source_columns": ["age"],
            "description": "Bin age into buckets.",
            "priority": "low",
        }
        fs = FeatureSuggestion.model_validate(data)
        assert fs.priority == "low"

    def test_eda_report_from_dict(self, eda_report: EDAReport) -> None:
        data = eda_report.model_dump()
        reconstructed = EDAReport.model_validate(data)
        assert reconstructed.overall_summary == eda_report.overall_summary
        assert (
            reconstructed.descriptive.dataset_summary.rows
            == eda_report.descriptive.dataset_summary.rows
        )

    def test_eda_report_roundtrip_json(self, eda_report: EDAReport) -> None:
        json_str = eda_report.model_dump_json()
        reconstructed = EDAReport.model_validate_json(json_str)
        assert reconstructed == eda_report

    def test_dataset_summary_from_dict(self) -> None:
        data = {
            "rows": 500,
            "columns": 8,
            "numerical_columns": ["a"],
            "categorical_columns": ["b"],
            "datetime_columns": [],
            "missing_cells": 10,
            "duplicate_rows": 2,
        }
        summary = DatasetSummary.model_validate(data)
        assert summary.rows == 500
        assert summary.columns == 8
