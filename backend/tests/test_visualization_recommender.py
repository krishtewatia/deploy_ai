"""Tests for backend.app.eda.visualization_recommender — targeting 100% coverage.

Covers:
    1.  Histogram recommendation.
    2.  Boxplot recommendation.
    3.  Correlation heatmap recommendation.
    4.  Bar chart recommendation.
    5.  Missing value heatmap recommendation.
    6.  Line chart recommendation.
    7.  Multiple recommendations.
    8.  Empty report.
    9.  Invalid report.
    10. Deduplication.
    11. Serialization compatibility.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    MissingValueReport,
    StatisticsReport,
)
from backend.app.eda.schemas import (
    ChartType,
    VisualizationRecommendation,
)
from backend.app.eda.visualization_recommender import (
    VisualizationRecommender,
    VisualizationRecommenderError,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_report(
    *,
    column_profiles: dict[str, dict] | None = None,
    total_missing: int = 0,
    missing_by_column: dict[str, int] | None = None,
    missing_percentage: dict[str, float] | None = None,
) -> DatasetAnalysisReport:
    """Build a DatasetAnalysisReport with sensible defaults."""
    return DatasetAnalysisReport(
        missing_values=MissingValueReport(
            total_missing=total_missing,
            missing_by_column=missing_by_column or {},
            missing_percentage=missing_percentage or {},
        ),
        duplicates=DuplicateReport(
            duplicate_rows=0,
            duplicate_percentage=0.0,
        ),
        statistics=StatisticsReport(numerical_summary={}),
        imbalance=None,
        column_profiles=column_profiles or {},
    )


@pytest.fixture()
def recommender() -> VisualizationRecommender:
    """Return a VisualizationRecommender instance."""
    return VisualizationRecommender()


# ── Assertion helpers ───────────────────────────────────────────────────────


def _chart_types(recs: list[VisualizationRecommendation]) -> list[str]:
    """Extract chart_type values from a recommendation list."""
    return [r.chart_type for r in recs]


def _recs_of_type(
    recs: list[VisualizationRecommendation],
    chart_type: str,
) -> list[VisualizationRecommendation]:
    """Filter recommendations by chart type."""
    return [r for r in recs if r.chart_type == chart_type]


# ── 1. Histogram recommendation ────────────────────────────────────────────


def test_histogram_recommendation(recommender: VisualizationRecommender) -> None:
    """Each numerical column should produce a histogram recommendation."""
    report = _build_report(
        column_profiles={
            "age": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    histograms = _recs_of_type(result, ChartType.HISTOGRAM.value)
    assert len(histograms) == 2

    cols = {r.column_names[0] for r in histograms}
    assert cols == {"age", "salary"}

    for h in histograms:
        assert h.reason == "Visualize distribution and skewness."


# ── 2. Boxplot recommendation ──────────────────────────────────────────────


def test_boxplot_recommendation(recommender: VisualizationRecommender) -> None:
    """Each numerical column should produce a boxplot recommendation."""
    report = _build_report(
        column_profiles={
            "score": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    boxplots = _recs_of_type(result, ChartType.BOXPLOT.value)
    assert len(boxplots) == 1
    assert boxplots[0].column_names == ["score"]
    assert boxplots[0].reason == "Detect outliers and spread."


# ── 3. Correlation heatmap recommendation ───────────────────────────────────


def test_correlation_heatmap_recommendation(recommender: VisualizationRecommender) -> None:
    """Two or more numerical columns should produce a correlation heatmap."""
    report = _build_report(
        column_profiles={
            "a": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
            "b": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    heatmaps = _recs_of_type(result, ChartType.CORRELATION_HEATMAP.value)
    assert len(heatmaps) == 1
    assert sorted(heatmaps[0].column_names) == ["a", "b"]
    assert heatmaps[0].reason == "Analyze feature relationships."


def test_no_correlation_heatmap_single_column(recommender: VisualizationRecommender) -> None:
    """A single numerical column should NOT produce a correlation heatmap."""
    report = _build_report(
        column_profiles={
            "only": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    heatmaps = _recs_of_type(result, ChartType.CORRELATION_HEATMAP.value)
    assert heatmaps == []


# ── 4. Bar chart recommendation ────────────────────────────────────────────


def test_bar_chart_recommendation(recommender: VisualizationRecommender) -> None:
    """Each categorical column should produce a bar chart recommendation."""
    report = _build_report(
        column_profiles={
            "department": {"is_numeric": False, "is_categorical": True, "is_datetime": False},
            "city": {"is_numeric": False, "is_categorical": True, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    bars = _recs_of_type(result, ChartType.BAR_CHART.value)
    assert len(bars) == 2

    cols = {r.column_names[0] for r in bars}
    assert cols == {"department", "city"}

    for b in bars:
        assert b.reason == "Compare category frequencies."


# ── 5. Missing value heatmap recommendation ─────────────────────────────────


def test_missing_value_heatmap_recommendation(recommender: VisualizationRecommender) -> None:
    """Missing values should produce a missing-value heatmap recommendation."""
    report = _build_report(
        total_missing=10,
        missing_by_column={"age": 5, "salary": 5},
        missing_percentage={"age": 10.0, "salary": 10.0},
        column_profiles={
            "age": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    missing = _recs_of_type(result, ChartType.MISSING_VALUE_HEATMAP.value)
    assert len(missing) == 1
    assert sorted(missing[0].column_names) == ["age", "salary"]
    assert missing[0].reason == "Visualize missing-data patterns."


def test_no_missing_heatmap_when_clean(recommender: VisualizationRecommender) -> None:
    """No missing values should NOT produce a missing-value heatmap."""
    report = _build_report(
        column_profiles={
            "x": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    missing = _recs_of_type(result, ChartType.MISSING_VALUE_HEATMAP.value)
    assert missing == []


def test_no_missing_heatmap_when_missing_by_column_empty(
    recommender: VisualizationRecommender,
) -> None:
    """total_missing > 0 but empty missing_by_column should NOT produce a heatmap."""
    report = _build_report(
        total_missing=5,
        missing_by_column={},
        missing_percentage={},
        column_profiles={
            "x": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    missing = _recs_of_type(result, ChartType.MISSING_VALUE_HEATMAP.value)
    assert missing == []


# ── 6. Line chart recommendation ───────────────────────────────────────────


def test_line_chart_recommendation(recommender: VisualizationRecommender) -> None:
    """Datetime columns should produce line chart recommendations."""
    report = _build_report(
        column_profiles={
            "created_at": {"is_numeric": False, "is_categorical": False, "is_datetime": True},
        },
    )

    result = recommender.recommend(report)

    lines = _recs_of_type(result, ChartType.LINE_CHART.value)
    assert len(lines) == 1
    assert lines[0].column_names == ["created_at"]
    assert lines[0].reason == "Visualize temporal trends."


def test_no_line_chart_without_datetime(recommender: VisualizationRecommender) -> None:
    """No datetime columns should NOT produce line charts."""
    report = _build_report(
        column_profiles={
            "value": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)

    lines = _recs_of_type(result, ChartType.LINE_CHART.value)
    assert lines == []


# ── 7. Multiple recommendations ────────────────────────────────────────────


def test_multiple_recommendations(recommender: VisualizationRecommender) -> None:
    """A rich report should produce recommendations across multiple rules."""
    report = _build_report(
        total_missing=5,
        missing_by_column={"salary": 5},
        missing_percentage={"salary": 10.0},
        column_profiles={
            "age": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
            "salary": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
            "department": {"is_numeric": False, "is_categorical": True, "is_datetime": False},
            "hire_date": {"is_numeric": False, "is_categorical": False, "is_datetime": True},
        },
    )

    result = recommender.recommend(report)

    types = _chart_types(result)
    assert ChartType.HISTOGRAM.value in types
    assert ChartType.BOXPLOT.value in types
    assert ChartType.CORRELATION_HEATMAP.value in types
    assert ChartType.BAR_CHART.value in types
    assert ChartType.MISSING_VALUE_HEATMAP.value in types
    assert ChartType.LINE_CHART.value in types

    # 2 histograms + 2 boxplots + 1 heatmap + 1 bar + 1 missing + 1 line = 8
    assert len(result) == 8


# ── 8. Empty report ────────────────────────────────────────────────────────


def test_empty_report(recommender: VisualizationRecommender) -> None:
    """An empty report should produce no recommendations."""
    report = _build_report()

    result = recommender.recommend(report)

    assert result == []


# ── 9. Invalid report ──────────────────────────────────────────────────────


def test_invalid_report_type(recommender: VisualizationRecommender) -> None:
    """Passing a non-DatasetAnalysisReport should raise an error."""
    with pytest.raises(VisualizationRecommenderError, match="Expected a DatasetAnalysisReport"):
        recommender.recommend("not_a_report")  # type: ignore[arg-type]


def test_unexpected_error_wrapped(recommender: VisualizationRecommender) -> None:
    """A non-VisualizationRecommenderError should be wrapped."""
    with patch.object(
        VisualizationRecommender,
        "_recommend_histograms",
        side_effect=RuntimeError("unexpected boom"),
    ):
        report = _build_report(
            column_profiles={
                "a": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
            },
        )
        with pytest.raises(
            VisualizationRecommenderError,
            match="Failed to generate visualization recommendations",
        ):
            recommender.recommend(report)


def test_recommender_error_reraise(recommender: VisualizationRecommender) -> None:
    """A VisualizationRecommenderError raised internally is re-raised directly."""
    with patch.object(
        VisualizationRecommender,
        "_validate_input",
        side_effect=VisualizationRecommenderError("bad input"),
    ):
        report = _build_report()
        with pytest.raises(VisualizationRecommenderError, match="bad input"):
            recommender.recommend(report)


# ── 10. Deduplication ──────────────────────────────────────────────────────


def test_deduplication(recommender: VisualizationRecommender) -> None:
    """Identical recommendations should be deduplicated."""
    recs = [
        VisualizationRecommendation(
            chart_type=ChartType.HISTOGRAM,
            column_names=["age"],
            reason="Visualize distribution and skewness.",
        ),
        VisualizationRecommendation(
            chart_type=ChartType.HISTOGRAM,
            column_names=["age"],
            reason="Visualize distribution and skewness.",
        ),
    ]

    result = VisualizationRecommender._deduplicate(recs)

    assert len(result) == 1
    assert result[0].column_names == ["age"]


def test_deduplication_preserves_different(recommender: VisualizationRecommender) -> None:
    """Different recommendations should NOT be deduplicated."""
    recs = [
        VisualizationRecommendation(
            chart_type=ChartType.HISTOGRAM,
            column_names=["age"],
            reason="Visualize distribution and skewness.",
        ),
        VisualizationRecommendation(
            chart_type=ChartType.BOXPLOT,
            column_names=["age"],
            reason="Detect outliers and spread.",
        ),
    ]

    result = VisualizationRecommender._deduplicate(recs)

    assert len(result) == 2


# ── 11. Serialization compatibility ─────────────────────────────────────────


def test_serialization_compatibility(recommender: VisualizationRecommender) -> None:
    """Recommendations should serialise and deserialise cleanly."""
    report = _build_report(
        total_missing=3,
        missing_by_column={"x": 3},
        missing_percentage={"x": 10.0},
        column_profiles={
            "x": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
            "y": {"is_numeric": True, "is_categorical": False, "is_datetime": False},
        },
    )

    result = recommender.recommend(report)
    assert len(result) > 0

    for rec in result:
        data = rec.model_dump()
        reconstructed = VisualizationRecommendation.model_validate(data)
        assert reconstructed.chart_type == rec.chart_type
        assert reconstructed.column_names == rec.column_names

        json_str = rec.model_dump_json()
        from_json = VisualizationRecommendation.model_validate_json(json_str)
        assert from_json == rec
