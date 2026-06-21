"""Tests for backend.app.eda.chart_generator — targeting 100% coverage.

Covers:
    1.  Histogram generation.
    2.  Boxplot generation.
    3.  Correlation heatmap generation.
    4.  Bar chart generation.
    5.  Missing value heatmap generation.
    6.  Line chart generation.
    7.  Unsupported chart type.
    8.  Missing column.
    9.  Empty dataframe.
    10. Output directory creation.
    11. File existence verification.
    12. Multiple chart generation.
    13. Failure cases (invalid types, plotting errors, directory creation errors).
    14. Dataset ID directory creation (provided ID vs automatic UUID).
    15. Isolation (different dataset IDs do not overwrite each other).
    16. dataset_chart_dir property.
    17. Batch generate_charts helper and its validation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from backend.app.eda.chart_generator import ChartGenerator, ChartGenerationError
from backend.app.eda.schemas import ChartType, VisualizationRecommendation


# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_sample_df() -> pd.DataFrame:
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "department": ["HR", "Eng", "Eng", "HR", "Sales"],
            "join_date": ["2020-01-01", "2021-02-01", "2022-03-01", "2023-04-01", "2024-05-01"],
            "missing_col": [1.0, None, 3.0, None, 5.0],
        }
    )


# ── 10. & 14. Output & Dataset Directory Creation Tests ──────────────────────


def test_output_directory_creation(tmp_path: Path) -> None:
    """It should automatically create the base and dataset-specific output directory if it doesn't exist."""
    target_dir = tmp_path / "new_reports" / "charts"
    assert not target_dir.exists()

    generator = ChartGenerator(output_dir=str(target_dir), dataset_id="test_dataset_123")
    assert target_dir.exists()
    assert generator.dataset_chart_dir == (target_dir / "test_dataset_123").resolve()
    assert generator.dataset_chart_dir.exists()


def test_automatic_uuid_directory_creation(tmp_path: Path) -> None:
    """It should automatically generate a unique 8-character hex dataset ID if not provided."""
    target_dir = tmp_path / "new_reports" / "charts"
    assert not target_dir.exists()

    generator = ChartGenerator(output_dir=str(target_dir))
    assert target_dir.exists()
    assert generator.dataset_id is not None
    assert len(generator.dataset_id) == 8
    assert generator.dataset_chart_dir == (target_dir / generator.dataset_id).resolve()
    assert generator.dataset_chart_dir.exists()


def test_output_directory_creation_failure() -> None:
    """It should raise ChartGenerationError if directory creation fails."""
    with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(ChartGenerationError, match="Failed to create dataset directory"):
            ChartGenerator(output_dir="/invalid/path/that/fails")


# ── 15. Charts Isolation Verification ───────────────────────────────────────


def test_charts_isolation_non_overwrite(tmp_path: Path) -> None:
    """Charts from different datasets should not overwrite each other."""
    df = _create_sample_df()
    rec = VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["salary"],
        reason="Visualize salary distribution.",
    )

    # Dataset A
    generator_a = ChartGenerator(output_dir=str(tmp_path), dataset_id="dataset_A")
    path_a = generator_a.generate_chart(df, rec)

    # Dataset B
    generator_b = ChartGenerator(output_dir=str(tmp_path), dataset_id="dataset_B")
    path_b = generator_b.generate_chart(df, rec)

    assert path_a != path_b
    assert Path(path_a).exists()
    assert Path(path_b).exists()


# ── 16. dataset_chart_dir Property ──────────────────────────────────────────


def test_dataset_chart_dir_property(tmp_path: Path) -> None:
    """It should expose the correct path via the dataset_chart_dir property."""
    generator = ChartGenerator(output_dir=str(tmp_path), dataset_id="dataset_XYZ")
    assert isinstance(generator.dataset_chart_dir, Path)
    assert generator.dataset_chart_dir == (tmp_path / "dataset_XYZ").resolve()


# ── 1. Histogram Generation ─────────────────────────────────────────────────


def test_histogram_generation(tmp_path: Path) -> None:
    """It should generate a histogram for a numerical column and save it as PNG in dataset dir."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path), dataset_id="hist_ds")

    rec = VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["salary"],
        reason="Visualize salary distribution.",
    )

    path_str = generator.generate_chart(df, rec)
    saved_path = Path(path_str)

    assert saved_path.is_absolute()
    assert saved_path.exists()
    assert saved_path.name == "salary_histogram.png"
    assert saved_path.parent == generator.dataset_chart_dir


def test_histogram_generation_empty_data(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if column has only NaN values."""
    df = pd.DataFrame({"empty_num": [None, None, None]})
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["empty_num"],
        reason="Visualize empty distribution.",
    )

    with pytest.raises(ChartGenerationError, match="has no valid numerical data"):
        generator.generate_chart(df, rec)


# ── 2. Boxplot Generation ───────────────────────────────────────────────────


def test_boxplot_generation(tmp_path: Path) -> None:
    """It should generate a boxplot for a numerical column and save it as PNG."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.BOXPLOT,
        column_names=["age"],
        reason="Detect age outliers.",
    )

    path_str = generator.generate_chart(df, rec)
    saved_path = Path(path_str)

    assert saved_path.is_absolute()
    assert saved_path.exists()
    assert saved_path.name == "age_boxplot.png"
    assert saved_path.parent == generator.dataset_chart_dir


def test_boxplot_generation_empty_data(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if column has only NaN values for boxplot."""
    df = pd.DataFrame({"empty_num": [None, None, None]})
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.BOXPLOT,
        column_names=["empty_num"],
        reason="Visualize empty boxplot.",
    )

    with pytest.raises(ChartGenerationError, match="has no valid numerical data"):
        generator.generate_chart(df, rec)


# ── 3. Correlation Heatmap Generation ───────────────────────────────────────


def test_correlation_heatmap_generation(tmp_path: Path) -> None:
    """It should generate a correlation heatmap for numerical columns."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.CORRELATION_HEATMAP,
        column_names=["age", "salary"],
        reason="Analyze relationship.",
    )

    path_str = generator.generate_chart(df, rec)
    saved_path = Path(path_str)

    assert saved_path.is_absolute()
    assert saved_path.exists()
    assert saved_path.name == "correlation_heatmap.png"
    assert saved_path.parent == generator.dataset_chart_dir


def test_correlation_heatmap_insufficient_numeric(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if less than 2 numeric columns are provided."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.CORRELATION_HEATMAP,
        column_names=["age", "department"],  # Only 'age' is numeric
        reason="Analyze relationship.",
    )

    with pytest.raises(ChartGenerationError, match="requires at least 2 numerical columns"):
        generator.generate_chart(df, rec)


def test_correlation_heatmap_empty_matrix(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if correlation cannot be computed."""
    df = pd.DataFrame(
        {
            "col1": [1.0, 1.0, 1.0],  # Zero variance
            "col2": [2.0, 2.0, 2.0],  # Zero variance
        }
    )
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.CORRELATION_HEATMAP,
        column_names=["col1", "col2"],
        reason="Analyze relationship.",
    )

    with pytest.raises(ChartGenerationError, match="Cannot compute correlation matrix"):
        generator.generate_chart(df, rec)


# ── 4. Bar Chart Generation ─────────────────────────────────────────────────


def test_bar_chart_generation(tmp_path: Path) -> None:
    """It should generate a bar chart for a categorical column."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.BAR_CHART,
        column_names=["department"],
        reason="Compare frequencies.",
    )

    path_str = generator.generate_chart(df, rec)
    saved_path = Path(path_str)

    assert saved_path.is_absolute()
    assert saved_path.exists()
    assert saved_path.name == "department_bar_chart.png"
    assert saved_path.parent == generator.dataset_chart_dir


def test_bar_chart_generation_empty_data(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if column has only NaN values."""
    df = pd.DataFrame({"empty_cat": [None, None, None]})
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.BAR_CHART,
        column_names=["empty_cat"],
        reason="Visualize empty bar chart.",
    )

    with pytest.raises(ChartGenerationError, match="has no data to plot a bar chart"):
        generator.generate_chart(df, rec)


# ── 5. Missing Value Heatmap Generation ─────────────────────────────────────


def test_missing_value_heatmap_generation(tmp_path: Path) -> None:
    """It should generate a missing value heatmap and save it as PNG."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.MISSING_VALUE_HEATMAP,
        column_names=["missing_col"],
        reason="Show missing patterns.",
    )

    path_str = generator.generate_chart(df, rec)
    saved_path = Path(path_str)

    assert saved_path.is_absolute()
    assert saved_path.exists()
    assert saved_path.name == "missing_value_heatmap.png"
    assert saved_path.parent == generator.dataset_chart_dir


# ── 6. Line Chart Generation ────────────────────────────────────────────────


def test_line_chart_generation(tmp_path: Path) -> None:
    """It should generate a line chart for a temporal column and save it as PNG."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.LINE_CHART,
        column_names=["join_date"],
        reason="Temporal trends.",
    )

    path_str = generator.generate_chart(df, rec)
    saved_path = Path(path_str)

    assert saved_path.is_absolute()
    assert saved_path.exists()
    assert saved_path.name == "join_date_line_chart.png"
    assert saved_path.parent == generator.dataset_chart_dir


def test_line_chart_unparseable_datetime(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if the column values cannot be parsed as datetime."""
    df = pd.DataFrame({"bad_date": ["not a date", "another bad one"]})
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.LINE_CHART,
        column_names=["bad_date"],
        reason="Temporal trends.",
    )

    with patch("pandas.to_datetime", side_effect=ValueError("could not parse")):
        with pytest.raises(ChartGenerationError, match="could not be parsed as datetime"):
            generator.generate_chart(df, rec)


def test_line_chart_empty_temporal_data(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if the temporal column is completely empty after dropna."""
    df = pd.DataFrame({"join_date": [None, None]})
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.LINE_CHART,
        column_names=["join_date"],
        reason="Temporal trends.",
    )

    with pytest.raises(ChartGenerationError, match="No valid temporal data"):
        generator.generate_chart(df, rec)


# ── 7. Unsupported Chart Type ───────────────────────────────────────────────


def test_unsupported_chart_type(tmp_path: Path) -> None:
    """It should raise ChartGenerationError for unsupported chart types like scatterplot or pie_chart."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.SCATTERPLOT,
        column_names=["age", "salary"],
        reason="Show correlation scatter.",
    )

    with pytest.raises(ChartGenerationError, match="Unsupported chart type: scatterplot"):
        generator.generate_chart(df, rec)


# ── 8. Missing Column ───────────────────────────────────────────────────────


def test_missing_column(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if a specified column is not in the DataFrame."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["non_existent_column"],
        reason="Visualize missing distribution.",
    )

    with pytest.raises(ChartGenerationError, match="not found in DataFrame"):
        generator.generate_chart(df, rec)


# ── 9. Empty Dataframe ──────────────────────────────────────────────────────


def test_empty_dataframe(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if the input DataFrame is empty."""
    df = pd.DataFrame()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["age"],
        reason="Visualize distribution.",
    )

    with pytest.raises(ChartGenerationError, match="Cannot generate chart from an empty DataFrame"):
        generator.generate_chart(df, rec)


# ── 11. File Existence Verification ─────────────────────────────────────────


def test_file_existence_verification(tmp_path: Path) -> None:
    """It should verify that the returned path is absolute and the file actually exists on the filesystem."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    rec = VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["age"],
        reason="Visualize age distribution.",
    )

    path_str = generator.generate_chart(df, rec)
    saved_path = Path(path_str)

    assert saved_path.is_absolute()
    assert saved_path.exists()
    assert saved_path.is_file()


# ── 12. & 17. Multiple & Batch Chart Generation ─────────────────────────────


def test_multiple_chart_generation(tmp_path: Path) -> None:
    """It should support generating multiple charts without conflicts or memory leaks."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    recs = [
        VisualizationRecommendation(
            chart_type=ChartType.HISTOGRAM,
            column_names=["age"],
            reason="Age distribution.",
        ),
        VisualizationRecommendation(
            chart_type=ChartType.BOXPLOT,
            column_names=["salary"],
            reason="Salary spread.",
        ),
        VisualizationRecommendation(
            chart_type=ChartType.BAR_CHART,
            column_names=["department"],
            reason="Department frequencies.",
        ),
    ]

    paths = []
    for rec in recs:
        path = generator.generate_chart(df, rec)
        paths.append(Path(path))

    # Verify all files are created
    for path in paths:
        assert path.exists()
        assert path.is_file()

    # Verify they have unique, deterministic names
    assert len(set(paths)) == 3
    filenames = {p.name for p in paths}
    assert filenames == {"age_histogram.png", "salary_boxplot.png", "department_bar_chart.png"}


def test_batch_generate_charts(tmp_path: Path) -> None:
    """It should generate multiple charts in batch using the generate_charts method."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path), dataset_id="batch_dataset")

    recs = [
        VisualizationRecommendation(
            chart_type=ChartType.HISTOGRAM,
            column_names=["age"],
            reason="Age distribution.",
        ),
        VisualizationRecommendation(
            chart_type=ChartType.BOXPLOT,
            column_names=["salary"],
            reason="Salary spread.",
        ),
    ]

    paths = generator.generate_charts(df, recs)
    assert len(paths) == 2

    for p in paths:
        saved_path = Path(p)
        assert saved_path.is_absolute()
        assert saved_path.exists()
        assert saved_path.parent == generator.dataset_chart_dir

    names = {Path(p).name for p in paths}
    assert names == {"age_histogram.png", "salary_boxplot.png"}


def test_batch_generate_charts_invalid_input(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if recommendations is not a list."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))

    with pytest.raises(ChartGenerationError, match="Input 'recommendations' must be a list"):
        generator.generate_charts(df, "not a list")  # type: ignore


# ── Extra Input Validation & Unexpected Failures ────────────────────────────


def test_invalid_dataframe_type(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if the df is not a pandas DataFrame."""
    generator = ChartGenerator(output_dir=str(tmp_path))
    rec = VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["age"],
        reason="Age distribution.",
    )
    with pytest.raises(ChartGenerationError, match="Input 'df' must be a pandas DataFrame"):
        generator.generate_chart("not a dataframe", rec)  # type: ignore


def test_invalid_recommendation_type(tmp_path: Path) -> None:
    """It should raise ChartGenerationError if the recommendation is not a VisualizationRecommendation."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))
    with pytest.raises(ChartGenerationError, match="Input 'recommendation' must be a VisualizationRecommendation"):
        generator.generate_chart(df, "not a recommendation")  # type: ignore


def test_unexpected_plotting_failures_are_wrapped(tmp_path: Path) -> None:
    """It should wrap general plotting runtime exceptions in ChartGenerationError."""
    df = _create_sample_df()
    generator = ChartGenerator(output_dir=str(tmp_path))
    rec = VisualizationRecommendation(
        chart_type=ChartType.HISTOGRAM,
        column_names=["age"],
        reason="Age distribution.",
    )

    with patch("matplotlib.pyplot.subplots", side_effect=RuntimeError("unexpected plotting error")):
        with pytest.raises(ChartGenerationError, match="Failed to generate chart for type 'histogram'"):
            generator.generate_chart(df, rec)
