"""Tests for backend.app.dataset_intelligence.context_builder.

Covers DatasetContextBuilder with realistic DatasetAnalysisReport objects
built from the ACTUAL existing analysis schemas.
"""

import json
from collections import OrderedDict

import pytest
from pydantic import ValidationError

from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    MissingValueReport,
    StatisticsReport,
)
from backend.app.dataset_intelligence.context_builder import (
    DatasetContextBuilder,
    DatasetContextBuilderError,
)
from backend.app.dataset_intelligence.schemas import DatasetContext


# ── Helpers ────────────────────────────────────────────────────────────


def _make_column_profile(
    *,
    dtype: str = "int64",
    unique_values: int = 80,
    unique_percentage: float = 8.0,
    missing_count: int = 5,
    missing_percentage: float = 0.5,
    sample_values: list | None = None,
    is_numeric: bool = True,
    is_categorical: bool = False,
    is_datetime: bool = False,
) -> dict:
    """Return a column profile dict matching ColumnProfiler output."""
    return {
        "dtype": dtype,
        "unique_values": unique_values,
        "unique_percentage": unique_percentage,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "sample_values": sample_values if sample_values is not None else [1, 2, 3],
        "is_numeric": is_numeric,
        "is_categorical": is_categorical,
        "is_datetime": is_datetime,
    }


def _make_analysis_report(
    *,
    column_profiles: dict | None = None,
    total_missing: int = 5,
    missing_by_column: dict | None = None,
    missing_percentage: dict | None = None,
    duplicate_rows: int = 10,
    duplicate_percentage: float = 1.0,
    numerical_summary: dict | None = None,
) -> DatasetAnalysisReport:
    """Return a realistic DatasetAnalysisReport."""
    if column_profiles is None:
        column_profiles = {
            "age": _make_column_profile(
                dtype="int64",
                unique_values=80,
                unique_percentage=8.0,
                missing_count=5,
                missing_percentage=0.5,
                sample_values=[21, 34, 55],
                is_numeric=True,
            ),
            "city": _make_column_profile(
                dtype="object",
                unique_values=5,
                unique_percentage=0.5,
                missing_count=0,
                missing_percentage=0.0,
                sample_values=["NYC", "LA", "Chicago"],
                is_numeric=False,
                is_categorical=True,
            ),
        }

    if missing_by_column is None:
        missing_by_column = {"age": 5}
    if missing_percentage is None:
        missing_percentage = {"age": 0.5}
    if numerical_summary is None:
        numerical_summary = {
            "age": {
                "mean": 42.3,
                "median": 40.0,
                "std": 15.2,
                "min": 18.0,
                "max": 90.0,
            }
        }

    return DatasetAnalysisReport(
        missing_values=MissingValueReport(
            total_missing=total_missing,
            missing_by_column=missing_by_column,
            missing_percentage=missing_percentage,
        ),
        duplicates=DuplicateReport(
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
        ),
        statistics=StatisticsReport(numerical_summary=numerical_summary),
        imbalance=None,
        column_profiles=column_profiles,
    )


def _default_build_kwargs(
    analysis_report: DatasetAnalysisReport | None = None,
) -> dict:
    """Return default keyword arguments for builder.build()."""
    return {
        "dataset_id": "ds-001",
        "file_name": "sales.csv",
        "row_count": 1000,
        "column_count": 2,
        "memory_usage_bytes": 80_000,
        "analysis_report": analysis_report or _make_analysis_report(),
    }


# ══════════════════════════════════════════════════════════════════════
# Builder instantiation
# ══════════════════════════════════════════════════════════════════════


class TestDatasetContextBuilderBasics:
    """Basic builder tests."""

    def test_builder_instantiation(self):
        builder = DatasetContextBuilder()
        assert builder is not None

    def test_successful_build(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        assert isinstance(ctx, DatasetContext)


# ══════════════════════════════════════════════════════════════════════
# DatasetBasicInfo mapping
# ══════════════════════════════════════════════════════════════════════


class TestBasicInfoMapping:
    """Tests that top-level metadata maps into DatasetBasicInfo correctly."""

    def test_basic_info_fields(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        assert ctx.basic_info.dataset_id == "ds-001"
        assert ctx.basic_info.file_name == "sales.csv"
        assert ctx.basic_info.row_count == 1000
        assert ctx.basic_info.column_count == 2
        assert ctx.basic_info.memory_usage_bytes == 80_000

    def test_invalid_dataset_id_fails(self):
        builder = DatasetContextBuilder()
        kwargs = _default_build_kwargs()
        kwargs["dataset_id"] = ""
        with pytest.raises(ValidationError, match="dataset_id"):
            builder.build(**kwargs)

    def test_negative_row_count_fails(self):
        builder = DatasetContextBuilder()
        kwargs = _default_build_kwargs()
        kwargs["row_count"] = -1
        with pytest.raises(ValidationError, match="row_count"):
            builder.build(**kwargs)


# ══════════════════════════════════════════════════════════════════════
# Column context mapping
# ══════════════════════════════════════════════════════════════════════


class TestColumnContextMapping:
    """Tests for column profile → ColumnContext conversion."""

    def test_numeric_column_mapped_correctly(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        age_col = ctx.columns[0]  # "age" is first in profile dict
        assert age_col.name == "age"
        assert age_col.dtype == "int64"
        assert age_col.is_numeric is True
        assert age_col.is_categorical is False
        assert age_col.is_datetime is False
        assert age_col.missing_count == 5
        assert age_col.missing_percentage == 0.5
        assert age_col.sample_values == [21, 34, 55]

    def test_categorical_column_mapped_correctly(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        city_col = ctx.columns[1]  # "city" is second
        assert city_col.name == "city"
        assert city_col.dtype == "object"
        assert city_col.is_numeric is False
        assert city_col.is_categorical is True
        assert city_col.is_datetime is False
        assert city_col.statistics is None

    def test_datetime_column_mapped_correctly(self):
        profiles = OrderedDict({
            "created_at": _make_column_profile(
                dtype="datetime64[ns]",
                unique_values=950,
                unique_percentage=95.0,
                missing_count=2,
                missing_percentage=0.2,
                sample_values=["2024-01-01", "2024-06-15"],
                is_numeric=False,
                is_categorical=False,
                is_datetime=True,
            ),
        })
        report = _make_analysis_report(
            column_profiles=profiles,
            total_missing=2,
            missing_by_column={"created_at": 2},
            missing_percentage={"created_at": 0.2},
            numerical_summary={},
        )
        builder = DatasetContextBuilder()
        kwargs = _default_build_kwargs(report)
        kwargs["column_count"] = 1
        ctx = builder.build(**kwargs)
        dt_col = ctx.columns[0]
        assert dt_col.name == "created_at"
        assert dt_col.is_datetime is True
        assert dt_col.is_numeric is False
        assert dt_col.statistics is None

    def test_unique_values_maps_to_unique_count(self):
        """The existing profile key 'unique_values' must map to ColumnContext.unique_count."""
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        age_col = ctx.columns[0]
        # ColumnProfiler sets unique_values=80 for "age"
        assert age_col.unique_count == 80
        assert age_col.unique_percentage == 8.0

    def test_column_order_preserved(self):
        profiles = OrderedDict({
            "z_col": _make_column_profile(),
            "a_col": _make_column_profile(
                is_numeric=False, is_categorical=True, dtype="object"
            ),
            "m_col": _make_column_profile(),
        })
        report = _make_analysis_report(
            column_profiles=profiles,
            total_missing=15,
            missing_by_column={"z_col": 5, "a_col": 5, "m_col": 5},
            missing_percentage={"z_col": 0.5, "a_col": 0.5, "m_col": 0.5},
            numerical_summary={
                "z_col": {"mean": 1, "median": 1, "std": 0, "min": 1, "max": 1},
                "m_col": {"mean": 2, "median": 2, "std": 0, "min": 2, "max": 2},
            },
        )
        builder = DatasetContextBuilder()
        kwargs = _default_build_kwargs(report)
        kwargs["column_count"] = 3
        ctx = builder.build(**kwargs)
        assert [c.name for c in ctx.columns] == ["z_col", "a_col", "m_col"]


# ══════════════════════════════════════════════════════════════════════
# Numerical statistics mapping
# ══════════════════════════════════════════════════════════════════════


class TestStatisticsMapping:
    """Tests for StatisticsReport → ColumnStatistics conversion."""

    def test_numerical_column_gets_statistics(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        age_col = ctx.columns[0]
        assert age_col.statistics is not None
        assert age_col.statistics.mean == 42.3
        assert age_col.statistics.median == 40.0
        assert age_col.statistics.std == 15.2
        assert age_col.statistics.min == 18.0
        assert age_col.statistics.max == 90.0

    def test_categorical_column_gets_no_statistics(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        city_col = ctx.columns[1]
        assert city_col.statistics is None

    def test_multiple_numerical_columns_get_own_statistics(self):
        profiles = OrderedDict({
            "age": _make_column_profile(sample_values=[21, 34]),
            "income": _make_column_profile(
                unique_values=500,
                unique_percentage=50.0,
                missing_count=10,
                missing_percentage=1.0,
                sample_values=[30000, 80000],
            ),
        })
        report = _make_analysis_report(
            column_profiles=profiles,
            total_missing=15,
            missing_by_column={"age": 5, "income": 10},
            missing_percentage={"age": 0.5, "income": 1.0},
            numerical_summary={
                "age": {"mean": 42.0, "median": 40.0, "std": 15.0, "min": 18.0, "max": 90.0},
                "income": {"mean": 60000.0, "median": 55000.0, "std": 20000.0, "min": 20000.0, "max": 150000.0},
            },
        )
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs(report))

        age_stats = ctx.columns[0].statistics
        income_stats = ctx.columns[1].statistics

        assert age_stats is not None
        assert age_stats.mean == 42.0
        assert income_stats is not None
        assert income_stats.mean == 60000.0
        assert income_stats.max == 150000.0


# ══════════════════════════════════════════════════════════════════════
# Missing data summary mapping
# ══════════════════════════════════════════════════════════════════════


class TestMissingDataMapping:
    """Tests for MissingValueReport → MissingDataSummary conversion."""

    def test_total_missing_cells_mapped(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        assert ctx.missing_data.total_missing_cells == 5

    def test_only_columns_with_missing_included(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        # "age" has 5 missing, "city" has 0
        assert ctx.missing_data.columns_with_missing == ["age"]

    def test_zero_missing_values(self):
        profiles = OrderedDict({
            "col_a": _make_column_profile(missing_count=0, missing_percentage=0.0),
        })
        report = _make_analysis_report(
            column_profiles=profiles,
            total_missing=0,
            missing_by_column={},
            missing_percentage={},
            numerical_summary={
                "col_a": {"mean": 1, "median": 1, "std": 0, "min": 1, "max": 1},
            },
        )
        builder = DatasetContextBuilder()
        kwargs = _default_build_kwargs(report)
        kwargs["column_count"] = 1
        ctx = builder.build(**kwargs)
        assert ctx.missing_data.total_missing_cells == 0
        assert ctx.missing_data.columns_with_missing == []

    def test_columns_with_missing_preserves_profile_order(self):
        profiles = OrderedDict({
            "z_col": _make_column_profile(missing_count=3, missing_percentage=0.3),
            "a_col": _make_column_profile(missing_count=7, missing_percentage=0.7),
            "m_col": _make_column_profile(missing_count=0, missing_percentage=0.0),
        })
        report = _make_analysis_report(
            column_profiles=profiles,
            total_missing=10,
            missing_by_column={"z_col": 3, "a_col": 7},
            missing_percentage={"z_col": 0.3, "a_col": 0.7},
            numerical_summary={
                "z_col": {"mean": 1, "median": 1, "std": 0, "min": 1, "max": 1},
                "a_col": {"mean": 2, "median": 2, "std": 0, "min": 2, "max": 2},
                "m_col": {"mean": 3, "median": 3, "std": 0, "min": 3, "max": 3},
            },
        )
        builder = DatasetContextBuilder()
        kwargs = _default_build_kwargs(report)
        kwargs["column_count"] = 3
        ctx = builder.build(**kwargs)
        # z_col before a_col, m_col excluded (0 missing)
        assert ctx.missing_data.columns_with_missing == ["z_col", "a_col"]


# ══════════════════════════════════════════════════════════════════════
# Duplicate summary mapping
# ══════════════════════════════════════════════════════════════════════


class TestDuplicateMapping:
    """Tests for DuplicateReport → DuplicateSummary conversion."""

    def test_duplicate_rows_mapped(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        assert ctx.duplicates.duplicate_rows == 10

    def test_duplicate_percentage_mapped(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        assert ctx.duplicates.duplicate_percentage == 1.0

    def test_zero_duplicates(self):
        report = _make_analysis_report(
            duplicate_rows=0,
            duplicate_percentage=0.0,
        )
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs(report))
        assert ctx.duplicates.duplicate_rows == 0
        assert ctx.duplicates.duplicate_percentage == 0.0


# ══════════════════════════════════════════════════════════════════════
# Target candidates
# ══════════════════════════════════════════════════════════════════════


class TestTargetCandidates:
    """Tests for Stage 2 target_candidates behavior."""

    def test_target_candidates_always_empty(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        assert ctx.target_candidates == []


# ══════════════════════════════════════════════════════════════════════
# Schema version
# ══════════════════════════════════════════════════════════════════════


class TestSchemaVersion:
    """Tests for default schema_version."""

    def test_default_schema_version(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        assert ctx.schema_version == "1.0"


# ══════════════════════════════════════════════════════════════════════
# Serialization
# ══════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Tests that the built DatasetContext serializes cleanly."""

    def test_model_dump_json_produces_valid_json(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        json_str = ctx.model_dump_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == "1.0"
        assert len(parsed["columns"]) == 2

    def test_model_dump_round_trip(self):
        builder = DatasetContextBuilder()
        ctx = builder.build(**_default_build_kwargs())
        json_str = ctx.model_dump_json()
        reconstructed = DatasetContext.model_validate_json(json_str)
        assert ctx == reconstructed


# ══════════════════════════════════════════════════════════════════════
# Error handling
# ══════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Tests for builder error handling."""

    def test_empty_column_profiles_raises(self):
        report = _make_analysis_report(column_profiles={})
        builder = DatasetContextBuilder()
        with pytest.raises(DatasetContextBuilderError, match="column_profiles is empty"):
            builder.build(**_default_build_kwargs(report))

    def test_missing_profile_key_raises(self):
        # Profile dict missing "dtype" key
        bad_profile = _make_column_profile()
        del bad_profile["dtype"]
        profiles = {"broken_col": bad_profile}
        report = _make_analysis_report(
            column_profiles=profiles,
            total_missing=0,
            missing_by_column={},
            missing_percentage={},
            numerical_summary={},
        )
        builder = DatasetContextBuilder()
        kwargs = _default_build_kwargs(report)
        kwargs["column_count"] = 1
        with pytest.raises(DatasetContextBuilderError, match="missing required keys"):
            builder.build(**kwargs)

    def test_missing_multiple_profile_keys_lists_all(self):
        bad_profile = {
            "sample_values": [1],
            "is_numeric": True,
            "is_categorical": False,
            "is_datetime": False,
        }
        profiles = {"incomplete_col": bad_profile}
        report = _make_analysis_report(
            column_profiles=profiles,
            total_missing=0,
            missing_by_column={},
            missing_percentage={},
            numerical_summary={},
        )
        builder = DatasetContextBuilder()
        kwargs = _default_build_kwargs(report)
        kwargs["column_count"] = 1
        with pytest.raises(DatasetContextBuilderError, match="missing required keys"):
            builder.build(**kwargs)
