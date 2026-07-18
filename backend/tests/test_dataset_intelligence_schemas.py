"""Tests for backend.app.dataset_intelligence.schemas.

Covers all seven Pydantic v2 models, validation rules, serialization,
and edge cases as specified in Stage 1.
"""

import json

import pytest
from pydantic import ValidationError

from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    ColumnStatistics,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
    TargetCandidateSummary,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_basic_info(**overrides) -> dict:
    """Return a valid DatasetBasicInfo dict with optional overrides."""
    defaults = {
        "dataset_id": "ds-001",
        "file_name": "sales.csv",
        "row_count": 1000,
        "column_count": 10,
        "memory_usage_bytes": 80_000,
    }
    defaults.update(overrides)
    return defaults


def _make_column_context(*, name: str = "age", **overrides) -> dict:
    """Return a valid ColumnContext dict for a numeric column."""
    defaults = {
        "name": name,
        "dtype": "int64",
        "is_numeric": True,
        "is_categorical": False,
        "is_datetime": False,
        "missing_count": 5,
        "missing_percentage": 0.5,
        "unique_count": 80,
        "unique_percentage": 8.0,
        "sample_values": [21, 34, 55, 67, 78],
        "statistics": {
            "mean": 42.3,
            "median": 40.0,
            "std": 15.2,
            "min": 18.0,
            "max": 90.0,
        },
    }
    defaults.update(overrides)
    return defaults


def _make_categorical_column(*, name: str = "city", **overrides) -> dict:
    """Return a valid ColumnContext dict for a categorical column."""
    defaults = {
        "name": name,
        "dtype": "object",
        "is_numeric": False,
        "is_categorical": True,
        "is_datetime": False,
        "missing_count": 0,
        "missing_percentage": 0.0,
        "unique_count": 5,
        "unique_percentage": 0.5,
        "sample_values": ["NYC", "LA", "Chicago", "Houston", "Phoenix"],
        "statistics": None,
    }
    defaults.update(overrides)
    return defaults


def _make_datetime_column(*, name: str = "created_at", **overrides) -> dict:
    """Return a valid ColumnContext dict for a datetime column."""
    defaults = {
        "name": name,
        "dtype": "datetime64[ns]",
        "is_numeric": False,
        "is_categorical": False,
        "is_datetime": True,
        "missing_count": 2,
        "missing_percentage": 0.2,
        "unique_count": 950,
        "unique_percentage": 95.0,
        "sample_values": ["2024-01-01", "2024-06-15", "2024-12-31"],
        "statistics": None,
    }
    defaults.update(overrides)
    return defaults


def _make_dataset_context(**overrides) -> dict:
    """Return a valid DatasetContext dict with optional overrides."""
    defaults = {
        "basic_info": _make_basic_info(),
        "columns": [
            _make_column_context(name="age"),
            _make_categorical_column(name="city"),
        ],
        "missing_data": {
            "total_missing_cells": 5,
            "columns_with_missing": ["age"],
        },
        "duplicates": {
            "duplicate_rows": 10,
            "duplicate_percentage": 1.0,
        },
        "target_candidates": [
            {
                "column_name": "city",
                "unique_count": 5,
                "unique_percentage": 0.5,
                "reason": "Low cardinality categorical column.",
            }
        ],
    }
    defaults.update(overrides)
    return defaults


# ══════════════════════════════════════════════════════════════════════
# 1. DatasetBasicInfo
# ══════════════════════════════════════════════════════════════════════


class TestDatasetBasicInfo:
    """Tests for the DatasetBasicInfo model."""

    def test_valid_creation(self):
        info = DatasetBasicInfo(**_make_basic_info())
        assert info.dataset_id == "ds-001"
        assert info.file_name == "sales.csv"
        assert info.row_count == 1000
        assert info.column_count == 10
        assert info.memory_usage_bytes == 80_000

    def test_empty_dataset_id_rejected(self):
        with pytest.raises(ValidationError, match="dataset_id"):
            DatasetBasicInfo(**_make_basic_info(dataset_id=""))

    def test_empty_file_name_rejected(self):
        with pytest.raises(ValidationError, match="file_name"):
            DatasetBasicInfo(**_make_basic_info(file_name=""))

    def test_negative_row_count_rejected(self):
        with pytest.raises(ValidationError, match="row_count"):
            DatasetBasicInfo(**_make_basic_info(row_count=-1))

    def test_negative_column_count_rejected(self):
        with pytest.raises(ValidationError, match="column_count"):
            DatasetBasicInfo(**_make_basic_info(column_count=-5))

    def test_negative_memory_usage_rejected(self):
        with pytest.raises(ValidationError, match="memory_usage_bytes"):
            DatasetBasicInfo(**_make_basic_info(memory_usage_bytes=-100))

    def test_zero_counts_allowed(self):
        info = DatasetBasicInfo(
            **_make_basic_info(row_count=0, column_count=0, memory_usage_bytes=0)
        )
        assert info.row_count == 0
        assert info.column_count == 0
        assert info.memory_usage_bytes == 0


# ══════════════════════════════════════════════════════════════════════
# 2. ColumnStatistics
# ══════════════════════════════════════════════════════════════════════


class TestColumnStatistics:
    """Tests for the ColumnStatistics model."""

    def test_valid_statistics(self):
        stats = ColumnStatistics(
            mean=42.3, median=40.0, std=15.2, min=18.0, max=90.0
        )
        assert stats.mean == 42.3
        assert stats.median == 40.0
        assert stats.std == 15.2
        assert stats.min == 18.0
        assert stats.max == 90.0

    def test_all_none_defaults(self):
        stats = ColumnStatistics()
        assert stats.mean is None
        assert stats.median is None
        assert stats.std is None
        assert stats.min is None
        assert stats.max is None

    def test_partial_values(self):
        stats = ColumnStatistics(mean=10.5, max=100.0)
        assert stats.mean == 10.5
        assert stats.median is None
        assert stats.max == 100.0

    def test_json_serializable(self):
        stats = ColumnStatistics(mean=1.0, median=2.0, std=0.5, min=0.0, max=3.0)
        data = json.loads(stats.model_dump_json())
        assert data["mean"] == 1.0
        assert data["min"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# 3. ColumnContext
# ══════════════════════════════════════════════════════════════════════


class TestColumnContext:
    """Tests for the ColumnContext model."""

    def test_valid_numeric_column(self):
        col = ColumnContext(**_make_column_context())
        assert col.name == "age"
        assert col.is_numeric is True
        assert col.is_categorical is False
        assert col.is_datetime is False
        assert col.statistics is not None
        assert col.statistics.mean == 42.3

    def test_valid_categorical_column(self):
        col = ColumnContext(**_make_categorical_column())
        assert col.name == "city"
        assert col.is_numeric is False
        assert col.is_categorical is True
        assert col.statistics is None
        assert "NYC" in col.sample_values

    def test_valid_datetime_column(self):
        col = ColumnContext(**_make_datetime_column())
        assert col.name == "created_at"
        assert col.is_datetime is True
        assert col.is_numeric is False
        assert col.is_categorical is False

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError, match="name"):
            ColumnContext(**_make_column_context(name=""))

    def test_empty_dtype_rejected(self):
        with pytest.raises(ValidationError, match="dtype"):
            ColumnContext(**_make_column_context(dtype=""))

    def test_negative_missing_count_rejected(self):
        with pytest.raises(ValidationError, match="missing_count"):
            ColumnContext(**_make_column_context(missing_count=-1))

    def test_negative_unique_count_rejected(self):
        with pytest.raises(ValidationError, match="unique_count"):
            ColumnContext(**_make_column_context(unique_count=-10))

    def test_missing_percentage_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="missing_percentage"):
            ColumnContext(**_make_column_context(missing_percentage=-0.1))

    def test_missing_percentage_above_100_rejected(self):
        with pytest.raises(ValidationError, match="missing_percentage"):
            ColumnContext(**_make_column_context(missing_percentage=100.1))

    def test_unique_percentage_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="unique_percentage"):
            ColumnContext(**_make_column_context(unique_percentage=-5.0))

    def test_unique_percentage_above_100_rejected(self):
        with pytest.raises(ValidationError, match="unique_percentage"):
            ColumnContext(**_make_column_context(unique_percentage=101.0))

    def test_empty_sample_values_allowed(self):
        col = ColumnContext(**_make_column_context(sample_values=[]))
        assert col.sample_values == []

    def test_sample_values_with_mixed_types(self):
        col = ColumnContext(
            **_make_column_context(sample_values=[1, "two", 3.0, True, None])
        )
        assert len(col.sample_values) == 5

    def test_boundary_percentages(self):
        col = ColumnContext(
            **_make_column_context(
                missing_percentage=0.0,
                unique_percentage=100.0,
            )
        )
        assert col.missing_percentage == 0.0
        assert col.unique_percentage == 100.0

    def test_non_serializable_sample_values_rejected(self):
        with pytest.raises(ValidationError, match="sample_values must be JSON serializable"):
            ColumnContext(
                **_make_column_context(sample_values=[object()])
            )



# ══════════════════════════════════════════════════════════════════════
# 4. MissingDataSummary
# ══════════════════════════════════════════════════════════════════════


class TestMissingDataSummary:
    """Tests for the MissingDataSummary model."""

    def test_valid_creation(self):
        summary = MissingDataSummary(
            total_missing_cells=42,
            columns_with_missing=["age", "income"],
        )
        assert summary.total_missing_cells == 42
        assert summary.columns_with_missing == ["age", "income"]

    def test_zero_missing(self):
        summary = MissingDataSummary(
            total_missing_cells=0,
            columns_with_missing=[],
        )
        assert summary.total_missing_cells == 0

    def test_negative_total_rejected(self):
        with pytest.raises(ValidationError, match="total_missing_cells"):
            MissingDataSummary(
                total_missing_cells=-1,
                columns_with_missing=[],
            )


# ══════════════════════════════════════════════════════════════════════
# 5. DuplicateSummary
# ══════════════════════════════════════════════════════════════════════


class TestDuplicateSummary:
    """Tests for the DuplicateSummary model."""

    def test_valid_creation(self):
        dup = DuplicateSummary(duplicate_rows=15, duplicate_percentage=1.5)
        assert dup.duplicate_rows == 15
        assert dup.duplicate_percentage == 1.5

    def test_negative_rows_rejected(self):
        with pytest.raises(ValidationError, match="duplicate_rows"):
            DuplicateSummary(duplicate_rows=-1, duplicate_percentage=0.0)

    def test_percentage_above_100_rejected(self):
        with pytest.raises(ValidationError, match="duplicate_percentage"):
            DuplicateSummary(duplicate_rows=0, duplicate_percentage=100.5)

    def test_percentage_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="duplicate_percentage"):
            DuplicateSummary(duplicate_rows=0, duplicate_percentage=-0.1)


# ══════════════════════════════════════════════════════════════════════
# 6. TargetCandidateSummary
# ══════════════════════════════════════════════════════════════════════


class TestTargetCandidateSummary:
    """Tests for the TargetCandidateSummary model."""

    def test_valid_creation(self):
        candidate = TargetCandidateSummary(
            column_name="species",
            unique_count=3,
            unique_percentage=0.3,
            reason="Low cardinality categorical column.",
        )
        assert candidate.column_name == "species"
        assert candidate.unique_count == 3

    def test_empty_column_name_rejected(self):
        with pytest.raises(ValidationError, match="column_name"):
            TargetCandidateSummary(
                column_name="",
                unique_count=3,
                unique_percentage=0.3,
                reason="test",
            )

    def test_empty_reason_rejected(self):
        with pytest.raises(ValidationError, match="reason"):
            TargetCandidateSummary(
                column_name="target",
                unique_count=2,
                unique_percentage=0.2,
                reason="",
            )


# ══════════════════════════════════════════════════════════════════════
# 7. DatasetContext
# ══════════════════════════════════════════════════════════════════════


class TestDatasetContext:
    """Tests for the top-level DatasetContext model."""

    def test_valid_creation(self):
        ctx = DatasetContext(**_make_dataset_context())
        assert ctx.schema_version == "1.0"
        assert ctx.basic_info.dataset_id == "ds-001"
        assert len(ctx.columns) == 2
        assert ctx.missing_data.total_missing_cells == 5
        assert ctx.duplicates.duplicate_rows == 10
        assert len(ctx.target_candidates) == 1

    def test_default_schema_version(self):
        ctx = DatasetContext(**_make_dataset_context())
        assert ctx.schema_version == "1.0"

    def test_custom_schema_version(self):
        ctx = DatasetContext(**_make_dataset_context(schema_version="2.0"))
        assert ctx.schema_version == "2.0"

    def test_empty_columns_rejected(self):
        with pytest.raises(ValidationError, match="columns"):
            DatasetContext(**_make_dataset_context(columns=[]))

    def test_duplicate_column_names_rejected(self):
        dup_columns = [
            _make_column_context(name="age"),
            _make_column_context(name="age"),
        ]
        with pytest.raises(ValidationError, match="Duplicate column names"):
            DatasetContext(**_make_dataset_context(columns=dup_columns))

    def test_empty_target_candidates_allowed(self):
        ctx = DatasetContext(**_make_dataset_context(target_candidates=[]))
        assert ctx.target_candidates == []

    def test_model_dump(self):
        ctx = DatasetContext(**_make_dataset_context())
        dumped = ctx.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["schema_version"] == "1.0"
        assert isinstance(dumped["columns"], list)
        assert dumped["basic_info"]["dataset_id"] == "ds-001"
        assert dumped["columns"][0]["name"] == "age"
        assert dumped["columns"][0]["statistics"]["mean"] == 42.3

    def test_model_dump_json_produces_valid_json(self):
        ctx = DatasetContext(**_make_dataset_context())
        json_str = ctx.model_dump_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == "1.0"
        assert len(parsed["columns"]) == 2

    def test_nested_models_serialize_correctly(self):
        ctx = DatasetContext(**_make_dataset_context())
        dumped = ctx.model_dump()

        # ColumnContext nested statistics
        age_col = dumped["columns"][0]
        assert age_col["statistics"]["mean"] == 42.3
        assert age_col["statistics"]["max"] == 90.0

        # Categorical column has None statistics
        city_col = dumped["columns"][1]
        assert city_col["statistics"] is None

        # MissingDataSummary
        assert dumped["missing_data"]["total_missing_cells"] == 5
        assert dumped["missing_data"]["columns_with_missing"] == ["age"]

        # DuplicateSummary
        assert dumped["duplicates"]["duplicate_rows"] == 10

        # TargetCandidateSummary
        assert dumped["target_candidates"][0]["column_name"] == "city"

    def test_round_trip_json_serialization(self):
        """Serialize to JSON and reconstruct — models must be identical."""
        original = DatasetContext(**_make_dataset_context())
        json_str = original.model_dump_json()
        reconstructed = DatasetContext.model_validate_json(json_str)
        assert original == reconstructed
