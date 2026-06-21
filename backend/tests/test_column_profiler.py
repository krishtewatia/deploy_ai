"""Tests for backend.app.analysis.column_profiler — targeting 100 % coverage.

Covers:
* Numerical columns
* Categorical columns (object + Categorical dtype)
* Datetime columns
* Boolean columns
* Missing values
* Empty DataFrame
* Sample values limit (≤ 5)
* Percentage calculations (rounded to 2 decimals)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.analysis.column_profiler import ColumnProfiler


@pytest.fixture()
def profiler() -> ColumnProfiler:
    return ColumnProfiler()


# ── Numerical columns ──────────────────────────────────────────────────────


class TestNumericalColumns:
    def test_int_column(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"age": [20, 25, 30, 35, 40]})
        result = profiler.profile(df)

        col = result["age"]
        assert col["is_numeric"] is True
        assert col["is_categorical"] is False
        assert col["is_datetime"] is False
        assert col["unique_values"] == 5
        assert col["missing_count"] == 0

    def test_float_column(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"salary": [50000.5, 60000.0, 70000.75]})
        result = profiler.profile(df)

        col = result["salary"]
        assert col["is_numeric"] is True
        assert col["dtype"] == "float64"


# ── Categorical columns ───────────────────────────────────────────────────


class TestCategoricalColumns:
    def test_object_column(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"city": ["NYC", "LA", "NYC", "SF"]})
        result = profiler.profile(df)

        col = result["city"]
        assert col["is_categorical"] is True
        assert col["is_numeric"] is False
        assert col["unique_values"] == 3

    def test_pandas_categorical_dtype(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({
            "grade": pd.Categorical(["A", "B", "A", "C"])
        })
        result = profiler.profile(df)

        col = result["grade"]
        assert col["is_categorical"] is True

    def test_bool_column(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"flag": [True, False, True]})
        result = profiler.profile(df)

        col = result["flag"]
        assert col["is_categorical"] is True
        assert col["is_numeric"] is True  # booleans are also numeric in pandas


# ── Datetime columns ──────────────────────────────────────────────────────


class TestDatetimeColumns:
    def test_datetime_column(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({
            "created_at": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        })
        result = profiler.profile(df)

        col = result["created_at"]
        assert col["is_datetime"] is True
        assert col["is_categorical"] is False
        assert col["is_numeric"] is False
        assert col["sample_values"] == ["2024-01-01T00:00:00", "2024-02-01T00:00:00", "2024-03-01T00:00:00"]
        assert all(isinstance(v, str) for v in col["sample_values"])


# ── Missing values ─────────────────────────────────────────────────────────


class TestMissingValues:
    def test_missing_count_and_percentage(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"val": [1, None, 3, None, 5]})
        result = profiler.profile(df)

        col = result["val"]
        assert col["missing_count"] == 2
        assert col["missing_percentage"] == 40.0

    def test_no_missing(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"val": [1, 2, 3]})
        result = profiler.profile(df)

        col = result["val"]
        assert col["missing_count"] == 0
        assert col["missing_percentage"] == 0.0

    def test_all_missing(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"val": [None, None, None]})
        result = profiler.profile(df)

        col = result["val"]
        assert col["missing_count"] == 3
        assert col["missing_percentage"] == 100.0
        assert col["unique_values"] == 0
        assert col["sample_values"] == []


# ── Empty DataFrame ────────────────────────────────────────────────────────


class TestEmptyDataFrame:
    def test_empty_df_returns_empty_dict(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame()
        result = profiler.profile(df)
        assert result == {}

    def test_df_with_columns_but_no_rows(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="object")})
        result = profiler.profile(df)

        assert "a" in result
        assert "b" in result
        assert result["a"]["missing_count"] == 0
        assert result["a"]["missing_percentage"] == 0.0
        assert result["a"]["unique_values"] == 0
        assert result["a"]["unique_percentage"] == 0.0
        assert result["a"]["sample_values"] == []


# ── Sample values limit ───────────────────────────────────────────────────


class TestSampleValuesLimit:
    def test_max_5_sample_values(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"x": list(range(100))})
        result = profiler.profile(df)

        assert len(result["x"]["sample_values"]) <= 5

    def test_fewer_than_5_unique(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = profiler.profile(df)

        assert result["x"]["sample_values"] == [1, 2, 3]

    def test_sample_values_are_native_python_types(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"x": np.array([10, 20, 30])})
        result = profiler.profile(df)

        for v in result["x"]["sample_values"]:
            assert isinstance(v, int)

    def test_sample_excludes_nulls(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"x": [None, "a", None, "b"]})
        result = profiler.profile(df)

        assert None not in result["x"]["sample_values"]
        assert set(result["x"]["sample_values"]) == {"a", "b"}


# ── Percentage calculations ────────────────────────────────────────────────


class TestPercentageCalculations:
    def test_unique_percentage(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3]})
        result = profiler.profile(df)

        # 3 unique out of 5 → 60.0%
        assert result["x"]["unique_percentage"] == 60.0

    def test_missing_percentage_rounding(self, profiler: ColumnProfiler) -> None:
        df = pd.DataFrame({"x": [1, None, 3, 4, 5, 6]})
        result = profiler.profile(df)

        # 1 missing out of 6 → 16.666… → 16.67
        assert result["x"]["missing_percentage"] == 16.67
