"""Tests for numerical scaling preprocessing handler."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.preprocessing_engine.scaling_handler import (
    ScalingHandler,
    ScalingHandlerError,
)


@pytest.fixture()
def handler() -> ScalingHandler:
    """Return a scaling handler instance."""
    return ScalingHandler()


def test_standard_scaling(handler: ScalingHandler) -> None:
    """Standard scaling should apply (x - mean) / std."""
    df = pd.DataFrame({"salary": [10.0, 20.0, 30.0]})

    result = handler.apply(
        df,
        {"salary": {"strategy": "standard_scaling", "reason": "Scale-sensitive model."}},
    )

    assert result["salary"].tolist() == [pytest.approx(-1.0), pytest.approx(0.0), pytest.approx(1.0)]


def test_minmax_scaling(handler: ScalingHandler) -> None:
    """Min-max scaling should apply (x - min) / (max - min)."""
    df = pd.DataFrame({"salary": [10.0, 20.0, 30.0]})

    result = handler.apply(
        df,
        {"salary": {"strategy": "minmax_scaling", "reason": "Normalize range."}},
    )

    assert result["salary"].tolist() == [0.0, 0.5, 1.0]


def test_no_scaling(handler: ScalingHandler) -> None:
    """No-scaling strategy should leave the column unchanged."""
    df = pd.DataFrame({"salary": [10.0, 20.0, 30.0]})

    result = handler.apply(
        df,
        {"salary": {"strategy": "no_scaling", "reason": "Already scaled."}},
    )

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_non_existent_column(handler: ScalingHandler) -> None:
    """Actions for missing columns should be ignored."""
    df = pd.DataFrame({"salary": [10.0, 20.0, 30.0]})

    result = handler.apply(
        df,
        {"bonus": {"strategy": "standard_scaling", "reason": "Missing column."}},
    )

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_non_numeric_column(handler: ScalingHandler) -> None:
    """Non-numeric columns should be ignored safely."""
    df = pd.DataFrame({"department": ["AI", "ML", "DS"]})

    result = handler.apply(
        df,
        {"department": {"strategy": "standard_scaling", "reason": "Not numeric."}},
    )

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_zero_variance_column(handler: ScalingHandler) -> None:
    """Zero-variance columns should remain unchanged."""
    df = pd.DataFrame({"salary": [10.0, 10.0, 10.0]})

    result = handler.apply(
        df,
        {"salary": {"strategy": "standard_scaling", "reason": "Constant column."}},
    )

    pd.testing.assert_frame_equal(result, df)


def test_zero_range_column(handler: ScalingHandler) -> None:
    """Min-max scaling should safely leave zero-range columns unchanged."""
    df = pd.DataFrame({"salary": [10.0, 10.0, 10.0]})

    result = handler.apply(
        df,
        {"salary": {"strategy": "minmax_scaling", "reason": "Constant column."}},
    )

    pd.testing.assert_frame_equal(result, df)


def test_empty_dataframe(handler: ScalingHandler) -> None:
    """Empty DataFrames should be returned safely as new DataFrames."""
    df = pd.DataFrame()

    result = handler.apply(
        df,
        {"salary": {"strategy": "standard_scaling", "reason": "No data."}},
    )

    assert result.empty
    assert result is not df


def test_no_actions_returns_copy(handler: ScalingHandler) -> None:
    """Empty action mappings should return a copy of the input DataFrame."""
    df = pd.DataFrame({"salary": [10.0, 20.0, 30.0]})

    result = handler.apply(df, {})

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_invalid_strategy(handler: ScalingHandler) -> None:
    """Unsupported strategies should raise the handler-specific exception."""
    df = pd.DataFrame({"salary": [10.0, 20.0, 30.0]})

    with pytest.raises(ScalingHandlerError) as exc_info:
        handler.apply(
            df,
            {"salary": {"strategy": "robust_scaling", "reason": "Invalid."}},
        )

    assert "Failed to apply scaling strategy" in str(exc_info.value)


def test_original_dataframe_unchanged(handler: ScalingHandler) -> None:
    """Applying scaling should not mutate the original DataFrame."""
    df = pd.DataFrame({"salary": [10.0, 20.0, 30.0], "age": [1.0, 2.0, 3.0]})
    original = df.copy(deep=True)

    result = handler.apply(
        df,
        {
            "salary": {"strategy": "standard_scaling", "reason": "Scale-sensitive model."},
            "age": {"strategy": "minmax_scaling", "reason": "Normalize range."},
        },
    )

    pd.testing.assert_frame_equal(df, original)
    assert result is not df
    assert result["salary"].tolist() == [pytest.approx(-1.0), pytest.approx(0.0), pytest.approx(1.0)]
    assert result["age"].tolist() == [0.0, 0.5, 1.0]
