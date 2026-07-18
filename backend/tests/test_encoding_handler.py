"""Tests for categorical encoding preprocessing handler."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.preprocessing_engine.encoding_handler import (
    EncodingHandler,
    EncodingHandlerError,
)


@pytest.fixture()
def handler() -> EncodingHandler:
    """Return an encoding handler instance."""
    return EncodingHandler()


def test_one_hot_encoding(handler: EncodingHandler) -> None:
    """One-hot encoding should create prefixed dummy columns."""
    df = pd.DataFrame({"department": ["AI", "ML", "AI"]})

    result = handler.apply(
        df,
        {"department": {"strategy": "one_hot_encode", "reason": "Nominal feature."}},
    )

    expected = pd.DataFrame(
        {
            "department_AI": [1, 0, 1],
            "department_ML": [0, 1, 0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
    assert "department" not in result.columns


def test_label_encoding(handler: EncodingHandler) -> None:
    """Label encoding should use deterministic sorted label ordering."""
    df = pd.DataFrame({"role": ["ML", "AI", "DS", "AI"]})

    result = handler.apply(
        df,
        {"role": {"strategy": "label_encode", "reason": "Ordinal-compatible feature."}},
    )

    assert result["role"].tolist() == [2, 0, 1, 0]


def test_multiple_columns(handler: EncodingHandler) -> None:
    """Multiple encoding actions should be applied sequentially."""
    df = pd.DataFrame(
        {
            "department": ["AI", "ML", "AI"],
            "role": ["ML", "AI", "DS"],
            "salary": [10, 20, 30],
        }
    )

    result = handler.apply(
        df,
        {
            "department": {"strategy": "one_hot_encode", "reason": "Nominal feature."},
            "role": {"strategy": "label_encode", "reason": "Compact labels."},
        },
    )

    assert result["role"].tolist() == [2, 0, 1]
    assert result["salary"].tolist() == [10, 20, 30]
    assert "department" not in result.columns
    assert "department_AI" in result.columns
    assert "department_ML" in result.columns


def test_non_existent_column(handler: EncodingHandler) -> None:
    """Actions for missing columns should be ignored."""
    df = pd.DataFrame({"department": ["AI", "ML"]})

    result = handler.apply(
        df,
        {"missing": {"strategy": "one_hot_encode", "reason": "Unknown column."}},
    )

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_empty_dataframe(handler: EncodingHandler) -> None:
    """Empty DataFrames should be returned safely as new DataFrames."""
    df = pd.DataFrame()

    result = handler.apply(
        df,
        {"department": {"strategy": "one_hot_encode", "reason": "No data."}},
    )

    assert result.empty
    assert result is not df


def test_no_actions_returns_copy(handler: EncodingHandler) -> None:
    """Empty action mappings should return a copy of the input DataFrame."""
    df = pd.DataFrame({"department": ["AI", "ML"]})

    result = handler.apply(df, {})

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_invalid_strategy(handler: EncodingHandler) -> None:
    """Unsupported strategies should raise the handler-specific exception."""
    df = pd.DataFrame({"department": ["AI", "ML"]})

    with pytest.raises(EncodingHandlerError) as exc_info:
        handler.apply(
            df,
            {"department": {"strategy": "binary_encode", "reason": "Invalid."}},
        )

    assert "Failed to apply encoding strategy" in str(exc_info.value)


def test_original_dataframe_unchanged(handler: EncodingHandler) -> None:
    """Applying encoding should not mutate the original DataFrame."""
    df = pd.DataFrame({"department": ["AI", "ML", "AI"], "role": ["ML", "AI", "DS"]})
    original = df.copy(deep=True)

    result = handler.apply(
        df,
        {
            "department": {"strategy": "one_hot_encode", "reason": "Nominal feature."},
            "role": {"strategy": "label_encode", "reason": "Compact labels."},
        },
    )

    pd.testing.assert_frame_equal(df, original)
    assert result is not df
    assert "department" not in result.columns
    assert result["role"].tolist() == [2, 0, 1]


def test_label_encoding_preserves_missing_values(handler: EncodingHandler) -> None:
    """Label encoding should leave missing values as missing."""
    df = pd.DataFrame({"role": ["ML", None, "AI"]})

    result = handler.apply(
        df,
        {"role": {"strategy": "label_encode", "reason": "Contains missing values."}},
    )

    assert result["role"].tolist()[0] == 1.0
    assert pd.isna(result["role"].tolist()[1])
    assert result["role"].tolist()[2] == 0.0
