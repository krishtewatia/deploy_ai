"""Tests for missing-value preprocessing handler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.preprocessing_engine.missing_handler import (
    MissingValueHandler,
    MissingValueHandlerError,
)


@pytest.fixture()
def handler() -> MissingValueHandler:
    """Return a missing-value handler instance."""
    return MissingValueHandler()


def test_mean_imputation(handler: MissingValueHandler) -> None:
    """Mean imputation should fill missing values with the column mean."""
    df = pd.DataFrame({"salary": [100.0, 200.0, np.nan, 400.0]})

    result = handler.apply(
        df,
        {"salary": {"strategy": "mean_imputation", "reason": "Numeric column."}},
    )

    assert result["salary"].tolist() == [100.0, 200.0, pytest.approx(700.0 / 3), 400.0]


def test_median_imputation(handler: MissingValueHandler) -> None:
    """Median imputation should fill missing values with the column median."""
    df = pd.DataFrame({"age": [10.0, 20.0, np.nan, 100.0]})

    result = handler.apply(
        df,
        {"age": {"strategy": "median_imputation", "reason": "Skewed column."}},
    )

    assert result["age"].tolist() == [10.0, 20.0, 20.0, 100.0]


def test_mode_imputation(handler: MissingValueHandler) -> None:
    """Mode imputation should fill missing values with the first mode."""
    df = pd.DataFrame({"department": ["Sales", None, "HR", "Sales"]})

    result = handler.apply(
        df,
        {"department": {"strategy": "mode_imputation", "reason": "Categorical column."}},
    )

    assert result["department"].tolist() == ["Sales", "Sales", "HR", "Sales"]


def test_mode_imputation_all_missing_column(handler: MissingValueHandler) -> None:
    """Mode imputation should safely leave all-missing columns unchanged."""
    df = pd.DataFrame({"department": [None, None]})

    result = handler.apply(
        df,
        {"department": {"strategy": "mode_imputation", "reason": "No observed mode."}},
    )

    assert result["department"].isna().all()


def test_drop_column(handler: MissingValueHandler) -> None:
    """Drop-column strategy should remove the selected column."""
    df = pd.DataFrame({"legacy_code": [None, "A"], "salary": [100, 200]})

    result = handler.apply(
        df,
        {"legacy_code": {"strategy": "drop_column", "reason": "Too sparse."}},
    )

    assert "legacy_code" not in result.columns
    assert "salary" in result.columns


def test_non_existent_column(handler: MissingValueHandler) -> None:
    """Actions for missing columns should be ignored."""
    df = pd.DataFrame({"salary": [100.0, np.nan]})

    result = handler.apply(
        df,
        {"bonus": {"strategy": "mean_imputation", "reason": "Missing column."}},
    )

    pd.testing.assert_frame_equal(result, df)


def test_empty_dataframe(handler: MissingValueHandler) -> None:
    """Empty DataFrames should be returned safely as new DataFrames."""
    df = pd.DataFrame()

    result = handler.apply(
        df,
        {"salary": {"strategy": "mean_imputation", "reason": "No data."}},
    )

    assert result.empty
    assert result is not df


def test_no_actions_returns_copy(handler: MissingValueHandler) -> None:
    """Empty action mappings should return a copy of the input DataFrame."""
    df = pd.DataFrame({"salary": [100.0, np.nan]})

    result = handler.apply(df, {})

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_original_dataframe_unchanged(handler: MissingValueHandler) -> None:
    """Applying actions should not mutate the original DataFrame."""
    df = pd.DataFrame({"salary": [100.0, 200.0, np.nan], "department": ["A", None, "B"]})
    original = df.copy(deep=True)

    result = handler.apply(
        df,
        {
            "salary": {"strategy": "mean_imputation", "reason": "Numeric column."},
            "department": {"strategy": "drop_column", "reason": "Drop column."},
        },
    )

    pd.testing.assert_frame_equal(df, original)
    assert result is not df
    assert "department" not in result.columns
    assert result["salary"].isna().sum() == 0


def test_invalid_strategy_raises_custom_error(handler: MissingValueHandler) -> None:
    """Unsupported strategies should raise the handler-specific exception."""
    df = pd.DataFrame({"salary": [100.0, np.nan]})

    with pytest.raises(MissingValueHandlerError):
        handler.apply(
            df,
            {"salary": {"strategy": "unknown_strategy", "reason": "Invalid."}},
        )
