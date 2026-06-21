"""Tests for duplicate-row preprocessing handler."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.preprocessing_engine.duplicate_handler import (
    DuplicateHandler,
    DuplicateHandlerError,
)


@pytest.fixture()
def handler() -> DuplicateHandler:
    """Return a duplicate-row handler instance."""
    return DuplicateHandler()


def test_remove_duplicates(handler: DuplicateHandler) -> None:
    """Remove-duplicates strategy should drop duplicate rows."""
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 3],
            "city": ["A", "A", "B", "C", "C"],
        }
    )

    result = handler.apply(df, "remove_duplicates")

    expected = pd.DataFrame({"id": [1, 2, 3], "city": ["A", "B", "C"]}, index=[0, 2, 3])
    pd.testing.assert_frame_equal(result, expected)
    assert result is not df


def test_keep_duplicates(handler: DuplicateHandler) -> None:
    """Keep-duplicates strategy should return an unchanged copy."""
    df = pd.DataFrame({"id": [1, 1], "city": ["A", "A"]})

    result = handler.apply(df, "keep_duplicates")

    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_empty_dataframe(handler: DuplicateHandler) -> None:
    """Empty DataFrames should be handled safely."""
    df = pd.DataFrame()

    result = handler.apply(df, "remove_duplicates")

    assert result.empty
    assert result is not df


def test_invalid_strategy(handler: DuplicateHandler) -> None:
    """Unsupported strategies should raise the handler-specific exception."""
    df = pd.DataFrame({"id": [1, 1]})

    with pytest.raises(DuplicateHandlerError) as exc_info:
        handler.apply(df, "drop_duplicates")

    assert "Unsupported duplicate-row strategy" in str(exc_info.value)


def test_original_dataframe_unchanged(handler: DuplicateHandler) -> None:
    """Applying duplicate handling should not mutate the original DataFrame."""
    df = pd.DataFrame({"id": [1, 1, 2], "city": ["A", "A", "B"]})
    original = df.copy(deep=True)

    result = handler.apply(df, "remove_duplicates")

    pd.testing.assert_frame_equal(df, original)
    assert len(result) == 2
    assert len(df) == 3
