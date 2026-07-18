"""Tests for backend.app.preprocessing_engine.pipeline_executor."""

from __future__ import annotations

from unittest.mock import patch
import pandas as pd
import pytest

from backend.app.preprocessing_engine.pipeline_executor import (
    PipelineExecutionError,
    PipelineExecutor,
)
from backend.app.preprocessing_engine.schemas import (
    ColumnAction,
    DuplicateStrategy,
    EncodingStrategy,
    ExecutionPlan,
    MissingValueStrategy,
    ScalingStrategy,
)


def test_full_pipeline_execution() -> None:
    """Test executing a full pipeline with all stages populated."""
    # Setup dataframe with:
    # - missing values in A
    # - duplicate rows (row 1, row 3, row 4 are duplicates)
    # - categorical values in B
    # - numerical values in C for scaling
    df = pd.DataFrame({
        "A": [1.0, 2.0, None, 2.0, 2.0],
        "B": ["cat", "dog", "cat", "dog", "dog"],
        "C": [10.0, 20.0, 30.0, 20.0, 20.0],
    })

    plan = ExecutionPlan(
        missing_values={
            "A": ColumnAction(
                strategy=MissingValueStrategy.MEAN_IMPUTATION.value,
                reason="Mean imputation for numeric",
            )
        },
        duplicates_action=DuplicateStrategy.REMOVE_DUPLICATES,
        encoding={
            "B": ColumnAction(
                strategy=EncodingStrategy.LABEL_ENCODE.value,
                reason="Label encode categorical",
            )
        },
        scaling={
            "C": ColumnAction(
                strategy=ScalingStrategy.MINMAX_SCALING.value,
                reason="MinMax scale target numeric",
            )
        },
    )

    executor = PipelineExecutor()
    result = executor.execute(df, plan)

    # Assertions:
    # 1. Missing Value Stage on 'A':
    # Mean of non-null in A is (1.0 + 2.0 + 2.0 + 2.0) / 4 = 1.75.
    # Row 3 (index 2) had None, so it becomes 1.75.
    # 2. Duplicate Stage:
    # Duplicates are dropped.
    # Original A: [1.0, 2.0, 1.75, 2.0, 2.0]
    # Original B: ["cat", "dog", "cat", "dog", "dog"]
    # Original C: [10.0, 20.0, 30.0, 20.0, 20.0]
    # Rows at indices 3 and 4 are duplicate of row 1 (index 1).
    # Resulting DataFrame has rows at indices 0, 1, 2.
    # 3. Encoding Stage on 'B':
    # Unique values: "cat" and "dog".
    # Sorted order: "cat" (0), "dog" (1).
    # 'B' column becomes: [0, 1, 0].
    # 4. Scaling Stage on 'C':
    # Remaining rows C: [10.0, 20.0, 30.0].
    # Min is 10.0, Max is 30.0.
    # MinMax scaling: (x - 10) / 20.
    # 'C' column becomes: [0.0, 0.5, 1.0].

    assert len(result) == 3
    assert list(result.index) == [0, 1, 2]
    assert result.loc[2, "A"] == 1.75
    assert list(result["B"]) == [0, 1, 0]
    assert list(result["C"]) == [0.0, 0.5, 1.0]


def test_missing_values_only() -> None:
    """Test pipeline execution with only missing values actions."""
    df = pd.DataFrame({
        "A": [1.0, None, 3.0],
        "B": ["cat", "dog", "cat"],
    })

    plan = ExecutionPlan(
        missing_values={
            "A": ColumnAction(
                strategy=MissingValueStrategy.MEAN_IMPUTATION.value,
                reason="Mean imputation",
            )
        },
        duplicates_action=DuplicateStrategy.KEEP_DUPLICATES,
        encoding={},
        scaling={},
    )

    executor = PipelineExecutor()
    result = executor.execute(df, plan)

    # Missing value imputed (mean = 2.0).
    # Duplicates kept, no encoding, no scaling.
    assert len(result) == 3
    assert result.loc[1, "A"] == 2.0
    assert list(result["B"]) == ["cat", "dog", "cat"]


def test_encoding_only() -> None:
    """Test pipeline execution with only encoding actions."""
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0],
        "B": ["cat", "dog", "cat"],
    })

    plan = ExecutionPlan(
        missing_values={},
        duplicates_action=DuplicateStrategy.KEEP_DUPLICATES,
        encoding={
            "B": ColumnAction(
                strategy=EncodingStrategy.LABEL_ENCODE.value,
                reason="Label encode",
            )
        },
        scaling={},
    )

    executor = PipelineExecutor()
    result = executor.execute(df, plan)

    assert len(result) == 3
    assert list(result["B"]) == [0, 1, 0]
    assert list(result["A"]) == [1.0, 2.0, 3.0]


def test_scaling_only() -> None:
    """Test pipeline execution with only scaling actions."""
    df = pd.DataFrame({
        "A": [10.0, 20.0, 30.0],
        "B": ["cat", "dog", "cat"],
    })

    plan = ExecutionPlan(
        missing_values={},
        duplicates_action=DuplicateStrategy.KEEP_DUPLICATES,
        encoding={},
        scaling={
            "A": ColumnAction(
                strategy=ScalingStrategy.MINMAX_SCALING.value,
                reason="MinMax scaling",
            )
        },
    )

    executor = PipelineExecutor()
    result = executor.execute(df, plan)

    assert len(result) == 3
    assert list(result["A"]) == [0.0, 0.5, 1.0]
    assert list(result["B"]) == ["cat", "dog", "cat"]


def test_empty_dataframe() -> None:
    """Test pipeline execution with an empty DataFrame."""
    df = pd.DataFrame(columns=["A", "B"])
    plan = ExecutionPlan(
        missing_values={
            "A": ColumnAction(
                strategy=MissingValueStrategy.MEAN_IMPUTATION.value,
                reason="Mean imputation",
            )
        },
        duplicates_action=DuplicateStrategy.REMOVE_DUPLICATES,
        encoding={
            "B": ColumnAction(
                strategy=EncodingStrategy.LABEL_ENCODE.value,
                reason="Label encode",
            )
        },
        scaling={
            "A": ColumnAction(
                strategy=ScalingStrategy.MINMAX_SCALING.value,
                reason="MinMax scaling",
            )
        },
    )

    executor = PipelineExecutor()
    result = executor.execute(df, plan)

    assert result.empty
    assert list(result.columns) == ["A", "B"]
    assert result is not df


def test_original_dataframe_unchanged() -> None:
    """Test that the original DataFrame is not mutated during pipeline execution."""
    df = pd.DataFrame({
        "A": [1.0, None, 1.0],
        "B": ["cat", "dog", "cat"],
    })

    df_copy = df.copy(deep=True)

    plan = ExecutionPlan(
        missing_values={
            "A": ColumnAction(
                strategy=MissingValueStrategy.MEAN_IMPUTATION.value,
                reason="Mean imputation",
            )
        },
        duplicates_action=DuplicateStrategy.REMOVE_DUPLICATES,
        encoding={
            "B": ColumnAction(
                strategy=EncodingStrategy.LABEL_ENCODE.value,
                reason="Label encode",
            )
        },
        scaling={},
    )

    executor = PipelineExecutor()
    result = executor.execute(df, plan)

    # Assert original DataFrame df is identical to its copy
    pd.testing.assert_frame_equal(df, df_copy)
    assert result is not df


def test_missing_handler_failure_propagation() -> None:
    """Test that missing value handler failure is wrapped in PipelineExecutionError."""
    df = pd.DataFrame({"A": [1.0, None]})
    plan = ExecutionPlan(
        missing_values={
            "A": ColumnAction(
                strategy=MissingValueStrategy.MEAN_IMPUTATION.value,
                reason="Mean imputation",
            )
        },
        duplicates_action=DuplicateStrategy.KEEP_DUPLICATES,
    )

    executor = PipelineExecutor()
    with patch.object(
        executor._missing_value_handler,
        "apply",
        side_effect=RuntimeError("Handler failed"),
    ):
        with pytest.raises(PipelineExecutionError) as exc_info:
            executor.execute(df, plan)
        assert "Missing value handling stage failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_duplicate_handler_failure_propagation() -> None:
    """Test that duplicate handler failure is wrapped in PipelineExecutionError."""
    df = pd.DataFrame({"A": [1.0, 2.0]})
    plan = ExecutionPlan(
        missing_values={},
        duplicates_action=DuplicateStrategy.REMOVE_DUPLICATES,
    )

    executor = PipelineExecutor()
    with patch.object(
        executor._duplicate_handler,
        "apply",
        side_effect=RuntimeError("Handler failed"),
    ):
        with pytest.raises(PipelineExecutionError) as exc_info:
            executor.execute(df, plan)
        assert "Duplicate handling stage failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_encoding_handler_failure_propagation() -> None:
    """Test that encoding handler failure is wrapped in PipelineExecutionError."""
    df = pd.DataFrame({"A": ["cat", "dog"]})
    plan = ExecutionPlan(
        missing_values={},
        duplicates_action=DuplicateStrategy.KEEP_DUPLICATES,
        encoding={
            "A": ColumnAction(
                strategy=EncodingStrategy.LABEL_ENCODE.value,
                reason="Label encode",
            )
        },
    )

    executor = PipelineExecutor()
    with patch.object(
        executor._encoding_handler,
        "apply",
        side_effect=RuntimeError("Handler failed"),
    ):
        with pytest.raises(PipelineExecutionError) as exc_info:
            executor.execute(df, plan)
        assert "Categorical encoding stage failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_scaling_handler_failure_propagation() -> None:
    """Test that scaling handler failure is wrapped in PipelineExecutionError."""
    df = pd.DataFrame({"A": [1.0, 2.0]})
    plan = ExecutionPlan(
        missing_values={},
        duplicates_action=DuplicateStrategy.KEEP_DUPLICATES,
        encoding={},
        scaling={
            "A": ColumnAction(
                strategy=ScalingStrategy.MINMAX_SCALING.value,
                reason="MinMax scaling",
            )
        },
    )

    executor = PipelineExecutor()
    with patch.object(
        executor._scaling_handler,
        "apply",
        side_effect=RuntimeError("Handler failed"),
    ):
        with pytest.raises(PipelineExecutionError) as exc_info:
            executor.execute(df, plan)
        assert "Numerical scaling stage failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
