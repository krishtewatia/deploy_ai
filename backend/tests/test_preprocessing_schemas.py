"""Tests for preprocessing execution engine schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.preprocessing_engine.schemas import (
    ColumnAction,
    DuplicateStrategy,
    EncodingStrategy,
    ExecutionPlan,
    MissingValueStrategy,
    ScalingStrategy,
)


@pytest.fixture()
def valid_plan_payload() -> dict:
    """Return a complete valid execution plan payload."""
    return {
        "missing_values": {
            "age": {
                "strategy": "median_imputation",
                "reason": "Median is robust to outliers.",
            },
            "legacy_code": {
                "strategy": "drop_column",
                "reason": "Column has too many missing values.",
            },
        },
        "duplicates_action": "remove_duplicates",
        "encoding": {
            "department": {
                "strategy": "one_hot_encode",
                "reason": "Low-cardinality nominal feature.",
            },
            "city": {
                "strategy": "label_encode",
                "reason": "Compact representation is acceptable.",
            },
        },
        "scaling": {
            "salary": {
                "strategy": "standard_scaling",
                "reason": "Scale-sensitive downstream model.",
            },
            "score": {
                "strategy": "no_scaling",
                "reason": "Already normalized.",
            },
        },
    }


class TestEnums:
    """Validate preprocessing strategy enums."""

    def test_enum_values(self) -> None:
        assert MissingValueStrategy.MEAN_IMPUTATION.value == "mean_imputation"
        assert MissingValueStrategy.MEDIAN_IMPUTATION.value == "median_imputation"
        assert MissingValueStrategy.MODE_IMPUTATION.value == "mode_imputation"
        assert MissingValueStrategy.DROP_COLUMN.value == "drop_column"
        assert DuplicateStrategy.REMOVE_DUPLICATES.value == "remove_duplicates"
        assert DuplicateStrategy.KEEP_DUPLICATES.value == "keep_duplicates"
        assert EncodingStrategy.ONE_HOT_ENCODE.value == "one_hot_encode"
        assert EncodingStrategy.LABEL_ENCODE.value == "label_encode"
        assert ScalingStrategy.STANDARD_SCALING.value == "standard_scaling"
        assert ScalingStrategy.MINMAX_SCALING.value == "minmax_scaling"
        assert ScalingStrategy.NO_SCALING.value == "no_scaling"

    def test_invalid_enum_value_raises(self) -> None:
        with pytest.raises(ValueError):
            MissingValueStrategy("unknown_strategy")


class TestColumnAction:
    """Validate column action schema behavior."""

    def test_valid_schema_creation(self) -> None:
        action = ColumnAction(
            strategy="mean_imputation",
            reason="Column is numeric and approximately symmetric.",
        )

        assert action.strategy == "mean_imputation"
        assert action.reason == "Column is numeric and approximately symmetric."

    def test_invalid_strategy_value_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ColumnAction(strategy="unsupported_strategy", reason="Invalid.")

        assert "Unsupported column strategy" in str(exc_info.value)

    def test_empty_strategy_raises(self) -> None:
        with pytest.raises(ValidationError):
            ColumnAction(strategy="", reason="Missing strategy.")

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValidationError):
            ColumnAction(strategy="mode_imputation", reason="")

    def test_extra_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ColumnAction(
                strategy="mode_imputation",
                reason="Valid reason.",
                unexpected=True,
            )

    def test_assignment_validation(self) -> None:
        action = ColumnAction(strategy="mode_imputation", reason="Categorical mode.")

        with pytest.raises(ValidationError):
            action.strategy = "bad_strategy"


class TestExecutionPlan:
    """Validate execution plan schema behavior."""

    def test_valid_schema_creation(self, valid_plan_payload: dict) -> None:
        plan = ExecutionPlan(**valid_plan_payload)

        assert plan.duplicates_action == "remove_duplicates"
        assert isinstance(plan.missing_values["age"], ColumnAction)
        assert plan.encoding["department"].strategy == "one_hot_encode"
        assert plan.scaling["salary"].strategy == "standard_scaling"

    def test_enum_validation_accepts_enum_instance(self) -> None:
        plan = ExecutionPlan(duplicates_action=DuplicateStrategy.KEEP_DUPLICATES)

        assert plan.duplicates_action == "keep_duplicates"
        assert plan.missing_values == {}
        assert plan.encoding == {}
        assert plan.scaling == {}

    def test_invalid_duplicate_strategy_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionPlan(duplicates_action="drop_duplicates")

    def test_invalid_missing_value_section_strategy_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                duplicates_action="keep_duplicates",
                missing_values={
                    "department": {
                        "strategy": "one_hot_encode",
                        "reason": "Wrong section.",
                    }
                },
            )

        assert "missing_values.department" in str(exc_info.value)

    def test_invalid_encoding_section_strategy_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                duplicates_action="keep_duplicates",
                encoding={
                    "department": {
                        "strategy": "median_imputation",
                        "reason": "Wrong section.",
                    }
                },
            )

        assert "encoding.department" in str(exc_info.value)

    def test_invalid_scaling_section_strategy_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                duplicates_action="keep_duplicates",
                scaling={
                    "salary": {
                        "strategy": "label_encode",
                        "reason": "Wrong section.",
                    }
                },
            )

        assert "scaling.salary" in str(exc_info.value)

    def test_none_sections_default_to_empty_dicts(self) -> None:
        plan = ExecutionPlan(
            duplicates_action="keep_duplicates",
            missing_values=None,
            encoding=None,
            scaling=None,
        )

        assert plan.missing_values == {}
        assert plan.encoding == {}
        assert plan.scaling == {}

    def test_non_dict_section_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionPlan(
                duplicates_action="keep_duplicates",
                missing_values=["age"],
            )

    def test_serialization(self, valid_plan_payload: dict) -> None:
        plan = ExecutionPlan(**valid_plan_payload)
        dumped = plan.model_dump()

        assert dumped == valid_plan_payload

    def test_deserialization(self, valid_plan_payload: dict) -> None:
        plan = ExecutionPlan.model_validate(valid_plan_payload)

        assert plan.model_dump() == valid_plan_payload

    def test_json_round_trip(self, valid_plan_payload: dict) -> None:
        plan = ExecutionPlan(**valid_plan_payload)
        restored = ExecutionPlan.model_validate_json(plan.model_dump_json())

        assert restored.model_dump() == valid_plan_payload

    def test_extra_field_raises(self, valid_plan_payload: dict) -> None:
        payload = {**valid_plan_payload, "unexpected": True}

        with pytest.raises(ValidationError):
            ExecutionPlan(**payload)

    def test_assignment_validation(self) -> None:
        plan = ExecutionPlan(duplicates_action="keep_duplicates")

        with pytest.raises(ValidationError):
            plan.duplicates_action = "drop_duplicates"
