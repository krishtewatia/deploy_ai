"""Schemas for the preprocessing execution engine.

The execution engine consumes structured plans produced by the AI
recommendation layer. These schemas keep that contract strongly typed while
remaining compatible with the string strategy identifiers returned by the LLM.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MissingValueStrategy(str, Enum):
    """Supported strategies for handling missing values."""

    MEAN_IMPUTATION = "mean_imputation"
    MEDIAN_IMPUTATION = "median_imputation"
    MODE_IMPUTATION = "mode_imputation"
    DROP_COLUMN = "drop_column"


class DuplicateStrategy(str, Enum):
    """Supported strategies for handling duplicate rows."""

    REMOVE_DUPLICATES = "remove_duplicates"
    KEEP_DUPLICATES = "keep_duplicates"


class EncodingStrategy(str, Enum):
    """Supported strategies for encoding categorical columns."""

    ONE_HOT_ENCODE = "one_hot_encode"
    LABEL_ENCODE = "label_encode"


class ScalingStrategy(str, Enum):
    """Supported strategies for scaling numerical columns."""

    STANDARD_SCALING = "standard_scaling"
    MINMAX_SCALING = "minmax_scaling"
    NO_SCALING = "no_scaling"


_ALL_COLUMN_STRATEGIES: set[str] = {
    *(strategy.value for strategy in MissingValueStrategy),
    *(strategy.value for strategy in EncodingStrategy),
    *(strategy.value for strategy in ScalingStrategy),
}


class ColumnAction(BaseModel):
    """Recommended execution action for a single column."""

    model_config = {
        "extra": "forbid",
        "frozen": False,
        "populate_by_name": True,
        "validate_assignment": True,
    }

    strategy: str = Field(
        ...,
        min_length=1,
        description="Machine-readable preprocessing strategy identifier.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Human-readable rationale for choosing the strategy.",
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        """Ensure column-level actions use a known execution strategy."""
        if value not in _ALL_COLUMN_STRATEGIES:
            raise ValueError(f"Unsupported column strategy: {value}")
        return value


class ExecutionPlan(BaseModel):
    """Executable preprocessing plan grouped by transformation category."""

    model_config = {
        "extra": "forbid",
        "frozen": False,
        "populate_by_name": True,
        "use_enum_values": True,
        "validate_assignment": True,
    }

    missing_values: dict[str, ColumnAction] = Field(
        default_factory=dict,
        description="Per-column missing value actions keyed by column name.",
    )
    duplicates_action: DuplicateStrategy = Field(
        ...,
        description="Global duplicate-row handling strategy.",
    )
    encoding: dict[str, ColumnAction] = Field(
        default_factory=dict,
        description="Per-column categorical encoding actions keyed by column name.",
    )
    scaling: dict[str, ColumnAction] = Field(
        default_factory=dict,
        description="Per-column numeric scaling actions keyed by column name.",
    )

    @model_validator(mode="after")
    def validate_section_strategies(self) -> "ExecutionPlan":
        """Ensure strategies appear only in compatible execution sections."""
        self._validate_actions(
            section_name="missing_values",
            actions=self.missing_values,
            allowed={strategy.value for strategy in MissingValueStrategy},
        )
        self._validate_actions(
            section_name="encoding",
            actions=self.encoding,
            allowed={strategy.value for strategy in EncodingStrategy},
        )
        self._validate_actions(
            section_name="scaling",
            actions=self.scaling,
            allowed={strategy.value for strategy in ScalingStrategy},
        )
        return self

    @staticmethod
    def _validate_actions(
        section_name: str,
        actions: dict[str, ColumnAction],
        allowed: set[str],
    ) -> None:
        """Validate all actions in a plan section against allowed strategies."""
        for column_name, action in actions.items():
            if action.strategy not in allowed:
                raise ValueError(
                    f"Invalid strategy '{action.strategy}' for "
                    f"{section_name}.{column_name}"
                )

    @field_validator("missing_values", "encoding", "scaling", mode="before")
    @classmethod
    def validate_column_mapping(cls, value: Any) -> Any:
        """Require transformation sections to be dictionaries."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Column action sections must be dictionaries.")
        return value
