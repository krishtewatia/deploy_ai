"""Tests for backend.app.ai_engine.schemas — targeting 100 % coverage.

Covers:
* Valid schema creation (happy-path)
* Nested schema validation
* ``model_dump`` round-trip verification
* Validation errors for missing / invalid fields
* Default-factory behaviour for optional dict fields
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.ai_engine.schemas import (
    CleaningPlan,
    ColumnRecommendation,
    RecommendationResponse,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_column_recommendation() -> dict:
    """Minimal valid payload for a ``ColumnRecommendation``."""
    return {
        "strategy": "median_imputation",
        "reason": "Column contains outliers and is right-skewed.",
    }


@pytest.fixture()
def sample_cleaning_plan(sample_column_recommendation: dict) -> dict:
    """Minimal valid payload for a ``CleaningPlan``."""
    return {
        "missing_values": {"age": sample_column_recommendation},
        "duplicates_action": "drop",
        "encoding": {"city": {"strategy": "one_hot_encoding", "reason": "Low cardinality."}},
        "scaling": {"income": {"strategy": "standard_scaling", "reason": "Normally distributed."}},
    }


@pytest.fixture()
def sample_recommendation_response(sample_cleaning_plan: dict) -> dict:
    """Minimal valid payload for a ``RecommendationResponse``."""
    return {
        "cleaning_plan": sample_cleaning_plan,
        "overall_reasoning": (
            "The dataset is moderately clean with 5% missing values."
        ),
    }


# ── ColumnRecommendation ───────────────────────────────────────────────────


class TestColumnRecommendation:
    """Unit tests for :class:`ColumnRecommendation`."""

    def test_valid_creation(self, sample_column_recommendation: dict) -> None:
        rec = ColumnRecommendation(**sample_column_recommendation)
        assert rec.strategy == "median_imputation"
        assert "outliers" in rec.reason

    def test_model_dump(self, sample_column_recommendation: dict) -> None:
        rec = ColumnRecommendation(**sample_column_recommendation)
        dumped = rec.model_dump()
        assert isinstance(dumped, dict)
        assert dumped == sample_column_recommendation

    def test_missing_strategy_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ColumnRecommendation(reason="some reason")  # type: ignore[call-arg]
        assert "strategy" in str(exc_info.value)

    def test_missing_reason_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ColumnRecommendation(strategy="mean_imputation")  # type: ignore[call-arg]
        assert "reason" in str(exc_info.value)

    def test_empty_strategy_raises(self) -> None:
        with pytest.raises(ValidationError):
            ColumnRecommendation(strategy="", reason="valid reason")

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValidationError):
            ColumnRecommendation(strategy="valid_strategy", reason="")

    def test_model_json_schema_has_descriptions(self) -> None:
        schema = ColumnRecommendation.model_json_schema()
        props = schema["properties"]
        assert "description" in props["strategy"]
        assert "description" in props["reason"]


# ── CleaningPlan ────────────────────────────────────────────────────────────


class TestCleaningPlan:
    """Unit tests for :class:`CleaningPlan`."""

    def test_valid_creation(self, sample_cleaning_plan: dict) -> None:
        plan = CleaningPlan(**sample_cleaning_plan)
        assert plan.duplicates_action == "drop"
        assert "age" in plan.missing_values
        assert isinstance(plan.missing_values["age"], ColumnRecommendation)

    def test_nested_validation(self, sample_cleaning_plan: dict) -> None:
        plan = CleaningPlan(**sample_cleaning_plan)
        assert plan.encoding["city"].strategy == "one_hot_encoding"
        assert plan.scaling["income"].strategy == "standard_scaling"

    def test_model_dump(self, sample_cleaning_plan: dict) -> None:
        plan = CleaningPlan(**sample_cleaning_plan)
        dumped = plan.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["duplicates_action"] == "drop"
        assert dumped["missing_values"]["age"]["strategy"] == "median_imputation"
        assert dumped["encoding"]["city"]["strategy"] == "one_hot_encoding"
        assert dumped["scaling"]["income"]["strategy"] == "standard_scaling"

    def test_defaults_for_dict_fields(self) -> None:
        plan = CleaningPlan(duplicates_action="keep_first")
        assert plan.missing_values == {}
        assert plan.encoding == {}
        assert plan.scaling == {}

    def test_missing_duplicates_action_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CleaningPlan()  # type: ignore[call-arg]
        assert "duplicates_action" in str(exc_info.value)

    def test_empty_duplicates_action_raises(self) -> None:
        with pytest.raises(ValidationError):
            CleaningPlan(duplicates_action="")

    def test_invalid_nested_recommendation_raises(self) -> None:
        with pytest.raises(ValidationError):
            CleaningPlan(
                duplicates_action="drop",
                missing_values={"col": {"strategy": ""}},  # empty strategy
            )


# ── RecommendationResponse ─────────────────────────────────────────────────


class TestRecommendationResponse:
    """Unit tests for :class:`RecommendationResponse`."""

    def test_valid_creation(self, sample_recommendation_response: dict) -> None:
        resp = RecommendationResponse(**sample_recommendation_response)
        assert isinstance(resp.cleaning_plan, CleaningPlan)
        assert "moderately clean" in resp.overall_reasoning

    def test_nested_schema_validation(
        self, sample_recommendation_response: dict
    ) -> None:
        resp = RecommendationResponse(**sample_recommendation_response)
        plan = resp.cleaning_plan
        assert plan.missing_values["age"].strategy == "median_imputation"
        assert plan.encoding["city"].reason == "Low cardinality."

    def test_model_dump(self, sample_recommendation_response: dict) -> None:
        resp = RecommendationResponse(**sample_recommendation_response)
        dumped = resp.model_dump()
        assert isinstance(dumped, dict)
        assert "cleaning_plan" in dumped
        assert "overall_reasoning" in dumped
        # Verify full round-trip
        resp2 = RecommendationResponse(**dumped)
        assert resp2.model_dump() == dumped

    def test_missing_cleaning_plan_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            RecommendationResponse(overall_reasoning="Some reasoning.")  # type: ignore[call-arg]
        assert "cleaning_plan" in str(exc_info.value)

    def test_missing_overall_reasoning_raises(
        self, sample_cleaning_plan: dict
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            RecommendationResponse(cleaning_plan=sample_cleaning_plan)  # type: ignore[call-arg]
        assert "overall_reasoning" in str(exc_info.value)

    def test_empty_overall_reasoning_raises(
        self, sample_cleaning_plan: dict
    ) -> None:
        with pytest.raises(ValidationError):
            RecommendationResponse(
                cleaning_plan=sample_cleaning_plan,
                overall_reasoning="",
            )

    def test_model_json_schema_structure(self) -> None:
        schema = RecommendationResponse.model_json_schema()
        assert "properties" in schema
        assert "cleaning_plan" in schema["properties"]
        assert "overall_reasoning" in schema["properties"]
