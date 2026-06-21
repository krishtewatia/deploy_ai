"""Tests for backend.app.ai_engine.plan_parser — targeting 100 % coverage.

Covers:
* Valid JSON response
* Empty response
* Whitespace-only response
* Invalid JSON
* Missing required fields (schema validation)
* Invalid schema structure (wrong types)
* Markdown code-fence stripping
* Code-fence with ``json`` tag
"""

from __future__ import annotations

import json

import pytest

from backend.app.ai_engine.plan_parser import (
    InvalidJSONError,
    InvalidRecommendationError,
    PlanParser,
)
from backend.app.ai_engine.schemas import RecommendationResponse


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def parser() -> PlanParser:
    return PlanParser()


@pytest.fixture()
def valid_payload() -> dict:
    """A payload that satisfies the RecommendationResponse schema."""
    return {
        "cleaning_plan": {
            "missing_values": {
                "age": {
                    "strategy": "median_imputation",
                    "reason": "Contains outliers.",
                }
            },
            "duplicates_action": "remove_duplicates",
            "encoding": {},
            "scaling": {
                "age": {
                    "strategy": "standard_scaling",
                    "reason": "Numerical feature.",
                }
            },
        },
        "overall_reasoning": "Recommended preprocessing plan.",
    }


@pytest.fixture()
def valid_json(valid_payload: dict) -> str:
    return json.dumps(valid_payload)


# ── Valid response ──────────────────────────────────────────────────────────


class TestValidResponse:
    """Happy-path: well-formed JSON that matches the schema."""

    def test_returns_recommendation_response(
        self, parser: PlanParser, valid_json: str
    ) -> None:
        result = parser.parse(valid_json)
        assert isinstance(result, RecommendationResponse)

    def test_cleaning_plan_fields(
        self, parser: PlanParser, valid_json: str
    ) -> None:
        result = parser.parse(valid_json)
        plan = result.cleaning_plan
        assert plan.duplicates_action == "remove_duplicates"
        assert plan.missing_values["age"].strategy == "median_imputation"
        assert plan.scaling["age"].strategy == "standard_scaling"
        assert plan.encoding == {}

    def test_overall_reasoning(
        self, parser: PlanParser, valid_json: str
    ) -> None:
        result = parser.parse(valid_json)
        assert result.overall_reasoning == "Recommended preprocessing plan."


# ── Empty / whitespace response ─────────────────────────────────────────────


class TestEmptyResponse:
    """Empty or whitespace-only strings must raise InvalidJSONError."""

    def test_empty_string(self, parser: PlanParser) -> None:
        with pytest.raises(InvalidJSONError, match="empty"):
            parser.parse("")

    def test_whitespace_only(self, parser: PlanParser) -> None:
        with pytest.raises(InvalidJSONError, match="empty"):
            parser.parse("   \n\t  ")


# ── Invalid JSON ────────────────────────────────────────────────────────────


class TestInvalidJSON:
    """Strings that are not valid JSON must raise InvalidJSONError."""

    def test_plain_text(self, parser: PlanParser) -> None:
        with pytest.raises(InvalidJSONError, match="not valid JSON"):
            parser.parse("This is not JSON at all.")

    def test_truncated_json(self, parser: PlanParser) -> None:
        with pytest.raises(InvalidJSONError, match="not valid JSON"):
            parser.parse('{"cleaning_plan": {')


# ── Missing required fields ────────────────────────────────────────────────


class TestMissingFields:
    """Valid JSON that is missing required schema fields."""

    def test_missing_overall_reasoning(self, parser: PlanParser) -> None:
        payload = {
            "cleaning_plan": {
                "missing_values": {},
                "duplicates_action": "drop",
                "encoding": {},
                "scaling": {},
            }
        }
        with pytest.raises(InvalidRecommendationError, match="schema"):
            parser.parse(json.dumps(payload))

    def test_missing_cleaning_plan(self, parser: PlanParser) -> None:
        payload = {"overall_reasoning": "Some reasoning."}
        with pytest.raises(InvalidRecommendationError, match="schema"):
            parser.parse(json.dumps(payload))


# ── Invalid schema structure ───────────────────────────────────────────────


class TestInvalidSchemaStructure:
    """JSON is valid but types / values don't match the schema."""

    def test_wrong_type_for_cleaning_plan(self, parser: PlanParser) -> None:
        payload = {
            "cleaning_plan": "not a dict",
            "overall_reasoning": "Some reasoning.",
        }
        with pytest.raises(InvalidRecommendationError, match="schema"):
            parser.parse(json.dumps(payload))

    def test_empty_duplicates_action(self, parser: PlanParser) -> None:
        payload = {
            "cleaning_plan": {
                "missing_values": {},
                "duplicates_action": "",
                "encoding": {},
                "scaling": {},
            },
            "overall_reasoning": "Some reasoning.",
        }
        with pytest.raises(InvalidRecommendationError, match="schema"):
            parser.parse(json.dumps(payload))


# ── Markdown code-fence stripping ──────────────────────────────────────────


class TestCodeFenceStripping:
    """LLMs sometimes wrap JSON in markdown fences — parser must handle it."""

    def test_triple_backtick_json_tag(
        self, parser: PlanParser, valid_json: str
    ) -> None:
        wrapped = f"```json\n{valid_json}\n```"
        result = parser.parse(wrapped)
        assert isinstance(result, RecommendationResponse)

    def test_triple_backtick_no_tag(
        self, parser: PlanParser, valid_json: str
    ) -> None:
        wrapped = f"```\n{valid_json}\n```"
        result = parser.parse(wrapped)
        assert isinstance(result, RecommendationResponse)

    def test_extra_whitespace_around_fences(
        self, parser: PlanParser, valid_json: str
    ) -> None:
        wrapped = f"  \n```json\n  {valid_json}  \n```  \n"
        result = parser.parse(wrapped)
        assert isinstance(result, RecommendationResponse)
