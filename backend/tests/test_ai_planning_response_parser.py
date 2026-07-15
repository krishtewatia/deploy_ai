"""Tests for AI Response Parser (Stage 7D5)."""

import json
import pytest

from backend.app.ai_planning.response_parser import AIResponseParser, AIResponseParseError
from backend.app.ai_planning.schemas import AIDecisionProposal


# ── Helpers ────────────────────────────────────────────────────────────


def _valid_proposal_json() -> str:
    """Return a valid minimal AIDecisionProposal JSON string."""
    return json.dumps({
        "proposal_set_id": "ps_01",
        "baseline_plan_id": "bp_01",
        "dataset_id": "ds_01",
        "request_id": "req_01",
        "problem_definition_id": "pd_01",
        "compute_capability_id": "cap_01",
        "summary": "No changes needed.",
    })


def _valid_proposal_with_proposals_json() -> str:
    """Return a proposal with actual proposals."""
    return json.dumps({
        "proposal_set_id": "ps_01",
        "baseline_plan_id": "bp_01",
        "dataset_id": "ds_01",
        "request_id": "req_01",
        "problem_definition_id": "pd_01",
        "compute_capability_id": "cap_01",
        "summary": "Add model candidate.",
        "model_candidate_proposals": [
            {
                "proposal_id": "mc_01",
                "action": "add",
                "model_family": "random_forest",
                "parameters": {"n_estimators": 100},
                "search_strategy": "none",
                "search_space": {},
                "reason": "Add random forest for comparison.",
                "confidence": "high",
            }
        ],
    })


# ── Tests ──────────────────────────────────────────────────────────────


class TestAIResponseParser:
    def test_parse_valid_json(self):
        parser = AIResponseParser()
        result = parser.parse(_valid_proposal_json())
        assert isinstance(result, AIDecisionProposal)
        assert result.proposal_set_id == "ps_01"

    def test_parse_valid_with_proposals(self):
        parser = AIResponseParser()
        result = parser.parse(_valid_proposal_with_proposals_json())
        assert len(result.model_candidate_proposals) == 1
        assert result.model_candidate_proposals[0].model_family.value == "random_forest"

    def test_empty_response_rejected(self):
        parser = AIResponseParser()
        with pytest.raises(AIResponseParseError, match="empty"):
            parser.parse("")

    def test_whitespace_only_response_rejected(self):
        parser = AIResponseParser()
        with pytest.raises(AIResponseParseError, match="empty"):
            parser.parse("   \n\t  ")

    def test_invalid_json_rejected(self):
        parser = AIResponseParser()
        with pytest.raises(AIResponseParseError, match="not valid JSON"):
            parser.parse("this is not json at all")

    def test_valid_json_invalid_schema_rejected(self):
        parser = AIResponseParser()
        with pytest.raises(AIResponseParseError, match="does not match"):
            parser.parse('{"not": "a proposal"}')

    def test_invalid_enum_value_rejected(self):
        parser = AIResponseParser()
        data = json.dumps({
            "proposal_set_id": "ps_01",
            "baseline_plan_id": "bp_01",
            "dataset_id": "ds_01",
            "request_id": "req_01",
            "problem_definition_id": "pd_01",
            "compute_capability_id": "cap_01",
            "summary": "Bad.",
            "model_candidate_proposals": [
                {
                    "proposal_id": "mc_01",
                    "action": "invalid_action",
                    "model_family": "random_forest",
                    "reason": "Test.",
                    "confidence": "high",
                }
            ],
        })
        with pytest.raises(AIResponseParseError, match="does not match"):
            parser.parse(data)

    def test_missing_required_field_rejected(self):
        parser = AIResponseParser()
        data = json.dumps({
            "baseline_plan_id": "bp_01",
            "dataset_id": "ds_01",
            "request_id": "req_01",
            "problem_definition_id": "pd_01",
            "compute_capability_id": "cap_01",
            "summary": "Missing proposal_set_id.",
        })
        with pytest.raises(AIResponseParseError, match="does not match"):
            parser.parse(data)

    def test_markdown_json_fence_stripped(self):
        parser = AIResponseParser()
        fenced = "```json\n" + _valid_proposal_json() + "\n```"
        result = parser.parse(fenced)
        assert isinstance(result, AIDecisionProposal)

    def test_markdown_plain_fence_stripped(self):
        parser = AIResponseParser()
        fenced = "```\n" + _valid_proposal_json() + "\n```"
        result = parser.parse(fenced)
        assert isinstance(result, AIDecisionProposal)

    def test_no_fence_no_stripping(self):
        parser = AIResponseParser()
        raw = _valid_proposal_json()
        result = parser.parse(raw)
        assert isinstance(result, AIDecisionProposal)

    def test_parse_error_is_exception(self):
        err = AIResponseParseError("test")
        assert isinstance(err, Exception)

    def test_whitespace_around_valid_json(self):
        parser = AIResponseParser()
        raw = "\n  " + _valid_proposal_json() + "\n  "
        result = parser.parse(raw)
        assert isinstance(result, AIDecisionProposal)

    def test_duplicate_proposal_ids_rejected_via_schema(self):
        parser = AIResponseParser()
        data = json.dumps({
            "proposal_set_id": "ps_01",
            "baseline_plan_id": "bp_01",
            "dataset_id": "ds_01",
            "request_id": "req_01",
            "problem_definition_id": "pd_01",
            "compute_capability_id": "cap_01",
            "summary": "Dup IDs.",
            "preprocessing_proposals": [
                {"proposal_id": "dup", "action": "add", "operation": "impute_mean",
                 "columns": [], "reason": "R.", "confidence": "low"},
            ],
            "model_candidate_proposals": [
                {"proposal_id": "dup", "action": "add", "model_family": "knn",
                 "reason": "R.", "confidence": "low"},
            ],
        })
        with pytest.raises(AIResponseParseError, match="does not match"):
            parser.parse(data)
