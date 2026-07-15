"""Unit tests for AIModelCriticResponseParser."""

from __future__ import annotations

import json
import pytest

from backend.app.ai_model_critic.schemas import CritiqueGrade, ModelCritique
from backend.app.ai_model_critic.response_parser import AIModelCriticResponseParser


def _make_valid_json_string() -> str:
    data = {
        "overall_grade": "A",
        "production_ready": True,
        "confidence": 0.90,
        "strengths": ["Robust", "Fast"],
        "weaknesses": ["None"],
        "risks": ["Minor drift"],
        "recommendations": ["Retrain regularly"],
        "warnings": ["Check column distribution"],
        "summary": "Good baseline performance.",
    }
    return json.dumps(data)


class TestAIModelCriticResponseParser:
    """Tests covering LLM text parser sanitization and model mapping."""

    def test_parse_valid_json(self):
        """Verify standard JSON text is parsed correctly."""
        text = _make_valid_json_string()
        parser = AIModelCriticResponseParser()
        critique = parser.parse(text, report_id="rep_123")

        assert isinstance(critique, ModelCritique)
        assert critique.report_id == "rep_123"
        assert critique.overall_grade == CritiqueGrade.A
        assert critique.production_ready is True
        assert critique.critique_id.startswith("critique_")

    def test_parse_json_with_markdown_blocks(self):
        """Verify parser can strip markdown code block tags if returned by LLM."""
        raw_json = _make_valid_json_string()
        text = f"```json\n{raw_json}\n```"
        
        parser = AIModelCriticResponseParser()
        critique = parser.parse(text, report_id="rep_123")
        assert critique.report_id == "rep_123"
        assert critique.overall_grade == CritiqueGrade.A

    def test_parse_rejects_empty(self):
        """Verify empty responses raise ValueError."""
        parser = AIModelCriticResponseParser()
        with pytest.raises(ValueError, match="cannot be empty"):
            parser.parse("", report_id="rep_1")
        with pytest.raises(ValueError, match="cannot be empty"):
            parser.parse("  ", report_id="rep_1")

    def test_parse_rejects_malformed_json(self):
        """Verify malformed JSON raises ValueError."""
        parser = AIModelCriticResponseParser()
        with pytest.raises(ValueError, match="Response is not valid JSON"):
            parser.parse("{invalid-json}", report_id="rep_1")

    def test_parse_rejects_non_object(self):
        """Verify JSON arrays or literals raise ValueError."""
        parser = AIModelCriticResponseParser()
        with pytest.raises(ValueError, match="JSON top-level structure must be a dict"):
            parser.parse('[{"a": 1}]', report_id="rep_1")

    def test_parse_rejects_invalid_schema(self):
        """Verify JSON with missing keys or invalid grades raises ValueError."""
        parser = AIModelCriticResponseParser()
        bad_data = {
            "overall_grade": "Z",  # Invalid grade choice
            "production_ready": True,
        }
        with pytest.raises(ValueError, match="Pydantic schema validation failed"):
            parser.parse(json.dumps(bad_data), report_id="rep_1")
