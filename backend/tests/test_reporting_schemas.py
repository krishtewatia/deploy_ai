"""Unit tests for ExecutiveReport schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from backend.app.reporting.schemas import ExecutiveReport


def _make_valid_report_dict() -> dict:
    return {
        "report_id": "exec_01",
        "title": "Exec Report",
        "generated_timestamp": "2026-07-15T00:00:00Z",
        "problem_summary": {"type": "classification"},
        "dataset_summary": {"features": ["a"]},
        "pipeline_summary": {"cv": 5},
        "models_summary": [{"family": "lr"}],
        "champion_summary": {"metric": 0.95},
        "optimization_summary": {"actions": []},
        "ai_review": {"strengths": []},
        "governance_summary": {"winner": "BASELINE"},
        "deployment_summary": {"production_ready": True},
        "warnings": [],
        "recommendations": [],
        "executive_summary": "AutoML Run complete.",
    }


class TestReportingSchemas:
    """Tests covering validation rules on ExecutiveReport schema."""

    def test_valid_report_instantiation(self):
        """Verify valid ExecutiveReport is successfully parsed."""
        d = _make_valid_report_dict()
        rep = ExecutiveReport(**d)
        assert rep.report_id == "exec_01"
        assert rep.title == "Exec Report"
        assert rep.executive_summary == "AutoML Run complete."

    def test_empty_strings_rejected(self):
        """Verify empty strings in required fields are rejected."""
        d = _make_valid_report_dict()
        d["title"] = "   "
        with pytest.raises(ValidationError, match="cannot be empty"):
            ExecutiveReport(**d)

    def test_non_string_field_rejected(self):
        """Verify non-string values passed to string fields are rejected."""
        d = _make_valid_report_dict()
        d["executive_summary"] = 123  # non-string
        with pytest.raises(ValidationError, match="must be a string"):
            ExecutiveReport(**d)
