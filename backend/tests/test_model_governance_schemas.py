"""Unit tests for ChampionDecision schema validation."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from backend.app.ml_execution.execution_report import ExecutionReport
from backend.app.model_governance.schemas import ChampionDecision, Winner


def _make_valid_decision_dict() -> dict:
    mock_report = MagicMock(spec=ExecutionReport)
    return {
        "decision_id": "dec_01",
        "baseline_report_id": "rep_base",
        "retrained_report_id": "rep_retrain",
        "winner": Winner.RETRAINED,
        "winner_report": mock_report,
        "improvement_detected": True,
        "metric_name": "f1",
        "baseline_metric": 0.85,
        "retrained_metric": 0.88,
        "relative_improvement": 0.035,
        "decision_reason": "Retrained model scored higher.",
        "production_ready": True,
        "comparison_timestamp": "2026-07-15T00:00:00Z",
    }


class TestModelGovernanceSchemas:
    """Tests covering validation rules on Winner enum and ChampionDecision schema."""

    def test_valid_decision_instantiation(self):
        """Verify valid ChampionDecision is successfully parsed."""
        d = _make_valid_decision_dict()
        dec = ChampionDecision(**d)
        assert dec.decision_id == "dec_01"
        assert dec.winner == Winner.RETRAINED
        assert dec.improvement_detected is True
        assert dec.production_ready is True

    def test_winner_enum_invalid(self):
        """Verify invalid Winner enum values are rejected."""
        d = _make_valid_decision_dict()
        d["winner"] = "INVALID"
        with pytest.raises(ValidationError):
            ChampionDecision(**d)

    def test_empty_strings_rejected(self):
        """Verify empty strings in required fields are rejected."""
        d = _make_valid_decision_dict()
        d["decision_reason"] = "   "
        with pytest.raises(ValidationError, match="cannot be empty"):
            ChampionDecision(**d)

    def test_non_string_field_rejected(self):
        """Verify non-string values passed to string fields are rejected."""
        d = _make_valid_decision_dict()
        d["decision_reason"] = 123  # non-string
        with pytest.raises(ValidationError, match="must be a string"):
            ChampionDecision(**d)
