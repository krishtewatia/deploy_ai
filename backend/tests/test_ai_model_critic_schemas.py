"""Unit tests for ModelCritique schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.ai_model_critic.schemas import CritiqueGrade, ModelCritique


def _make_valid_critique_dict() -> dict:
    return {
        "critique_id": "crit_01",
        "report_id": "report_01",
        "overall_grade": "A+",
        "production_ready": True,
        "confidence": 0.95,
        "strengths": ["Fast inference", "Low overfitting"],
        "weaknesses": ["Minor skew on column B"],
        "risks": ["Out of distribution drift risk"],
        "recommendations": ["Retrain with more data next week"],
        "warnings": ["Imbalanced validation fold"],
        "summary": "The model performs outstandingly.",
    }


class TestModelCritiqueSchema:
    """Tests covering validation logic in ModelCritique pydantic schema."""

    def test_valid_critique_validation(self):
        """Verify valid critique inputs parse correctly."""
        d = _make_valid_critique_dict()
        crit = ModelCritique(**d)
        assert crit.critique_id == "crit_01"
        assert crit.overall_grade == CritiqueGrade.A_PLUS
        assert crit.production_ready is True
        assert crit.confidence == 0.95
        assert len(crit.strengths) == 2

    def test_grade_enum_invalid(self):
        """Verify invalid grade throws ValidationError."""
        d = _make_valid_critique_dict()
        d["overall_grade"] = "INVALID"
        with pytest.raises(ValidationError):
            ModelCritique(**d)

    def test_confidence_out_of_bounds(self):
        """Verify confidence bounds limits (0.0 to 1.0) are enforced."""
        d = _make_valid_critique_dict()
        
        # Too low
        d["confidence"] = -0.1
        with pytest.raises(ValidationError):
            ModelCritique(**d)

        # Too high
        d["confidence"] = 1.1
        with pytest.raises(ValidationError):
            ModelCritique(**d)

    def test_validation_rejects_empty_strings(self):
        """Verify empty strings are stripped and rejected."""
        d = _make_valid_critique_dict()
        d["summary"] = "   "
        with pytest.raises(ValidationError, match="cannot be empty"):
            ModelCritique(**d)

    def test_validation_rejects_non_list_fields(self):
        """Verify list fields reject non-list structures."""
        d = _make_valid_critique_dict()
        d["strengths"] = "not-a-list"
        with pytest.raises(ValidationError, match="must be a list"):
            ModelCritique(**d)

    def test_validation_rejects_empty_items_in_lists(self):
        """Verify empty items inside list fields are rejected."""
        d = _make_valid_critique_dict()
        d["strengths"] = ["Valid strength", "  "]
        with pytest.raises(ValidationError, match="cannot be empty"):
            ModelCritique(**d)
