"""Unit tests for optimization schemas validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from backend.app.ai_model_optimizer.schemas import (
    OptimizationAction,
    OptimizationActionType,
    OptimizationResult,
)


class TestOptimizerSchemas:
    """Tests covering validation rules on OptimizationAction and OptimizationResult."""

    def test_optimization_action_valid(self):
        """Verify valid OptimizationAction instantiation."""
        action = OptimizationAction(
            action_id="act_01",
            action_type=OptimizationActionType.CHANGE_CV_FOLDS,
            target=None,
            replacement=None,
            parameters={"folds": 5},
            reason="Increase validation stability",
            confidence=0.9,
        )
        assert action.action_id == "act_01"
        assert action.action_type == "CHANGE_CV_FOLDS"
        assert action.confidence == 0.9

    def test_optimization_action_invalid_type(self):
        """Verify invalid action type is rejected."""
        with pytest.raises(ValidationError):
            OptimizationAction(
                action_id="act_01",
                action_type="INVALID_TYPE",
                reason="some reason",
                confidence=0.9,
            )

    def test_optimization_action_confidence_out_of_bounds(self):
        """Verify confidence boundary checks."""
        with pytest.raises(ValidationError):
            OptimizationAction(
                action_id="act_01",
                action_type=OptimizationActionType.NO_ACTION,
                reason="r",
                confidence=-0.1,
            )
        with pytest.raises(ValidationError):
            OptimizationAction(
                action_id="act_01",
                action_type=OptimizationActionType.NO_ACTION,
                reason="r",
                confidence=1.1,
            )

    def test_optimization_action_empty_strings(self):
        """Verify empty strings in required fields are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            OptimizationAction(
                action_id="   ",
                action_type=OptimizationActionType.NO_ACTION,
                reason="reason",
                confidence=0.5,
            )

        with pytest.raises(ValidationError, match="cannot be empty"):
            OptimizationAction(
                action_id="act_01",
                action_type=OptimizationActionType.NO_ACTION,
                reason="  ",
                confidence=0.5,
            )

    def test_optimization_action_non_string(self):
        """Verify non-string values passed to string fields are rejected."""
        with pytest.raises(ValidationError, match="must be a string"):
            OptimizationAction(
                action_id=123,  # non-string
                action_type=OptimizationActionType.NO_ACTION,
                reason="reason",
                confidence=0.5,
            )
