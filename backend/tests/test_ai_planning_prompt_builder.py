"""Tests for AI Prompt Builder (Stage 7D4)."""

import json
import pytest

from backend.app.ai_planning.prompt_builder import AIPlanningPromptBuilder


# ── Fixtures ───────────────────────────────────────────────────────────


def _minimal_context() -> dict:
    """A minimal valid planning context dict."""
    return {
        "dataset": {
            "dataset_id": "ds_01",
            "row_count": 100,
            "column_count": 2,
            "columns": [
                {"name": "x", "dtype": "float64", "is_numeric": True, "is_categorical": False,
                 "is_datetime": False, "missing_percentage": 0.0, "unique_count": 100,
                 "unique_percentage": 100.0, "sample_values": [1.0, 2.0]},
                {"name": "y", "dtype": "int64", "is_numeric": True, "is_categorical": True,
                 "is_datetime": False, "missing_percentage": 0.0, "unique_count": 2,
                 "unique_percentage": 2.0, "sample_values": [0, 1]},
            ],
        },
        "user_goal": {
            "goal": "Predict y",
            "target_column": "y",
            "problem_type_preference": "auto",
            "primary_metric": None,
            "excluded_columns": [],
            "additional_context": None,
        },
        "resolved_problem": {
            "target_column": "y",
            "problem_type": "classification",
            "feature_columns": ["x"],
            "excluded_columns": [],
            "primary_metric": "f1",
            "warnings": [],
        },
        "compute_capabilities": {
            "compute_tier": "standard",
            "memory_constraint": "moderate",
            "safe_parallel_workers": 4,
            "gpu_acceleration_available": False,
            "accelerator_type": "none",
            "warnings": [],
        },
        "baseline_plan": {
            "preprocessing_steps": [],
            "feature_engineering_steps": [],
            "feature_selection": {"method": "none", "candidate_columns": ["x"],
                                   "max_features": None, "reason": "No selection."},
            "split_plan": {"strategy": "stratified", "test_size": 0.2, "validation_size": 0.0,
                           "random_state": 42, "shuffle": True, "stratify_column": "y",
                           "time_column": None},
            "model_candidates": [{"candidate_id": "m1", "model_family": "logistic_regression",
                                   "parameters": {}, "search_strategy": "none",
                                   "search_space": {}, "reason": "Baseline."}],
            "evaluation_plan": {"primary_metric": "f1", "secondary_metrics": [],
                                "cross_validation_folds": 5},
            "execution_constraints": {"parallel_workers": 4, "use_gpu_acceleration": False,
                                      "accelerator_type": "none", "compute_tier": "standard"},
            "warnings": [],
        },
    }


# ── Tests ──────────────────────────────────────────────────────────────


class TestAIPlanningPromptBuilder:
    def test_returns_dict_with_keys(self):
        builder = AIPlanningPromptBuilder()
        result = builder.build(planning_context=_minimal_context())
        assert isinstance(result, dict)
        assert "system_prompt" in result
        assert "user_prompt" in result

    def test_system_prompt_contains_rules(self):
        builder = AIPlanningPromptBuilder()
        result = builder.build(planning_context=_minimal_context())
        sp = result["system_prompt"]
        assert "AIDecisionProposal" in sp
        assert "Return ONLY valid JSON" in sp
        assert "Do NOT return Python code" in sp
        assert "Do NOT invent dataset columns" in sp
        assert "Respect the resolved target column" in sp
        assert "hardware constraints" in sp
        assert "no-change proposal" in sp

    def test_system_prompt_contains_json_schema(self):
        builder = AIPlanningPromptBuilder()
        result = builder.build(planning_context=_minimal_context())
        sp = result["system_prompt"]
        assert "EXPECTED RESPONSE SCHEMA" in sp
        # The JSON schema should be embedded
        assert "properties" in sp

    def test_user_prompt_contains_context(self):
        builder = AIPlanningPromptBuilder()
        ctx = _minimal_context()
        result = builder.build(planning_context=ctx)
        up = result["user_prompt"]
        assert "ds_01" in up
        assert "Predict y" in up
        assert "PLANNING CONTEXT" in up

    def test_user_prompt_is_valid_with_embedded_json(self):
        builder = AIPlanningPromptBuilder()
        ctx = _minimal_context()
        result = builder.build(planning_context=ctx)
        up = result["user_prompt"]
        # Extract JSON from user prompt
        json_start = up.index("{")
        json_str = up[json_start:]
        parsed = json.loads(json_str)
        assert parsed["dataset"]["dataset_id"] == "ds_01"

    def test_prompts_are_strings(self):
        builder = AIPlanningPromptBuilder()
        result = builder.build(planning_context=_minimal_context())
        assert isinstance(result["system_prompt"], str)
        assert isinstance(result["user_prompt"], str)

    def test_deterministic(self):
        builder = AIPlanningPromptBuilder()
        ctx = _minimal_context()
        r1 = builder.build(planning_context=ctx)
        r2 = builder.build(planning_context=ctx)
        assert r1 == r2

    def test_does_not_mutate_context(self):
        import copy
        builder = AIPlanningPromptBuilder()
        ctx = _minimal_context()
        orig = copy.deepcopy(ctx)
        builder.build(planning_context=ctx)
        assert ctx == orig
