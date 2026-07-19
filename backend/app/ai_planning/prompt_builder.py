"""AI Prompt Builder.

Converts the structured AI planning context into system and user prompts
for an AI provider.  The prompts instruct the AI to return only structured
JSON matching the AIDecisionProposal schema.
"""

from __future__ import annotations

import json
from typing import Any

from backend.app.ai_planning.schemas import AIDecisionProposal


class AIPlanningPromptBuilder:
    """Builds system and user prompts for AI-assisted ML planning.

    The prompts are deterministic and contain no provider-specific logic.
    """

    def build(
        self,
        *,
        planning_context: dict[str, Any],
    ) -> dict[str, str]:
        """Build system and user prompts from the planning context.

        Args:
            planning_context: The context dictionary from
                :class:`AIPlanningContextBuilder`.

        Returns:
            Dictionary with ``"system_prompt"`` and ``"user_prompt"`` keys.
        """
        return {
            "system_prompt": self._build_system_prompt(planning_context),
            "user_prompt": self._build_user_prompt(planning_context),
        }

    def _build_system_prompt(self, context: dict[str, Any] | None = None) -> str:
        """Build the system prompt with AI rules and constraints."""
        context = context or {}
        bp_id = context.get("baseline_plan", {}).get("plan_id", "baseline_plan_id")
        ds_id = context.get("dataset", {}).get("dataset_id", "dataset_id")
        req_id = context.get("user_goal", {}).get("request_id", "request_id")
        pd_id = context.get("resolved_problem", {}).get("problem_definition_id", "problem_definition_id")
        cc_id = context.get("compute_capabilities", {}).get("compute_capability_id", "compute_capability_id")

        return f"""You are an expert ML planning advisor for DeployAI, a local-first machine learning automation platform.

You are reviewing an existing deterministic baseline ML plan and may propose improvements ONLY through the structured AIDecisionProposal schema.

RULES:
1. Return ONLY valid JSON matching the AIDecisionProposal properties. Do NOT output markdown formatting, code block fences, schema definitions like '$defs', and Do NOT return Python code.
2. Fill in the specific linkage IDs from the planning context (baseline_plan_id, dataset_id, request_id, etc.).
3. Do NOT invent dataset columns that do not exist.
4. Respect the resolved target column — do NOT change it.
5. Respect hardware constraints and excluded columns — do NOT use excluded columns as features.
6. Every proposed change MUST include a unique 'proposal_id', a valid 'action' ("add", "remove", or "replace"), a reason string, and confidence ("low", "medium", or "high").
7. All proposal_id values must be unique across all categories (e.g. "prep_1", "mc_1").
8. A valid no-change proposal (empty proposal lists) is acceptable if the baseline is optimal.
9. Empty arrays are allowed only for top-level proposal lists. Every preprocessing proposal needs at least one existing column. Every feature-engineering proposal needs at least one existing input column and one new output column. Omit a proposal when those names are unknown.

ALLOWED ENUM VALUES:
- action: "add", "remove", "replace"
- confidence: "low", "medium", "high"
- preprocessing operation: "drop_column", "impute_mean", "impute_median", "impute_mode", "impute_constant", "one_hot_encode", "ordinal_encode", "standard_scale", "minmax_scale", "robust_scale", "passthrough"
- feature engineering operation: "interaction", "polynomial", "ratio", "difference", "datetime_parts", "log_transform", "custom"
- model family: "linear_regression", "logistic_regression", "ridge", "lasso", "decision_tree", "random_forest", "gradient_boosting", "extra_trees", "knn", "svm"
- feature selection method: "none", "variance_threshold", "correlation_filter", "mutual_information", "model_based"

EXPECTED RESPONSE SCHEMA (AIDecisionProposal properties):
Example of a valid proposal response JSON:
{{
  "proposal_set_id": "prop_set_001",
  "baseline_plan_id": "{bp_id}",
  "dataset_id": "{ds_id}",
  "request_id": "{req_id}",
  "problem_definition_id": "{pd_id}",
  "compute_capability_id": "{cc_id}",
  "preprocessing_proposals": [],
  "feature_engineering_proposals": [],
  "model_candidate_proposals": [
    {{
      "proposal_id": "mc_1",
      "action": "add",
      "model_family": "random_forest",
      "parameters": {{}},
      "search_strategy": "random",
      "search_space": {{}},
      "reason": "Tree ensemble for non-linear feature interactions",
      "confidence": "high"
    }}
  ],
  "feature_selection_proposal": {{
    "method": "none",
    "reason": "All features are informative",
    "confidence": "high"
  }},
  "evaluation_proposal": {{
    "reason": "Use standard stratified k-fold evaluation",
    "confidence": "high"
  }},
  "warnings": [],
  "summary": "Proposed standard scaling and random forest candidate for baseline optimization."
}}"""

    def _build_user_prompt(self, context: dict[str, Any]) -> str:
        """Build the user prompt containing the planning context."""
        context_json = json.dumps(context, indent=2)

        return (
            "Below is the complete ML planning context including dataset metadata, "
            "user goals, the resolved problem definition, compute capabilities, "
            "and the current deterministic baseline plan.\n\n"
            "Review this context and return a JSON AIDecisionProposal with any "
            "improvements you recommend. If no improvements are needed, return a "
            "valid proposal with empty proposal lists.\n\n"
            "PLANNING CONTEXT:\n"
            f"{context_json}"
        )
