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
            "system_prompt": self._build_system_prompt(),
            "user_prompt": self._build_user_prompt(planning_context),
        }

    def _build_system_prompt(self) -> str:
        """Build the system prompt with AI rules and constraints."""
        schema = AIDecisionProposal.model_json_schema()
        schema_json = json.dumps(schema, indent=2)

        return (
            "You are an expert ML planning advisor for DeployAI, a local-first "
            "machine learning automation platform.\n\n"
            "You are reviewing an existing deterministic baseline ML plan and may "
            "propose improvements ONLY through the structured proposal schema.\n\n"
            "RULES:\n"
            "1. Return ONLY valid JSON matching the AIDecisionProposal schema.\n"
            "2. Do NOT return markdown fences, code blocks, or explanatory text.\n"
            "3. Do NOT return Python code, scripts, or shell commands.\n"
            "4. Do NOT invent dataset columns that do not exist.\n"
            "5. Respect the resolved target column — do NOT change it.\n"
            "6. Respect excluded columns — do NOT use them as features.\n"
            "7. Respect hardware constraints — do NOT override execution safety limits.\n"
            "8. Avoid unnecessary changes — only propose improvements with clear rationale.\n"
            "9. A valid no-change proposal (empty proposal lists) is acceptable.\n"
            "10. Every proposed change MUST include a reason and confidence level.\n"
            "11. All proposal_id values must be unique across all categories.\n"
            "12. Your response MUST parse as valid JSON conforming to AIDecisionProposal.\n\n"
            "EXPECTED RESPONSE SCHEMA:\n"
            f"{schema_json}"
        )

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
