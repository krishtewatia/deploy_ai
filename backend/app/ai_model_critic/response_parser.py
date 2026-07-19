"""Response parser converting raw LLM completion responses into structured ModelCritique schemas."""

from __future__ import annotations

import json
import uuid
from backend.app.ai_model_critic.schemas import ModelCritique


class AIModelCriticResponseParser:
    """Validates and parses LLM text completion outputs into a validated ModelCritique model."""

    def parse(self, text: str, report_id: str) -> ModelCritique:
        """Parse raw response text, clean markdown backticks, load JSON, and validate against ModelCritique.

        Args:
            text: Raw completion string from the LLM.
            report_id: The authoritative report ID that must be linked.

        Returns:
            A validated Pydantic ModelCritique instance.

        Raises:
            ValueError: If the input text is empty, is not valid JSON,
                        or does not comply with the ModelCritique schema.
        """
        if not text or not text.strip():
            raise ValueError("Response text cannot be empty or None")

        cleaned = text.strip()

        # Remove markdown code block wraps if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            parsed_json = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"Response is not valid JSON: {e}") from e

        if not isinstance(parsed_json, dict):
            raise ValueError("JSON top-level structure must be a dictionary/object")

        # Ensure critique_id is populated
        if "critique_id" not in parsed_json or not parsed_json["critique_id"]:
            parsed_json["critique_id"] = f"critique_{uuid.uuid4().hex}"

        # Authoritatively bind the correct report ID
        parsed_json["report_id"] = report_id

        # Ensure list fields are lists of strings
        for list_field in ["key_insights", "strengths", "weaknesses", "deployment_risks", "recommendations"]:
            if list_field in parsed_json:
                val = parsed_json[list_field]
                if isinstance(val, str):
                    parsed_json[list_field] = [val] if val.strip() else []
                elif isinstance(val, list):
                    parsed_json[list_field] = [str(item) for item in val if str(item).strip()]
                else:
                    parsed_json[list_field] = [str(val)]
            else:
                parsed_json[list_field] = []

        try:
            return ModelCritique(**parsed_json)
        except Exception as e:
            raise ValueError(f"Pydantic schema validation failed: {e}") from e
