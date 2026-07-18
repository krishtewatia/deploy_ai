"""AI Response Parser.

Converts raw AI provider text output into a validated AIDecisionProposal.
Handles common edge cases like markdown fences around JSON.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from backend.app.ai_planning.schemas import AIDecisionProposal


class AIResponseParseError(Exception):
    """Raised when the AI response cannot be parsed into a valid proposal."""


class AIResponseParser:
    """Parses and validates raw AI text responses into AIDecisionProposal.

    Security constraints:
    - No eval() or exec()
    - No dynamic imports
    - No execution of returned content
    - No arbitrary Python repair
    """

    def parse(self, raw_response: str) -> AIDecisionProposal:
        """Parse the raw AI response into a validated AIDecisionProposal.

        Args:
            raw_response: Raw text output from an AI provider.

        Returns:
            A validated AIDecisionProposal instance.

        Raises:
            AIResponseParseError: If the response cannot be parsed.
        """
        if not raw_response or not raw_response.strip():
            raise AIResponseParseError("AI response is empty.")

        cleaned = self._strip_markdown_fences(raw_response.strip())

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise AIResponseParseError(
                f"AI response is not valid JSON: {exc}"
            ) from exc

        try:
            proposal = AIDecisionProposal.model_validate(data)
        except ValidationError as exc:
            raise AIResponseParseError(
                f"AI response does not match AIDecisionProposal schema: {exc}"
            ) from exc

        return proposal

    def _strip_markdown_fences(self, text: str) -> str:
        """Strip common markdown JSON fences from the response.

        Handles patterns like:
        - ```json\\n...\\n```
        - ```\\n...\\n```
        """
        # Match ```json ... ``` or ``` ... ```
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
        match = re.match(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
