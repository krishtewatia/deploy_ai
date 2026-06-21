"""Parse raw Groq LLM responses into validated recommendation objects.

This module sits between the :class:`~backend.app.ai_engine.groq_client.GroqClient`
(which returns raw text) and the rest of the application (which expects
typed :class:`~backend.app.ai_engine.schemas.RecommendationResponse` instances).

It handles common LLM quirks such as markdown code-fences wrapping the
JSON payload, and surfaces clear exceptions for every failure mode.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from pydantic import ValidationError

from backend.app.ai_engine.schemas import RecommendationResponse

logger = logging.getLogger(__name__)

# ── Regex to strip markdown code fences ─────────────────────────────────────
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)


# ── Custom exceptions ──────────────────────────────────────────────────────

class PlanParserError(Exception):
    """Base exception for all :class:`PlanParser` failures."""


class InvalidJSONError(PlanParserError):
    """Raised when the LLM response is not valid JSON."""


class InvalidRecommendationError(PlanParserError):
    """Raised when the JSON does not conform to the expected schema."""


# ── Parser ──────────────────────────────────────────────────────────────────

class PlanParser:
    """Convert a raw LLM text response into a validated
    :class:`RecommendationResponse`.

    The parser is resilient to common LLM output artefacts:

    * Leading / trailing whitespace
    * Markdown ````` `` ``` ````` code fences (with or without ``json`` tag)
    * Empty / whitespace-only responses

    Usage::

        parser   = PlanParser()
        response = parser.parse(raw_llm_text)
    """

    # ── public API ──────────────────────────────────────────────────────

    def parse(self, raw_response: str) -> RecommendationResponse:
        """Parse *raw_response* into a :class:`RecommendationResponse`.

        Parameters
        ----------
        raw_response:
            The raw string returned by the LLM (e.g. from
            :meth:`GroqClient.generate_recommendations`).

        Returns
        -------
        RecommendationResponse
            A fully-validated recommendation object.

        Raises
        ------
        InvalidJSONError
            The response could not be decoded as JSON.
        InvalidRecommendationError
            The JSON is syntactically valid but does not match the
            :class:`RecommendationResponse` schema.
        """
        logger.info("Parsing raw LLM response (%d characters).", len(raw_response))

        cleaned = self._clean_response(raw_response)
        data = self._decode_json(cleaned)
        return self._validate(data)

    # ── private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _clean_response(raw: str) -> str:
        """Strip whitespace and markdown code fences from *raw*.

        Raises
        ------
        InvalidJSONError
            If the response is empty or whitespace-only after cleaning.
        """
        text = raw.strip()

        if not text:
            logger.error("Received empty response from LLM.")
            raise InvalidJSONError("LLM returned an empty response.")

        # Strip markdown fences if present
        match = _CODE_FENCE_RE.search(text)
        if match:
            text = match.group(1).strip()
            logger.debug("Stripped markdown code fence from response.")

        return text

    @staticmethod
    def _decode_json(text: str) -> Dict[str, Any]:
        """Decode *text* as JSON.

        Raises
        ------
        InvalidJSONError
            If *text* is not valid JSON.
        """
        try:
            data = json.loads(text)
            logger.debug("JSON decoded successfully.")
            return data
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON from LLM: %s", exc)
            raise InvalidJSONError(
                f"LLM response is not valid JSON: {exc}"
            ) from exc

    @staticmethod
    def _validate(data: Dict[str, Any]) -> RecommendationResponse:
        """Validate *data* against the :class:`RecommendationResponse` schema.

        Raises
        ------
        InvalidRecommendationError
            If schema validation fails.
        """
        try:
            response = RecommendationResponse.model_validate(data)
            logger.info("Recommendation response validated successfully.")
            return response
        except ValidationError as exc:
            logger.error("Schema validation failed: %s", exc)
            raise InvalidRecommendationError(
                f"LLM response does not match the expected schema: {exc}"
            ) from exc
