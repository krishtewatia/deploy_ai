"""Orchestration service for AI-powered preprocessing recommendations.

This module provides the single entry-point that wires together
:class:`PromptBuilder`, :class:`GroqClient`, and :class:`PlanParser`
to convert a :class:`DatasetAnalysisReport` into a validated
:class:`RecommendationResponse` in one call.

Usage::

    from backend.app.ai_engine.recommendation_service import RecommendationService

    service  = RecommendationService()
    response = service.generate_recommendations(analysis_report)
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.ai_engine.groq_client import GroqClient, GroqClientError
from backend.app.ai_engine.plan_parser import PlanParser, PlanParserError
from backend.app.ai_engine.prompt_builder import PromptBuilder
from backend.app.ai_engine.schemas import RecommendationResponse
from backend.app.analysis.schemas import DatasetAnalysisReport

logger = logging.getLogger(__name__)


# ── Custom exception ────────────────────────────────────────────────────────

class RecommendationServiceError(Exception):
    """Raised when the recommendation pipeline fails at any stage."""


# ── Service ─────────────────────────────────────────────────────────────────

class RecommendationService:
    """End-to-end orchestrator for AI preprocessing recommendations.

    Parameters
    ----------
    prompt_builder:
        Optional :class:`PromptBuilder` instance.  A default is created
        when *None*.
    groq_client:
        Optional :class:`GroqClient` instance.  A default is created
        when *None*.
    plan_parser:
        Optional :class:`PlanParser` instance.  A default is created
        when *None*.

    All three dependencies can be injected for testing or to swap
    implementations without modifying consumer code.
    """

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        groq_client: Optional[GroqClient] = None,
        plan_parser: Optional[PlanParser] = None,
    ) -> None:
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._groq_client = groq_client or GroqClient()
        self._plan_parser = plan_parser or PlanParser()
        logger.info("RecommendationService initialised.")

    # ── public API ──────────────────────────────────────────────────────

    def generate_recommendations(
        self,
        analysis_report: DatasetAnalysisReport,
    ) -> RecommendationResponse:
        """Generate a validated preprocessing plan from *analysis_report*.

        Pipeline
        --------
        1. **Build prompt** — serialise the analysis report into an
           LLM-ready prompt.
        2. **Call Groq** — send the prompt to the LLM and receive raw text.
        3. **Parse & validate** — decode the JSON and validate it against
           the :class:`RecommendationResponse` schema.

        Parameters
        ----------
        analysis_report:
            The dataset analysis produced by the analysis service.

        Returns
        -------
        RecommendationResponse
            A fully-validated recommendation object.

        Raises
        ------
        RecommendationServiceError
            Wraps any downstream exception from the prompt builder,
            Groq client, or plan parser, preserving the original context.
        """
        try:
            # Step 1: Build prompt
            logger.info("Step 1/3 — Building prompt from analysis report.")
            prompt = self._prompt_builder.build_prompt(analysis_report)
            logger.debug("Prompt built (%d characters).", len(prompt))

            # Step 2: Send to Groq
            logger.info("Step 2/3 — Sending prompt to Groq LLM.")
            raw_response = self._groq_client.generate_recommendations(prompt)
            logger.debug("Raw response received (%d characters).", len(raw_response))

            # Step 3: Parse and validate
            logger.info("Step 3/3 — Parsing and validating LLM response.")
            recommendation = self._plan_parser.parse(raw_response)
            logger.info("Recommendation generated successfully.")
            return recommendation

        except (GroqClientError, PlanParserError) as exc:
            logger.error("Recommendation pipeline failed: %s", exc)
            raise RecommendationServiceError(
                f"Failed to generate recommendations: {exc}"
            ) from exc

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in recommendation pipeline.")
            raise RecommendationServiceError(
                f"Unexpected error during recommendation generation: {exc}"
            ) from exc
