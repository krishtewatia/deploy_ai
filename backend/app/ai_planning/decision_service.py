"""AI Decision Service.

Orchestrates the AI-assisted ML planning pipeline. Converts input artifacts
into context, builds prompts, runs the AI provider, parses the response, and
merges proposals with the baseline plan to produce a final validated plan.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from backend.app.ai_planning.context_builder import AIPlanningContextBuilder
from backend.app.ai_planning.prompt_builder import AIPlanningPromptBuilder
from backend.app.ai_planning.proposal_merger import ProposalMerger, ProposalMergerError
from backend.app.ai_planning.providers.base import AIProvider, AIProviderError
from backend.app.ai_planning.response_parser import AIResponseParser, AIResponseParseError
from backend.app.ai_planning.schemas import AIAssistedPlanningResult, AIDecisionProposal
from backend.app.compute_capabilities.schemas import ComputeCapabilities
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import MLPlan
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import ProblemDefinition

logger = logging.getLogger(__name__)


class AIDecisionServiceError(Exception):
    """Raised when any step of the AI decision orchestration fails."""


class AIDecisionService:
    """Orchestrates the AI-assisted ML planning workflow.

    Uses dependency injection for the AI text provider.
    Enforces artifact type validation and error propagation.
    """

    def __init__(
        self,
        *,
        provider: AIProvider,
        context_builder: Optional[AIPlanningContextBuilder] = None,
        prompt_builder: Optional[AIPlanningPromptBuilder] = None,
        parser: Optional[AIResponseParser] = None,
        merger: Optional[ProposalMerger] = None,
    ) -> None:
        """Initialize the decision service with injected dependencies.

        Args:
            provider: Concrete implementation of AIProvider.
            context_builder: Builder to construct planning context.
            prompt_builder: Builder to construct prompts.
            parser: Parser to parse raw AI responses.
            merger: Merger to merge proposals into the baseline.
        """
        if not isinstance(provider, AIProvider):
            raise TypeError("provider must be an instance of AIProvider")

        self._provider = provider
        self._context_builder = context_builder or AIPlanningContextBuilder()
        self._prompt_builder = prompt_builder or AIPlanningPromptBuilder()
        self._parser = parser or AIResponseParser()
        self._merger = merger or ProposalMerger()

    def run_planning(
        self,
        *,
        dataset_context: DatasetContext,
        user_request: UserMLRequest,
        problem_definition: ProblemDefinition,
        compute_capabilities: ComputeCapabilities,
        baseline_plan: MLPlan,
    ) -> AIAssistedPlanningResult:
        """Run the end-to-end AI planning orchestration.

        Args:
            dataset_context: Target dataset metadata.
            user_request: User intent and goals.
            problem_definition: Resolved ML problem statement.
            compute_capabilities: System compute capabilities constraint facts.
            baseline_plan: Deterministic baseline MLPlan.

        Returns:
            AIAssistedPlanningResult containing baseline ID, proposal,
            final plan, and whether changes were applied.

        Raises:
            AIDecisionServiceError: If any pipeline step fails.
        """
        # 1. Type validation
        if not isinstance(dataset_context, DatasetContext):
            raise TypeError("dataset_context must be a DatasetContext")
        if not isinstance(user_request, UserMLRequest):
            raise TypeError("user_request must be a UserMLRequest")
        if not isinstance(problem_definition, ProblemDefinition):
            raise TypeError("problem_definition must be a ProblemDefinition")
        if not isinstance(compute_capabilities, ComputeCapabilities):
            raise TypeError("compute_capabilities must be a ComputeCapabilities")
        if not isinstance(baseline_plan, MLPlan):
            raise TypeError("baseline_plan must be an MLPlan")

        # 2. Build Context
        try:
            context = self._context_builder.build(
                dataset_context=dataset_context,
                user_request=user_request,
                problem_definition=problem_definition,
                compute_capabilities=compute_capabilities,
                baseline_plan=baseline_plan,
            )
        except Exception as exc:
            raise AIDecisionServiceError(
                f"Failed to build AI planning context: {exc}"
            ) from exc

        # 3. Build Prompts
        try:
            prompts = self._prompt_builder.build(planning_context=context)
            system_prompt = prompts["system_prompt"]
            user_prompt = prompts["user_prompt"]
        except Exception as exc:
            raise AIDecisionServiceError(
                f"Failed to build prompts: {exc}"
            ) from exc

        # 4. Generate AI Completion
        try:
            raw_response = self._provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except AIProviderError as exc:
            raise AIDecisionServiceError(
                f"AI Provider failed to generate response: {exc}"
            ) from exc
        except Exception as exc:
            raise AIDecisionServiceError(
                f"Unexpected provider generation error: {exc}"
            ) from exc

        # 5. Parse Response
        try:
            proposal = self._parser.parse(raw_response)
            # Ensure strict artifact linkage matching with baseline plan
            proposal.baseline_plan_id = baseline_plan.plan_id
            proposal.dataset_id = baseline_plan.dataset_id
            proposal.request_id = baseline_plan.request_id
            proposal.problem_definition_id = baseline_plan.problem_definition_id
            proposal.compute_capability_id = baseline_plan.compute_capability_id
        except AIResponseParseError as exc:
            logger.warning(
                "AI planning response was unusable; continuing with the "
                "deterministic baseline plan: %s",
                exc,
            )
            return self._baseline_fallback(
                baseline_plan=baseline_plan,
                reason="AI response was unusable; deterministic baseline plan retained.",
            )
        except Exception as exc:
            raise AIDecisionServiceError(
                f"Unexpected response parsing error: {exc}"
            ) from exc

        # 6. Merge & Validate
        try:
            final_plan = self._merger.merge(
                baseline_plan=baseline_plan,
                ai_proposal=proposal,
                dataset_context=dataset_context,
                problem_definition=problem_definition,
                compute_capabilities=compute_capabilities,
            )
        except ProposalMergerError as exc:
            logger.warning(
                "AI planning proposal was unsafe or invalid; continuing with "
                "the deterministic baseline plan: %s",
                exc,
            )
            return self._baseline_fallback(
                baseline_plan=baseline_plan,
                reason="AI proposal was invalid; deterministic baseline plan retained.",
            )
        except Exception as exc:
            raise AIDecisionServiceError(
                f"Unexpected plan merge error: {exc}"
            ) from exc

        # 7. Check if changes were applied
        applied = self._has_changes(proposal)

        return AIAssistedPlanningResult(
            baseline_plan_id=baseline_plan.plan_id,
            proposal=proposal,
            final_plan=final_plan,
            applied=applied,
        )

    @staticmethod
    def _baseline_fallback(
        *,
        baseline_plan: MLPlan,
        reason: str,
    ) -> AIAssistedPlanningResult:
        """Return a valid no-change result when optional AI advice is unusable."""
        proposal = AIDecisionProposal(
            proposal_set_id=f"fallback_{uuid.uuid4().hex}",
            baseline_plan_id=baseline_plan.plan_id,
            dataset_id=baseline_plan.dataset_id,
            request_id=baseline_plan.request_id,
            problem_definition_id=baseline_plan.problem_definition_id,
            compute_capability_id=baseline_plan.compute_capability_id,
            summary=reason,
        )
        return AIAssistedPlanningResult(
            baseline_plan_id=baseline_plan.plan_id,
            proposal=proposal,
            final_plan=baseline_plan,
            applied=False,
        )

    def _has_changes(self, proposal: AIDecisionProposal) -> bool:
        """Determine if non-trivial changes were proposed."""
        return bool(
            proposal.preprocessing_proposals
            or proposal.feature_engineering_proposals
            or proposal.model_candidate_proposals
            or proposal.feature_selection_proposal
            or proposal.evaluation_proposal
        )
