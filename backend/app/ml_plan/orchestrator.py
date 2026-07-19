"""High-level ML Planning Orchestrator.

Coordinates ProblemResolver, BaselineMLPlanner, optional AIDecisionService,
and final MLPlanValidator to generate, assist, and validate execution plans.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from backend.app.ai_planning.providers.base import AIProvider
from backend.app.compute_capabilities.schemas import ComputeCapabilities
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import MLPlan
from backend.app.ml_plan.validator import MLPlanValidator, MLPlanValidationResult
from backend.app.ml_plan.baseline_planner import BaselineMLPlanner
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.resolver import ProblemResolver
from backend.app.problem_definition.schemas import ProblemDefinition


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlanningMode(str, Enum):
    """Execution planning mode targeting deterministic or AI-assisted results."""

    DETERMINISTIC = "deterministic"
    AI_ASSISTED = "ai_assisted"


# ---------------------------------------------------------------------------
# MLPlanningResult Schema
# ---------------------------------------------------------------------------


class MLPlanningResult(BaseModel):
    """The structured result returned by the high-level orchestrator."""

    planning_id: str = Field(
        ...,
        description="Unique identifier for this orchestration planning run.",
    )
    mode: PlanningMode = Field(
        ...,
        description="Target planning mode utilized.",
    )
    problem_definition: ProblemDefinition = Field(
        ...,
        description="The resolved problem definition schema.",
    )
    baseline_plan: MLPlan = Field(
        ...,
        description="The deterministic baseline plan created first.",
    )
    final_plan: MLPlan = Field(
        ...,
        description="The final output plan (baseline plan or merged AI plan).",
    )
    ai_assistance_used: bool = Field(
        ...,
        description="True if the AI planning pipeline was executed.",
    )
    ai_changes_applied: bool = Field(
        ...,
        description="True if non-trivial AI improvements were successfully merged.",
    )
    ai_result: Optional[Any] = Field(
        default=None,
        description="Details of the AI decision proposal, if used.",
    )
    validation_result: MLPlanValidationResult = Field(
        ...,
        description="Validation outcome of the final plan.",
    )

    @field_validator("planning_id", mode="before")
    @classmethod
    def _validate_planning_id(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("planning_id must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("planning_id cannot be empty or whitespace-only")
        return stripped


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class MLPlanningOrchestratorError(Exception):
    """Raised when the high-level planning orchestration workflow fails."""


# ---------------------------------------------------------------------------
# Orchestrator Class
# ---------------------------------------------------------------------------


class MLPlanningOrchestrator:
    """Orchestrates the entire machine learning planning workflow."""

    def __init__(
        self,
        *,
        problem_resolver: Optional[ProblemResolver] = None,
        baseline_planner: Optional[BaselineMLPlanner] = None,
        validator: Optional[MLPlanValidator] = None,
    ) -> None:
        """Initialize MLPlanningOrchestrator with optional dependency injection.

        Args:
            problem_resolver: Optional custom ProblemResolver.
            baseline_planner: Optional custom BaselineMLPlanner.
            validator: Optional custom MLPlanValidator.
        """
        if problem_resolver is not None and not isinstance(problem_resolver, ProblemResolver):
            raise TypeError("problem_resolver must be a ProblemResolver")
        if baseline_planner is not None and not isinstance(baseline_planner, BaselineMLPlanner):
            raise TypeError("baseline_planner must be a BaselineMLPlanner")
        if validator is not None and not isinstance(validator, MLPlanValidator):
            raise TypeError("validator must be a MLPlanValidator")

        self._problem_resolver = problem_resolver or ProblemResolver()
        self._baseline_planner = baseline_planner or BaselineMLPlanner()
        self._validator = validator or MLPlanValidator()

    def create_plan(
        self,
        *,
        dataset_context: DatasetContext,
        user_request: UserMLRequest,
        compute_capabilities: ComputeCapabilities,
        mode: PlanningMode = PlanningMode.DETERMINISTIC,
        ai_provider: Optional[AIProvider] = None,
    ) -> MLPlanningResult:
        """Resolve problem definition and build a validated final plan.

        Args:
            dataset_context: Target dataset metadata profile.
            user_request: User requests/intent preferences.
            compute_capabilities: System compute capabilities.
            mode: Target planning mode (deterministic or ai_assisted).
            ai_provider: Required if mode is AI_ASSISTED.

        Returns:
            The structured planning execution result.

        Raises:
            MLPlanningOrchestratorError: If problem resolution, baseline planning,
                AI assistance, or final validation fails.
        """
        # 1. Type validation
        if not isinstance(dataset_context, DatasetContext):
            raise TypeError("dataset_context must be a DatasetContext")
        if not isinstance(user_request, UserMLRequest):
            raise TypeError("user_request must be a UserMLRequest")
        if not isinstance(compute_capabilities, ComputeCapabilities):
            raise TypeError("compute_capabilities must be a ComputeCapabilities")
        if not isinstance(mode, PlanningMode):
            raise TypeError("mode must be a PlanningMode")
        if ai_provider is not None and not isinstance(ai_provider, AIProvider):
            raise TypeError("ai_provider must be an AIProvider")

        # 2. Require AI provider if AI-assisted mode is requested
        if mode == PlanningMode.AI_ASSISTED and ai_provider is None:
            raise MLPlanningOrchestratorError(
                "AI-assisted planning failed: AI provider is required for AI-assisted planning mode."
            )

        # 3. Resolve problem
        try:
            problem_definition = self._problem_resolver.resolve(
                dataset_context=dataset_context,
                user_request=user_request,
            )
        except Exception as exc:
            raise MLPlanningOrchestratorError(
                f"Problem resolution failed: {exc}"
            ) from exc

        # 4. Generate deterministic baseline plan
        try:
            baseline_plan = self._baseline_planner.create_plan(
                dataset_context=dataset_context,
                user_request=user_request,
                problem_definition=problem_definition,
                compute_capabilities=compute_capabilities,
            )
        except Exception as exc:
            raise MLPlanningOrchestratorError(
                f"Baseline planning failed: {exc}"
            ) from exc

        # 5. Handle Optional AI Planning Flow
        ai_result: Optional[AIAssistedPlanningResult] = None
        final_plan: MLPlan = baseline_plan

        if mode == PlanningMode.AI_ASSISTED:
            assert ai_provider is not None
            try:
                from backend.app.ai_planning.decision_service import AIDecisionService
                decision_service = AIDecisionService(provider=ai_provider)
                ai_result = decision_service.run_planning(
                    dataset_context=dataset_context,
                    user_request=user_request,
                    problem_definition=problem_definition,
                    compute_capabilities=compute_capabilities,
                    baseline_plan=baseline_plan,
                )
                final_plan = ai_result.final_plan
            except Exception as exc:
                logger.warning(
                    "AI-assisted planning proposal was unsafe or invalid; continuing with "
                    "the deterministic baseline plan: %s",
                    exc,
                )
                final_plan = baseline_plan

        # 6. Final safety validation gate
        try:
            validation_result = self._validator.validate(
                plan=final_plan,
                dataset_context=dataset_context,
                problem_definition=problem_definition,
                compute_capabilities=compute_capabilities,
            )
        except Exception as exc:
            raise MLPlanningOrchestratorError(
                f"Final MLPlan validation failed: Unexpected error during validation: {exc}"
            ) from exc

        if not validation_result.is_valid:
            error_messages = "; ".join(
                f"[{e.code}] {e.message}" for e in validation_result.errors
            )
            raise MLPlanningOrchestratorError(
                f"Final MLPlan validation failed: {error_messages}"
            )

        # 7. Construct result
        planning_run_id = f"planning_{uuid.uuid4().hex}"
        ai_used = (mode == PlanningMode.AI_ASSISTED)
        ai_applied = (ai_result.applied if ai_result is not None else False)

        return MLPlanningResult(
            planning_id=planning_run_id,
            mode=mode,
            problem_definition=problem_definition,
            baseline_plan=baseline_plan,
            final_plan=final_plan,
            ai_assistance_used=ai_used,
            ai_changes_applied=ai_applied,
            ai_result=ai_result,
            validation_result=validation_result,
        )
