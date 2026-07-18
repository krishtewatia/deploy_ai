"""Model optimizer service orchestrating recommendation parsing and plan adjustment."""

from __future__ import annotations

import uuid
from backend.app.ai_model_critic.schemas import ModelCritique
from backend.app.ml_plan.schemas import MLPlan
from backend.app.ai_model_optimizer.schemas import OptimizationResult
from backend.app.ai_model_optimizer.recommendation_mapper import AIRecommendationMapper
from backend.app.ai_model_optimizer.plan_optimizer import AIPlanOptimizer


class AIModelOptimizerError(Exception):
    """Raised when plan optimization fails."""

    pass


class AIModelOptimizer:
    """Orchestrates the conversion of AI reviews into deterministic MLPlan updates."""

    def __init__(self) -> None:
        self.mapper = AIRecommendationMapper()
        self.optimizer = AIPlanOptimizer()

    def optimize(self, critique: ModelCritique, baseline_plan: MLPlan) -> OptimizationResult:
        """Translate critique recommendations into optimization actions and optimize baseline_plan.

        Args:
            critique: Output from AIModelCritic.
            baseline_plan: Target MLPlan baseline.

        Returns:
            OptimizationResult containing optimized MLPlan and applied actions.

        Raises:
            AIModelOptimizerError: On validation violations, identity mismatches,
                                  or plan optimization failures.
        """
        # 1. Validation checks
        if critique is None:
            raise AIModelOptimizerError("critique cannot be None")
        if baseline_plan is None:
            raise AIModelOptimizerError("baseline_plan cannot be None")

        if not isinstance(critique, ModelCritique):
            raise AIModelOptimizerError("critique must be a ModelCritique instance")
        if not isinstance(baseline_plan, MLPlan):
            raise AIModelOptimizerError("baseline_plan must be an MLPlan instance")

        # Identity validation check:
        # The critique report_id should bind to the baseline plan.
        # Since report_id is formatted as 'report_{plan_id}_xxx' or contain the plan_id.
        if baseline_plan.plan_id not in critique.report_id:
            raise AIModelOptimizerError(
                f"Identity mismatch: critique report_id '{critique.report_id}' "
                f"does not match baseline_plan plan_id '{baseline_plan.plan_id}'"
            )

        # 2. Run Recommendation Mapper
        try:
            actions = self.mapper.map_recommendations(critique)
        except Exception as e:
            raise AIModelOptimizerError(f"Recommendation mapping failed: {e}") from e

        # 3. Run Plan Optimizer
        try:
            optimized_plan = self.optimizer.optimize(baseline_plan, actions)
        except Exception as e:
            raise AIModelOptimizerError(f"Plan optimization failed: {e}") from e

        # 4. Construct OptimizationResult summary description
        summary_parts = []
        for action in actions:
            if action.action_type != "NO_ACTION":
                summary_parts.append(f"- {action.action_type}: {action.reason}")
        if not summary_parts:
            summary = "No optimization actions applied based on recommendations."
        else:
            summary = "Optimized MLPlan successfully:\n" + "\n".join(summary_parts)

        return OptimizationResult(
            optimization_id=f"opt_res_{uuid.uuid4().hex[:8]}",
            baseline_plan_id=baseline_plan.plan_id,
            optimized_plan=optimized_plan,
            actions=actions,
            summary=summary,
        )
