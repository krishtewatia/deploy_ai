"""Deterministic plan optimizer applying OptimizationActions to MLPlans."""

from __future__ import annotations

import copy
import uuid
from typing import List
from backend.app.ml_plan.schemas import (
    MLPlan,
    ModelCandidate,
    ModelFamily,
    SearchStrategy,
    PreprocessingStep,
    PreprocessingOperation,
    FeatureSelectionMethod,
    MLPlanWarning,
)
from backend.app.ai_model_optimizer.schemas import OptimizationAction, OptimizationActionType


class AIPlanOptimizer:
    """Applies a list of OptimizationActions to an MLPlan deterministically."""

    def optimize(self, baseline_plan: MLPlan, actions: List[OptimizationAction]) -> MLPlan:
        """Apply actions to baseline_plan, returning a new optimized MLPlan instance.

        Args:
            baseline_plan: The original MLPlan.
            actions: Mapped OptimizationActions to apply.

        Returns:
            A new optimized MLPlan instance.

        Raises:
            ValueError: On validation violations, invalid targets, or empty candidate list.
        """
        # Ensure deep copy to preserve original plan state
        optimized_plan = copy.deepcopy(baseline_plan)

        # 1. Validation for duplicate action IDs
        action_ids = set()
        for action in actions:
            if action.action_id in action_ids:
                raise ValueError(f"Duplicate action ID detected: {action.action_id}")
            action_ids.add(action.action_id)

        # Apply actions deterministically
        for action in actions:
            # Type safety check
            try:
                action_type = OptimizationActionType(action.action_type)
            except ValueError as e:
                raise ValueError(f"Unknown action type: {action.action_type}") from e

            if action_type == OptimizationActionType.CHANGE_CV_FOLDS:
                # Modify evaluation folds
                folds = 5
                if action.parameters and "folds" in action.parameters:
                    folds = int(action.parameters["folds"])
                else:
                    folds = optimized_plan.evaluation_plan.cross_validation_folds + 1

                if folds < 2:
                    raise ValueError(f"Cross-validation folds must be >= 2, got {folds}")
                optimized_plan.evaluation_plan.cross_validation_folds = folds

            elif action_type == OptimizationActionType.REPLACE_PREPROCESSING:
                # Replace scaling preprocessing step
                if action.target != "scale":
                    raise ValueError(f"Invalid preprocessing target: {action.target}. Supported target is 'scale'")

                # Find any existing scaling step
                scaling_step_found = False
                for step in optimized_plan.preprocessing_steps:
                    if step.operation in (
                        PreprocessingOperation.STANDARD_SCALE,
                        PreprocessingOperation.ROBUST_SCALE,
                        PreprocessingOperation.MINMAX_SCALE,
                    ):
                        step.operation = PreprocessingOperation(action.replacement)
                        scaling_step_found = True

                if not scaling_step_found:
                    raise ValueError("Invalid replacement target: no existing scaling step found in MLPlan")

            elif action_type == OptimizationActionType.CHANGE_FEATURE_SELECTION:
                # Replace feature selection method
                try:
                    method = FeatureSelectionMethod(action.replacement)
                except ValueError as e:
                    raise ValueError(f"Invalid feature selection method: {action.replacement}") from e
                optimized_plan.feature_selection.method = method

            elif action_type == OptimizationActionType.CHANGE_SEARCH_STRATEGY:
                # Replace search strategy on all candidates
                try:
                    strategy = SearchStrategy(action.replacement)
                except ValueError as e:
                    raise ValueError(f"Invalid search strategy: {action.replacement}") from e
                for candidate in optimized_plan.model_candidates:
                    candidate.search_strategy = strategy

            elif action_type == OptimizationActionType.CHANGE_SEARCH_SPACE:
                # Update search spaces of the candidates if specified
                if action.parameters and "search_space" in action.parameters:
                    space = action.parameters["search_space"]
                    for candidate in optimized_plan.model_candidates:
                        candidate.search_space = space

            elif action_type == OptimizationActionType.ADD_MODEL:
                # Add model candidate
                try:
                    family = ModelFamily(action.replacement)
                except ValueError as e:
                    raise ValueError(f"Invalid model family to add: {action.replacement}") from e
                
                # Check if it already exists to avoid redundant duplication
                exists = any(c.model_family == family for c in optimized_plan.model_candidates)
                if not exists:
                    new_cand = ModelCandidate(
                        candidate_id=f"opt_model_{uuid.uuid4().hex[:6]}",
                        model_family=family,
                        search_strategy=SearchStrategy.NONE,
                        reason=action.reason,
                    )
                    optimized_plan.model_candidates.append(new_cand)

            elif action_type == OptimizationActionType.REMOVE_MODEL:
                # Remove model candidate by family
                try:
                    family = ModelFamily(action.target)
                except ValueError as e:
                    raise ValueError(f"Invalid model family to remove: {action.target}") from e

                original_len = len(optimized_plan.model_candidates)
                optimized_plan.model_candidates = [
                    c for c in optimized_plan.model_candidates if c.model_family != family
                ]

                if len(optimized_plan.model_candidates) == original_len:
                    raise ValueError(f"Invalid replacement target: Model family '{action.target}' not found in candidate list")

                if not optimized_plan.model_candidates:
                    raise ValueError("Invalid optimization: Removal of model candidate resulted in an empty candidate list")

            elif action_type == OptimizationActionType.ADD_WARNING:
                # Append warning to MLPlan
                optimized_plan.warnings.append(
                    MLPlanWarning(
                        code="OPT_CRITIC_WARNING",
                        message=action.reason,
                    )
                )

            elif action_type == OptimizationActionType.NO_ACTION:
                # Do nothing
                pass

        # 2. Identity preservation asserts
        assert optimized_plan.dataset_id == baseline_plan.dataset_id, "dataset_id mismatch"
        assert optimized_plan.request_id == baseline_plan.request_id, "request_id mismatch"
        assert optimized_plan.problem_definition_id == baseline_plan.problem_definition_id, "problem_definition_id mismatch"
        assert optimized_plan.compute_capability_id == baseline_plan.compute_capability_id, "compute_capability_id mismatch"
        assert optimized_plan.target_column == baseline_plan.target_column, "target_column mismatch"
        assert optimized_plan.feature_columns == baseline_plan.feature_columns, "feature_columns mismatch"

        return optimized_plan
