"""Recommendation mapper translating textual recommendations into OptimizationAction schemas."""

from __future__ import annotations

from typing import List
from backend.app.ai_model_critic.schemas import ModelCritique
from backend.app.ai_model_optimizer.schemas import OptimizationAction, OptimizationActionType


class AIRecommendationMapper:
    """Translates text recommendations from ModelCritique objects into structured OptimizationActions."""

    def map_recommendations(self, critique: ModelCritique) -> List[OptimizationAction]:
        """Convert list of recommendations into mapped deterministic actions.

        Args:
            critique: Input ModelCritique with recommendations list.

        Returns:
            List of structured OptimizationAction objects.
        """
        actions = []
        for idx, rec in enumerate(critique.recommendations):
            rec_clean = rec.strip()
            rec_lower = rec_clean.lower()
            action_id = f"action_{critique.critique_id}_{idx}"
            confidence = critique.confidence if critique.confidence is not None else 0.8

            if "increase cross validation folds" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.CHANGE_CV_FOLDS,
                        reason=rec_clean,
                        confidence=confidence,
                        parameters={"folds": 5},
                    )
                )
            elif "use robustscaler" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.REPLACE_PREPROCESSING,
                        target="scale",
                        replacement="robust_scale",
                        reason=rec_clean,
                        confidence=confidence,
                    )
                )
            elif "use standardscaler" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.REPLACE_PREPROCESSING,
                        target="scale",
                        replacement="standard_scale",
                        reason=rec_clean,
                        confidence=confidence,
                    )
                )
            elif "enable feature selection" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.CHANGE_FEATURE_SELECTION,
                        replacement="mutual_information",
                        reason=rec_clean,
                        confidence=confidence,
                    )
                )
            elif "use random search" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.CHANGE_SEARCH_STRATEGY,
                        replacement="random",
                        reason=rec_clean,
                        confidence=confidence,
                    )
                )
            elif "use grid search" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.CHANGE_SEARCH_STRATEGY,
                        replacement="grid",
                        reason=rec_clean,
                        confidence=confidence,
                    )
                )
            elif "add gradient boosting" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.ADD_MODEL,
                        replacement="gradient_boosting",
                        reason=rec_clean,
                        confidence=confidence,
                    )
                )
            elif "remove random forest" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.REMOVE_MODEL,
                        target="random_forest",
                        reason=rec_clean,
                        confidence=confidence,
                    )
                )
            elif "tree models overfit" in rec_lower:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.ADD_WARNING,
                        reason=rec_clean,
                        confidence=confidence,
                    )
                )
            else:
                actions.append(
                    OptimizationAction(
                        action_id=action_id,
                        action_type=OptimizationActionType.NO_ACTION,
                        reason=f"Ignored unknown recommendation: {rec_clean}",
                        confidence=1.0,
                    )
                )

        return actions
