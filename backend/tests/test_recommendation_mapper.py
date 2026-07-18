"""Unit tests for AIRecommendationMapper."""

from __future__ import annotations

from backend.app.ai_model_critic.schemas import CritiqueGrade, ModelCritique
from backend.app.ai_model_optimizer.schemas import OptimizationActionType
from backend.app.ai_model_optimizer.recommendation_mapper import AIRecommendationMapper


def _make_critique(recommendations: list[str]) -> ModelCritique:
    return ModelCritique(
        critique_id="crit_01",
        report_id="report_plan_01_abc",
        overall_grade=CritiqueGrade.B_PLUS,
        production_ready=False,
        confidence=0.85,
        strengths=["Baseline accuracy"],
        weaknesses=["Slight overfitting"],
        risks=["Drift"],
        recommendations=recommendations,
        warnings=[],
        summary="Baseline review.",
    )


class TestAIRecommendationMapper:
    """Tests covering recommendation text mapping behavior to OptimizationActions."""

    def test_map_all_known_recommendations(self):
        """Verify all supported recommendation patterns map to the correct action types."""
        recs = [
            "Increase cross validation folds to 5",
            "Use RobustScaler to scale input columns",
            "Use StandardScaler in preprocessing pipeline",
            "Enable feature selection using mutual information",
            "Use Random Search for tuning",
            "Use Grid Search for hyperparameter tuning",
            "Add Gradient Boosting classifier to model candidates",
            "Remove Random Forest candidate from selection pool",
            "Tree models overfit easily, watch out",
        ]
        critique = _make_critique(recs)
        mapper = AIRecommendationMapper()
        actions = mapper.map_recommendations(critique)

        assert len(actions) == 9
        assert actions[0].action_type == OptimizationActionType.CHANGE_CV_FOLDS
        assert actions[0].parameters == {"folds": 5}
        
        assert actions[1].action_type == OptimizationActionType.REPLACE_PREPROCESSING
        assert actions[1].replacement == "robust_scale"

        assert actions[2].action_type == OptimizationActionType.REPLACE_PREPROCESSING
        assert actions[2].replacement == "standard_scale"

        assert actions[3].action_type == OptimizationActionType.CHANGE_FEATURE_SELECTION
        assert actions[3].replacement == "mutual_information"

        assert actions[4].action_type == OptimizationActionType.CHANGE_SEARCH_STRATEGY
        assert actions[4].replacement == "random"

        assert actions[5].action_type == OptimizationActionType.CHANGE_SEARCH_STRATEGY
        assert actions[5].replacement == "grid"

        assert actions[6].action_type == OptimizationActionType.ADD_MODEL
        assert actions[6].replacement == "gradient_boosting"

        assert actions[7].action_type == OptimizationActionType.REMOVE_MODEL
        assert actions[7].target == "random_forest"

        assert actions[8].action_type == OptimizationActionType.ADD_WARNING

    def test_map_unknown_recommendation(self):
        """Verify unknown recommendations map to NO_ACTION action type."""
        critique = _make_critique(["Tune learning rate", "Unknown optimization tip"])
        mapper = AIRecommendationMapper()
        actions = mapper.map_recommendations(critique)

        assert len(actions) == 2
        assert actions[0].action_type == OptimizationActionType.NO_ACTION
        assert actions[1].action_type == OptimizationActionType.NO_ACTION
