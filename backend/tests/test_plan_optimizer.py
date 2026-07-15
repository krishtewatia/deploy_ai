"""Unit tests for AIPlanOptimizer."""

from __future__ import annotations

import pytest

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.ml_plan.schemas import (
    MLPlan,
    MLPlanStatus,
    ProblemType,
    SplitStrategy,
    DatasetSplitPlan,
    FeatureSelectionPlan,
    FeatureSelectionMethod,
    EvaluationPlan,
    ExecutionConstraints,
    ModelCandidate,
    ModelFamily,
    SearchStrategy,
    PreprocessingStep,
    PreprocessingOperation,
)
from backend.app.ai_model_optimizer.schemas import OptimizationAction, OptimizationActionType
from backend.app.ai_model_optimizer.plan_optimizer import AIPlanOptimizer


# ── Helper Builders ───────────────────────────────────────────────────


def _make_sample_plan(
    plan_id: str = "plan_01",
    preprocessing: list[PreprocessingStep] | None = None,
    candidates: list[ModelCandidate] | None = None,
) -> MLPlan:
    if preprocessing is None:
        preprocessing = [
            PreprocessingStep(
                step_id="step_scale",
                operation=PreprocessingOperation.STANDARD_SCALE,
                columns=["feat_a"],
                reason="Scale baseline features",
            )
        ]
    if candidates is None:
        candidates = [
            ModelCandidate(
                candidate_id="model_001",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline",
            ),
            ModelCandidate(
                candidate_id="model_002",
                model_family=ModelFamily.RANDOM_FOREST,
                search_strategy=SearchStrategy.NONE,
                reason="Provides forest model",
            ),
        ]

    return MLPlan(
        plan_id=plan_id,
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        problem_type=ProblemType.CLASSIFICATION,
        target_column="species",
        feature_columns=["feat_a"],
        preprocessing_steps=preprocessing,
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["feat_a"],
            max_features=None,
            reason="No selection",
        ),
        split_plan=DatasetSplitPlan(
            strategy=SplitStrategy.RANDOM,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        ),
        model_candidates=candidates,
        evaluation_plan=EvaluationPlan(
            primary_metric="f1",
            secondary_metrics=[],
            cross_validation_folds=2,
        ),
        execution_constraints=ExecutionConstraints(
            parallel_workers=1,
            use_gpu_acceleration=False,
            accelerator_type=AcceleratorType.NONE,
            compute_tier=ComputeTier.STANDARD,
        ),
        status=MLPlanStatus.READY,
    )


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestAIPlanOptimizer:
    """Tests covering deterministic MLPlan modifications based on OptimizationAction lists."""

    def test_change_cv_folds_action(self):
        """Verify modifying CV folds works."""
        plan = _make_sample_plan()
        action_param = OptimizationAction(
            action_id="act_cv",
            action_type=OptimizationActionType.CHANGE_CV_FOLDS,
            parameters={"folds": 5},
            reason="Increase folds",
            confidence=0.9,
        )
        action_incr = OptimizationAction(
            action_id="act_cv_incr",
            action_type=OptimizationActionType.CHANGE_CV_FOLDS,
            reason="Increment folds",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        
        # Test explicit folds setting
        opt_plan = opt.optimize(plan, [action_param])
        assert opt_plan.evaluation_plan.cross_validation_folds == 5

        # Test increment folds setting
        opt_plan_incr = opt.optimize(plan, [action_incr])
        assert opt_plan_incr.evaluation_plan.cross_validation_folds == 3

    def test_change_cv_folds_validation_fails(self):
        """Verify CV folds < 2 raises ValueError."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_cv",
            action_type=OptimizationActionType.CHANGE_CV_FOLDS,
            parameters={"folds": 1},
            reason="Invalid folds",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError, match="folds must be >= 2"):
            opt.optimize(plan, [action])

    def test_replace_preprocessing_action(self):
        """Verify replacing standard scaler with robust scaler works."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_scale",
            action_type=OptimizationActionType.REPLACE_PREPROCESSING,
            target="scale",
            replacement="robust_scale",
            reason="Robust to outliers",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt_plan = opt.optimize(plan, [action])

        assert opt_plan.preprocessing_steps[0].operation == PreprocessingOperation.ROBUST_SCALE

    def test_replace_preprocessing_invalid_target(self):
        """Verify invalid replacement target raises ValueError."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_scale",
            action_type=OptimizationActionType.REPLACE_PREPROCESSING,
            target="invalid_target",
            replacement="robust_scale",
            reason="Robust to outliers",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError, match="Invalid preprocessing target"):
            opt.optimize(plan, [action])

    def test_replace_preprocessing_missing_scaler_in_plan(self):
        """Verify trying to replace scaler when none exists in plan raises ValueError."""
        plan = _make_sample_plan(preprocessing=[])  # Empty preprocessing
        action = OptimizationAction(
            action_id="act_scale",
            action_type=OptimizationActionType.REPLACE_PREPROCESSING,
            target="scale",
            replacement="robust_scale",
            reason="Robust to outliers",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError, match="no existing scaling step found in MLPlan"):
            opt.optimize(plan, [action])

    def test_change_feature_selection(self):
        """Verify changing feature selection method works."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_feat",
            action_type=OptimizationActionType.CHANGE_FEATURE_SELECTION,
            replacement="mutual_information",
            reason="Filter features",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt_plan = opt.optimize(plan, [action])
        assert opt_plan.feature_selection.method == FeatureSelectionMethod.MUTUAL_INFORMATION

    def test_change_feature_selection_invalid(self):
        """Verify invalid feature selection method raises ValueError."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_feat",
            action_type=OptimizationActionType.CHANGE_FEATURE_SELECTION,
            replacement="invalid_method",
            reason="Filter features",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError):
            opt.optimize(plan, [action])

    def test_change_search_strategy(self):
        """Verify changing search strategy updates all candidate strategies."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_search",
            action_type=OptimizationActionType.CHANGE_SEARCH_STRATEGY,
            replacement="grid",
            reason="Use grid search",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt_plan = opt.optimize(plan, [action])
        for c in opt_plan.model_candidates:
            assert c.search_strategy == SearchStrategy.GRID

    def test_change_search_strategy_invalid(self):
        """Verify invalid search strategy raises ValueError."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_search",
            action_type=OptimizationActionType.CHANGE_SEARCH_STRATEGY,
            replacement="invalid_strategy",
            reason="Use strategy",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError):
            opt.optimize(plan, [action])

    def test_change_search_space(self):
        """Verify changing search space updates candidates search space parameters."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_space",
            action_type=OptimizationActionType.CHANGE_SEARCH_SPACE,
            parameters={"search_space": {"C": [0.1, 1.0]}},
            reason="Tune C",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt_plan = opt.optimize(plan, [action])
        for c in opt_plan.model_candidates:
            assert c.search_space == {"C": [0.1, 1.0]}

    def test_add_model_candidate(self):
        """Verify adding a model candidate works."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_add",
            action_type=OptimizationActionType.ADD_MODEL,
            replacement="gradient_boosting",
            reason="Better boosting performance",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt_plan = opt.optimize(plan, [action])
        assert len(opt_plan.model_candidates) == 3
        assert any(c.model_family == ModelFamily.GRADIENT_BOOSTING for c in opt_plan.model_candidates)

    def test_add_model_candidate_already_exists(self):
        """Verify adding a candidate that is already present is skipped/idempotent."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_add",
            action_type=OptimizationActionType.ADD_MODEL,
            replacement="logistic_regression",
            reason="Logistic is already there",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt_plan = opt.optimize(plan, [action])
        assert len(opt_plan.model_candidates) == 2

    def test_add_model_candidate_invalid(self):
        """Verify adding invalid model family raises ValueError."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_add",
            action_type=OptimizationActionType.ADD_MODEL,
            replacement="invalid_family",
            reason="Add fake model",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError):
            opt.optimize(plan, [action])

    def test_remove_model_candidate(self):
        """Verify removing a model candidate works."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_remove",
            action_type=OptimizationActionType.REMOVE_MODEL,
            target="random_forest",
            reason="Random Forest overfits",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt_plan = opt.optimize(plan, [action])
        assert len(opt_plan.model_candidates) == 1
        assert opt_plan.model_candidates[0].model_family == ModelFamily.LOGISTIC_REGRESSION

    def test_remove_model_candidate_not_found(self):
        """Verify trying to remove a candidate not in plan raises ValueError."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_remove",
            action_type=OptimizationActionType.REMOVE_MODEL,
            target="gradient_boosting",  # Not in sample plan
            reason="Remove gradient boosting",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError, match="Model family 'gradient_boosting' not found"):
            opt.optimize(plan, [action])

    def test_remove_model_candidate_resulting_empty_list(self):
        """Verify trying to remove all model candidates raises ValueError."""
        plan = _make_sample_plan()
        # Remove logistic regression
        act1 = OptimizationAction(
            action_id="act_remove1",
            action_type=OptimizationActionType.REMOVE_MODEL,
            target="logistic_regression",
            reason="r1",
            confidence=0.9,
        )
        # Remove random forest
        act2 = OptimizationAction(
            action_id="act_remove2",
            action_type=OptimizationActionType.REMOVE_MODEL,
            target="random_forest",
            reason="r2",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        with pytest.raises(ValueError, match="empty candidate list"):
            opt.optimize(plan, [act1, act2])

    def test_add_warning(self):
        """Verify adding warning appends an MLPlanWarning."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_warn",
            action_type=OptimizationActionType.ADD_WARNING,
            reason="Beware of overfit risk",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt_plan = opt.optimize(plan, [action])
        assert len(opt_plan.warnings) == 1
        assert opt_plan.warnings[0].message == "Beware of overfit risk"

    def test_duplicate_action_ids(self):
        """Verify duplicate action IDs in OptimizationAction list raise ValueError."""
        plan = _make_sample_plan()
        act1 = OptimizationAction(
            action_id="act_dup",
            action_type=OptimizationActionType.CHANGE_CV_FOLDS,
            parameters={"folds": 3},
            reason="r1",
            confidence=0.9,
        )
        act2 = OptimizationAction(
            action_id="act_dup",
            action_type=OptimizationActionType.CHANGE_CV_FOLDS,
            parameters={"folds": 4},
            reason="r2",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        with pytest.raises(ValueError, match="Duplicate action ID detected"):
            opt.optimize(plan, [act1, act2])

    def test_non_mutation(self):
        """Verify the optimizer does not mutate the baseline MLPlan."""
        plan = _make_sample_plan()
        import copy
        plan_copy = copy.deepcopy(plan)

        action = OptimizationAction(
            action_id="act_cv",
            action_type=OptimizationActionType.CHANGE_CV_FOLDS,
            parameters={"folds": 5},
            reason="Increase folds",
            confidence=0.9,
        )

        opt = AIPlanOptimizer()
        opt.optimize(plan, [action])
        assert plan == plan_copy

    def test_remove_model_invalid_family_target(self):
        """Verify trying to remove an invalid model family targets raises ValueError."""
        plan = _make_sample_plan()
        action = OptimizationAction(
            action_id="act_remove",
            action_type=OptimizationActionType.REMOVE_MODEL,
            target="invalid-model-family",
            reason="Remove invalid family",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError, match="Invalid model family to remove"):
            opt.optimize(plan, [action])

    def test_unknown_action_type_validation(self):
        """Verify plan optimizer catches unknown action types if bypassed."""
        plan = _make_sample_plan()
        
        # Bypassing validation with model_construct
        action = OptimizationAction.model_construct(
            action_id="act_unknown",
            action_type="UNKNOWN_ACTION_TYPE",
            reason="Unknown action",
            confidence=0.9,
        )
        opt = AIPlanOptimizer()
        with pytest.raises(ValueError, match="Unknown action type"):
            opt.optimize(plan, [action])
