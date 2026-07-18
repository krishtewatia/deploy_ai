"""Unit tests for ModelFactory."""

from __future__ import annotations

import copy
from unittest.mock import patch
import pytest

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from backend.app.compute_capabilities.schemas import AcceleratorType, ComputeTier
from backend.app.ml_plan.schemas import (
    DatasetSplitPlan,
    EvaluationPlan,
    ExecutionConstraints,
    FeatureSelectionMethod,
    FeatureSelectionPlan,
    MLPlan,
    ModelCandidate,
    ModelFamily,
    ProblemType,
    SearchStrategy,
    SplitStrategy,
)
from backend.app.ml_execution.model_factory import (
    ModelFactory,
    ModelFactoryError,
    ModelFactoryResult,
)


# ── Helper Builders ───────────────────────────────────────────────────

def _make_ml_plan(
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    model_candidates: list[ModelCandidate] = None,
) -> MLPlan:
    if model_candidates is None:
        model_candidates = [
            ModelCandidate(
                candidate_id="model_rf",
                model_family=ModelFamily.RANDOM_FOREST,
                parameters={"n_estimators": 10},
                search_strategy=SearchStrategy.NONE,
                search_space={},
                reason="Baseline random forest.",
            )
        ]

    return MLPlan(
        plan_id="plan_01",
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        problem_type=problem_type,
        target_column="target",
        feature_columns=["A", "B"],
        preprocessing_steps=[],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["A", "B"],
            max_features=2,
            parameters={},
            reason="No feature selection required.",
        ),
        split_plan=DatasetSplitPlan(
            strategy=SplitStrategy.RANDOM,
            test_size=0.2,
            validation_size=0.1,
            random_state=42,
            shuffle=True,
        ),
        model_candidates=model_candidates,
        evaluation_plan=EvaluationPlan(
            primary_metric="accuracy",
            secondary_metrics=["f1"],
            cross_validation_folds=5,
        ),
        execution_constraints=ExecutionConstraints(
            parallel_workers=4,
            use_gpu_acceleration=False,
            accelerator_type=AcceleratorType.NONE,
            compute_tier=ComputeTier.STANDARD,
        ),
    )


# ── Test Suite ──────────────────────────────────────────────────────────────

class TestModelFactory:
    """Tests covering ModelFactory instantiation, parameters, and validations."""

    def test_factory_build_none_input(self):
        """Verify factory build rejects None input."""
        factory = ModelFactory()
        with pytest.raises(ModelFactoryError, match="plan cannot be None"):
            factory.build(None)

    def test_factory_build_wrong_input_type(self):
        """Verify factory build rejects wrong input types."""
        factory = ModelFactory()
        with pytest.raises(ModelFactoryError, match="plan must be an MLPlan instance"):
            factory.build("not-a-plan")

    def test_factory_build_empty_candidate_list(self):
        """Verify empty plan.model_candidates raises ModelFactoryError."""
        plan = _make_ml_plan()
        factory = ModelFactory()
        with patch.object(plan, "model_candidates", []):
            with pytest.raises(ModelFactoryError, match="plan.model_candidates list cannot be empty"):
                factory.build(plan)

    def test_factory_build_empty_candidate_id(self):
        """Verify empty candidate ID raises ModelFactoryError."""
        plan = _make_ml_plan()
        factory = ModelFactory()
        with patch.object(plan.model_candidates[0], "candidate_id", ""):
            with pytest.raises(ModelFactoryError, match="candidate_id cannot be empty or whitespace-only"):
                factory.build(plan)

    def test_factory_build_duplicate_candidate_ids(self):
        """Verify duplicate candidate IDs raise ModelFactoryError."""
        c1 = ModelCandidate(candidate_id="model_rf", model_family=ModelFamily.RANDOM_FOREST, parameters={}, reason="rf")
        c2 = ModelCandidate(candidate_id="model_dt", model_family=ModelFamily.DECISION_TREE, parameters={}, reason="dt")
        plan = _make_ml_plan(model_candidates=[c1, c2])

        factory = ModelFactory()
        with patch.object(c2, "candidate_id", "model_rf"):
            with pytest.raises(ModelFactoryError, match="Duplicate candidate_id 'model_rf' found"):
                factory.build(plan)

    def test_factory_build_duplicate_families(self):
        """Verify duplicate model families raise ModelFactoryError."""
        c1 = ModelCandidate(candidate_id="model_rf1", model_family=ModelFamily.RANDOM_FOREST, parameters={}, reason="rf1")
        c2 = ModelCandidate(candidate_id="model_rf2", model_family=ModelFamily.DECISION_TREE, parameters={}, reason="dt")
        plan = _make_ml_plan(model_candidates=[c1, c2])

        factory = ModelFactory()
        with patch.object(c2, "model_family", ModelFamily.RANDOM_FOREST):
            with pytest.raises(ModelFactoryError, match="Duplicate model_family 'random_forest' found"):
                factory.build(plan)

    def test_factory_build_parameters_not_dictionary(self):
        """Verify non-dictionary parameters raise ModelFactoryError."""
        plan = _make_ml_plan()
        factory = ModelFactory()
        with patch.object(plan.model_candidates[0], "parameters", "not-a-dict"):
            with pytest.raises(ModelFactoryError, match="must be a dictionary"):
                factory.build(plan)

    def test_factory_build_unsupported_family_for_problem_type(self):
        """Verify regression-only family is rejected in classification, and vice versa."""
        # 1. Classification plan with Linear Regression
        plan_classif = _make_ml_plan(problem_type=ProblemType.CLASSIFICATION)
        factory = ModelFactory()
        with patch.object(plan_classif.model_candidates[0], "model_family", ModelFamily.LINEAR_REGRESSION):
            with pytest.raises(ModelFactoryError, match="is not supported for problem type 'classification'"):
                factory.build(plan_classif)

        # 2. Regression plan with Logistic Regression
        plan_reg = _make_ml_plan(problem_type=ProblemType.REGRESSION)
        with patch.object(plan_reg.model_candidates[0], "model_family", ModelFamily.LOGISTIC_REGRESSION):
            with pytest.raises(ModelFactoryError, match="is not supported for problem type 'regression'"):
                factory.build(plan_reg)

    def test_factory_build_unsupported_problem_type_fallback(self):
        """Verify unknown problem types raise ModelFactoryError."""
        plan = _make_ml_plan()
        factory = ModelFactory()
        with patch.object(plan, "problem_type", "UNKNOWN_PROBLEM_TYPE"):
            with pytest.raises(ModelFactoryError, match="Unsupported problem type"):
                factory.build(plan)

    def test_factory_build_constructor_failure(self):
        """Verify constructor failures raise ModelFactoryError."""
        # Invalid keyword argument for RandomForest
        c = ModelCandidate(
            candidate_id="model_rf",
            model_family=ModelFamily.RANDOM_FOREST,
            parameters={"invalid_parameter_name": 123},
            reason="rf",
        )
        plan = _make_ml_plan(model_candidates=[c])
        factory = ModelFactory()
        with pytest.raises(ModelFactoryError, match="Failed to construct model"):
            factory.build(plan)

    def test_factory_build_regression_models(self):
        """Verify all 8 supported regression estimators are constructed correctly."""
        estimators_to_test = [
            (ModelFamily.LINEAR_REGRESSION, LinearRegression, {}),
            (ModelFamily.RIDGE, Ridge, {"alpha": 1.0, "random_state": 42}),
            (ModelFamily.LASSO, Lasso, {"alpha": 0.5, "random_state": 42}),
            (ModelFamily.DECISION_TREE, DecisionTreeRegressor, {"max_depth": 3, "random_state": 42}),
            (ModelFamily.RANDOM_FOREST, RandomForestRegressor, {"n_estimators": 5, "random_state": 42}),
            (ModelFamily.GRADIENT_BOOSTING, GradientBoostingRegressor, {"n_estimators": 5, "random_state": 42}),
            (ModelFamily.KNN, KNeighborsRegressor, {"n_neighbors": 3}),
            (ModelFamily.SVM, SVR, {"C": 1.0}),
        ]

        for idx, (family, expected_class, params) in enumerate(estimators_to_test):
            c = ModelCandidate(
                candidate_id=f"model_{idx}",
                model_family=family,
                parameters=params,
                reason="regression model test",
            )
            plan = _make_ml_plan(
                problem_type=ProblemType.REGRESSION,
                model_candidates=[c],
            )
            factory = ModelFactory()
            res = factory.build(plan)

            assert isinstance(res, ModelFactoryResult)
            assert f"model_{idx}" in res.models
            model = res.models[f"model_{idx}"]
            assert isinstance(model, expected_class)
            
            # Check parameter forwarding
            for param_key, param_val in params.items():
                assert getattr(model, param_key) == param_val

    def test_factory_build_classification_models(self):
        """Verify all 6 supported classification estimators are constructed correctly."""
        estimators_to_test = [
            (ModelFamily.LOGISTIC_REGRESSION, LogisticRegression, {"C": 1.0, "random_state": 42}),
            (ModelFamily.DECISION_TREE, DecisionTreeClassifier, {"max_depth": 3, "random_state": 42}),
            (ModelFamily.RANDOM_FOREST, RandomForestClassifier, {"n_estimators": 5, "random_state": 42}),
            (ModelFamily.GRADIENT_BOOSTING, GradientBoostingClassifier, {"n_estimators": 5, "random_state": 42}),
            (ModelFamily.KNN, KNeighborsClassifier, {"n_neighbors": 3}),
            (ModelFamily.SVM, SVC, {"C": 1.0, "random_state": 42}),
        ]

        for idx, (family, expected_class, params) in enumerate(estimators_to_test):
            c = ModelCandidate(
                candidate_id=f"model_{idx}",
                model_family=family,
                parameters=params,
                reason="classification model test",
            )
            plan = _make_ml_plan(
                problem_type=ProblemType.CLASSIFICATION,
                model_candidates=[c],
            )
            factory = ModelFactory()
            res = factory.build(plan)

            assert isinstance(res, ModelFactoryResult)
            assert f"model_{idx}" in res.models
            model = res.models[f"model_{idx}"]
            assert isinstance(model, expected_class)
            
            # Check parameter forwarding
            for param_key, param_val in params.items():
                assert getattr(model, param_key) == param_val

    def test_non_mutation(self):
        """Verify that building models does not mutate the plan input."""
        plan = _make_ml_plan()
        plan_orig = copy.deepcopy(plan)

        factory = ModelFactory()
        res = factory.build(plan)

        assert isinstance(res, ModelFactoryResult)
        assert plan == plan_orig

    def test_determinism(self):
        """Verify that building twice with identical parameters constructs equivalent estimators."""
        plan = _make_ml_plan()
        factory = ModelFactory()
        res1 = factory.build(plan)
        res2 = factory.build(plan)

        assert res1.candidate_ids == res2.candidate_ids
        assert res1.model_families == res2.model_families
        assert res1.execution_summary == res2.execution_summary
        
        # Verify estimators are distinct instances but configured identically
        for cid in res1.candidate_ids:
            est1 = res1.models[cid]
            est2 = res2.models[cid]
            assert est1 is not est2
            assert est1.get_params() == est2.get_params()
