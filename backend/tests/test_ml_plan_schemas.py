"""Unit tests for MLPlan schemas.

Covers all 90 specified test cases for MLPlan and its sub-schemas.
"""

import json
import pytest
from pydantic import ValidationError

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.ml_plan.schemas import (
    DatasetSplitPlan,
    EvaluationPlan,
    ExecutionConstraints,
    FeatureEngineeringOperation,
    FeatureEngineeringStep,
    FeatureSelectionMethod,
    FeatureSelectionPlan,
    MLPlan,
    MLPlanConfirmationItem,
    MLPlanStatus,
    MLPlanWarning,
    ModelCandidate,
    ModelFamily,
    PreprocessingOperation,
    PreprocessingStep,
    SearchStrategy,
    SplitStrategy,
)
from backend.app.problem_definition.schemas import ProblemType


# ── Helper Constructors ────────────────────────────────────────────────


def _make_valid_preprocessing_step(**overrides) -> dict:
    base = {
        "step_id": "prep_01",
        "operation": PreprocessingOperation.IMPUTE_MEDIAN,
        "columns": ["age", "salary"],
        "parameters": {"strategy": "median"},
        "reason": "Numeric columns contain missing values.",
    }
    base.update(overrides)
    return base


def _make_valid_feature_engineering_step(**overrides) -> dict:
    base = {
        "step_id": "fe_01",
        "operation": FeatureEngineeringOperation.INTERACTION,
        "input_columns": ["col_a", "col_b"],
        "output_columns": ["col_a_col_b"],
        "parameters": {},
        "reason": "Interaction of physical parameters.",
    }
    base.update(overrides)
    return base


def _make_valid_feature_selection_plan(**overrides) -> dict:
    base = {
        "method": FeatureSelectionMethod.VARIANCE_THRESHOLD,
        "candidate_columns": ["col_a", "col_b", "col_c"],
        "max_features": 2,
        "parameters": {"threshold": 0.01},
        "reason": "Filter out zero variance features.",
    }
    base.update(overrides)
    return base


def _make_valid_split_plan(**overrides) -> dict:
    base = {
        "strategy": SplitStrategy.RANDOM,
        "test_size": 0.2,
        "validation_size": 0.1,
        "random_state": 42,
        "shuffle": True,
        "stratify_column": None,
        "time_column": None,
    }
    base.update(overrides)
    return base


def _make_valid_model_candidate(**overrides) -> dict:
    base = {
        "candidate_id": "model_01",
        "model_family": ModelFamily.RANDOM_FOREST,
        "parameters": {"n_estimators": 100},
        "search_strategy": SearchStrategy.NONE,
        "search_space": {},
        "reason": "Stable baseline random forest model.",
    }
    base.update(overrides)
    return base


def _make_valid_evaluation_plan(**overrides) -> dict:
    base = {
        "primary_metric": "accuracy",
        "secondary_metrics": ["f1", "precision"],
        "cross_validation_folds": 5,
    }
    base.update(overrides)
    return base


def _make_valid_execution_constraints(**overrides) -> dict:
    base = {
        "parallel_workers": 4,
        "use_gpu_acceleration": False,
        "accelerator_type": AcceleratorType.NONE,
        "compute_tier": ComputeTier.STANDARD,
    }
    base.update(overrides)
    return base


def _make_valid_ml_plan_dict(**overrides) -> dict:
    base = {
        "plan_id": "plan_001",
        "dataset_id": "ds_001",
        "request_id": "req_001",
        "problem_definition_id": "pd_001",
        "compute_capability_id": "cap_001",
        "problem_type": ProblemType.CLASSIFICATION,
        "target_column": "target",
        "feature_columns": ["col_a", "col_b", "col_c"],
        "preprocessing_steps": [PreprocessingStep(**_make_valid_preprocessing_step())],
        "feature_engineering_steps": [FeatureEngineeringStep(**_make_valid_feature_engineering_step())],
        "feature_selection": FeatureSelectionPlan(**_make_valid_feature_selection_plan()),
        "split_plan": DatasetSplitPlan(**_make_valid_split_plan()),
        "model_candidates": [ModelCandidate(**_make_valid_model_candidate())],
        "evaluation_plan": EvaluationPlan(**_make_valid_evaluation_plan()),
        "execution_constraints": ExecutionConstraints(**_make_valid_execution_constraints()),
        "status": MLPlanStatus.DRAFT,
        "warnings": [],
        "confirmation_items": [],
    }
    base.update(overrides)
    return base


# ── Tests ──────────────────────────────────────────────────────────────


class TestMLPlanEnums:
    """Tests covering enum serialization and values."""

    # 1. All MLPlanStatus values
    @pytest.mark.parametrize("status", ["draft", "ready", "needs_confirmation", "blocked"])
    def test_ml_plan_statuses(self, status):
        assert MLPlanStatus(status) == status

    # 2. All PreprocessingOperation values
    @pytest.mark.parametrize("op", [
        "drop_column", "impute_mean", "impute_median", "impute_mode",
        "impute_constant", "one_hot_encode", "ordinal_encode",
        "standard_scale", "minmax_scale", "robust_scale",
        "datetime_extract", "passthrough"
    ])
    def test_preprocessing_operations(self, op):
        assert PreprocessingOperation(op) == op

    # 3. All FeatureEngineeringOperation values
    @pytest.mark.parametrize("op", [
        "interaction", "polynomial", "ratio", "difference",
        "datetime_parts", "log_transform", "custom"
    ])
    def test_feature_engineering_operations(self, op):
        assert FeatureEngineeringOperation(op) == op

    # 4. All FeatureSelectionMethod values
    @pytest.mark.parametrize("method", [
        "none", "variance_threshold", "correlation_filter",
        "mutual_information", "model_based"
    ])
    def test_feature_selection_methods(self, method):
        assert FeatureSelectionMethod(method) == method

    # 5. All SplitStrategy values
    @pytest.mark.parametrize("strat", ["random", "stratified", "time_based"])
    def test_split_strategies(self, strat):
        assert SplitStrategy(strat) == strat

    # 6. All ModelFamily values
    @pytest.mark.parametrize("fam", [
        "linear_regression", "logistic_regression", "ridge", "lasso",
        "decision_tree", "random_forest", "gradient_boosting",
        "extra_trees", "knn", "svm"
    ])
    def test_model_families(self, fam):
        assert ModelFamily(fam) == fam

    # 7. All SearchStrategy values
    @pytest.mark.parametrize("strat", ["none", "grid", "random"])
    def test_search_strategies(self, strat):
        assert SearchStrategy(strat) == strat


class TestPreprocessingStep:
    """Tests covering PreprocessingStep model."""

    # 8. Valid preprocessing step
    def test_valid_step(self):
        step = PreprocessingStep(**_make_valid_preprocessing_step())
        assert step.step_id == "prep_01"

    # 9. Empty step_id rejected
    def test_empty_step_id_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            PreprocessingStep(**_make_valid_preprocessing_step(step_id=""))
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            PreprocessingStep(**_make_valid_preprocessing_step(step_id="  "))

    # 10. Empty columns rejected
    def test_empty_columns_rejected(self):
        with pytest.raises(ValidationError, match="columns list must contain at least one column"):
            PreprocessingStep(**_make_valid_preprocessing_step(columns=[]))

    # 11. Duplicate columns rejected
    def test_duplicate_columns_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate column name detected"):
            PreprocessingStep(**_make_valid_preprocessing_step(columns=["age", "age"]))

    # 12. Whitespace normalization
    def test_whitespace_normalization(self):
        step = PreprocessingStep(**_make_valid_preprocessing_step(step_id="  prep_02\n", columns=[" age ", " salary\t"]))
        assert step.step_id == "prep_02"
        assert step.columns == ["age", "salary"]

    # 13. JSON parameters accepted
    def test_json_parameters_accepted(self):
        step = PreprocessingStep(**_make_valid_preprocessing_step(parameters={"threshold": 0.5}))
        assert step.parameters == {"threshold": 0.5}

    # 14. Non-JSON parameters rejected
    def test_non_json_parameters_rejected(self):
        # Set list with set inside or custom un-serializable objects
        with pytest.raises(ValidationError, match="parameters must be JSON serializable"):
            PreprocessingStep(**_make_valid_preprocessing_step(parameters={"custom": {1, 2}}))

    # 15. Independent parameters dictionaries
    def test_independent_parameters_dicts(self):
        s1 = PreprocessingStep(**_make_valid_preprocessing_step())
        s2 = PreprocessingStep(**_make_valid_preprocessing_step())
        s1.parameters["foo"] = "bar"
        assert "foo" not in s2.parameters


class TestFeatureEngineeringStep:
    """Tests covering FeatureEngineeringStep model."""

    # 16. Valid step
    def test_valid_fe_step(self):
        step = FeatureEngineeringStep(**_make_valid_feature_engineering_step())
        assert step.step_id == "fe_01"

    # 17. Empty input columns rejected
    def test_empty_input_columns_rejected(self):
        with pytest.raises(ValidationError, match="input_columns list must contain at least one column"):
            FeatureEngineeringStep(**_make_valid_feature_engineering_step(input_columns=[]))

    # 18. Empty output columns rejected
    def test_empty_output_columns_rejected(self):
        with pytest.raises(ValidationError, match="output_columns list must contain at least one column"):
            FeatureEngineeringStep(**_make_valid_feature_engineering_step(output_columns=[]))

    # 19. Duplicate input columns rejected
    def test_duplicate_input_columns_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate column name detected"):
            FeatureEngineeringStep(**_make_valid_feature_engineering_step(input_columns=["a", "a"]))

    # 20. Duplicate output columns rejected
    def test_duplicate_output_columns_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate column name detected"):
            FeatureEngineeringStep(**_make_valid_feature_engineering_step(output_columns=["b", "b"]))

    # 21. Non-JSON parameters rejected
    def test_non_json_parameters_rejected(self):
        with pytest.raises(ValidationError, match="parameters must be JSON serializable"):
            FeatureEngineeringStep(**_make_valid_feature_engineering_step(parameters={"non_json": object()}))


class TestFeatureSelectionPlan:
    """Tests covering FeatureSelectionPlan model."""

    # 22. method=none accepted
    def test_method_none_accepted(self):
        plan = FeatureSelectionPlan(**_make_valid_feature_selection_plan(method=FeatureSelectionMethod.NONE))
        assert plan.method == FeatureSelectionMethod.NONE

    # 23. Valid enabled method
    def test_valid_selection_method(self):
        plan = FeatureSelectionPlan(**_make_valid_feature_selection_plan(method=FeatureSelectionMethod.MODEL_BASED))
        assert plan.method == FeatureSelectionMethod.MODEL_BASED

    # 24. Empty candidate columns rejected
    def test_empty_candidate_columns_rejected(self):
        with pytest.raises(ValidationError, match="candidate_columns list must contain at least one column"):
            FeatureSelectionPlan(**_make_valid_feature_selection_plan(candidate_columns=[]))

    # 25. Duplicate candidates rejected
    def test_duplicate_candidates_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate column name detected"):
            FeatureSelectionPlan(**_make_valid_feature_selection_plan(candidate_columns=["a", "a"]))

    # 26. max_features < 1 rejected
    def test_max_features_below_one_rejected(self):
        with pytest.raises(ValidationError, match="max_features must be >= 1"):
            FeatureSelectionPlan(**_make_valid_feature_selection_plan(max_features=0))

    # 27. max_features greater than candidate count rejected
    def test_max_features_greater_than_candidates_rejected(self):
        with pytest.raises(ValidationError, match="max_features cannot exceed the number of candidate columns"):
            FeatureSelectionPlan(**_make_valid_feature_selection_plan(max_features=4))

    # 28. max_features equal to candidate count accepted
    def test_max_features_equal_to_candidates_accepted(self):
        plan = FeatureSelectionPlan(**_make_valid_feature_selection_plan(max_features=3))
        assert plan.max_features == 3


class TestDatasetSplitPlan:
    """Tests covering DatasetSplitPlan model."""

    # 29. Valid random split
    def test_valid_random_split(self):
        plan = DatasetSplitPlan(**_make_valid_split_plan(strategy=SplitStrategy.RANDOM))
        assert plan.strategy == SplitStrategy.RANDOM

    # 30. Valid stratified split
    def test_valid_stratified_split(self):
        plan = DatasetSplitPlan(
            **_make_valid_split_plan(strategy=SplitStrategy.STRATIFIED, stratify_column="class_label")
        )
        assert plan.strategy == SplitStrategy.STRATIFIED
        assert plan.stratify_column == "class_label"

    # 31. Valid time-based split
    def test_valid_time_split(self):
        plan = DatasetSplitPlan(
            **_make_valid_split_plan(strategy=SplitStrategy.TIME_BASED, time_column="timestamp", shuffle=False)
        )
        assert plan.strategy == SplitStrategy.TIME_BASED
        assert plan.time_column == "timestamp"
        assert plan.shuffle is False

    # 32. test_size <= 0 rejected
    def test_test_size_too_small_rejected(self):
        with pytest.raises(ValidationError, match="test_size must be strictly between 0.0 and 1.0"):
            DatasetSplitPlan(**_make_valid_split_plan(test_size=0.0))

    # 33. test_size >= 1 rejected
    def test_test_size_too_large_rejected(self):
        with pytest.raises(ValidationError, match="test_size must be strictly between 0.0 and 1.0"):
            DatasetSplitPlan(**_make_valid_split_plan(test_size=1.0))

    # 34. validation_size < 0 rejected
    def test_validation_size_negative_rejected(self):
        with pytest.raises(ValidationError, match="validation_size must be between 0.0 and 1.0"):
            DatasetSplitPlan(**_make_valid_split_plan(validation_size=-0.1))

    # 35. test + validation >= 1 rejected
    def test_sizes_sum_exceeds_one_rejected(self):
        with pytest.raises(ValidationError, match="Sum of test_size and validation_size must be less than 1.0"):
            DatasetSplitPlan(**_make_valid_split_plan(test_size=0.6, validation_size=0.4))

    # 36. random + stratify_column rejected
    def test_random_with_stratify_column_rejected(self):
        with pytest.raises(ValidationError, match="stratify_column must be None for random split strategy"):
            DatasetSplitPlan(**_make_valid_split_plan(strategy=SplitStrategy.RANDOM, stratify_column="target"))

    # 37. random + time_column rejected
    def test_random_with_time_column_rejected(self):
        with pytest.raises(ValidationError, match="time_column must be None for random split strategy"):
            DatasetSplitPlan(**_make_valid_split_plan(strategy=SplitStrategy.RANDOM, time_column="date"))

    # 38. stratified without stratify_column rejected
    def test_stratified_missing_column_rejected(self):
        with pytest.raises(ValidationError, match="stratify_column is required for stratified split strategy"):
            DatasetSplitPlan(**_make_valid_split_plan(strategy=SplitStrategy.STRATIFIED))

    # 39. stratified with time_column rejected
    def test_stratified_with_time_column_rejected(self):
        with pytest.raises(ValidationError, match="time_column must be None for stratified split strategy"):
            DatasetSplitPlan(
                **_make_valid_split_plan(strategy=SplitStrategy.STRATIFIED, stratify_column="lbl", time_column="dt")
            )

    # 40. time_based without time_column rejected
    def test_time_split_missing_column_rejected(self):
        with pytest.raises(ValidationError, match="time_column is required for time_based split strategy"):
            DatasetSplitPlan(**_make_valid_split_plan(strategy=SplitStrategy.TIME_BASED, shuffle=False))

    # 41. time_based with stratify_column rejected
    def test_time_split_with_stratify_column_rejected(self):
        with pytest.raises(ValidationError, match="stratify_column must be None for time_based split strategy"):
            DatasetSplitPlan(
                **_make_valid_split_plan(
                    strategy=SplitStrategy.TIME_BASED, time_column="dt", stratify_column="lbl", shuffle=False
                )
            )

    # 42. time_based with shuffle=True rejected
    def test_time_split_with_shuffle_rejected(self):
        with pytest.raises(ValidationError, match="shuffle must be False for time_based split strategy"):
            DatasetSplitPlan(
                **_make_valid_split_plan(strategy=SplitStrategy.TIME_BASED, time_column="dt", shuffle=True)
            )

    # 43. Boundary-valid split accepted
    def test_split_plan_boundary_valid(self):
        plan = DatasetSplitPlan(
            **_make_valid_split_plan(test_size=0.99, validation_size=0.0)
        )
        assert plan.test_size == 0.99
        assert plan.validation_size == 0.0


class TestModelCandidate:
    """Tests covering ModelCandidate model."""

    # 44. Valid no-search candidate
    def test_valid_no_search_candidate(self):
        cand = ModelCandidate(**_make_valid_model_candidate())
        assert cand.search_strategy == SearchStrategy.NONE

    # 45. Valid grid-search candidate
    def test_valid_grid_search_candidate(self):
        cand = ModelCandidate(
            **_make_valid_model_candidate(
                search_strategy=SearchStrategy.GRID, search_space={"n_estimators": [50, 100, 200]}
            )
        )
        assert cand.search_strategy == SearchStrategy.GRID

    # 46. Valid random-search candidate
    def test_valid_random_search_candidate(self):
        cand = ModelCandidate(
            **_make_valid_model_candidate(
                search_strategy=SearchStrategy.RANDOM, search_space={"max_depth": [3, 5, None]}
            )
        )
        assert cand.search_strategy == SearchStrategy.RANDOM

    # 47. Empty candidate_id rejected
    def test_empty_candidate_id_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ModelCandidate(**_make_valid_model_candidate(candidate_id=" "))

    # 48. search_strategy=none with non-empty search_space rejected
    def test_no_search_with_search_space_rejected(self):
        with pytest.raises(ValidationError, match="search_space must be empty when search_strategy is 'none'"):
            ModelCandidate(**_make_valid_model_candidate(search_space={"max_depth": [3]}))

    # 49. grid with empty search_space rejected
    def test_grid_with_empty_search_space_rejected(self):
        with pytest.raises(ValidationError, match="search_space cannot be empty when search_strategy is 'grid'"):
            ModelCandidate(**_make_valid_model_candidate(search_strategy=SearchStrategy.GRID, search_space={}))

    # 50. random with empty search_space rejected
    def test_random_with_empty_search_space_rejected(self):
        with pytest.raises(ValidationError, match="search_space cannot be empty when search_strategy is 'random'"):
            ModelCandidate(**_make_valid_model_candidate(search_strategy=SearchStrategy.RANDOM, search_space={}))

    # 51. Non-JSON parameters rejected
    def test_non_json_parameters_rejected(self):
        with pytest.raises(ValidationError, match="parameters must be JSON serializable"):
            ModelCandidate(**_make_valid_model_candidate(parameters={"val": {1, 2}}))

    # 52. Non-JSON search_space rejected
    def test_non_json_search_space_rejected(self):
        with pytest.raises(ValidationError, match="search_space must be JSON serializable"):
            ModelCandidate(
                **_make_valid_model_candidate(
                    search_strategy=SearchStrategy.GRID, search_space={"val": object()}
                )
            )


class TestEvaluationPlan:
    """Tests covering EvaluationPlan model."""

    # 53. Valid evaluation plan
    def test_valid_evaluation_plan(self):
        plan = EvaluationPlan(**_make_valid_evaluation_plan())
        assert plan.primary_metric == "accuracy"

    # 54. Empty primary metric rejected
    def test_empty_primary_metric_rejected(self):
        with pytest.raises(ValidationError, match="primary_metric cannot be empty"):
            EvaluationPlan(**_make_valid_evaluation_plan(primary_metric=""))

    # 55. Duplicate secondary metrics rejected
    def test_duplicate_secondary_metrics_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate secondary metric detected"):
            EvaluationPlan(**_make_valid_evaluation_plan(secondary_metrics=["precision", "precision"]))

    # 56. Primary metric duplicated in secondary metrics rejected
    def test_primary_duplicated_in_secondary_rejected(self):
        with pytest.raises(ValidationError, match="cannot appear in secondary_metrics"):
            EvaluationPlan(**_make_valid_evaluation_plan(primary_metric="f1", secondary_metrics=["f1"]))

    # 57. cross_validation_folds < 2 rejected
    def test_cv_folds_below_two_rejected(self):
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 2"):
            EvaluationPlan(**_make_valid_evaluation_plan(cross_validation_folds=1))

    # 58. Independent secondary metric lists
    def test_independent_secondary_metric_lists(self):
        p1 = EvaluationPlan(**_make_valid_evaluation_plan())
        p2 = EvaluationPlan(**_make_valid_evaluation_plan())
        p1.secondary_metrics.append("recall")
        assert "recall" not in p2.secondary_metrics


class TestExecutionConstraints:
    """Tests covering ExecutionConstraints model."""

    # 59. Valid CPU execution constraints
    def test_valid_cpu_constraints(self):
        cons = ExecutionConstraints(**_make_valid_execution_constraints())
        assert cons.use_gpu_acceleration is False
        assert cons.accelerator_type == AcceleratorType.NONE

    # 60. Valid CUDA execution constraints
    def test_valid_cuda_constraints(self):
        cons = ExecutionConstraints(
            **_make_valid_execution_constraints(
                use_gpu_acceleration=True, accelerator_type=AcceleratorType.CUDA
            )
        )
        assert cons.use_gpu_acceleration is True
        assert cons.accelerator_type == AcceleratorType.CUDA

    # 61. parallel_workers < 1 rejected
    def test_workers_below_one_rejected(self):
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
            ExecutionConstraints(**_make_valid_execution_constraints(parallel_workers=0))

    # 62. GPU false + CUDA rejected
    def test_gpu_false_with_cuda_rejected(self):
        with pytest.raises(ValidationError, match="accelerator_type must be 'none' when use_gpu_acceleration is False"):
            ExecutionConstraints(
                **_make_valid_execution_constraints(
                    use_gpu_acceleration=False, accelerator_type=AcceleratorType.CUDA
                )
            )

    # 63. GPU true + NONE rejected
    def test_gpu_true_with_none_rejected(self):
        with pytest.raises(ValidationError, match="accelerator_type cannot be 'none' when use_gpu_acceleration is True"):
            ExecutionConstraints(
                **_make_valid_execution_constraints(
                    use_gpu_acceleration=True, accelerator_type=AcceleratorType.NONE
                )
            )


class TestWarningsConfirmation:
    """Tests covering MLPlanWarning and MLPlanConfirmationItem."""

    # 64. Valid MLPlanWarning
    def test_valid_warning(self):
        warn = MLPlanWarning(code="W01", message="Low system resources")
        assert warn.code == "W01"

    # 65. Empty warning code rejected
    def test_empty_warning_code_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            MLPlanWarning(code="", message="msg")

    # 66. Empty warning message rejected
    def test_empty_warning_msg_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            MLPlanWarning(code="W01", message=" ")

    # 67. Valid confirmation item
    def test_valid_confirmation_item(self):
        item = MLPlanConfirmationItem(key="train", question="Proceed?", reason="Required decision")
        assert item.key == "train"

    # 68. Empty confirmation fields rejected
    def test_empty_confirmation_fields_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            MLPlanConfirmationItem(key="", question="?", reason="reason")


class TestMainMLPlan:
    """Tests covering the aggregated MLPlan model."""

    # 69. Minimal valid draft MLPlan
    def test_minimal_valid_draft_plan(self):
        plan = MLPlan(**_make_valid_ml_plan_dict())
        assert plan.status == MLPlanStatus.DRAFT

    # 70. Valid ready MLPlan
    def test_valid_ready_plan(self):
        plan = MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.READY))
        assert plan.status == MLPlanStatus.READY

    # 71. Valid needs-confirmation MLPlan
    def test_valid_needs_confirmation_plan(self):
        item = MLPlanConfirmationItem(key="gpu", question="Use GPU?", reason="Speed")
        plan = MLPlan(
            **_make_valid_ml_plan_dict(status=MLPlanStatus.NEEDS_CONFIRMATION, confirmation_items=[item])
        )
        assert plan.status == MLPlanStatus.NEEDS_CONFIRMATION

    # 72. Valid blocked MLPlan
    def test_valid_blocked_plan(self):
        plan = MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.BLOCKED))
        assert plan.status == MLPlanStatus.BLOCKED

    # 73. Empty IDs rejected
    def test_empty_ids_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            MLPlan(**_make_valid_ml_plan_dict(plan_id=" "))

    # 74. Empty target column rejected
    def test_empty_target_column_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            MLPlan(**_make_valid_ml_plan_dict(target_column=" "))

    # 75. Empty feature_columns rejected
    def test_empty_feature_columns_rejected(self):
        with pytest.raises(ValidationError, match="feature_columns list must contain at least one column"):
            MLPlan(**_make_valid_ml_plan_dict(feature_columns=[]))

    # 76. Duplicate feature columns rejected
    def test_duplicate_feature_columns_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate column name detected"):
            MLPlan(**_make_valid_ml_plan_dict(feature_columns=["col_a", "col_a"]))

    # 77. Target in feature_columns rejected
    def test_target_in_features_rejected(self):
        with pytest.raises(ValidationError, match="cannot appear in feature_columns"):
            MLPlan(**_make_valid_ml_plan_dict(target_column="col_a", feature_columns=["col_a", "col_b"]))

    # 78. Empty model_candidates rejected
    def test_empty_candidates_rejected(self):
        with pytest.raises(ValidationError, match="model_candidates must contain at least one candidate"):
            MLPlan(**_make_valid_ml_plan_dict(model_candidates=[]))

    # 79. Duplicate model candidate IDs rejected
    def test_duplicate_candidates_rejected(self):
        cand1 = ModelCandidate(**_make_valid_model_candidate(candidate_id="model_x"))
        cand2 = ModelCandidate(**_make_valid_model_candidate(candidate_id="model_x"))
        with pytest.raises(ValidationError, match="Duplicate model candidate_id detected"):
            MLPlan(**_make_valid_ml_plan_dict(model_candidates=[cand1, cand2]))

    # 80. Duplicate preprocessing step IDs rejected
    def test_duplicate_prep_steps_rejected(self):
        step1 = PreprocessingStep(**_make_valid_preprocessing_step(step_id="prep_x"))
        step2 = PreprocessingStep(**_make_valid_preprocessing_step(step_id="prep_x"))
        with pytest.raises(ValidationError, match="Duplicate preprocessing step_id detected"):
            MLPlan(**_make_valid_ml_plan_dict(preprocessing_steps=[step1, step2]))

    # 81. Duplicate feature engineering step IDs rejected
    def test_duplicate_fe_steps_rejected(self):
        step1 = FeatureEngineeringStep(**_make_valid_feature_engineering_step(step_id="fe_x"))
        step2 = FeatureEngineeringStep(**_make_valid_feature_engineering_step(step_id="fe_x"))
        with pytest.raises(ValidationError, match="Duplicate feature engineering step_id detected"):
            MLPlan(**_make_valid_ml_plan_dict(feature_engineering_steps=[step1, step2]))

    # 82. ready + confirmation items rejected
    def test_ready_with_confirmation_items_rejected(self):
        item = MLPlanConfirmationItem(key="cv", question="Folds?", reason="validation")
        with pytest.raises(ValidationError, match="confirmation_items must be empty when status is 'ready'"):
            MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.READY, confirmation_items=[item]))

    # 83. needs_confirmation without confirmation items rejected
    def test_needs_confirmation_without_items_rejected(self):
        with pytest.raises(ValidationError, match="confirmation_items cannot be empty when status is 'needs_confirmation'"):
            MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.NEEDS_CONFIRMATION, confirmation_items=[]))

    # 84. blocked with or without confirmation items accepted
    def test_blocked_confirmation_items_any_accepted(self):
        p1 = MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.BLOCKED, confirmation_items=[]))
        assert p1.status == MLPlanStatus.BLOCKED

        item = MLPlanConfirmationItem(key="mem", question="High RAM?", reason="limit")
        p2 = MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.BLOCKED, confirmation_items=[item]))
        assert len(p2.confirmation_items) == 1

    # 85. draft with or without confirmation items accepted
    def test_draft_confirmation_items_any_accepted(self):
        p1 = MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.DRAFT, confirmation_items=[]))
        assert p1.status == MLPlanStatus.DRAFT

        item = MLPlanConfirmationItem(key="mem", question="High RAM?", reason="limit")
        p2 = MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.DRAFT, confirmation_items=[item]))
        assert len(p2.confirmation_items) == 1

    # 86. model_dump() works
    def test_model_dump_plan(self):
        plan = MLPlan(**_make_valid_ml_plan_dict())
        assert isinstance(plan.model_dump(), dict)

    # 87. model_dump_json() produces valid JSON
    def test_model_dump_json_plan(self):
        plan = MLPlan(**_make_valid_ml_plan_dict())
        assert isinstance(json.loads(plan.model_dump_json()), dict)

    # 88. All enums serialize as strings
    def test_enums_serialize_as_strings(self):
        plan = MLPlan(**_make_valid_ml_plan_dict(status=MLPlanStatus.DRAFT))
        parsed = json.loads(plan.model_dump_json())
        assert parsed["status"] == "draft"
        assert parsed["problem_type"] == "classification"
        assert parsed["split_plan"]["strategy"] == "random"

    # 89. Nested schemas serialize correctly
    def test_nested_schemas_serialize(self):
        plan = MLPlan(**_make_valid_ml_plan_dict())
        parsed = json.loads(plan.model_dump_json())
        assert isinstance(parsed["split_plan"], dict)
        assert isinstance(parsed["preprocessing_steps"][0], dict)
        assert isinstance(parsed["model_candidates"][0], dict)

    # 90. Mutable default lists are independent
    def test_independent_mutable_defaults(self):
        p1 = MLPlan(**_make_valid_ml_plan_dict())
        p2 = MLPlan(**_make_valid_ml_plan_dict())
        p1.preprocessing_steps.append(PreprocessingStep(**_make_valid_preprocessing_step(step_id="prep_new")))
        assert len(p1.preprocessing_steps) == 2
        assert len(p2.preprocessing_steps) == 1


class TestMLPlanTypeCoverage:
    """Extra tests to ensure all type-validation branch targets are covered."""

    def test_invalid_types_preprocessing_step(self):
        with pytest.raises(ValidationError, match="columns must be a list of strings"):
            PreprocessingStep(**_make_valid_preprocessing_step(columns="not-a-list"))
        with pytest.raises(ValidationError, match="at index 0 must be a string"):
            PreprocessingStep(**_make_valid_preprocessing_step(columns=[123]))
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            PreprocessingStep(**_make_valid_preprocessing_step(columns=["  "]))
        with pytest.raises(ValidationError, match="Field must be a string"):
            PreprocessingStep(**_make_valid_preprocessing_step(step_id=123))

    def test_invalid_types_feature_engineering_step(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            FeatureEngineeringStep(**_make_valid_feature_engineering_step(step_id=123))
        with pytest.raises(ValidationError, match="input_columns must be a list of strings"):
            FeatureEngineeringStep(**_make_valid_feature_engineering_step(input_columns="not-a-list"))
        with pytest.raises(ValidationError, match="Field cannot be empty or whitespace-only"):
            FeatureEngineeringStep(**_make_valid_feature_engineering_step(step_id="  "))

    def test_invalid_types_feature_selection_plan(self):
        with pytest.raises(ValidationError, match="reason must be a string"):
            FeatureSelectionPlan(**_make_valid_feature_selection_plan(reason=123))
        with pytest.raises(ValidationError, match="reason cannot be empty or whitespace-only"):
            FeatureSelectionPlan(**_make_valid_feature_selection_plan(reason="  "))

    def test_invalid_types_dataset_split_plan(self):
        with pytest.raises(ValidationError, match="Column name must be a string"):
            DatasetSplitPlan(**_make_valid_split_plan(stratify_column=123))
        with pytest.raises(ValidationError, match="Column name cannot be empty or whitespace-only"):
            DatasetSplitPlan(**_make_valid_split_plan(stratify_column=" "))

    def test_invalid_types_model_candidate(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            ModelCandidate(**_make_valid_model_candidate(candidate_id=123))

    def test_invalid_types_evaluation_plan(self):
        with pytest.raises(ValidationError, match="primary_metric must be a string"):
            EvaluationPlan(**_make_valid_evaluation_plan(primary_metric=123))
        with pytest.raises(ValidationError, match="secondary_metrics must be a list of strings"):
            EvaluationPlan(**_make_valid_evaluation_plan(secondary_metrics="not-a-list"))
        with pytest.raises(ValidationError, match="at index 0 must be a string"):
            EvaluationPlan(**_make_valid_evaluation_plan(secondary_metrics=[123]))
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            EvaluationPlan(**_make_valid_evaluation_plan(secondary_metrics=[" "]))

    def test_invalid_types_plan_warning(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            MLPlanWarning(code=123, message="msg")

    def test_invalid_types_confirmation_item(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            MLPlanConfirmationItem(key=123, question="?", reason="reason")

    def test_invalid_types_main_plan(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            MLPlan(**_make_valid_ml_plan_dict(plan_id=123))

