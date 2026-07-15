"""Unit tests for MLExecutionOrchestrator."""

from __future__ import annotations

import copy
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
)
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)
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
)
from backend.app.ml_plan.baseline_planner import BaselineMLPlanner
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ResolutionStatus,
    TargetSource,
)
from backend.app.ml_execution.orchestrator import (
    MLExecutionOrchestrator,
    MLExecutionOrchestratorError,
    MLExecutionResult,
)
from backend.app.ml_execution.evaluation_engine import EvaluationResult


# ── Helper Builders ───────────────────────────────────────────────────


def _make_column_context(
    name: str,
    dtype: str = "float64",
    is_numeric: bool = True,
    is_categorical: bool = False,
    is_datetime: bool = False,
) -> ColumnContext:
    return ColumnContext(
        name=name,
        dtype=dtype,
        is_numeric=is_numeric,
        is_categorical=is_categorical,
        is_datetime=is_datetime,
        missing_count=0,
        missing_percentage=0.0,
        unique_count=50,
        unique_percentage=5.0,
        sample_values=[1.0, 2.0],
    )


def _make_dataset_context(
    dataset_id: str = "ds_01",
    columns: list[ColumnContext] | None = None,
) -> DatasetContext:
    if columns is None:
        columns = [
            _make_column_context("feat_a"),
            _make_column_context("feat_b"),
            _make_column_context(
                "target", dtype="int64", is_numeric=True, is_categorical=True,
            ),
        ]

    basic_info = DatasetBasicInfo(
        dataset_id=dataset_id,
        file_name="data.csv",
        row_count=1000,
        column_count=len(columns),
        memory_usage_bytes=50000,
    )
    return DatasetContext(
        basic_info=basic_info,
        columns=columns,
        missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
    )


def _make_classification_dataframe(rows: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "feat_a": np.random.randn(rows),
        "feat_b": np.random.randn(rows),
        "target": np.array([0, 1] * (rows // 2)),
    })


def _make_regression_dataframe(rows: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "feat_a": np.random.randn(rows),
        "feat_b": np.random.randn(rows),
        "target": np.random.randn(rows) * 10 + 50,
    })


def _make_classification_plan(
    candidates: list[ModelCandidate] | None = None,
    primary_metric: str = "accuracy",
) -> MLPlan:
    if candidates is None:
        candidates = [
            ModelCandidate(
                candidate_id="lr_01",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline",
            ),
        ]

    return MLPlan(
        plan_id="plan_01",
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        problem_type=ProblemType.CLASSIFICATION,
        target_column="target",
        feature_columns=["feat_a", "feat_b"],
        preprocessing_steps=[],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["feat_a", "feat_b"],
            max_features=None,
            reason="No selection",
        ),
        split_plan=DatasetSplitPlan(
            strategy=SplitStrategy.STRATIFIED,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify_column="target",
        ),
        model_candidates=candidates,
        evaluation_plan=EvaluationPlan(
            primary_metric=primary_metric,
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


def _make_regression_plan(
    candidates: list[ModelCandidate] | None = None,
    primary_metric: str = "mae",
) -> MLPlan:
    if candidates is None:
        candidates = [
            ModelCandidate(
                candidate_id="linreg_01",
                model_family=ModelFamily.LINEAR_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline",
            ),
        ]

    return MLPlan(
        plan_id="plan_02",
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        problem_type=ProblemType.REGRESSION,
        target_column="target",
        feature_columns=["feat_a", "feat_b"],
        preprocessing_steps=[],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["feat_a", "feat_b"],
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
            primary_metric=primary_metric,
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


class TestMLExecutionOrchestrator:
    """Tests covering the full ML execution orchestrator pipeline."""

    def test_classification_pipeline(self):
        """Verify full classification pipeline with LogisticRegression."""
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        plan = _make_classification_plan()

        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )

        assert isinstance(result, MLExecutionResult)
        assert result.plan_id == "plan_01"
        assert result.problem_definition_id == "pd_01"
        assert result.best_candidate_id == "lr_01"
        assert result.best_model is not None
        assert isinstance(result.best_evaluation, EvaluationResult)
        assert result.execution_duration_seconds > 0.0
        assert "lr_01" in result.candidate_results
        assert result.best_evaluation.primary_metric == "accuracy"

    def test_regression_pipeline(self):
        """Verify full regression pipeline with LinearRegression."""
        df = _make_regression_dataframe(100)
        context = _make_dataset_context()
        plan = _make_regression_plan()

        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )

        assert isinstance(result, MLExecutionResult)
        assert result.plan_id == "plan_02"
        assert result.best_candidate_id == "linreg_01"
        assert result.best_evaluation.primary_metric == "mae"
        assert "mae" in result.best_evaluation.all_metrics
        assert "mse" in result.best_evaluation.all_metrics
        assert "rmse" in result.best_evaluation.all_metrics
        assert "r2" in result.best_evaluation.all_metrics

    def test_multiple_candidates_higher_is_better(self):
        """Verify best model selection with higher-is-better metric (accuracy)."""
        candidates = [
            ModelCandidate(
                candidate_id="lr_01",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline 1",
            ),
            ModelCandidate(
                candidate_id="rf_01",
                model_family=ModelFamily.RANDOM_FOREST,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline 2",
            ),
        ]
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        plan = _make_classification_plan(candidates=candidates, primary_metric="accuracy")

        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )

        assert len(result.candidate_results) == 2
        assert result.best_candidate_id in ("lr_01", "rf_01")
        # Verify best was chosen by highest accuracy
        best_score = result.best_evaluation.primary_metric_value
        for cid, eval_res in result.candidate_results.items():
            assert eval_res.primary_metric_value <= best_score

    def test_multiple_candidates_lower_is_better(self):
        """Verify best model selection with lower-is-better metric (mae)."""
        candidates = [
            ModelCandidate(
                candidate_id="linreg_01",
                model_family=ModelFamily.LINEAR_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline 1",
            ),
            ModelCandidate(
                candidate_id="ridge_01",
                model_family=ModelFamily.RIDGE,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline 2",
            ),
        ]
        df = _make_regression_dataframe(100)
        context = _make_dataset_context()
        plan = _make_regression_plan(candidates=candidates, primary_metric="mae")

        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )

        assert len(result.candidate_results) == 2
        # Verify best was chosen by lowest mae
        best_score = result.best_evaluation.primary_metric_value
        for cid, eval_res in result.candidate_results.items():
            assert eval_res.primary_metric_value >= best_score

    def test_execution_summary_populated(self):
        """Verify execution_summary contains expected keys."""
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        plan = _make_classification_plan()

        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )

        summary = result.execution_summary
        assert "plan_id" in summary
        assert "candidates_evaluated" in summary
        assert "best_candidate_id" in summary
        assert "primary_metric" in summary
        assert "best_score" in summary
        assert "duration_seconds" in summary

    def test_subsystem_failure_propagation(self):
        """Verify subsystem failures are wrapped in MLExecutionOrchestratorError."""
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        plan = _make_classification_plan()

        with patch(
            "backend.app.ml_execution.orchestrator.SplitExecutor"
        ) as mock_split_cls:
            mock_split = MagicMock()
            mock_split.execute.side_effect = ValueError("Split boom")
            mock_split_cls.return_value = mock_split

            orchestrator = MLExecutionOrchestrator()
            with pytest.raises(MLExecutionOrchestratorError, match="Pipeline execution failed"):
                orchestrator.execute(
                    dataframe=df, dataset_context=context, plan=plan,
                )

    def test_validation_rejects_none(self):
        """Verify None inputs raise MLExecutionOrchestratorError."""
        orchestrator = MLExecutionOrchestrator()
        df = _make_classification_dataframe(10)
        ctx = _make_dataset_context()
        plan = _make_classification_plan()

        with pytest.raises(MLExecutionOrchestratorError, match="dataframe cannot be None"):
            orchestrator.execute(dataframe=None, dataset_context=ctx, plan=plan)

        with pytest.raises(MLExecutionOrchestratorError, match="dataset_context cannot be None"):
            orchestrator.execute(dataframe=df, dataset_context=None, plan=plan)

        with pytest.raises(MLExecutionOrchestratorError, match="plan cannot be None"):
            orchestrator.execute(dataframe=df, dataset_context=ctx, plan=None)

    def test_validation_rejects_wrong_types(self):
        """Verify wrong input types raise MLExecutionOrchestratorError."""
        orchestrator = MLExecutionOrchestrator()
        df = _make_classification_dataframe(10)
        ctx = _make_dataset_context()
        plan = _make_classification_plan()

        with pytest.raises(MLExecutionOrchestratorError, match="dataframe must be a pandas"):
            orchestrator.execute(dataframe="not-df", dataset_context=ctx, plan=plan)

        with pytest.raises(MLExecutionOrchestratorError, match="dataset_context must be a DatasetContext"):
            orchestrator.execute(dataframe=df, dataset_context="not-ctx", plan=plan)

        with pytest.raises(MLExecutionOrchestratorError, match="plan must be an MLPlan"):
            orchestrator.execute(dataframe=df, dataset_context=ctx, plan="not-plan")

    def test_validation_rejects_empty_dataframe(self):
        """Verify empty dataframe raises MLExecutionOrchestratorError."""
        orchestrator = MLExecutionOrchestrator()
        ctx = _make_dataset_context()
        plan = _make_classification_plan()

        with pytest.raises(MLExecutionOrchestratorError, match="dataframe cannot be empty"):
            orchestrator.execute(
                dataframe=pd.DataFrame(),
                dataset_context=ctx,
                plan=plan,
            )

    def test_validation_rejects_empty_candidates(self):
        """Verify empty model_candidates raises MLExecutionOrchestratorError."""
        orchestrator = MLExecutionOrchestrator()
        df = _make_classification_dataframe(10)
        ctx = _make_dataset_context()
        plan = _make_classification_plan(candidates=[
            ModelCandidate(
                candidate_id="lr_01",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline",
            ),
        ])
        # Force empty candidates after validation
        plan.model_candidates = []

        with pytest.raises(MLExecutionOrchestratorError, match="at least one model candidate"):
            orchestrator.execute(dataframe=df, dataset_context=ctx, plan=plan)

    def test_non_mutation(self):
        """Verify that execute does not mutate inputs."""
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        plan = _make_classification_plan()

        df_orig = df.copy()
        plan_id_orig = plan.plan_id
        ctx_id_orig = context.basic_info.dataset_id

        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )

        assert isinstance(result, MLExecutionResult)
        pd.testing.assert_frame_equal(df, df_orig)
        assert plan.plan_id == plan_id_orig
        assert context.basic_info.dataset_id == ctx_id_orig

    def test_rmse_lower_is_better(self):
        """Verify rmse metric uses lower-is-better selection."""
        candidates = [
            ModelCandidate(
                candidate_id="linreg_01",
                model_family=ModelFamily.LINEAR_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline 1",
            ),
            ModelCandidate(
                candidate_id="ridge_01",
                model_family=ModelFamily.RIDGE,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline 2",
            ),
        ]
        df = _make_regression_dataframe(100)
        context = _make_dataset_context()
        plan = _make_regression_plan(candidates=candidates, primary_metric="rmse")

        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )

        best_score = result.best_evaluation.primary_metric_value
        for cid, eval_res in result.candidate_results.items():
            assert eval_res.primary_metric_value >= best_score

    def test_f1_higher_is_better(self):
        """Verify f1 metric uses higher-is-better selection."""
        candidates = [
            ModelCandidate(
                candidate_id="lr_01",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline 1",
            ),
            ModelCandidate(
                candidate_id="rf_01",
                model_family=ModelFamily.RANDOM_FOREST,
                search_strategy=SearchStrategy.NONE,
                reason="Provides baseline 2",
            ),
        ]
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        plan = _make_classification_plan(candidates=candidates, primary_metric="f1")

        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )

        best_score = result.best_evaluation.primary_metric_value
        for cid, eval_res in result.candidate_results.items():
            assert eval_res.primary_metric_value <= best_score

    def test_pipeline_with_feature_engineering_and_selection_and_tuning(self):
        """Verify pipeline executes feature engineering, feature selection, and hyperparameter tuning."""
        from backend.app.ml_plan.schemas import (
            FeatureEngineeringStep,
            FeatureEngineeringOperation,
            FeatureSelectionPlan,
            FeatureSelectionMethod,
        )
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        
        candidates = [
            ModelCandidate(
                candidate_id="lr_01",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                search_strategy=SearchStrategy.GRID,
                search_space={"C": [0.1, 1.0]},
                reason="Tuned logistic regression",
            ),
        ]
        
        plan = _make_classification_plan(candidates=candidates)
        plan.feature_engineering_steps = [
            FeatureEngineeringStep(
                step_id="step_interaction",
                operation=FeatureEngineeringOperation.INTERACTION,
                input_columns=["feat_a", "feat_b"],
                output_columns=["feat_a_mult_feat_b"],
                reason="Interaction step",
            )
        ]
        plan.feature_selection = FeatureSelectionPlan(
            method=FeatureSelectionMethod.VARIANCE_THRESHOLD,
            candidate_columns=["feat_a", "feat_b", "feat_a_mult_feat_b"],
            max_features=None,
            parameters={"threshold": 0.01},
            reason="Select high variance features",
        )
        
        orchestrator = MLExecutionOrchestrator()
        result = orchestrator.execute(
            dataframe=df, dataset_context=context, plan=plan,
        )
        
        assert isinstance(result, MLExecutionResult)
        assert result.best_candidate_id == "lr_01"
        assert result.best_evaluation.primary_metric_value > 0.0

    def test_orchestrator_raises_explicit_orchestrator_error(self):
        """Verify explicit MLExecutionOrchestratorError is propagated directly without wrapping."""
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        plan = _make_classification_plan()

        with patch(
            "backend.app.ml_execution.orchestrator.SplitExecutor"
        ) as mock_split_cls:
            mock_split = MagicMock()
            mock_split.execute.side_effect = MLExecutionOrchestratorError("Direct orchestrator failure")
            mock_split_cls.return_value = mock_split

            orchestrator = MLExecutionOrchestrator()
            with pytest.raises(MLExecutionOrchestratorError, match="Direct orchestrator failure"):
                orchestrator.execute(
                    dataframe=df, dataset_context=context, plan=plan,
                )

    def test_preprocessing_pipeline_builder_non_dataframe(self):
        """Verify orchestrator converts non-dataframe preprocessing outputs to dataframes."""
        df = _make_classification_dataframe(100)
        context = _make_dataset_context()
        plan = _make_classification_plan()

        # Case 1: get_feature_names_out returns a list of column names
        mock_pipeline_1 = MagicMock()
        mock_pipeline_1.fit_transform.return_value = np.random.randn(80, 2)
        mock_pipeline_1.transform.return_value = np.random.randn(20, 2)
        mock_pipeline_1.get_feature_names_out.return_value = ["feat_a", "feat_b"]

        # Case 2: get_feature_names_out raises an exception
        mock_pipeline_2 = MagicMock()
        mock_pipeline_2.fit_transform.return_value = np.random.randn(80, 2)
        mock_pipeline_2.transform.return_value = np.random.randn(20, 2)
        mock_pipeline_2.get_feature_names_out.side_effect = RuntimeError("Failed to get names")

        for mock_pipe in (mock_pipeline_1, mock_pipeline_2):
            with patch(
                "backend.app.ml_execution.orchestrator.PreprocessingPipelineBuilder"
            ) as mock_builder_cls:
                mock_builder = MagicMock()
                mock_builder.build.return_value = mock_pipe
                mock_builder_cls.return_value = mock_builder

                orchestrator = MLExecutionOrchestrator()
                result = orchestrator.execute(
                    dataframe=df, dataset_context=context, plan=plan,
                )
                assert isinstance(result, MLExecutionResult)

    def test_lower_is_better_metric_selection(self):
        """Verify selection logic for lower-is-better metric when second candidate is better."""
        candidates = [
            ModelCandidate(
                candidate_id="cand_1",
                model_family=ModelFamily.LINEAR_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="c1",
            ),
            ModelCandidate(
                candidate_id="cand_2",
                model_family=ModelFamily.RIDGE,
                search_strategy=SearchStrategy.NONE,
                reason="c2",
            ),
        ]
        df = _make_regression_dataframe(100)
        context = _make_dataset_context()
        plan = _make_regression_plan(candidates=candidates, primary_metric="mae")

        eval_res_1 = MagicMock(spec=EvaluationResult)
        eval_res_1.primary_metric_value = 10.0
        eval_res_1.primary_metric = "mae"

        eval_res_2 = MagicMock(spec=EvaluationResult)
        eval_res_2.primary_metric_value = 5.0
        eval_res_2.primary_metric = "mae"

        with patch(
            "backend.app.ml_execution.orchestrator.EvaluationEngine"
        ) as mock_eval_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate.side_effect = [eval_res_1, eval_res_2]
            mock_eval_cls.return_value = mock_eval

            orchestrator = MLExecutionOrchestrator()
            result = orchestrator.execute(
                dataframe=df, dataset_context=context, plan=plan,
            )
            assert result.best_candidate_id == "cand_2"
            assert result.best_evaluation.primary_metric_value == 5.0

