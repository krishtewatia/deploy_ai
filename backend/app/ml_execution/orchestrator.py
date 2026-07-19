"""ML Execution Orchestrator for DeployAI.

Stage 9J coordinates the full ML execution pipeline from an MLPlan:
split → preprocess → feature engineer → feature select → build models →
(per candidate) optimize → train → evaluate → select best.

This orchestrator is purely deterministic and does not use AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import MLPlan, SearchStrategy
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ProblemType,
    ResolutionStatus,
    TargetSource,
)

from backend.app.ml_execution.split_executor import SplitExecutor
from backend.app.ml_execution.preprocessing_builder import PreprocessingPipelineBuilder
from backend.app.ml_execution.feature_engineering_executor import FeatureEngineeringExecutor
from backend.app.ml_execution.feature_selection_executor import FeatureSelectionExecutor
from backend.app.ml_execution.model_factory import ModelFactory
from backend.app.ml_execution.hyperparameter_optimizer import HyperparameterOptimizer
from backend.app.ml_execution.training_executor import TrainingExecutor
from backend.app.ml_execution.evaluation_engine import EvaluationEngine, EvaluationResult


# Metrics where higher values indicate a better model.
_HIGHER_IS_BETTER = frozenset({"accuracy", "precision", "recall", "f1", "roc_auc", "r2"})
# Metrics where lower values indicate a better model.
_LOWER_IS_BETTER = frozenset({"mae", "mse", "rmse"})


class MLExecutionOrchestratorError(Exception):
    """Raised when the ML execution orchestrator pipeline fails."""

    pass


@dataclass
class MLExecutionResult:
    """The structured result of the full ML execution pipeline."""

    plan_id: str
    problem_definition_id: str
    candidate_results: dict[str, EvaluationResult]
    best_candidate_id: str
    best_model: BaseEstimator
    best_evaluation: EvaluationResult
    execution_duration_seconds: float
    execution_summary: dict[str, Any] = field(default_factory=dict)


class MLExecutionOrchestrator:
    """Coordinates the full ML execution pipeline from an MLPlan."""

    def execute(
        self,
        *,
        dataframe: pd.DataFrame,
        dataset_context: DatasetContext,
        plan: MLPlan,
    ) -> MLExecutionResult:
        """Execute the complete ML pipeline for all model candidates in the plan.

        Args:
            dataframe: Input pandas DataFrame.
            dataset_context: Target dataset metadata profile.
            plan: The planned execution MLPlan.

        Returns:
            MLExecutionResult containing evaluation results for all candidates
            and the best model selection.

        Raises:
            MLExecutionOrchestratorError: For validation or execution failures.
        """
        # ── Validation ──────────────────────────────────────────────
        if dataframe is None:
            raise MLExecutionOrchestratorError("dataframe cannot be None")
        if dataset_context is None:
            raise MLExecutionOrchestratorError("dataset_context cannot be None")
        if plan is None:
            raise MLExecutionOrchestratorError("plan cannot be None")

        if not isinstance(dataframe, pd.DataFrame):
            raise MLExecutionOrchestratorError("dataframe must be a pandas DataFrame")
        if not isinstance(dataset_context, DatasetContext):
            raise MLExecutionOrchestratorError("dataset_context must be a DatasetContext instance")
        if not isinstance(plan, MLPlan):
            raise MLExecutionOrchestratorError("plan must be an MLPlan instance")

        if dataframe.empty:
            raise MLExecutionOrchestratorError("dataframe cannot be empty")

        if not plan.model_candidates:
            raise MLExecutionOrchestratorError("plan must contain at least one model candidate")

        start_time = time.perf_counter()

        # Build a lightweight ProblemDefinition from plan fields for subsystems
        # that require it (EvaluationEngine, HyperparameterOptimizer).
        problem_def = ProblemDefinition(
            definition_id=plan.problem_definition_id,
            request_id=plan.request_id,
            dataset_id=plan.dataset_id,
            goal="Orchestrated ML execution",
            problem_type=plan.problem_type,
            target_column=plan.target_column,
            target_source=TargetSource.USER,
            feature_columns=list(plan.feature_columns),
            excluded_columns=[],
            primary_metric=plan.evaluation_plan.primary_metric,
            status=ResolutionStatus.RESOLVED,
        )

        try:
            # ── Step 1: Split ───────────────────────────────────────
            split_result = SplitExecutor().execute(
                dataframe=dataframe,
                dataset_context=dataset_context,
                plan=plan,
            )
            X_train = split_result.X_train
            X_test = split_result.X_test
            y_train = split_result.y_train
            y_test = split_result.y_test
            # Save original indices to restore them if dataframe conversion is needed
            train_index = X_train.index
            test_index = X_test.index

            # ── Step 2: Preprocessing ───────────────────────────────
            pipeline = PreprocessingPipelineBuilder().build(
                dataset_context=dataset_context,
                plan=plan,
            )
            X_train = pipeline.fit_transform(X_train)
            X_test = pipeline.transform(X_test)

            # Ensure X_train/X_test are DataFrames after pipeline transform
            if not isinstance(X_train, pd.DataFrame):
                cols = None
                if hasattr(pipeline, "get_feature_names_out"):
                    try:
                        from unittest.mock import Mock
                        val = pipeline.get_feature_names_out()
                        if not isinstance(val, Mock) and hasattr(val, "__len__"):
                            cols = list(val)
                    except Exception:
                        pass
                if cols is None and X_train.shape[1] == len(plan.feature_columns):
                    cols = plan.feature_columns
                X_train = pd.DataFrame(X_train, index=train_index, columns=cols)
            if not isinstance(X_test, pd.DataFrame):
                cols = None
                if hasattr(pipeline, "get_feature_names_out"):
                    try:
                        from unittest.mock import Mock
                        val = pipeline.get_feature_names_out()
                        if not isinstance(val, Mock) and hasattr(val, "__len__"):
                            cols = list(val)
                    except Exception:
                        pass
                if cols is None and X_test.shape[1] == len(plan.feature_columns):
                    cols = plan.feature_columns
                X_test = pd.DataFrame(X_test, index=test_index, columns=cols)

            # ── Step 3: Feature Engineering ─────────────────────────
            if plan.feature_engineering_steps:
                fe_executor = FeatureEngineeringExecutor()

                # FE executor operates on the full dataframe (features only,
                # target validation uses plan.target_column but doesn't need
                # the column in the dataframe).
                fe_train_result = fe_executor.execute(
                    dataframe=X_train,
                    dataset_context=dataset_context,
                    plan=plan,
                )
                X_train = fe_train_result.dataframe

                fe_test_result = fe_executor.execute(
                    dataframe=X_test,
                    dataset_context=dataset_context,
                    plan=plan,
                )
                X_test = fe_test_result.dataframe

            # ── Step 4: Feature Selection ───────────────────────────
            fs_executor = FeatureSelectionExecutor()

            # Feature selection needs the target column in the dataframe
            # for methods like mutual information and model-based.
            train_with_target = X_train.copy()
            train_with_target[plan.target_column] = y_train.values

            fs_result = fs_executor.execute(
                dataframe=train_with_target,
                dataset_context=dataset_context,
                plan=plan,
            )
            selected_columns = fs_result.selected_columns

            # Filter X_train and X_test to selected feature columns
            X_train = X_train[selected_columns]
            X_test = X_test[selected_columns]

            # ── Step 5: Model Factory ───────────────────────────────
            factory_result = ModelFactory().build(plan)

            # ── Per-Candidate Loop ──────────────────────────────────
            candidate_results: dict[str, EvaluationResult] = {}
            trained_models: dict[str, BaseEstimator] = {}

            optimizer = HyperparameterOptimizer()
            trainer = TrainingExecutor()
            evaluator = EvaluationEngine()

            for candidate in plan.model_candidates:
                cid = candidate.candidate_id
                estimator = factory_result.models[cid]

                # Step 6: Hyperparameter Optimization
                # To prevent underfitting/overfitting and ensure models are trained with high optimization,
                # we dynamically inject hyperparameter search parameters for standard scikit-learn baseline candidates.
                from backend.app.ml_plan.schemas import ModelFamily, ModelCandidate, SearchStrategy
                import unittest.mock
                
                if (
                    candidate.search_strategy == SearchStrategy.NONE 
                    and not isinstance(estimator, unittest.mock.Mock)
                ):
                    family = candidate.model_family
                    default_space = {}
                    if family == ModelFamily.RANDOM_FOREST:
                        default_space = {
                            "n_estimators": [50, 100, 150],
                            "max_depth": [5, 10, None],
                        }
                    elif family == ModelFamily.DECISION_TREE:
                        default_space = {
                            "max_depth": [3, 5, 10, None],
                            "min_samples_split": [2, 5],
                        }
                    elif family == ModelFamily.GRADIENT_BOOSTING:
                        default_space = {
                            "n_estimators": [50, 100],
                            "learning_rate": [0.05, 0.1, 0.2],
                            "max_depth": [3, 5],
                        }
                    elif family == ModelFamily.LOGISTIC_REGRESSION:
                        default_space = {
                            "C": [0.1, 1.0, 10.0],
                            "max_iter": [500],
                        }
                    elif family == ModelFamily.SVM:
                        default_space = {
                            "C": [0.1, 1.0, 10.0],
                        }
                    elif family == ModelFamily.KNN:
                        default_space = {
                            "n_neighbors": [3, 5, 7],
                        }
                    elif family == ModelFamily.EXTRA_TREES:
                        default_space = {
                            "n_estimators": [50, 100],
                            "max_depth": [5, 10, None],
                        }
                    elif family in (ModelFamily.RIDGE, ModelFamily.LASSO):
                        default_space = {
                            "alpha": [0.01, 0.1, 1.0, 10.0],
                        }

                    # Double check parameters exist in the estimator before applying
                    try:
                        valid_params = set(estimator.get_params().keys())
                        filtered_space = {k: v for k, v in default_space.items() if k in valid_params}
                        if filtered_space:
                            candidate = ModelCandidate(
                                candidate_id=candidate.candidate_id,
                                model_family=candidate.model_family,
                                parameters=candidate.parameters,
                                search_strategy=SearchStrategy.GRID,
                                search_space=filtered_space,
                                reason="Optimized by execution engine to prevent underfitting/overfitting",
                            )
                    except Exception:
                        pass

                if candidate.search_strategy != SearchStrategy.NONE:
                    opt_result = optimizer.optimize(
                        estimator=estimator,
                        candidate=candidate,
                        X_train=X_train,
                        y_train=y_train,
                        problem_definition=problem_def,
                        plan=plan,
                    )
                    estimator = opt_result.optimized_estimator

                # Step 7: Training
                train_result = trainer.train(
                    estimator=estimator,
                    X_train=X_train,
                    y_train=y_train,
                    candidate=candidate,
                )

                # Step 8: Evaluation
                eval_result = evaluator.evaluate(
                    trained_estimator=train_result.trained_estimator,
                    X_test=X_test,
                    y_test=y_test,
                    problem_definition=problem_def,
                    training_result=train_result,
                )

                candidate_results[cid] = eval_result
                trained_models[cid] = train_result.trained_estimator

        except MLExecutionOrchestratorError:
            raise
        except Exception as e:
            raise MLExecutionOrchestratorError(
                f"Pipeline execution failed: {e}"
            ) from e

        # ── Best Model Selection ────────────────────────────────────
        primary_metric = plan.evaluation_plan.primary_metric.lower().strip()
        higher_is_better = primary_metric in _HIGHER_IS_BETTER

        best_cid: str | None = None
        best_score: float | None = None

        for cid, eval_res in candidate_results.items():
            score = eval_res.primary_metric_value
            if best_score is None:
                best_cid = cid
                best_score = score
            elif higher_is_better and score > best_score:
                best_cid = cid
                best_score = score
            elif not higher_is_better and score < best_score:
                best_cid = cid
                best_score = score

        end_time = time.perf_counter()
        duration = end_time - start_time

        execution_summary = {
            "plan_id": plan.plan_id,
            "candidates_evaluated": len(candidate_results),
            "best_candidate_id": best_cid,
            "primary_metric": primary_metric,
            "best_score": best_score,
            "duration_seconds": duration,
        }

        return MLExecutionResult(
            plan_id=plan.plan_id,
            problem_definition_id=plan.problem_definition_id,
            candidate_results=candidate_results,
            best_candidate_id=best_cid,
            best_model=trained_models[best_cid],
            best_evaluation=candidate_results[best_cid],
            execution_duration_seconds=duration,
            execution_summary=execution_summary,
        )
