"""AI Planning Context Builder.

Converts upstream structured artifacts into a compact, JSON-serializable
context dictionary suitable for LLM consumption.
"""

from __future__ import annotations

from typing import Any

from backend.app.compute_capabilities.schemas import ComputeCapabilities
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import MLPlan
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import ProblemDefinition


class AIPlanningContextBuilder:
    """Builds a compact JSON-serializable context for AI planning prompts.

    The context preserves the information necessary for AI reasoning
    while avoiding irrelevant data, raw DataFrames, or secrets.
    """

    def build(
        self,
        *,
        dataset_context: DatasetContext,
        user_request: UserMLRequest,
        problem_definition: ProblemDefinition,
        compute_capabilities: ComputeCapabilities,
        baseline_plan: MLPlan,
    ) -> dict[str, Any]:
        """Build the planning context dictionary.

        Args:
            dataset_context: Analysed dataset metadata.
            user_request: User ML request preferences.
            problem_definition: Resolved problem definition.
            compute_capabilities: System compute constraints.
            baseline_plan: Deterministic baseline ML plan.

        Returns:
            A JSON-serializable dictionary containing all context sections.
        """
        return {
            "dataset": self._build_dataset_section(dataset_context),
            "user_goal": self._build_user_goal_section(user_request),
            "resolved_problem": self._build_problem_section(problem_definition),
            "compute_capabilities": self._build_compute_section(compute_capabilities),
            "baseline_plan": self._build_baseline_plan_section(baseline_plan),
        }

    def _build_dataset_section(self, ctx: DatasetContext) -> dict[str, Any]:
        """Build the dataset metadata section."""
        columns = []
        is_large_dataset = len(ctx.columns) > 15
        sample_limit = 2 if is_large_dataset else 5

        for col in ctx.columns:
            col_data: dict[str, Any] = {
                "name": col.name,
                "dtype": col.dtype,
                "is_numeric": col.is_numeric,
                "is_categorical": col.is_categorical,
                "is_datetime": col.is_datetime,
                "missing_percentage": col.missing_percentage,
                "unique_count": col.unique_count,
                "unique_percentage": col.unique_percentage,
                "sample_values": col.sample_values[:sample_limit] if col.sample_values else [],
            }
            if col.statistics is not None and not is_large_dataset:
                col_data["statistics"] = {
                    "mean": col.statistics.mean,
                    "median": col.statistics.median,
                    "std": col.statistics.std,
                    "min": col.statistics.min,
                    "max": col.statistics.max,
                }
            columns.append(col_data)

        return {
            "dataset_id": ctx.basic_info.dataset_id,
            "row_count": ctx.basic_info.row_count,
            "column_count": ctx.basic_info.column_count,
            "columns": columns,
        }

    def _build_user_goal_section(self, req: UserMLRequest) -> dict[str, Any]:
        """Build the user goal section."""
        return {
            "request_id": req.request_id,
            "goal": req.goal,
            "target_column": req.target_column,
            "problem_type_preference": req.problem_type.value,
            "primary_metric": req.primary_metric,
            "excluded_columns": req.excluded_columns,
            "additional_context": req.additional_context,
        }

    def _build_problem_section(self, pd: ProblemDefinition) -> dict[str, Any]:
        """Build the resolved problem section."""
        warnings_list = []
        for w in pd.warnings:
            warnings_list.append({"code": w.code, "message": w.message})

        return {
            "problem_definition_id": pd.definition_id,
            "target_column": pd.target_column,
            "problem_type": pd.problem_type.value,
            "feature_columns": pd.feature_columns,
            "excluded_columns": pd.excluded_columns,
            "primary_metric": pd.primary_metric,
            "warnings": warnings_list,
        }

    def _build_compute_section(self, cc: ComputeCapabilities) -> dict[str, Any]:
        """Build the compute capabilities section."""
        warnings_list = []
        for w in cc.warnings:
            warnings_list.append({"code": w.code, "message": w.message})

        return {
            "compute_capability_id": cc.capability_id,
            "compute_tier": cc.compute_tier.value,
            "memory_constraint": cc.memory_constraint.value,
            "safe_parallel_workers": cc.safe_parallel_workers,
            "gpu_acceleration_available": cc.gpu_acceleration_available,
            "accelerator_type": cc.accelerator_type.value,
            "warnings": warnings_list,
        }

    def _build_baseline_plan_section(self, plan: MLPlan) -> dict[str, Any]:
        """Build the baseline plan section."""
        preprocessing = []
        for step in plan.preprocessing_steps:
            preprocessing.append({
                "step_id": step.step_id,
                "operation": step.operation.value,
                "columns": step.columns,
                "parameters": step.parameters,
                "reason": step.reason,
            })

        feature_engineering = []
        for step in plan.feature_engineering_steps:
            feature_engineering.append({
                "step_id": step.step_id,
                "operation": step.operation.value,
                "input_columns": step.input_columns,
                "output_columns": step.output_columns,
                "parameters": step.parameters,
                "reason": step.reason,
            })

        feature_selection = {
            "method": plan.feature_selection.method.value,
            "candidate_columns": plan.feature_selection.candidate_columns,
            "max_features": plan.feature_selection.max_features,
            "reason": plan.feature_selection.reason,
        }

        split_plan = {
            "strategy": plan.split_plan.strategy.value,
            "test_size": plan.split_plan.test_size,
            "validation_size": plan.split_plan.validation_size,
            "random_state": plan.split_plan.random_state,
            "shuffle": plan.split_plan.shuffle,
            "stratify_column": plan.split_plan.stratify_column,
            "time_column": plan.split_plan.time_column,
        }

        model_candidates = []
        for cand in plan.model_candidates:
            model_candidates.append({
                "candidate_id": cand.candidate_id,
                "model_family": cand.model_family.value,
                "parameters": cand.parameters,
                "search_strategy": cand.search_strategy.value,
                "search_space": cand.search_space,
                "reason": cand.reason,
            })

        evaluation = {
            "primary_metric": plan.evaluation_plan.primary_metric,
            "secondary_metrics": plan.evaluation_plan.secondary_metrics,
            "cross_validation_folds": plan.evaluation_plan.cross_validation_folds,
        }

        execution = {
            "parallel_workers": plan.execution_constraints.parallel_workers,
            "use_gpu_acceleration": plan.execution_constraints.use_gpu_acceleration,
            "accelerator_type": plan.execution_constraints.accelerator_type.value,
            "compute_tier": plan.execution_constraints.compute_tier.value,
        }

        warnings = []
        for w in plan.warnings:
            warnings.append({"code": w.code, "message": w.message})

        return {
            "plan_id": plan.plan_id,
            "preprocessing_steps": preprocessing,
            "feature_engineering_steps": feature_engineering,
            "feature_selection": feature_selection,
            "split_plan": split_plan,
            "model_candidates": model_candidates,
            "evaluation_plan": evaluation,
            "execution_constraints": execution,
            "warnings": warnings,
        }
