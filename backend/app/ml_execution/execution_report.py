"""Execution Report schemas and Builder for DeployAI.

Stage 10A aggregates all ML execution artifacts, metrics, and plan configurations
into a single immutable report structure for downstream AI analysis.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_execution.orchestrator import MLExecutionResult
from backend.app.ml_plan import MLPlan, ModelFamily, SearchStrategy
from backend.app.problem_definition.schemas import ProblemDefinition, ProblemType


class ExecutionReportBuilderError(Exception):
    """Raised when execution report building fails due to validation errors or input mismatches."""

    pass


class CandidateSummary(BaseModel):
    """Summarized execution details for a single trained model candidate."""

    model_config = ConfigDict(use_enum_values=True)

    candidate_id: str = Field(..., description="Unique algorithm candidate ID.")
    model_family: ModelFamily = Field(..., description="Classical learning algorithm family.")
    primary_metric: str = Field(..., description="Evaluation metric name.")
    primary_metric_value: float = Field(..., description="Metric score value.")
    training_duration: float = Field(..., description="Training time in seconds.")
    evaluation_duration: float = Field(..., description="Evaluation time in seconds.")
    search_strategy: SearchStrategy = Field(..., description="Tuning search strategy.")
    best_parameters: Dict[str, Any] = Field(default_factory=dict, description="Best parameters of the model.")


class ChampionSummary(BaseModel):
    """Summarized execution details for the selected champion model candidate."""

    model_config = ConfigDict(use_enum_values=True)

    candidate_id: str = Field(..., description="Unique algorithm candidate ID.")
    model_family: ModelFamily = Field(..., description="Classical learning algorithm family.")
    primary_metric: str = Field(..., description="Evaluation metric name.")
    primary_metric_value: float = Field(..., description="Metric score value.")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importances dictionary.")
    training_duration: float = Field(..., description="Training time in seconds.")
    evaluation_duration: float = Field(..., description="Evaluation time in seconds.")


class ExecutionReport(BaseModel):
    """Consolidated execution report containing all metadata and results from the run."""

    model_config = ConfigDict(use_enum_values=True)

    report_id: str = Field(..., description="Unique execution report identifier.")
    dataset_id: str = Field(..., description="Linkage ID to DatasetContext.")
    request_id: str = Field(..., description="Linkage ID to UserMLRequest.")
    problem_definition_id: str = Field(..., description="Linkage ID to ProblemDefinition.")
    plan_id: str = Field(..., description="Linkage ID to MLPlan.")
    execution_id: str = Field(..., description="Unique run identifier.")
    problem_type: ProblemType = Field(..., description="Type of target ML workload.")
    target_column: str = Field(..., description="Dependent variable target column name.")
    feature_columns: List[str] = Field(..., description="List of feature columns.")
    compute_tier: ComputeTier = Field(..., description="Compute resource tier snapshot.")
    accelerator_type: AcceleratorType = Field(..., description="Hardware accelerator vendor type.")
    candidate_summaries: List[CandidateSummary] = Field(..., description="Summarized details of all candidates.")
    champion_summary: ChampionSummary = Field(..., description="Details of the selected best candidate.")
    training_summary: Dict[str, Any] = Field(..., description="Aggregated training summary metadata.")
    evaluation_summary: Dict[str, Any] = Field(..., description="Consolidated evaluation summary metadata.")
    warnings: List[str] = Field(default_factory=list, description="Warnings consolidated across execution.")
    execution_duration: float = Field(..., description="Total execution time in seconds.")
    created_timestamp: str = Field(..., description="Report creation UTC timestamp (ISO format).")


class ExecutionReportBuilder:
    """Consolidates execution artifacts and inputs into a single immutable report object."""

    def build(
        self,
        *,
        dataset_context: DatasetContext,
        problem_definition: ProblemDefinition,
        plan: MLPlan,
        execution_result: MLExecutionResult,
    ) -> ExecutionReport:
        """Construct a validated, consolidated ExecutionReport.

        Args:
            dataset_context: Target dataset metadata profile.
            problem_definition: Resolved ML problem definition contract.
            plan: The planned execution MLPlan.
            execution_result: Result from ML Execution Orchestrator.

        Returns:
            An ExecutionReport containing candidate summaries, champion metadata, and diagnostics.

        Raises:
            ExecutionReportBuilderError: If validation fails, identity mismatch is found,
                                        or candidate details are missing.
        """
        # 1. Reject None inputs
        if dataset_context is None:
            raise ExecutionReportBuilderError("dataset_context cannot be None")
        if problem_definition is None:
            raise ExecutionReportBuilderError("problem_definition cannot be None")
        if plan is None:
            raise ExecutionReportBuilderError("plan cannot be None")
        if execution_result is None:
            raise ExecutionReportBuilderError("execution_result cannot be None")

        # 2. Reject wrong input types
        if not isinstance(dataset_context, DatasetContext):
            raise ExecutionReportBuilderError("dataset_context must be a DatasetContext instance")
        if not isinstance(problem_definition, ProblemDefinition):
            raise ExecutionReportBuilderError("problem_definition must be a ProblemDefinition instance")
        if not isinstance(plan, MLPlan):
            raise ExecutionReportBuilderError("plan must be an MLPlan instance")
        if not isinstance(execution_result, MLExecutionResult):
            raise ExecutionReportBuilderError("execution_result must be an MLExecutionResult instance")

        # 3. Identity consistency checks
        # dataset_id check
        if dataset_context.basic_info.dataset_id != problem_definition.dataset_id:
            raise ExecutionReportBuilderError("dataset_id mismatch between DatasetContext and ProblemDefinition")
        if dataset_context.basic_info.dataset_id != plan.dataset_id:
            raise ExecutionReportBuilderError("dataset_id mismatch between DatasetContext and MLPlan")
        
        # request_id check
        if problem_definition.request_id != plan.request_id:
            raise ExecutionReportBuilderError("request_id mismatch between ProblemDefinition and MLPlan")

        # problem_definition_id check
        if problem_definition.definition_id != plan.problem_definition_id:
            raise ExecutionReportBuilderError("problem_definition_id mismatch between ProblemDefinition and MLPlan")
        if problem_definition.definition_id != execution_result.problem_definition_id:
            raise ExecutionReportBuilderError("problem_definition_id mismatch between ProblemDefinition and MLExecutionResult")

        # plan_id check
        if plan.plan_id != execution_result.plan_id:
            raise ExecutionReportBuilderError("plan_id mismatch between MLPlan and MLExecutionResult")

        # 4. Check for missing candidate list
        if not plan.model_candidates:
            raise ExecutionReportBuilderError("plan.model_candidates list cannot be empty")
        if not execution_result.candidate_results:
            raise ExecutionReportBuilderError("execution_result.candidate_results cannot be empty")

        # 5. Check for missing champion
        best_cid = execution_result.best_candidate_id
        if not best_cid:
            raise ExecutionReportBuilderError("best_candidate_id in execution_result cannot be empty or None")
        if best_cid not in execution_result.candidate_results:
            raise ExecutionReportBuilderError(f"best_candidate_id '{best_cid}' not found in candidate results")

        # Ensure all candidate IDs in plan are evaluated
        for cand in plan.model_candidates:
            if cand.candidate_id not in execution_result.candidate_results:
                raise ExecutionReportBuilderError(
                    f"Candidate ID '{cand.candidate_id}' in plan is missing from execution_result.candidate_results"
                )

        # 6. Extract candidate summaries
        candidate_summaries = []
        for cand in plan.model_candidates:
            cid = cand.candidate_id
            eval_res = execution_result.candidate_results[cid]
            candidate_summaries.append(
                CandidateSummary(
                    candidate_id=cid,
                    model_family=cand.model_family,
                    primary_metric=eval_res.primary_metric,
                    primary_metric_value=eval_res.primary_metric_value,
                    training_duration=eval_res.training_duration,
                    evaluation_duration=eval_res.evaluation_duration,
                    search_strategy=cand.search_strategy,
                    best_parameters=eval_res.model_parameters,
                )
            )

        # 7. Extract champion summary
        best_eval = execution_result.best_evaluation
        # Double check candidate exists for best_cid to retrieve model_family
        best_candidate = next((c for c in plan.model_candidates if c.candidate_id == best_cid), None)
        if best_candidate is None:
            raise ExecutionReportBuilderError(f"Best candidate ID '{best_cid}' not found in plan.model_candidates")

        champion_summary = ChampionSummary(
            candidate_id=best_cid,
            model_family=best_candidate.model_family,
            primary_metric=best_eval.primary_metric,
            primary_metric_value=best_eval.primary_metric_value,
            feature_importance=best_eval.feature_importance,
            training_duration=best_eval.training_duration,
            evaluation_duration=best_eval.evaluation_duration,
        )

        # 8. Consolidate warnings list
        consolidated_warnings = []
        # Add warnings from problem_definition
        for w in problem_definition.warnings:
            consolidated_warnings.append(f"ProblemDefinition warning: {w.message}")
        # Add warnings from plan
        for w in plan.warnings:
            consolidated_warnings.append(f"MLPlan warning: {w.message}")
        # Add warnings from candidate evaluations
        for cid, eval_res in execution_result.candidate_results.items():
            for w in eval_res.warnings:
                consolidated_warnings.append(f"Candidate {cid} warning: {w}")

        # 9. Create training and evaluation summaries
        all_candidate_ids = [cand.candidate_id for cand in plan.model_candidates]
        
        training_summary = {
            "total_candidates": len(plan.model_candidates),
            "candidate_ids": all_candidate_ids,
            "trained_successfully": True,
            "best_candidate_id": best_cid,
        }

        evaluation_summary = {
            "evaluated_candidates": len(execution_result.candidate_results),
            "primary_metric": plan.evaluation_plan.primary_metric,
            "best_metric_value": best_eval.primary_metric_value,
        }

        # 10. Construct final ExecutionReport
        report_id = f"report_{uuid.uuid4().hex}"
        execution_id = getattr(execution_result, "execution_id", f"exec_{uuid.uuid4().hex}")
        created_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

        return ExecutionReport(
            report_id=report_id,
            dataset_id=dataset_context.basic_info.dataset_id,
            request_id=plan.request_id,
            problem_definition_id=problem_definition.definition_id,
            plan_id=plan.plan_id,
            execution_id=execution_id,
            problem_type=problem_definition.problem_type,
            target_column=plan.target_column,
            feature_columns=list(plan.feature_columns),
            compute_tier=plan.execution_constraints.compute_tier,
            accelerator_type=plan.execution_constraints.accelerator_type,
            candidate_summaries=candidate_summaries,
            champion_summary=champion_summary,
            training_summary=training_summary,
            evaluation_summary=evaluation_summary,
            warnings=consolidated_warnings,
            execution_duration=execution_result.execution_duration_seconds,
            created_timestamp=created_timestamp,
        )
