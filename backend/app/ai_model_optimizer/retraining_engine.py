"""Retraining Engine executing optimized MLPlans deterministically."""

from __future__ import annotations

import time
import uuid
import pandas as pd
from pydantic import BaseModel, ConfigDict

from backend.app.compute_capabilities.schemas import ComputeCapabilities
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.problem_definition.schemas import ProblemDefinition
from backend.app.ml_plan.schemas import MLPlan
from backend.app.ml_plan.validator import MLPlanValidator
from backend.app.ml_execution.orchestrator import MLExecutionOrchestrator, MLExecutionResult
from backend.app.ml_execution.execution_report import ExecutionReport, ExecutionReportBuilder


class RetrainingEngineError(Exception):
    """Raised when RetrainingEngine execution fails."""

    pass


class RetrainingResult(BaseModel):
    """Encapsulates retrained model metadata and evaluation artifacts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retraining_id: str
    optimized_plan_id: str
    execution_result: MLExecutionResult
    execution_report: ExecutionReport
    execution_duration: float


class RetrainingEngine:
    """Executes retraining on an optimized MLPlan deterministically."""

    def __init__(self) -> None:
        self._orchestrator = MLExecutionOrchestrator()
        self._report_builder = ExecutionReportBuilder()
        self._validator = MLPlanValidator()

    def retrain(
        self,
        *,
        dataframe: pd.DataFrame,
        dataset_context: DatasetContext,
        problem_definition: ProblemDefinition,
        compute_capabilities: ComputeCapabilities,
        optimized_plan: MLPlan,
    ) -> RetrainingResult:
        """Run retraining on the provided dataframe according to optimized_plan.

        Args:
            dataframe: Input pandas DataFrame.
            dataset_context: Context profile of the target dataset.
            problem_definition: Resolved problem definition settings.
            compute_capabilities: Detect system compute capabilities profile.
            optimized_plan: The optimized MLPlan to execute.

        Returns:
            A RetrainingResult enclosing execution reports and duration.

        Raises:
            RetrainingEngineError: On validation, identity, or execution failure.
        """
        # 1. Reject None inputs
        if dataframe is None:
            raise RetrainingEngineError("dataframe cannot be None")
        if dataset_context is None:
            raise RetrainingEngineError("dataset_context cannot be None")
        if problem_definition is None:
            raise RetrainingEngineError("problem_definition cannot be None")
        if compute_capabilities is None:
            raise RetrainingEngineError("compute_capabilities cannot be None")
        if optimized_plan is None:
            raise RetrainingEngineError("optimized_plan cannot be None")

        # 2. Reject wrong types
        if not isinstance(dataframe, pd.DataFrame):
            raise RetrainingEngineError("dataframe must be a pandas DataFrame")
        if not isinstance(dataset_context, DatasetContext):
            raise RetrainingEngineError("dataset_context must be a DatasetContext instance")
        if not isinstance(problem_definition, ProblemDefinition):
            raise RetrainingEngineError("problem_definition must be a ProblemDefinition instance")
        if not isinstance(compute_capabilities, ComputeCapabilities):
            raise RetrainingEngineError("compute_capabilities must be a ComputeCapabilities instance")
        if not isinstance(optimized_plan, MLPlan):
            raise RetrainingEngineError("optimized_plan must be an MLPlan instance")

        # 3. Identity checks
        # dataset_id check
        ds_id = dataset_context.basic_info.dataset_id
        if problem_definition.dataset_id != ds_id:
            raise RetrainingEngineError(
                f"Identity mismatch: problem_definition.dataset_id '{problem_definition.dataset_id}' "
                f"does not match dataset_context dataset_id '{ds_id}'"
            )
        if optimized_plan.dataset_id != ds_id:
            raise RetrainingEngineError(
                f"Identity mismatch: optimized_plan.dataset_id '{optimized_plan.dataset_id}' "
                f"does not match dataset_context dataset_id '{ds_id}'"
            )

        # request_id check
        if problem_definition.request_id != optimized_plan.request_id:
            raise RetrainingEngineError(
                f"Identity mismatch: problem_definition.request_id '{problem_definition.request_id}' "
                f"does not match optimized_plan request_id '{optimized_plan.request_id}'"
            )

        # definition_id check
        if problem_definition.definition_id != optimized_plan.problem_definition_id:
            raise RetrainingEngineError(
                f"Identity mismatch: problem_definition.definition_id '{problem_definition.definition_id}' "
                f"does not match optimized_plan problem_definition_id '{optimized_plan.problem_definition_id}'"
            )

        # compute_capability_id check
        if compute_capabilities.capability_id != optimized_plan.compute_capability_id:
            raise RetrainingEngineError(
                f"Identity mismatch: compute_capabilities.capability_id '{compute_capabilities.capability_id}' "
                f"does not match optimized_plan compute_capability_id '{optimized_plan.compute_capability_id}'"
            )

        # 4. Plan validation check
        val_res = self._validator.validate(
            optimized_plan,
            dataset_context,
            problem_definition,
            compute_capabilities,
        )
        if not val_res.is_valid:
            errors_str = "; ".join(f"[{e.severity.value}] {e.code}: {e.message}" for e in val_res.errors)
            raise RetrainingEngineError(f"Invalid optimized plan: {errors_str}")

        # Start timer
        start_time = time.perf_counter()

        # 5. Execute ML Execution Orchestrator
        try:
            exec_result = self._orchestrator.execute(
                dataframe=dataframe,
                dataset_context=dataset_context,
                plan=optimized_plan,
            )
        except Exception as e:
            raise RetrainingEngineError(f"ML Execution Orchestrator pipeline failure: {e}") from e

        # 6. Build Execution Report
        try:
            exec_report = self._report_builder.build(
                dataset_context=dataset_context,
                problem_definition=problem_definition,
                plan=optimized_plan,
                execution_result=exec_result,
            )
        except Exception as e:
            raise RetrainingEngineError(f"Execution Report Builder pipeline failure: {e}") from e

        duration = time.perf_counter() - start_time

        return RetrainingResult(
            retraining_id=f"retrain_{uuid.uuid4().hex[:8]}",
            optimized_plan_id=optimized_plan.plan_id,
            execution_result=exec_result,
            execution_report=exec_report,
            execution_duration=duration,
        )
