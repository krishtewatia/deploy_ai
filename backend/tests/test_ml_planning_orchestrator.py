"""Tests for the High-level ML Planning Orchestrator (Stage 7E)."""

import json
import pytest
import uuid
import copy
from unittest.mock import patch

from backend.app.ai_planning import (
    AIDecisionConfidence,
    AIDecisionProposal,
    AIEvaluationProposal,
    AIAssistedPlanningResult,
)
from backend.app.ai_planning.providers import AIProvider, AIProviderError
from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
)
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    ColumnStatistics,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)
from backend.app.ml_plan import (
    MLPlan,
    MLPlanValidator,
    PlanningMode,
    MLPlanningResult,
    MLPlanningOrchestrator,
    MLPlanningOrchestratorError,
    BaselineMLPlanner,
)
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition import ProblemResolver, ProblemResolverError
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ProblemType,
    ResolutionStatus,
    TargetSource,
)


# ── Fakes and Mocks ────────────────────────────────────────────────────


class FakeAIProvider(AIProvider):
    """Stub AI provider for orchestrator tests."""

    def __init__(self, response_text: str = "{}", should_raise: bool = False) -> None:
        self.response_text = response_text
        self.should_raise = should_raise
        self.call_count = 0

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        if self.should_raise:
            raise AIProviderError("AI provider error.")
        return self.response_text


class MockProblemResolver(ProblemResolver):
    """Mock problem resolver that can simulate failures."""

    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    def resolve(self, *, dataset_context: DatasetContext, user_request: UserMLRequest) -> ProblemDefinition:
        if self.should_fail:
            raise ProblemResolverError("Resolver failed manually.")
        return super().resolve(dataset_context=dataset_context, user_request=user_request)


# ── Fixtures ───────────────────────────────────────────────────────────


def _make_dataset_context() -> DatasetContext:
    columns = [
        ColumnContext(
            name="age", dtype="float64", is_numeric=True, is_categorical=False,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=50, unique_percentage=5.0, sample_values=[25.0, 30.0],
            statistics=ColumnStatistics(mean=35.0, median=34.0, std=10.0, min=18.0, max=80.0),
        ),
        ColumnContext(
            name="churn", dtype="int64", is_numeric=True, is_categorical=True,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=2, unique_percentage=0.2, sample_values=[0, 1],
        ),
    ]
    return DatasetContext(
        basic_info=DatasetBasicInfo(dataset_id="ds_01", file_name="data.csv",
                                    row_count=1000, column_count=2, memory_usage_bytes=10000),
        columns=columns,
        missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
    )


def _make_user_request() -> UserMLRequest:
    return UserMLRequest(
        request_id="req_01", goal="Predict churn",
        target_column="churn", additional_context=None,
    )


def _make_compute_capabilities() -> ComputeCapabilities:
    return ComputeCapabilities(
        capability_id="cap_01", hardware_profile_id="hw_01",
        compute_tier=ComputeTier.STANDARD, memory_constraint=MemoryConstraintLevel.MODERATE,
        cpu_training_available=True, gpu_acceleration_available=False,
        accelerator_type=AcceleratorType.NONE, safe_parallel_workers=4,
        max_parallel_workers=8, available_ram_mb_snapshot=4096, total_ram_mb=8192,
        warnings=[],
    )


def _make_valid_no_change_response(plan: MLPlan) -> str:
    return json.dumps({
        "proposal_set_id": "ps_01",
        "baseline_plan_id": plan.plan_id,
        "dataset_id": plan.dataset_id,
        "request_id": plan.request_id,
        "problem_definition_id": plan.problem_definition_id,
        "compute_capability_id": plan.compute_capability_id,
        "summary": "No changes proposed.",
    })


def _make_valid_change_response(plan: MLPlan) -> str:
    return json.dumps({
        "proposal_set_id": "ps_01",
        "baseline_plan_id": plan.plan_id,
        "dataset_id": plan.dataset_id,
        "request_id": plan.request_id,
        "problem_definition_id": plan.problem_definition_id,
        "compute_capability_id": plan.compute_capability_id,
        "summary": "Change folds.",
        "evaluation_proposal": {
            "primary_metric": None,
            "secondary_metrics": [],
            "cross_validation_folds": 3,
            "reason": "Reduce folds.",
            "confidence": "high",
        }
    })


# ── Tests ──────────────────────────────────────────────────────────────


class TestPlanningModeAndResult:
    """1-7: PlanningMode and MLPlanningResult Schema tests."""

    def test_planning_mode_values(self):
        assert PlanningMode.DETERMINISTIC.value == "deterministic"
        assert PlanningMode.AI_ASSISTED.value == "ai_assisted"

    def test_planning_mode_json_serialization(self):
        assert json.dumps(PlanningMode.DETERMINISTIC) == '"deterministic"'
        assert json.dumps(PlanningMode.AI_ASSISTED) == '"ai_assisted"'

    def test_invalid_planning_id_rejected(self):
        with pytest.raises(Exception):
            MLPlanningResult(
                planning_id="  ",
                mode=PlanningMode.DETERMINISTIC,
                problem_definition=None,  # type: ignore
                baseline_plan=None,  # type: ignore
                final_plan=None,  # type: ignore
                ai_assistance_used=False,
                ai_changes_applied=False,
                validation_result=None,  # type: ignore
            )


class TestDeterministicMode:
    """8-16: Deterministic mode orchestration workflow tests."""

    def test_successful_deterministic_flow(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        result = orchestrator.create_plan(
            dataset_context=ds,
            user_request=req,
            compute_capabilities=cc,
            mode=PlanningMode.DETERMINISTIC,
        )

        assert isinstance(result, MLPlanningResult)
        assert result.planning_id.startswith("planning_")
        assert result.mode == PlanningMode.DETERMINISTIC
        assert isinstance(result.problem_definition, ProblemDefinition)
        assert isinstance(result.baseline_plan, MLPlan)
        assert result.final_plan == result.baseline_plan
        assert result.ai_assistance_used is False
        assert result.ai_changes_applied is False
        assert result.ai_result is None
        assert result.validation_result.is_valid is True

    def test_deterministic_flow_ignores_supplied_provider(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()
        provider = FakeAIProvider()

        result = orchestrator.create_plan(
            dataset_context=ds,
            user_request=req,
            compute_capabilities=cc,
            mode=PlanningMode.DETERMINISTIC,
            ai_provider=provider,
        )

        assert result.ai_assistance_used is False
        assert provider.call_count == 0


class TestAIAssistedMode:
    """17-26: AI-assisted mode orchestration workflow tests."""

    def test_missing_provider_raises_error(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        with pytest.raises(MLPlanningOrchestratorError, match="AI provider is required"):
            orchestrator.create_plan(
                dataset_context=ds,
                user_request=req,
                compute_capabilities=cc,
                mode=PlanningMode.AI_ASSISTED,
                ai_provider=None,
            )

    @patch("uuid.uuid4")
    def test_ai_assisted_success_no_changes(self, mock_uuid):
        fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        mock_uuid.return_value = fixed_uuid

        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        # Build predicted baseline plan to match provider
        baseline_plan = orchestrator._baseline_planner.create_plan(
            dataset_context=ds,
            user_request=req,
            problem_definition=orchestrator._problem_resolver.resolve(dataset_context=ds, user_request=req),
            compute_capabilities=cc,
        )

        provider = FakeAIProvider(response_text=_make_valid_no_change_response(baseline_plan))

        result = orchestrator.create_plan(
            dataset_context=ds,
            user_request=req,
            compute_capabilities=cc,
            mode=PlanningMode.AI_ASSISTED,
            ai_provider=provider,
        )

        assert result.ai_assistance_used is True
        assert result.ai_changes_applied is False
        assert provider.call_count == 1
        assert result.final_plan.evaluation_plan.cross_validation_folds == baseline_plan.evaluation_plan.cross_validation_folds
        assert result.baseline_plan == baseline_plan
        assert result.ai_result is not None
        assert result.ai_result.applied is False

    @patch("uuid.uuid4")
    def test_ai_assisted_success_with_changes(self, mock_uuid):
        fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        mock_uuid.return_value = fixed_uuid

        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        baseline_plan = orchestrator._baseline_planner.create_plan(
            dataset_context=ds,
            user_request=req,
            problem_definition=orchestrator._problem_resolver.resolve(dataset_context=ds, user_request=req),
            compute_capabilities=cc,
        )

        provider = FakeAIProvider(response_text=_make_valid_change_response(baseline_plan))

        # Check immutability baseline plan reference comparison
        result = orchestrator.create_plan(
            dataset_context=ds,
            user_request=req,
            compute_capabilities=cc,
            mode=PlanningMode.AI_ASSISTED,
            ai_provider=provider,
        )

        assert result.ai_assistance_used is True
        assert result.ai_changes_applied is True
        assert result.final_plan.evaluation_plan.cross_validation_folds == 3
        # Baseline plan in result remains unchanged (original folds was 5)
        assert result.baseline_plan.evaluation_plan.cross_validation_folds == 5
        assert result.ai_result is not None
        assert result.ai_result.applied is True


class TestErrorHandling:
    """27-31: Boundary error wrapping and fallbacks tests."""

    def test_problem_resolver_failure_wrapped(self):
        resolver = MockProblemResolver(should_fail=True)
        orchestrator = MLPlanningOrchestrator(problem_resolver=resolver)
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        with pytest.raises(MLPlanningOrchestratorError, match="Problem resolution failed"):
            orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)

    def test_baseline_planner_failure_wrapped(self):
        class FailingPlanner(BaselineMLPlanner):
            def create_plan(self, **kwargs):
                raise ValueError("Planner crashed.")

        orchestrator = MLPlanningOrchestrator(baseline_planner=FailingPlanner())
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        with pytest.raises(MLPlanningOrchestratorError, match="Baseline planning failed"):
            orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)

    def test_ai_planning_failure_wrapped_no_fallback(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()
        provider = FakeAIProvider(should_raise=True)

        with pytest.raises(MLPlanningOrchestratorError, match="AI-assisted planning failed"):
            orchestrator.create_plan(
                dataset_context=ds,
                user_request=req,
                compute_capabilities=cc,
                mode=PlanningMode.AI_ASSISTED,
                ai_provider=provider,
            )

    def test_final_validation_failure_raises_orchestrator_error(self):
        class FailingValidator(MLPlanValidator):
            def validate(self, **kwargs):
                from backend.app.ml_plan.validator import MLPlanValidationResult
                return MLPlanValidationResult(
                    plan_id="plan_01",
                    is_valid=False,
                    errors=[{"code": "FORCED_ERR", "message": "Manual err.", "severity": "error"}],
                    warnings=[]
                )

        orchestrator = MLPlanningOrchestrator(validator=FailingValidator())
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        with pytest.raises(MLPlanningOrchestratorError, match="Final MLPlan validation failed"):
            orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)

    def test_final_validation_unexpected_crash_wrapped(self):
        class CrashingValidator(MLPlanValidator):
            def validate(self, **kwargs):
                raise RuntimeError("Validation code crashed.")

        orchestrator = MLPlanningOrchestrator(validator=CrashingValidator())
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        with pytest.raises(MLPlanningOrchestratorError, match="Final MLPlan validation failed: Unexpected error"):
            orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)


class TestIdentityConsistency:
    """32-35: Linkage and identity checks."""

    def test_identities_preserved(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        result = orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)

        assert result.problem_definition.dataset_id == ds.basic_info.dataset_id
        assert result.baseline_plan.dataset_id == ds.basic_info.dataset_id
        assert result.final_plan.dataset_id == ds.basic_info.dataset_id
        assert result.baseline_plan.request_id == req.request_id
        assert result.final_plan.request_id == req.request_id


class TestOrchestrationBehavior:
    """36-42: Orchestrator behavioral rules tests."""

    def test_inputs_not_mutated(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        ds_copy = copy.deepcopy(ds)
        req_copy = copy.deepcopy(req)
        cc_copy = copy.deepcopy(cc)

        orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)

        assert ds == ds_copy
        assert req == req_copy
        assert cc == cc_copy

    def test_unique_planning_ids(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        r1 = orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)
        r2 = orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)

        assert r1.planning_id != r2.planning_id
        assert r1.planning_id.startswith("planning_")

    def test_create_plan_parameter_type_checks(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        with pytest.raises(TypeError, match="dataset_context"):
            orchestrator.create_plan(dataset_context=object(), user_request=req, compute_capabilities=cc)  # type: ignore

        with pytest.raises(TypeError, match="user_request"):
            orchestrator.create_plan(dataset_context=ds, user_request=object(), compute_capabilities=cc)  # type: ignore

        with pytest.raises(TypeError, match="compute_capabilities"):
            orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=object())  # type: ignore

        with pytest.raises(TypeError, match="mode"):
            orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc, mode=object())  # type: ignore

        with pytest.raises(TypeError, match="ai_provider"):
            orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc, ai_provider=object())  # type: ignore

    def test_constructor_parameter_type_checks(self):
        with pytest.raises(TypeError, match="problem_resolver"):
            MLPlanningOrchestrator(problem_resolver=object())  # type: ignore

        with pytest.raises(TypeError, match="baseline_planner"):
            MLPlanningOrchestrator(baseline_planner=object())  # type: ignore

        with pytest.raises(TypeError, match="validator"):
            MLPlanningOrchestrator(validator=object())  # type: ignore

    def test_result_model_dump_json(self):
        orchestrator = MLPlanningOrchestrator()
        ds = _make_dataset_context()
        req = _make_user_request()
        cc = _make_compute_capabilities()

        result = orchestrator.create_plan(dataset_context=ds, user_request=req, compute_capabilities=cc)
        dumped_json = result.model_dump_json()
        loaded = json.loads(dumped_json)
        assert loaded["planning_id"] == result.planning_id
        assert loaded["mode"] == "deterministic"

        dumped_dict = result.model_dump()
        assert dumped_dict["planning_id"] == result.planning_id

    def test_result_non_string_planning_id_rejected(self):
        with pytest.raises(Exception):
            MLPlanningResult(
                planning_id=12345,  # type: ignore
                mode=PlanningMode.DETERMINISTIC,
                problem_definition=None,  # type: ignore
                baseline_plan=None,  # type: ignore
                final_plan=None,  # type: ignore
                ai_assistance_used=False,
                ai_changes_applied=False,
                validation_result=None,  # type: ignore
            )

