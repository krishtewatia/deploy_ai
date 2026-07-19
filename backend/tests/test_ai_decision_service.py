"""Tests for AI Decision Service (Stage 7D7)."""

import json
import pytest
import copy

from backend.app.ai_planning.decision_service import AIDecisionService, AIDecisionServiceError
from backend.app.ai_planning.schemas import (
    AIDecisionConfidence,
    AIDecisionProposal,
    AIAssistedPlanningResult,
    ProposalAction,
)
from backend.app.ai_planning.providers.base import AIProvider, AIProviderError
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
from backend.app.ml_plan.schemas import (
    DatasetSplitPlan,
    EvaluationPlan,
    ExecutionConstraints,
    FeatureSelectionMethod,
    FeatureSelectionPlan,
    MLPlan,
    MLPlanStatus,
    ModelCandidate,
    ModelFamily,
    PreprocessingOperation,
    PreprocessingStep,
    SearchStrategy,
    SplitStrategy,
)
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ProblemType,
    ResolutionStatus,
    TargetSource,
)


# ── Mocks / Stubs ──────────────────────────────────────────────────────


class MockAIProvider(AIProvider):
    """Stub provider to simulate success/failure completions."""

    def __init__(self, response_text: str = "", should_raise: bool = False) -> None:
        self.response_text = response_text
        self.should_raise = should_raise
        self.last_system_prompt = None
        self.last_user_prompt = None

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        if self.should_raise:
            raise AIProviderError("Simulated provider connection failure.")
        return self.response_text


# ── Fixtures ───────────────────────────────────────────────────────────


def _make_dataset_context() -> DatasetContext:
    columns = [
        ColumnContext(
            name="age", dtype="float64", is_numeric=True, is_categorical=False,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=50, unique_percentage=5.0, sample_values=[25.0, 30.0],
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


def _make_problem_definition() -> ProblemDefinition:
    return ProblemDefinition(
        definition_id="pd_01", request_id="req_01", dataset_id="ds_01",
        goal="Predict churn", problem_type=ProblemType.CLASSIFICATION,
        target_column="churn", target_source=TargetSource.USER,
        feature_columns=["age"], excluded_columns=[],
        primary_metric="f1", status=ResolutionStatus.RESOLVED,
        warnings=[],
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


def _make_baseline_plan() -> MLPlan:
    return MLPlan(
        plan_id="plan_01", dataset_id="ds_01", request_id="req_01",
        problem_definition_id="pd_01", compute_capability_id="cap_01",
        problem_type=ProblemType.CLASSIFICATION, target_column="churn",
        feature_columns=["age"],
        preprocessing_steps=[],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE, candidate_columns=["age"], reason="No selection."),
        split_plan=DatasetSplitPlan(strategy=SplitStrategy.STRATIFIED, test_size=0.2,
                                     stratify_column="churn"),
        model_candidates=[
            ModelCandidate(candidate_id="model_001", model_family=ModelFamily.LOGISTIC_REGRESSION,
                           parameters={"random_state": 42}, search_strategy=SearchStrategy.NONE,
                           search_space={}, reason="Baseline."),
        ],
        evaluation_plan=EvaluationPlan(primary_metric="f1", secondary_metrics=["accuracy"],
                                        cross_validation_folds=5),
        execution_constraints=ExecutionConstraints(
            parallel_workers=4, use_gpu_acceleration=False,
            accelerator_type=AcceleratorType.NONE, compute_tier=ComputeTier.STANDARD),
        status=MLPlanStatus.READY,
    )


def _make_valid_no_change_response() -> str:
    return json.dumps({
        "proposal_set_id": "ps_01",
        "baseline_plan_id": "plan_01",
        "dataset_id": "ds_01",
        "request_id": "req_01",
        "problem_definition_id": "pd_01",
        "compute_capability_id": "cap_01",
        "summary": "No changes needed.",
    })


def _make_valid_change_response() -> str:
    return json.dumps({
        "proposal_set_id": "ps_01",
        "baseline_plan_id": "plan_01",
        "dataset_id": "ds_01",
        "request_id": "req_01",
        "problem_definition_id": "pd_01",
        "compute_capability_id": "cap_01",
        "summary": "Adding model candidate.",
        "model_candidate_proposals": [
            {
                "proposal_id": "mc_rf",
                "action": "add",
                "model_family": "random_forest",
                "parameters": {},
                "search_strategy": "none",
                "search_space": {},
                "reason": "Add RF.",
                "confidence": "high"
            }
        ]
    })


# ── Tests ──────────────────────────────────────────────────────────────


class TestAIDecisionService:
    def test_instantiate_invalid_provider_raises_error(self):
        with pytest.raises(TypeError, match="provider must be an instance of AIProvider"):
            AIDecisionService(provider=object())

    def test_successful_no_change_flow(self):
        provider = MockAIProvider(response_text=_make_valid_no_change_response())
        service = AIDecisionService(provider=provider)

        ds = _make_dataset_context()
        req = _make_user_request()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()
        bp = _make_baseline_plan()

        result = service.run_planning(
            dataset_context=ds,
            user_request=req,
            problem_definition=pd,
            compute_capabilities=cc,
            baseline_plan=bp,
        )

        assert isinstance(result, AIAssistedPlanningResult)
        assert result.baseline_plan_id == bp.plan_id
        assert result.applied is False
        assert len(result.proposal.model_candidate_proposals) == 0
        assert result.final_plan.plan_id == bp.plan_id
        # Injected prompts are set correctly
        assert "DeployAI" in provider.last_system_prompt
        assert "PLANNING CONTEXT" in provider.last_user_prompt

    def test_successful_change_applied_flow(self):
        provider = MockAIProvider(response_text=_make_valid_change_response())
        service = AIDecisionService(provider=provider)

        result = service.run_planning(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )

        assert result.applied is True
        assert len(result.proposal.model_candidate_proposals) == 1
        assert len(result.final_plan.model_candidates) == 2
        assert result.final_plan.model_candidates[1].candidate_id == "ai_mc_rf"
        assert result.final_plan.model_candidates[1].model_family == ModelFamily.RANDOM_FOREST

    def test_incomplete_feature_engineering_proposal_uses_baseline_plan(self):
        """Malformed optional AI suggestions must not abort model training."""
        response = json.loads(_make_valid_no_change_response())
        response["feature_engineering_proposals"] = [
            {
                "proposal_id": "fe_incomplete_1",
                "action": "add",
                "operation": "ratio",
                "input_columns": [],
                "output_columns": [],
                "reason": "Incomplete feature suggestion.",
                "confidence": "medium",
            },
            {
                "proposal_id": "fe_incomplete_2",
                "action": "add",
                "operation": "polynomial",
                "input_columns": [],
                "output_columns": [],
                "reason": "Another incomplete feature suggestion.",
                "confidence": "medium",
            },
        ]
        service = AIDecisionService(
            provider=MockAIProvider(response_text=json.dumps(response))
        )

        result = service.run_planning(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )

        assert result.applied is False
        assert result.proposal.feature_engineering_proposals == []
        assert result.final_plan.feature_engineering_steps == []

    def test_partial_feature_selection_proposal_uses_safe_no_op(self):
        response = json.loads(_make_valid_no_change_response())
        response["feature_selection_proposal"] = {
            "reason": "Feature selection evaluation",
        }
        service = AIDecisionService(
            provider=MockAIProvider(response_text=json.dumps(response))
        )

        result = service.run_planning(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )

        assert result.final_plan.feature_selection.method == FeatureSelectionMethod.NONE

    def test_invalid_input_types_rejected(self):
        provider = MockAIProvider(response_text=_make_valid_no_change_response())
        service = AIDecisionService(provider=provider)

        ds = _make_dataset_context()
        req = _make_user_request()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()
        bp = _make_baseline_plan()

        with pytest.raises(TypeError, match="dataset_context"):
            service.run_planning(dataset_context=object(), user_request=req, problem_definition=pd,
                                 compute_capabilities=cc, baseline_plan=bp)

        with pytest.raises(TypeError, match="user_request"):
            service.run_planning(dataset_context=ds, user_request=object(), problem_definition=pd,
                                 compute_capabilities=cc, baseline_plan=bp)

        with pytest.raises(TypeError, match="problem_definition"):
            service.run_planning(dataset_context=ds, user_request=req, problem_definition=object(),
                                 compute_capabilities=cc, baseline_plan=bp)

        with pytest.raises(TypeError, match="compute_capabilities"):
            service.run_planning(dataset_context=ds, user_request=req, problem_definition=pd,
                                 compute_capabilities=object(), baseline_plan=bp)

        with pytest.raises(TypeError, match="baseline_plan"):
            service.run_planning(dataset_context=ds, user_request=req, problem_definition=pd,
                                 compute_capabilities=cc, baseline_plan=object())

    def test_provider_failure_wrapped(self):
        provider = MockAIProvider(should_raise=True)
        service = AIDecisionService(provider=provider)

        with pytest.raises(AIDecisionServiceError, match="AI Provider failed to generate response"):
            service.run_planning(
                dataset_context=_make_dataset_context(),
                user_request=_make_user_request(),
                problem_definition=_make_problem_definition(),
                compute_capabilities=_make_compute_capabilities(),
                baseline_plan=_make_baseline_plan(),
            )

    def test_parser_failure_uses_deterministic_baseline(self):
        provider = MockAIProvider(response_text="Not valid JSON response")
        service = AIDecisionService(provider=provider)

        result = service.run_planning(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )

        assert result.applied is False
        assert result.final_plan.plan_id == "plan_01"

    def test_merger_failure_uses_deterministic_baseline(self):
        # AI response contains invalid change (e.g. standard scale on invalid column name)
        bad_response = json.dumps({
            "proposal_set_id": "ps_01",
            "baseline_plan_id": "plan_01",
            "dataset_id": "ds_01",
            "request_id": "req_01",
            "problem_definition_id": "pd_01",
            "compute_capability_id": "cap_01",
            "summary": "Bad column scale.",
            "preprocessing_proposals": [
                {
                    "proposal_id": "p_bad",
                    "action": "add",
                    "operation": "standard_scale",
                    "columns": ["does_not_exist"],
                    "reason": "Scale missing column.",
                    "confidence": "high"
                }
            ]
        })
        provider = MockAIProvider(response_text=bad_response)
        service = AIDecisionService(provider=provider)

        result = service.run_planning(
            dataset_context=_make_dataset_context(),
            user_request=_make_user_request(),
            problem_definition=_make_problem_definition(),
            compute_capabilities=_make_compute_capabilities(),
            baseline_plan=_make_baseline_plan(),
        )

        assert result.applied is False
        assert result.final_plan.plan_id == "plan_01"

    def test_none_mutation_assurance(self):
        provider = MockAIProvider(response_text=_make_valid_change_response())
        service = AIDecisionService(provider=provider)

        ds = _make_dataset_context()
        req = _make_user_request()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()
        bp = _make_baseline_plan()

        orig_bp = copy.deepcopy(bp)

        service.run_planning(
            dataset_context=ds,
            user_request=req,
            problem_definition=pd,
            compute_capabilities=cc,
            baseline_plan=bp,
        )

        assert bp == orig_bp

    def test_context_builder_error_wrapped(self):
        """Simulate unexpected error during context building."""
        class FailingContextBuilder:
            def build(self, **kwargs):
                raise ValueError("Context builder crashed!")

        service = AIDecisionService(provider=MockAIProvider(), context_builder=FailingContextBuilder())
        with pytest.raises(AIDecisionServiceError, match="Failed to build AI planning context"):
            service.run_planning(
                dataset_context=_make_dataset_context(),
                user_request=_make_user_request(),
                problem_definition=_make_problem_definition(),
                compute_capabilities=_make_compute_capabilities(),
                baseline_plan=_make_baseline_plan(),
            )

    def test_prompt_builder_error_wrapped(self):
        """Simulate unexpected error during prompt building."""
        class FailingPromptBuilder:
            def build(self, **kwargs):
                raise ValueError("Prompt builder crashed!")

        service = AIDecisionService(provider=MockAIProvider(), prompt_builder=FailingPromptBuilder())
        with pytest.raises(AIDecisionServiceError, match="Failed to build prompts"):
            service.run_planning(
                dataset_context=_make_dataset_context(),
                user_request=_make_user_request(),
                problem_definition=_make_problem_definition(),
                compute_capabilities=_make_compute_capabilities(),
                baseline_plan=_make_baseline_plan(),
            )

    def test_unexpected_provider_error_wrapped(self):
        """Simulate general unexpected exception from provider.generate."""
        class GeneralFailingProvider(AIProvider):
            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                raise TypeError("Random unexpected error!")

        service = AIDecisionService(provider=GeneralFailingProvider())
        with pytest.raises(AIDecisionServiceError, match="Unexpected provider generation error"):
            service.run_planning(
                dataset_context=_make_dataset_context(),
                user_request=_make_user_request(),
                problem_definition=_make_problem_definition(),
                compute_capabilities=_make_compute_capabilities(),
                baseline_plan=_make_baseline_plan(),
            )

    def test_unexpected_parser_error_wrapped(self):
        """Simulate general unexpected exception from parser."""
        class GeneralFailingParser:
            def parse(self, text):
                raise ValueError("Random parsing crash!")

        service = AIDecisionService(provider=MockAIProvider(response_text="{}"), parser=GeneralFailingParser())
        with pytest.raises(AIDecisionServiceError, match="Unexpected response parsing error"):
            service.run_planning(
                dataset_context=_make_dataset_context(),
                user_request=_make_user_request(),
                problem_definition=_make_problem_definition(),
                compute_capabilities=_make_compute_capabilities(),
                baseline_plan=_make_baseline_plan(),
            )

    def test_unexpected_merger_error_wrapped(self):
        """Simulate general unexpected exception from merger."""
        class GeneralFailingMerger:
            def merge(self, **kwargs):
                raise KeyError("Random merger crash!")

        service = AIDecisionService(
            provider=MockAIProvider(response_text=_make_valid_no_change_response()),
            merger=GeneralFailingMerger(),
        )
        with pytest.raises(AIDecisionServiceError, match="Unexpected plan merge error"):
            service.run_planning(
                dataset_context=_make_dataset_context(),
                user_request=_make_user_request(),
                problem_definition=_make_problem_definition(),
                compute_capabilities=_make_compute_capabilities(),
                baseline_plan=_make_baseline_plan(),
            )
