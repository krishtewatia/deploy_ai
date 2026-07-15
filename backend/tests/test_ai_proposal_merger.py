"""Tests for Controlled Proposal Merger (Stage 7D6)."""

import pytest
import copy

from backend.app.ai_planning.proposal_merger import ProposalMerger, ProposalMergerError
from backend.app.ai_planning.schemas import (
    AIDecisionConfidence,
    AIDecisionProposal,
    AIEvaluationProposal,
    AIFeatureEngineeringProposal,
    AIFeatureSelectionProposal,
    AIModelCandidateProposal,
    AIPreprocessingProposal,
    ProposalAction,
)
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
    FeatureEngineeringOperation,
    FeatureEngineeringStep,
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
from backend.app.ml_plan.validator import MLPlanValidator


# ── Fixtures & Helpers ──────────────────────────────────────────────────


def _make_dataset_context() -> DatasetContext:
    columns = [
        ColumnContext(
            name="age", dtype="float64", is_numeric=True, is_categorical=False,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=50, unique_percentage=5.0, sample_values=[25.0, 30.0],
            statistics=ColumnStatistics(mean=35.0, median=34.0, std=10.0, min=18.0, max=80.0),
        ),
        ColumnContext(
            name="salary", dtype="float64", is_numeric=True, is_categorical=False,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=100, unique_percentage=10.0, sample_values=[50000.0, 60000.0],
            statistics=ColumnStatistics(mean=55000.0, median=54000.0, std=12000.0, min=20000.0, max=150000.0),
        ),
        ColumnContext(
            name="department", dtype="object", is_numeric=False, is_categorical=True,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=5, unique_percentage=0.5, sample_values=["sales", "engineering"],
        ),
        ColumnContext(
            name="churn", dtype="int64", is_numeric=True, is_categorical=True,
            is_datetime=False, missing_count=0, missing_percentage=0.0,
            unique_count=2, unique_percentage=0.2, sample_values=[0, 1],
        ),
    ]
    return DatasetContext(
        basic_info=DatasetBasicInfo(dataset_id="ds_01", file_name="data.csv",
                                    row_count=1000, column_count=4, memory_usage_bytes=50000),
        columns=columns,
        missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
    )


def _make_problem_definition() -> ProblemDefinition:
    return ProblemDefinition(
        definition_id="pd_01", request_id="req_01", dataset_id="ds_01",
        goal="Predict churn", problem_type=ProblemType.CLASSIFICATION,
        target_column="churn", target_source=TargetSource.USER,
        feature_columns=["age", "salary", "department"], excluded_columns=[],
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
        feature_columns=["age", "salary", "department"],
        preprocessing_steps=[
            PreprocessingStep(step_id="prep_001", operation=PreprocessingOperation.IMPUTE_MEDIAN,
                              columns=["age"], parameters={}, reason="Impute age."),
        ],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE, candidate_columns=["age", "salary", "department"],
            reason="No selection."),
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


def _make_empty_proposal() -> AIDecisionProposal:
    return AIDecisionProposal(
        proposal_set_id="ps_01",
        baseline_plan_id="plan_01",
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        summary="No changes proposed.",
    )


# ── Tests ──────────────────────────────────────────────────────────────


class TestProposalMerger:
    def test_merge_no_change_proposal_returns_valid_copy(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        ds = _make_dataset_context()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()

        candidate = merger.merge(
            baseline_plan=bp,
            ai_proposal=prop,
            dataset_context=ds,
            problem_definition=pd,
            compute_capabilities=cc,
        )

        assert candidate.plan_id == bp.plan_id
        # Plan is equal in content, but distinct object instances
        assert candidate is not bp
        assert len(candidate.preprocessing_steps) == 1
        assert candidate.model_candidates[0].model_family == ModelFamily.LOGISTIC_REGRESSION

    def test_baseline_plan_is_not_mutated(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        # Propose adding a preprocessing step
        prop.preprocessing_proposals.append(
            AIPreprocessingProposal(
                proposal_id="prep_new",
                action=ProposalAction.ADD,
                operation=PreprocessingOperation.STANDARD_SCALE,
                columns=["salary"],
                reason="Scale salary.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )

        ds = _make_dataset_context()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()

        # Before merge, baseline has 1 preprocessing step
        assert len(bp.preprocessing_steps) == 1

        candidate = merger.merge(
            baseline_plan=bp,
            ai_proposal=prop,
            dataset_context=ds,
            problem_definition=pd,
            compute_capabilities=cc,
        )

        # After merge, baseline must STILL have 1 preprocessing step (non-mutation)
        assert len(bp.preprocessing_steps) == 1
        # Candidate has 2 preprocessing steps
        assert len(candidate.preprocessing_steps) == 2
        assert candidate.preprocessing_steps[1].step_id == "ai_prep_new"
        assert candidate.preprocessing_steps[1].operation == PreprocessingOperation.STANDARD_SCALE

    def test_artifact_linkage_mismatches_raise_error(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        ds = _make_dataset_context()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()

        # baseline_plan_id mismatch
        prop = _make_empty_proposal()
        prop.baseline_plan_id = "wrong_plan_id"
        with pytest.raises(ProposalMergerError, match="baseline_plan_id"):
            merger.merge(baseline_plan=bp, ai_proposal=prop, dataset_context=ds,
                         problem_definition=pd, compute_capabilities=cc)

        # dataset_id mismatch
        prop = _make_empty_proposal()
        prop.dataset_id = "wrong_ds_id"
        with pytest.raises(ProposalMergerError, match="dataset_id"):
            merger.merge(baseline_plan=bp, ai_proposal=prop, dataset_context=ds,
                         problem_definition=pd, compute_capabilities=cc)

        # request_id mismatch
        prop = _make_empty_proposal()
        prop.request_id = "wrong_req_id"
        with pytest.raises(ProposalMergerError, match="request_id"):
            merger.merge(baseline_plan=bp, ai_proposal=prop, dataset_context=ds,
                         problem_definition=pd, compute_capabilities=cc)

        # problem_definition_id mismatch
        prop = _make_empty_proposal()
        prop.problem_definition_id = "wrong_pd_id"
        with pytest.raises(ProposalMergerError, match="problem_definition_id"):
            merger.merge(baseline_plan=bp, ai_proposal=prop, dataset_context=ds,
                         problem_definition=pd, compute_capabilities=cc)

        # compute_capability_id mismatch
        prop = _make_empty_proposal()
        prop.compute_capability_id = "wrong_cap_id"
        with pytest.raises(ProposalMergerError, match="compute_capability_id"):
            merger.merge(baseline_plan=bp, ai_proposal=prop, dataset_context=ds,
                         problem_definition=pd, compute_capabilities=cc)

    # ── Preprocessing Merge Actions ────────────────────────────────────

    def test_preprocessing_add(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.preprocessing_proposals.append(
            AIPreprocessingProposal(
                proposal_id="p_add",
                action=ProposalAction.ADD,
                operation=PreprocessingOperation.STANDARD_SCALE,
                columns=["salary"],
                reason="Scale.",
                confidence=AIDecisionConfidence.MEDIUM,
            )
        )
        candidate = merger.merge(baseline_plan=bp, ai_proposal=prop,
                                  dataset_context=_make_dataset_context(),
                                  problem_definition=_make_problem_definition(),
                                  compute_capabilities=_make_compute_capabilities())
        assert len(candidate.preprocessing_steps) == 2
        assert candidate.preprocessing_steps[1].step_id == "ai_p_add"
        assert candidate.preprocessing_steps[1].operation == PreprocessingOperation.STANDARD_SCALE
        assert candidate.preprocessing_steps[1].columns == ["salary"]

    def test_preprocessing_remove_valid(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        # bp has step impute_median for columns ["age"]
        prop = _make_empty_proposal()
        prop.preprocessing_proposals.append(
            AIPreprocessingProposal(
                proposal_id="p_rem",
                action=ProposalAction.REMOVE,
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["age"],
                reason="Remove imputation.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        candidate = merger.merge(baseline_plan=bp, ai_proposal=prop,
                                  dataset_context=_make_dataset_context(),
                                  problem_definition=_make_problem_definition(),
                                  compute_capabilities=_make_compute_capabilities())
        assert len(candidate.preprocessing_steps) == 0

    def test_preprocessing_remove_not_found_raises_error(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        # Try to remove a step that does not exist (operation mismatch or columns mismatch)
        prop.preprocessing_proposals.append(
            AIPreprocessingProposal(
                proposal_id="p_rem",
                action=ProposalAction.REMOVE,
                operation=PreprocessingOperation.STANDARD_SCALE,  # Not in bp
                columns=["age"],
                reason="Remove scale.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        with pytest.raises(ProposalMergerError, match="Cannot REMOVE preprocessing step"):
            merger.merge(baseline_plan=bp, ai_proposal=prop,
                         dataset_context=_make_dataset_context(),
                         problem_definition=_make_problem_definition(),
                         compute_capabilities=_make_compute_capabilities())

    def test_preprocessing_replace_valid(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        # Replace IMPUTE_MEDIAN on age with IMPUTE_MEAN on age
        prop.preprocessing_proposals.append(
            AIPreprocessingProposal(
                proposal_id="p_rep",
                action=ProposalAction.REPLACE,
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["age"],
                parameters={"strategy": "mean"},
                reason="Switch to mean.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        candidate = merger.merge(baseline_plan=bp, ai_proposal=prop,
                                  dataset_context=_make_dataset_context(),
                                  problem_definition=_make_problem_definition(),
                                  compute_capabilities=_make_compute_capabilities())
        assert len(candidate.preprocessing_steps) == 1
        step = candidate.preprocessing_steps[0]
        assert step.step_id == "ai_p_rep"
        assert step.operation == PreprocessingOperation.IMPUTE_MEDIAN
        assert step.parameters == {"strategy": "mean"}

    def test_preprocessing_replace_not_found_raises_error(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        # Try to replace step that does not exist
        prop.preprocessing_proposals.append(
            AIPreprocessingProposal(
                proposal_id="p_rep",
                action=ProposalAction.REPLACE,
                operation=PreprocessingOperation.ROBUST_SCALE,
                columns=["salary"],
                reason="Replace.",
                confidence=AIDecisionConfidence.LOW,
            )
        )
        with pytest.raises(ProposalMergerError, match="Cannot REPLACE preprocessing step"):
            merger.merge(baseline_plan=bp, ai_proposal=prop,
                         dataset_context=_make_dataset_context(),
                         problem_definition=_make_problem_definition(),
                         compute_capabilities=_make_compute_capabilities())

    # ── Feature Engineering Merge Actions ──────────────────────────────

    def test_feature_engineering_add(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.feature_engineering_proposals.append(
            AIFeatureEngineeringProposal(
                proposal_id="fe_add",
                action=ProposalAction.ADD,
                operation=FeatureEngineeringOperation.RATIO,
                input_columns=["salary", "age"],
                output_columns=["salary_age_ratio"],
                reason="Add ratio feature.",
                confidence=AIDecisionConfidence.MEDIUM,
            )
        )
        # Note: We need the output column to be valid/declared or it might fail validation
        # unless it is listed in features or parsed. Wait, the output columns of feature
        # engineering steps don't have to exist in dataset context, but they shouldn't conflict.
        # Let's see if this passes validation.
        candidate = merger.merge(baseline_plan=bp, ai_proposal=prop,
                                  dataset_context=_make_dataset_context(),
                                  problem_definition=_make_problem_definition(),
                                  compute_capabilities=_make_compute_capabilities())
        assert len(candidate.feature_engineering_steps) == 1
        assert candidate.feature_engineering_steps[0].step_id == "ai_fe_add"
        assert candidate.feature_engineering_steps[0].operation == FeatureEngineeringOperation.RATIO

    def test_feature_engineering_remove_and_replace(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        # Let's add a feature engineering step to bp first
        bp.feature_engineering_steps.append(
            FeatureEngineeringStep(
                step_id="fe_orig",
                operation=FeatureEngineeringOperation.LOG_TRANSFORM,
                input_columns=["salary"],
                output_columns=["log_salary"],
                reason="Log scale.",
            )
        )

        # 1. Test replace
        prop_rep = _make_empty_proposal()
        prop_rep.feature_engineering_proposals.append(
            AIFeatureEngineeringProposal(
                proposal_id="fe_rep",
                action=ProposalAction.REPLACE,
                operation=FeatureEngineeringOperation.LOG_TRANSFORM,
                input_columns=["salary"],
                output_columns=["log_salary"],
                parameters={"base": 10},
                reason="Use base 10.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        cand_rep = merger.merge(baseline_plan=bp, ai_proposal=prop_rep,
                                 dataset_context=_make_dataset_context(),
                                 problem_definition=_make_problem_definition(),
                                 compute_capabilities=_make_compute_capabilities())
        assert len(cand_rep.feature_engineering_steps) == 1
        assert cand_rep.feature_engineering_steps[0].step_id == "ai_fe_rep"
        assert cand_rep.feature_engineering_steps[0].parameters == {"base": 10}

        # 2. Test remove
        prop_rem = _make_empty_proposal()
        prop_rem.feature_engineering_proposals.append(
            AIFeatureEngineeringProposal(
                proposal_id="fe_rem",
                action=ProposalAction.REMOVE,
                operation=FeatureEngineeringOperation.LOG_TRANSFORM,
                input_columns=["salary"],
                output_columns=["log_salary"],
                reason="Remove log.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        cand_rem = merger.merge(baseline_plan=bp, ai_proposal=prop_rem,
                                 dataset_context=_make_dataset_context(),
                                 problem_definition=_make_problem_definition(),
                                 compute_capabilities=_make_compute_capabilities())
        assert len(cand_rem.feature_engineering_steps) == 0

    def test_feature_engineering_remove_not_found_raises_error(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.feature_engineering_proposals.append(
            AIFeatureEngineeringProposal(
                proposal_id="fe_rem",
                action=ProposalAction.REMOVE,
                operation=FeatureEngineeringOperation.LOG_TRANSFORM,
                input_columns=["salary"],
                output_columns=["log_salary"],
                reason="Remove non-existent.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        with pytest.raises(ProposalMergerError, match="Cannot REMOVE feature engineering step"):
            merger.merge(baseline_plan=bp, ai_proposal=prop,
                         dataset_context=_make_dataset_context(),
                         problem_definition=_make_problem_definition(),
                         compute_capabilities=_make_compute_capabilities())

    def test_feature_engineering_replace_not_found_raises_error(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.feature_engineering_proposals.append(
            AIFeatureEngineeringProposal(
                proposal_id="fe_rep",
                action=ProposalAction.REPLACE,
                operation=FeatureEngineeringOperation.LOG_TRANSFORM,
                input_columns=["salary"],
                output_columns=["log_salary"],
                reason="Replace non-existent.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        with pytest.raises(ProposalMergerError, match="Cannot REPLACE feature engineering step"):
            merger.merge(baseline_plan=bp, ai_proposal=prop,
                         dataset_context=_make_dataset_context(),
                         problem_definition=_make_problem_definition(),
                         compute_capabilities=_make_compute_capabilities())

    # ── Model Candidate Merge Actions ──────────────────────────────────

    def test_model_candidate_add(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.model_candidate_proposals.append(
            AIModelCandidateProposal(
                proposal_id="m_add",
                action=ProposalAction.ADD,
                model_family=ModelFamily.RANDOM_FOREST,
                reason="Add RF.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        candidate = merger.merge(baseline_plan=bp, ai_proposal=prop,
                                  dataset_context=_make_dataset_context(),
                                  problem_definition=_make_problem_definition(),
                                  compute_capabilities=_make_compute_capabilities())
        assert len(candidate.model_candidates) == 2
        assert candidate.model_candidates[1].candidate_id == "ai_m_add"
        assert candidate.model_candidates[1].model_family == ModelFamily.RANDOM_FOREST

    def test_model_candidate_remove_and_replace(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        # bp has model candidate LOGISTIC_REGRESSION
        assert bp.model_candidates[0].model_family == ModelFamily.LOGISTIC_REGRESSION

        # REPLACE with custom parameters
        prop_rep = _make_empty_proposal()
        prop_rep.model_candidate_proposals.append(
            AIModelCandidateProposal(
                proposal_id="m_rep",
                action=ProposalAction.REPLACE,
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                parameters={"C": 0.5},
                reason="Change C.",
                confidence=AIDecisionConfidence.MEDIUM,
            )
        )
        cand_rep = merger.merge(baseline_plan=bp, ai_proposal=prop_rep,
                                 dataset_context=_make_dataset_context(),
                                 problem_definition=_make_problem_definition(),
                                 compute_capabilities=_make_compute_capabilities())
        assert len(cand_rep.model_candidates) == 1
        assert cand_rep.model_candidates[0].candidate_id == "ai_m_rep"
        assert cand_rep.model_candidates[0].parameters == {"C": 0.5}

        # REMOVE model family (raises error since MLPlan requires >= 1 model candidate,
        # but let's see if we add another one first to keep it valid)
        bp.model_candidates.append(
            ModelCandidate(candidate_id="model_002", model_family=ModelFamily.DECISION_TREE, reason="DT.")
        )
        prop_rem = _make_empty_proposal()
        prop_rem.model_candidate_proposals.append(
            AIModelCandidateProposal(
                proposal_id="m_rem",
                action=ProposalAction.REMOVE,
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                reason="Remove LR.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        cand_rem = merger.merge(baseline_plan=bp, ai_proposal=prop_rem,
                                 dataset_context=_make_dataset_context(),
                                 problem_definition=_make_problem_definition(),
                                 compute_capabilities=_make_compute_capabilities())
        assert len(cand_rem.model_candidates) == 1
        assert cand_rem.model_candidates[0].model_family == ModelFamily.DECISION_TREE

    def test_model_candidate_remove_not_found_raises_error(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.model_candidate_proposals.append(
            AIModelCandidateProposal(
                proposal_id="m_rem",
                action=ProposalAction.REMOVE,
                model_family=ModelFamily.RANDOM_FOREST,  # Not in bp
                reason="Remove.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        with pytest.raises(ProposalMergerError, match="Cannot REMOVE model candidate"):
            merger.merge(baseline_plan=bp, ai_proposal=prop,
                         dataset_context=_make_dataset_context(),
                         problem_definition=_make_problem_definition(),
                         compute_capabilities=_make_compute_capabilities())

    def test_model_candidate_replace_not_found_raises_error(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.model_candidate_proposals.append(
            AIModelCandidateProposal(
                proposal_id="m_rep",
                action=ProposalAction.REPLACE,
                model_family=ModelFamily.RANDOM_FOREST,  # Not in bp
                reason="Replace.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        with pytest.raises(ProposalMergerError, match="Cannot REPLACE model candidate"):
            merger.merge(baseline_plan=bp, ai_proposal=prop,
                         dataset_context=_make_dataset_context(),
                         problem_definition=_make_problem_definition(),
                         compute_capabilities=_make_compute_capabilities())

    # ── Feature Selection Proposals ────────────────────────────────────

    def test_feature_selection_proposal(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.feature_selection_proposal = AIFeatureSelectionProposal(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            candidate_columns=["age", "salary"],
            max_features=2,
            parameters={"k": 2},
            reason="Select top features.",
            confidence=AIDecisionConfidence.HIGH,
        )
        candidate = merger.merge(baseline_plan=bp, ai_proposal=prop,
                                  dataset_context=_make_dataset_context(),
                                  problem_definition=_make_problem_definition(),
                                  compute_capabilities=_make_compute_capabilities())
        fs = candidate.feature_selection
        assert fs.method == FeatureSelectionMethod.MUTUAL_INFORMATION
        assert fs.candidate_columns == ["age", "salary"]
        assert fs.max_features == 2
        assert fs.parameters == {"k": 2}
        assert "[AI]" in fs.reason

    # ── Evaluation Proposals ───────────────────────────────────────────

    def test_evaluation_proposal(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        prop.evaluation_proposal = AIEvaluationProposal(
            primary_metric="accuracy",
            secondary_metrics=["precision", "recall"],
            cross_validation_folds=10,
            reason="Switch evaluation metric.",
            confidence=AIDecisionConfidence.MEDIUM,
        )
        # Note: validator requires primary_metric to match problem_definition primary_metric,
        # otherwise it raises a mismatch error. Let's make sure our problem definition
        # primary metric matches the proposed one to pass validation.
        pd = _make_problem_definition()
        pd.primary_metric = "accuracy"

        candidate = merger.merge(baseline_plan=bp, ai_proposal=prop,
                                  dataset_context=_make_dataset_context(),
                                  problem_definition=pd,
                                  compute_capabilities=_make_compute_capabilities())
        ep = candidate.evaluation_plan
        assert ep.primary_metric == "accuracy"
        assert ep.secondary_metrics == ["precision", "recall"]
        assert ep.cross_validation_folds == 10

    # ── Validation Gate Failures ───────────────────────────────────────

    def test_validation_gate_fails_rejections(self):
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        prop = _make_empty_proposal()
        # Add a preprocessing step for a column that does NOT exist in dataset context
        prop.preprocessing_proposals.append(
            AIPreprocessingProposal(
                proposal_id="p_bad",
                action=ProposalAction.ADD,
                operation=PreprocessingOperation.STANDARD_SCALE,
                columns=["non_existent_column"],
                reason="Scale invalid column.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )

        with pytest.raises(ProposalMergerError, match="failed validation"):
            merger.merge(baseline_plan=bp, ai_proposal=prop,
                         dataset_context=_make_dataset_context(),
                         problem_definition=_make_problem_definition(),
                         compute_capabilities=_make_compute_capabilities())

    def test_validation_construction_error_raises_merger_error(self):
        """Test if model construction fails (e.g. invalid types/hyperparameters)."""
        merger = ProposalMerger()
        bp = _make_baseline_plan()
        
        # We can trigger a ValidationError by removing the only model candidate, 
        # as MLPlan requires model_candidates to be non-empty.
        prop_remove_all = _make_empty_proposal()
        prop_remove_all.model_candidate_proposals.append(
            AIModelCandidateProposal(
                proposal_id="rm_all",
                action=ProposalAction.REMOVE,
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                reason="Remove LR.",
                confidence=AIDecisionConfidence.HIGH,
            )
        )
        # Reconstructing MLPlan with empty model_candidates should raise ValidationError
        with pytest.raises(ProposalMergerError, match="Failed to construct candidate MLPlan"):
            merger.merge(baseline_plan=bp, ai_proposal=prop_remove_all,
                         dataset_context=_make_dataset_context(),
                         problem_definition=_make_problem_definition(),
                         compute_capabilities=_make_compute_capabilities())

