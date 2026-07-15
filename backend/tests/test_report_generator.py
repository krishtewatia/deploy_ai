from __future__ import annotations

import pytest
from backend.app.analysis.report_generator import ReportGenerator
from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    ImbalanceReport,
    MissingValueReport,
    StatisticsReport,
)


@pytest.fixture
def mock_missing_values() -> MissingValueReport:
    return MissingValueReport(
        total_missing=5,
        missing_by_column={"col1": 5},
        missing_percentage={"col1": 50.0}
    )


@pytest.fixture
def mock_duplicates() -> DuplicateReport:
    return DuplicateReport(
        duplicate_rows=2,
        duplicate_percentage=20.0
    )


@pytest.fixture
def mock_statistics() -> StatisticsReport:
    return StatisticsReport(
        numerical_summary={
            "col1": {
                "mean": 10.0,
                "median": 10.0,
                "std": 1.0,
                "min": 9.0,
                "max": 11.0
            }
        }
    )


@pytest.fixture
def mock_imbalance() -> ImbalanceReport:
    return ImbalanceReport(
        imbalanced=True,
        distribution={"0": 9, "1": 1}
    )


def test_report_generator_with_imbalance(
    mock_missing_values: MissingValueReport,
    mock_duplicates: DuplicateReport,
    mock_statistics: StatisticsReport,
    mock_imbalance: ImbalanceReport,
) -> None:
    """Test generating a consolidated report including target class imbalance."""
    generator = ReportGenerator()
    report = generator.generate(
        missing_values=mock_missing_values,
        duplicates=mock_duplicates,
        statistics=mock_statistics,
        imbalance=mock_imbalance,
    )

    assert isinstance(report, DatasetAnalysisReport)
    # Verify all sections are preserved correctly
    assert report.missing_values == mock_missing_values
    assert report.duplicates == mock_duplicates
    assert report.statistics == mock_statistics
    assert report.imbalance == mock_imbalance


def test_report_generator_without_imbalance(
    mock_missing_values: MissingValueReport,
    mock_duplicates: DuplicateReport,
    mock_statistics: StatisticsReport,
) -> None:
    """Test generating a consolidated report without class imbalance."""
    generator = ReportGenerator()
    report = generator.generate(
        missing_values=mock_missing_values,
        duplicates=mock_duplicates,
        statistics=mock_statistics,
        imbalance=None,
    )

    assert isinstance(report, DatasetAnalysisReport)
    # Verify all sections are preserved correctly
    assert report.missing_values == mock_missing_values
    assert report.duplicates == mock_duplicates
    assert report.statistics == mock_statistics
    assert report.imbalance is None


def test_report_generator_with_column_profiles(
    mock_missing_values: MissingValueReport,
    mock_duplicates: DuplicateReport,
    mock_statistics: StatisticsReport,
) -> None:
    """Test generating a consolidated report with column profiles passed explicitly."""
    generator = ReportGenerator()
    column_profiles = {"col1": {"dtype": "int64", "unique_values": 2, "sample_values": [1, 2]}}
    report = generator.generate(
        missing_values=mock_missing_values,
        duplicates=mock_duplicates,
        statistics=mock_statistics,
        column_profiles=column_profiles,
    )

    assert isinstance(report, DatasetAnalysisReport)
    assert report.column_profiles == column_profiles


# ── EXECUTIVE REPORT GENERATOR UNIT TESTS ─────────────────────────────

import copy
from unittest.mock import MagicMock

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.ml_plan import ModelFamily, SearchStrategy
from backend.app.ml_plan.schemas import (
    MLPlan,
    MLPlanStatus,
    DatasetSplitPlan,
    SplitStrategy,
    FeatureSelectionPlan,
    FeatureSelectionMethod,
    EvaluationPlan,
    ExecutionConstraints,
    ModelCandidate,
)
from backend.app.compute_capabilities.schemas import AcceleratorType as CapAcceleratorType, ComputeTier as CapComputeTier
from backend.app.problem_definition.schemas import ProblemType
from backend.app.ml_execution.execution_report import (
    CandidateSummary,
    ChampionSummary,
    ExecutionReport,
)
from backend.app.ai_model_critic.schemas import ModelCritique, CritiqueGrade
from backend.app.ai_model_optimizer.schemas import (
    OptimizationResult,
    OptimizationAction,
    OptimizationActionType,
)
from backend.app.model_governance.schemas import ChampionDecision, Winner
from backend.app.reporting.report_generator import (
    ExecutiveReportGenerator,
    ExecutiveReportGeneratorError,
)
from backend.app.reporting.schemas import ExecutiveReport


def _make_mock_plan(plan_id: str = "plan_01", problem_type: ProblemType = ProblemType.CLASSIFICATION) -> MLPlan:
    return MLPlan(
        plan_id=plan_id,
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        problem_type=problem_type,
        target_column="species",
        feature_columns=["feat_a"],
        preprocessing_steps=[],
        feature_engineering_steps=[],
        feature_selection=FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=["feat_a"],
            reason="No selection",
        ),
        split_plan=DatasetSplitPlan(
            strategy=SplitStrategy.RANDOM,
            test_size=0.2,
        ),
        model_candidates=[
            ModelCandidate(
                candidate_id="model_01",
                model_family=ModelFamily.LOGISTIC_REGRESSION if problem_type == ProblemType.CLASSIFICATION else ModelFamily.LINEAR_REGRESSION,
                search_strategy=SearchStrategy.NONE,
                reason="Default candidate",
            )
        ],
        evaluation_plan=EvaluationPlan(
            primary_metric="f1" if problem_type == ProblemType.CLASSIFICATION else "mae",
            cross_validation_folds=3,
        ),
        execution_constraints=ExecutionConstraints(
            parallel_workers=1,
            use_gpu_acceleration=False,
            accelerator_type=CapAcceleratorType.NONE,
            compute_tier=CapComputeTier.STANDARD,
        ),
        status=MLPlanStatus.READY,
    )


def _make_mock_report(
    report_id: str = "rep_01",
    plan_id: str = "plan_01",
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    metric_value: float = 0.85,
) -> ExecutionReport:
    champ = ChampionSummary(
        candidate_id="model_01",
        model_family=ModelFamily.LOGISTIC_REGRESSION if problem_type == ProblemType.CLASSIFICATION else ModelFamily.LINEAR_REGRESSION,
        primary_metric="f1" if problem_type == ProblemType.CLASSIFICATION else "mae",
        primary_metric_value=metric_value,
        training_duration=0.5,
        evaluation_duration=0.05,
    )
    candidate = CandidateSummary(
        candidate_id="model_01",
        model_family=ModelFamily.LOGISTIC_REGRESSION if problem_type == ProblemType.CLASSIFICATION else ModelFamily.LINEAR_REGRESSION,
        primary_metric="f1" if problem_type == ProblemType.CLASSIFICATION else "mae",
        primary_metric_value=metric_value,
        training_duration=0.5,
        evaluation_duration=0.05,
        search_strategy=SearchStrategy.NONE,
    )
    return ExecutionReport(
        report_id=report_id,
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        plan_id=plan_id,
        execution_id="exec_01",
        problem_type=problem_type,
        target_column="species",
        feature_columns=["feat_a"],
        compute_tier=ComputeTier.STANDARD,
        accelerator_type=AcceleratorType.NONE,
        candidate_summaries=[candidate],
        champion_summary=champ,
        training_summary={"goal": "Optimize performance"},
        evaluation_summary={},
        warnings=["Some baseline warning"],
        execution_duration=1.2,
        created_timestamp="2026-07-15T00:00:00Z",
    )


def _make_mock_critique(report_id: str = "rep_01") -> ModelCritique:
    return ModelCritique(
        critique_id="crit_01",
        report_id=report_id,
        overall_grade=CritiqueGrade.A_PLUS,
        confidence=0.9,
        strengths=["Good features"],
        weaknesses=["None"],
        risks=["Low risks"],
        recommendations=["Increase cross validation folds"],
        warnings=["Critique warning"],
        summary="A solid critique.",
        production_ready=True,
    )


def _make_mock_optimization(baseline_plan_id: str = "plan_01", problem_type: ProblemType = ProblemType.CLASSIFICATION) -> OptimizationResult:
    opt_plan = _make_mock_plan(plan_id="plan_opt_01", problem_type=problem_type)
    action = OptimizationAction(
        action_id="act_01",
        action_type=OptimizationActionType.CHANGE_CV_FOLDS,
        target="evaluation_plan.cross_validation_folds",
        replacement="3",
        reason="Improve validation bounds",
        confidence=0.9,
    )
    return OptimizationResult(
        optimization_id="opt_01",
        baseline_plan_id=baseline_plan_id,
        optimized_plan=opt_plan,
        actions=[action],
        summary="Successfully optimized plan.",
    )


def _make_mock_governance(baseline_report_id: str = "rep_01", winner_report: ExecutionReport = None) -> ChampionDecision:
    if winner_report is None:
        winner_report = _make_mock_report(report_id="rep_opt_01", plan_id="plan_opt_01")
    return ChampionDecision(
        decision_id="dec_01",
        baseline_report_id=baseline_report_id,
        retrained_report_id="rep_opt_01",
        winner=Winner.RETRAINED,
        winner_report=winner_report,
        improvement_detected=True,
        metric_name="f1",
        baseline_metric=0.85,
        retrained_metric=0.90,
        relative_improvement=0.0588,
        decision_reason="Retrained model scored higher.",
        production_ready=True,
        comparison_timestamp="2026-07-15T00:00:00Z",
    )


class TestExecutiveReportGenerator:
    """Tests covering ExecutiveReportGenerator validation, mismatch checks, and formatting."""

    def test_generate_classification_success(self):
        """Verify successful generation of executive reports for classification workloads."""
        rep = _make_mock_report(problem_type=ProblemType.CLASSIFICATION)
        crit = _make_mock_critique(report_id="rep_01")
        opt = _make_mock_optimization(baseline_plan_id="plan_01", problem_type=ProblemType.CLASSIFICATION)
        gov = _make_mock_governance(baseline_report_id="rep_01")

        gen = ExecutiveReportGenerator()
        report = gen.generate(
            execution_report=rep,
            critique=crit,
            optimization=opt,
            governance=gov,
        )

        assert isinstance(report, ExecutiveReport)
        assert report.problem_summary["problem_type"] == ProblemType.CLASSIFICATION
        assert report.champion_summary["metric_name"] == "f1"
        assert report.champion_summary["production_readiness"] is True
        assert report.ai_review["overall_grade"] == CritiqueGrade.A_PLUS
        assert "Some baseline warning" in report.warnings
        assert "Critique warning" in report.warnings

    def test_generate_regression_success(self):
        """Verify successful generation of executive reports for regression workloads."""
        rep = _make_mock_report(problem_type=ProblemType.REGRESSION, metric_value=0.15)
        crit = _make_mock_critique(report_id="rep_01")
        opt = _make_mock_optimization(baseline_plan_id="plan_01", problem_type=ProblemType.REGRESSION)

        winner_rep = _make_mock_report(report_id="rep_opt_01", plan_id="plan_opt_01", problem_type=ProblemType.REGRESSION, metric_value=0.11)
        gov = _make_mock_governance(baseline_report_id="rep_01", winner_report=winner_rep)
        # Update governance metric detail for regression
        gov.metric_name = "mae"
        gov.baseline_metric = 0.15
        gov.retrained_metric = 0.11

        gen = ExecutiveReportGenerator()
        report = gen.generate(
            execution_report=rep,
            critique=crit,
            optimization=opt,
            governance=gov,
        )

        assert isinstance(report, ExecutiveReport)
        assert report.problem_summary["problem_type"] == ProblemType.REGRESSION
        assert report.champion_summary["metric_name"] == "mae"
        assert len(report.models_summary) == 1
        assert report.models_summary[0]["family"] == ModelFamily.LINEAR_REGRESSION

    def test_validation_rejects_none(self):
        """Verify None inputs raise ExecutiveReportGeneratorError."""
        rep = _make_mock_report()
        crit = _make_mock_critique()
        opt = _make_mock_optimization()
        gov = _make_mock_governance()

        gen = ExecutiveReportGenerator()

        with pytest.raises(ExecutiveReportGeneratorError, match="execution_report cannot be None"):
            gen.generate(execution_report=None, critique=crit, optimization=opt, governance=gov)

        with pytest.raises(ExecutiveReportGeneratorError, match="critique cannot be None"):
            gen.generate(execution_report=rep, critique=None, optimization=opt, governance=gov)

        with pytest.raises(ExecutiveReportGeneratorError, match="optimization cannot be None"):
            gen.generate(execution_report=rep, critique=crit, optimization=None, governance=gov)

        with pytest.raises(ExecutiveReportGeneratorError, match="governance cannot be None"):
            gen.generate(execution_report=rep, critique=crit, optimization=opt, governance=None)

    def test_validation_rejects_wrong_types(self):
        """Verify wrong input types raise ExecutiveReportGeneratorError."""
        rep = _make_mock_report()
        crit = _make_mock_critique()
        opt = _make_mock_optimization()
        gov = _make_mock_governance()

        gen = ExecutiveReportGenerator()

        with pytest.raises(ExecutiveReportGeneratorError, match="execution_report must be an ExecutionReport"):
            gen.generate(execution_report="not-report", critique=crit, optimization=opt, governance=gov)

        with pytest.raises(ExecutiveReportGeneratorError, match="critique must be a ModelCritique"):
            gen.generate(execution_report=rep, critique="not-critique", optimization=opt, governance=gov)

        with pytest.raises(ExecutiveReportGeneratorError, match="optimization must be an OptimizationResult"):
            gen.generate(execution_report=rep, critique=crit, optimization="not-opt", governance=gov)

        with pytest.raises(ExecutiveReportGeneratorError, match="governance must be a ChampionDecision"):
            gen.generate(execution_report=rep, critique=crit, optimization=opt, governance="not-gov")

    def test_identity_preservations_mismatches(self):
        """Verify identity mismatches across report, critique, optimization, and governance raise ExecutiveReportGeneratorError."""
        rep = _make_mock_report(report_id="rep_01", plan_id="plan_01")
        crit = _make_mock_critique(report_id="rep_01")
        opt = _make_mock_optimization(baseline_plan_id="plan_01")
        gov = _make_mock_governance(baseline_report_id="rep_01")

        gen = ExecutiveReportGenerator()

        # 1. Critique report_id mismatch
        crit_bad = _make_mock_critique(report_id="rep_mismatch")
        with pytest.raises(ExecutiveReportGeneratorError, match="Identity mismatch.*critique"):
            gen.generate(execution_report=rep, critique=crit_bad, optimization=opt, governance=gov)

        # 2. Optimization baseline_plan_id mismatch
        opt_bad = _make_mock_optimization(baseline_plan_id="plan_mismatch")
        with pytest.raises(ExecutiveReportGeneratorError, match="Identity mismatch.*optimization"):
            gen.generate(execution_report=rep, critique=crit, optimization=opt_bad, governance=gov)

        # 3. Governance baseline_report_id mismatch
        gov_bad = _make_mock_governance(baseline_report_id="rep_mismatch")
        with pytest.raises(ExecutiveReportGeneratorError, match="Identity mismatch.*governance"):
            gen.generate(execution_report=rep, critique=crit, optimization=opt, governance=gov_bad)

    def test_non_mutation(self):
        """Verify generate does not mutate input artifacts."""
        rep = _make_mock_report()
        crit = _make_mock_critique()
        opt = _make_mock_optimization()
        gov = _make_mock_governance()

        rep_copy = copy.deepcopy(rep)
        crit_copy = copy.deepcopy(crit)
        opt_copy = copy.deepcopy(opt)
        gov_copy = copy.deepcopy(gov)

        gen = ExecutiveReportGenerator()
        gen.generate(
            execution_report=rep,
            critique=crit,
            optimization=opt,
            governance=gov,
        )

        assert rep == rep_copy
        assert crit == crit_copy
        assert opt.baseline_plan_id == opt_copy.baseline_plan_id
        assert gov.baseline_report_id == gov_copy.baseline_report_id
