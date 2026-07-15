"""Unit tests for ChampionComparator."""

from __future__ import annotations

import copy
import pytest

from backend.app.compute_capabilities import AcceleratorType, ComputeTier
from backend.app.ml_plan import ModelFamily, SearchStrategy
from backend.app.problem_definition.schemas import ProblemType
from backend.app.ml_execution.execution_report import (
    CandidateSummary,
    ChampionSummary,
    ExecutionReport,
)
from backend.app.model_governance.schemas import ChampionDecision, Winner
from backend.app.model_governance.comparator import (
    ChampionComparator,
    ChampionComparatorError,
)


# ── Helper Builders ───────────────────────────────────────────────────


def _make_sample_report(
    report_id: str = "rep_01",
    dataset_id: str = "ds_01",
    definition_id: str = "pd_01",
    target_column: str = "species",
    primary_metric: str = "f1",
    primary_metric_value: float = 0.92,
    production_ready: bool = False,
) -> ExecutionReport:
    champ = ChampionSummary(
        candidate_id="model_01",
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        primary_metric=primary_metric,
        primary_metric_value=primary_metric_value,
        feature_importance={"feat_a": 0.8},
        training_duration=0.1,
        evaluation_duration=0.01,
    )
    return ExecutionReport(
        report_id=report_id,
        dataset_id=dataset_id,
        request_id="req_123",
        problem_definition_id=definition_id,
        plan_id="plan_789",
        execution_id="exec_999",
        problem_type=ProblemType.CLASSIFICATION,
        target_column=target_column,
        feature_columns=["feat_a"],
        compute_tier=ComputeTier.STANDARD,
        accelerator_type=AcceleratorType.NONE,
        candidate_summaries=[],
        champion_summary=champ,
        training_summary={"production_ready": production_ready},
        evaluation_summary={},
        warnings=[],
        execution_duration=0.5,
        created_timestamp="2026-07-15T00:00:00Z",
    )


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestChampionComparator:
    """Tests covering deterministic model champion comparisons and governance."""

    def test_compare_classification_higher_metric_wins(self):
        """Verify retrained model wins if metric score is higher (Classification/F1)."""
        base = _make_sample_report(report_id="rep_base", primary_metric_value=0.85, production_ready=True)
        retrain = _make_sample_report(report_id="rep_retrain", primary_metric_value=0.88, production_ready=True)

        comp = ChampionComparator()
        decision = comp.compare(base, retrain)

        assert isinstance(decision, ChampionDecision)
        assert decision.winner == Winner.RETRAINED
        assert decision.improvement_detected is True
        assert decision.production_ready is True
        assert decision.baseline_metric == 0.85
        assert decision.retrained_metric == 0.88
        assert pytest.approx(decision.relative_improvement, abs=1e-5) == 0.035294

    def test_compare_classification_baseline_wins(self):
        """Verify baseline wins if retrained score is lower."""
        base = _make_sample_report(report_id="rep_base", primary_metric_value=0.90, production_ready=True)
        retrain = _make_sample_report(report_id="rep_retrain", primary_metric_value=0.85, production_ready=True)

        comp = ChampionComparator()
        decision = comp.compare(base, retrain)

        assert decision.winner == Winner.BASELINE
        assert decision.improvement_detected is False
        assert decision.production_ready is True

    def test_compare_classification_tie_under_limit(self):
        """Verify tie Winner label is returned if difference is within 1e-9 tolerance."""
        base = _make_sample_report(report_id="rep_base", primary_metric_value=0.8500000001, production_ready=True)
        retrain = _make_sample_report(report_id="rep_retrain", primary_metric_value=0.8500000002, production_ready=True)

        comp = ChampionComparator()
        decision = comp.compare(base, retrain)

        assert decision.winner == Winner.TIE
        assert decision.winner_report.report_id == "rep_base"
        assert decision.improvement_detected is False
        assert decision.production_ready is True

    def test_compare_regression_lower_metric_wins(self):
        """Verify retrained model wins if metric score is lower (Regression/MAE)."""
        base = _make_sample_report(report_id="rep_base", primary_metric="mae", primary_metric_value=0.15, production_ready=True)
        retrain = _make_sample_report(report_id="rep_retrain", primary_metric="mae", primary_metric_value=0.12, production_ready=True)

        comp = ChampionComparator()
        decision = comp.compare(base, retrain)

        assert decision.winner == Winner.RETRAINED
        assert decision.improvement_detected is True
        assert pytest.approx(decision.relative_improvement, 1e-6) == 0.20

    def test_compare_regression_baseline_wins(self):
        """Verify baseline wins if retrained MAE score is higher (worse)."""
        base = _make_sample_report(report_id="rep_base", primary_metric="mae", primary_metric_value=0.10, production_ready=True)
        retrain = _make_sample_report(report_id="rep_retrain", primary_metric="mae", primary_metric_value=0.12, production_ready=True)

        comp = ChampionComparator()
        decision = comp.compare(base, retrain)

        assert decision.winner == Winner.BASELINE
        assert decision.improvement_detected is False

    def test_compare_regression_higher_r2_wins(self):
        """Verify R2 metric (Regression but higher-is-better) works correctly."""
        base = _make_sample_report(report_id="rep_base", primary_metric="r2", primary_metric_value=0.75, production_ready=True)
        retrain = _make_sample_report(report_id="rep_retrain", primary_metric="r2", primary_metric_value=0.82, production_ready=True)

        comp = ChampionComparator()
        decision = comp.compare(base, retrain)

        assert decision.winner == Winner.RETRAINED
        assert decision.improvement_detected is True

    def test_compare_zero_baseline_value_relative_improvement(self):
        """Verify relative improvement computation handles zero baseline value without division by zero error."""
        base = _make_sample_report(report_id="rep_base", primary_metric_value=0.0, production_ready=True)
        retrain = _make_sample_report(report_id="rep_retrain", primary_metric_value=0.80, production_ready=True)

        comp = ChampionComparator()
        decision = comp.compare(base, retrain)
        assert decision.relative_improvement == 0.0

    def test_validation_rejects_none(self):
        """Verify None inputs raise ChampionComparatorError."""
        report = _make_sample_report()
        comp = ChampionComparator()

        with pytest.raises(ChampionComparatorError, match="baseline_report cannot be None"):
            comp.compare(None, report)

        with pytest.raises(ChampionComparatorError, match="retrained_report cannot be None"):
            comp.compare(report, None)

    def test_validation_rejects_wrong_types(self):
        """Verify incorrect parameter types raise ChampionComparatorError."""
        report = _make_sample_report()
        comp = ChampionComparator()

        with pytest.raises(ChampionComparatorError, match="must be an ExecutionReport instance"):
            comp.compare("not-a-report", report)

        with pytest.raises(ChampionComparatorError, match="must be an ExecutionReport instance"):
            comp.compare(report, "not-a-report")

    def test_identity_preservations_mismatches(self):
        """Verify dataset, definition, and target column mismatches raise ChampionComparatorError."""
        comp = ChampionComparator()

        # 1. Dataset ID mismatch
        r1 = _make_sample_report(dataset_id="ds_base")
        r2 = _make_sample_report(dataset_id="ds_mismatch")
        with pytest.raises(ChampionComparatorError, match="Identity mismatch.*dataset_id"):
            comp.compare(r1, r2)

        # 2. ProblemDefinition ID mismatch
        r1 = _make_sample_report(definition_id="pd_base")
        r2 = _make_sample_report(definition_id="pd_mismatch")
        with pytest.raises(ChampionComparatorError, match="Identity mismatch.*problem_definition_id"):
            comp.compare(r1, r2)

        # 3. Target column mismatch
        r1 = _make_sample_report(target_column="y_base")
        r2 = _make_sample_report(target_column="y_mismatch")
        with pytest.raises(ChampionComparatorError, match="Identity mismatch.*target_column"):
            comp.compare(r1, r2)

    def test_metric_mismatches(self):
        """Verify mismatched primary metric names raise ChampionComparatorError."""
        r1 = _make_sample_report(primary_metric="f1")
        r2 = _make_sample_report(primary_metric="accuracy")

        comp = ChampionComparator()
        with pytest.raises(ChampionComparatorError, match="Metric mismatch"):
            comp.compare(r1, r2)

    def test_unsupported_metric_fails(self):
        """Verify unsupported metrics raise ChampionComparatorError."""
        r1 = _make_sample_report(primary_metric="unsupported_metric")
        r2 = _make_sample_report(primary_metric="unsupported_metric")

        comp = ChampionComparator()
        with pytest.raises(ChampionComparatorError, match="Unsupported primary metric"):
            comp.compare(r1, r2)

    def test_non_mutation(self):
        """Verify compare function does not mutate report inputs."""
        r1 = _make_sample_report(report_id="rep_1", primary_metric_value=0.85, production_ready=True)
        r2 = _make_sample_report(report_id="rep_2", primary_metric_value=0.90, production_ready=True)

        r1_copy = copy.deepcopy(r1)
        r2_copy = copy.deepcopy(r2)

        comp = ChampionComparator()
        comp.compare(r1, r2)

        assert r1 == r1_copy
        assert r2 == r2_copy
