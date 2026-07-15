"""Unit and integration tests for the Training Results Dashboard page (Stage 12I)."""

import os
import json
import pickle
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from PySide6.QtWidgets import QMessageBox, QTabWidget
from PySide6.QtCore import QThread
from frontend.app import DeployAIApplication
from frontend.pages.reports_page import ReportsPage
from frontend.pages.training_results import TrainingResultsPage

# Schemas
from backend.app.ml_plan.schemas import ModelFamily, ProblemType, ComputeTier, AcceleratorType, MLPlan
from backend.app.ml_execution.execution_report import ExecutionReport, CandidateSummary, ChampionSummary
from backend.app.ai_model_critic.schemas import ModelCritique, CritiqueGrade
from backend.app.ai_model_optimizer.schemas import OptimizationResult, OptimizationAction, OptimizationActionType
from backend.app.model_governance.schemas import ChampionDecision, Winner
from backend.app.reporting.schemas import ExecutiveReport


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """Setup and clean up the QApplication reuse state."""
    DeployAIApplication._allow_qapp_reuse = True
    DeployAIApplication._reset()
    yield
    DeployAIApplication._reset()
    DeployAIApplication._allow_qapp_reuse = False


def _write_mock_run_files(run_dir: Path, include_optional: bool = True) -> None:
    # 1. Execution Report
    champ = ChampionSummary(
        candidate_id="m_01",
        model_family=ModelFamily.RANDOM_FOREST,
        primary_metric="accuracy",
        primary_metric_value=0.85,
        feature_importance={"col1": 1.0},
        training_duration=1.5,
        evaluation_duration=0.2,
    )
    
    cand = CandidateSummary(
        candidate_id="m_01",
        model_family=ModelFamily.RANDOM_FOREST,
        primary_metric="accuracy",
        primary_metric_value=0.85,
        training_duration=1.5,
        evaluation_duration=0.2,
        search_strategy="none",
        best_parameters={},
    )
    
    report = ExecutionReport(
        report_id="rep_01",
        dataset_id="d_01",
        request_id="r_01",
        problem_definition_id="pd_01",
        plan_id="p_01",
        execution_id="exec_01",
        problem_type=ProblemType.CLASSIFICATION,
        target_column="target_col",
        feature_columns=["col1"],
        compute_tier=ComputeTier.STANDARD,
        accelerator_type=AcceleratorType.NONE,
        candidate_summaries=[cand],
        champion_summary=champ,
        training_summary={"total_candidates": 1},
        evaluation_summary={"train_score": 0.9, "test_score": 0.85, "confusion_matrix": [[10, 1], [2, 12]]},
        warnings=[],
        execution_duration=2.5,
        created_timestamp="2026-07-15T12:00:00Z",
    )
    
    with open(run_dir / "execution_report.json", "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=4))

    # 2. Metrics summary
    metrics_summary = {"m_01": 0.85}
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f)

    # 3. Training log
    with open(run_dir / "training.log", "w", encoding="utf-8") as f:
        f.write("Preparing dataset...\nEvaluating candidate model m_01...\nFinished.")

    # 4. Model pkl dummy
    with open(run_dir / "model.pkl", "wb") as f:
        f.write(b"dummy serialized estimator bytes")

    if include_optional:
        # 5. Critique report
        critique = ModelCritique(
            critique_id="crit_01",
            report_id="rep_01",
            overall_grade=CritiqueGrade.A,
            production_ready=True,
            confidence=0.95,
            strengths=["High accuracy"],
            weaknesses=["Slight overfitting"],
            risks=["Out-of-distribution drift"],
            recommendations=["Add CV"],
            warnings=["Low class representation"],
            summary="Excellent initial candidate.",
        )
        with open(run_dir / "critique.json", "w", encoding="utf-8") as f:
            f.write(critique.model_dump_json(indent=4))

        # 6. Optimization result
        plan_dict = {
            "plan_id": "p_01",
            "dataset_id": "d_01",
            "request_id": "r_01",
            "problem_definition_id": "pd_01",
            "compute_capability_id": "cc_01",
            "problem_type": "classification",
            "target_column": "target_col",
            "split_plan": {
                "strategy": "random",
                "test_size": 0.2,
                "validation_size": 0.1,
                "shuffle": True,
                "random_state": 42,
                "reason": "Standard split."
            },
            "feature_columns": ["col1"],
            "preprocessing_steps": [],
            "feature_engineering_steps": [],
            "feature_selection": {
                "method": "none",
                "candidate_columns": ["col1"],
                "max_features": None,
                "parameters": {},
                "reason": "None"
            },
            "model_candidates": [
                {
                    "candidate_id": "m_01",
                    "model_family": "random_forest",
                    "search_strategy": "none",
                    "parameters": {},
                    "search_space": {},
                    "reason": "Simple RF"
                }
            ],
            "evaluation_plan": {
                "primary_metric": "accuracy",
                "secondary_metrics": [],
                "cross_validation_folds": 5
            },
            "execution_constraints": {
                "parallel_workers": 4,
                "use_gpu_acceleration": False,
                "accelerator_type": "none",
                "compute_tier": "standard"
            }
        }
        plan = MLPlan.model_validate(plan_dict)
        action = OptimizationAction(
            action_id="act_01",
            action_type=OptimizationActionType.ADD_PREPROCESSING,
            target="col1",
            replacement="scaled_col1",
            parameters={"scale": True},
            reason="Scale numerical features",
            confidence=0.85,
        )
        opt = OptimizationResult(
            optimization_id="opt_01",
            baseline_plan_id="p_01",
            optimized_plan=plan,
            actions=[action],
            summary="Optimized baseline features.",
        )
        with open(run_dir / "optimization.json", "w", encoding="utf-8") as f:
            f.write(opt.model_dump_json(indent=4))

        # 7. Governance decision
        gov = ChampionDecision(
            decision_id="dec_01",
            baseline_report_id="rep_01",
            retrained_report_id="rep_02",
            winner=Winner.RETRAINED,
            winner_report=report,
            improvement_detected=True,
            metric_name="accuracy",
            baseline_metric=0.82,
            retrained_metric=0.85,
            relative_improvement=0.036,
            decision_reason="Retrained model metric is higher than baseline.",
            production_ready=True,
            comparison_timestamp="2026-07-15T12:00:00Z",
        )
        with open(run_dir / "governance.json", "w", encoding="utf-8") as f:
            f.write(gov.model_dump_json(indent=4))

        # 8. Executive report
        exec_rep = ExecutiveReport(
            report_id="ex_01",
            title="Executive Summary Report",
            generated_timestamp="2026-07-15T12:00:00Z",
            problem_summary={},
            dataset_summary={},
            pipeline_summary={},
            models_summary=[],
            champion_summary={},
            optimization_summary={},
            ai_review={},
            governance_summary={},
            deployment_summary={},
            warnings=[],
            recommendations=[],
            executive_summary="Highly performing model ready.",
        )
        with open(run_dir / "executive_report.json", "w", encoding="utf-8") as f:
            f.write(exec_rep.model_dump_json(indent=4))


def test_results_dashboard_creation_and_widgets() -> None:
    """Verify results dashboard page widgets are correctly instantiated."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = None
    
    page = TrainingResultsPage(orchestrator=mock_orch)
    assert page.lbl_title.text() == "Training Results Dashboard"
    assert page.tabs.count() == 7
    assert page.btn_open_folder.isEnabled() is True
    assert page.btn_export.isEnabled() is True
    assert page.btn_compare.isEnabled() is False
    assert page.btn_retrain.isEnabled() is False


def test_results_dashboard_empty_state() -> None:
    """Verify empty/placeholder values when no workspace or run is active."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = None

    page = TrainingResultsPage(orchestrator=mock_orch)
    page.refresh_results()
    assert page.lbl_champion.text() == "-"
    assert page.lbl_over_score.text() == "-"
    
    # Optional tabs are hidden
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_ai_review)) is False
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_optimization)) is False
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_governance)) is False


def test_results_dashboard_success_flow(tmp_path: Path) -> None:
    """Verify full success path with all optional files present."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    # Setup active run directory
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "run_2026_07_15_143012"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_mock_run_files(run_dir, include_optional=True)

    page = TrainingResultsPage(orchestrator=mock_orch)
    page.refresh_results()

    # Verify header summary
    assert page.lbl_champion.text() == "m_01"
    assert page.lbl_problem_type.text() == "Classification"
    assert page.lbl_total_duration.text() == "2.50s"
    assert page.lbl_best_duration.text() == "1.50s"

    # Verify Overview Tab
    assert page.lbl_over_score.text() == "0.8500"
    assert page.lbl_over_ready.text() == "YES (AI Approved)"
    assert page.lbl_over_grade.text() == "A"
    assert "dummy serialized" in page.lbl_meta_path.text() or page.lbl_meta_size.text() != "-"

    # Verify Metrics Tab
    assert page.lbl_met_primary.text() == "ACCURACY"
    assert page.lbl_met_train.text() == "0.9"
    assert page.lbl_met_test.text() == "0.85"
    assert page.tbl_fi.rowCount() == 1
    assert page.tbl_fi.item(0, 0).text() == "col1"

    # Verify optional tabs are visible and loaded
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_ai_review)) is True
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_optimization)) is True
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_governance)) is True

    # Check critique fields
    assert "critique_id" in page.critique.model_dump_json()
    assert page.lbl_ai_confidence.text() == "95.0%"

    # Check logs
    assert "Preparing dataset..." in page.txt_logs.toPlainText()


def test_results_dashboard_missing_optional_files(tmp_path: Path) -> None:
    """Verify that critique, optimization, and governance are safely hidden when files are missing."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    # Setup active run directory with required files only
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "run_2026_07_15_143012"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_mock_run_files(run_dir, include_optional=False)

    page = TrainingResultsPage(orchestrator=mock_orch)
    page.refresh_results()

    # Required files load correctly
    assert page.lbl_champion.text() == "m_01"
    assert page.lbl_over_ready.text() == "N/A"
    assert page.lbl_over_grade.text() == "N/A"

    # Optional tabs are hidden
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_ai_review)) is False
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_optimization)) is False
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_governance)) is False


def test_results_dashboard_corrupt_files_graceful_fallback(tmp_path: Path) -> None:
    """Verify that corrupt JSON reports raise warnings and trigger clean empty state fallback."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    # Setup active run directory with corrupt/invalid files
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "run_2026_07_15_143012"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "execution_report.json", "w", encoding="utf-8") as f:
        f.write("{corrupt: json_format}")
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        f.write("[]")
    with open(run_dir / "training.log", "w", encoding="utf-8") as f:
        f.write("Log trace")

    page = TrainingResultsPage(orchestrator=mock_orch)
    
    with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
        page.refresh_results()
        mock_warn.assert_called_once()
        # Clean empty fallback state verified
        assert page.lbl_champion.text() == "-"


def test_results_dashboard_latest_link_priority(tmp_path: Path) -> None:
    """Verify that 'latest' directory takes precedence over other timestamp folders."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Old timestamp folder
    run_old = runs_dir / "run_2026_07_15_143012"
    run_old.mkdir(parents=True, exist_ok=True)
    _write_mock_run_files(run_old, include_optional=False)

    # Latest link/directory
    run_latest = runs_dir / "latest"
    run_latest.mkdir(parents=True, exist_ok=True)
    _write_mock_run_files(run_latest, include_optional=True)

    page = TrainingResultsPage(orchestrator=mock_orch)
    page.refresh_results()

    # Verified loaded from 'latest' (optional tabs are visible)
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.tab_ai_review)) is True


def test_results_dashboard_open_folder_actions(tmp_path: Path) -> None:
    """Verify Open Folder action calls QDesktopServices correctly."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "run_2026_07_15_143012"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_mock_run_files(run_dir, include_optional=False)

    page = TrainingResultsPage(orchestrator=mock_orch)
    page.refresh_results()

    with patch("PySide6.QtGui.QDesktopServices.openUrl") as mock_open:
        page._on_open_folder()
        mock_open.assert_called_once()


def test_results_dashboard_export_report_actions(tmp_path: Path) -> None:
    """Verify Export Report action shows file dialog and copies report successfully."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "run_2026_07_15_143012"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_mock_run_files(run_dir, include_optional=False)

    page = TrainingResultsPage(orchestrator=mock_orch)
    page.refresh_results()

    target_export = tmp_path / "exported_report.json"

    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(target_export), "JSON Files (*.json)")), \
         patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
        page._on_export()
        mock_info.assert_called_once()
        assert target_export.exists()


def test_results_page_monkey_patched_reports_page(tmp_path: Path) -> None:
    """Verify that ReportsPage is monkey-patched correctly and hosts TrainingResultsPage."""
    app_obj = DeployAIApplication()
    
    # Mock window orchestrator
    mock_win = MagicMock()
    mock_win.orchestrator = MagicMock()
    mock_win.orchestrator.active_workspace_path = str(tmp_path)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "run_2026_07_15_143012"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_mock_run_files(run_dir, include_optional=False)

    # Instantiate ReportsPage
    page = ReportsPage()
    
    with patch.object(page, "window", return_value=mock_win):
        # Trigger showEvent
        page.show()
    
    # Verify dynamic patch container is instantiated
    assert hasattr(page, "_results_widget")
    assert isinstance(page._results_widget, TrainingResultsPage)
    assert page._results_widget.lbl_champion.text() == "m_01"
