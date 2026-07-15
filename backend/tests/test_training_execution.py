"""Unit and integration tests for Training Execution Page and worker thread (Stage 12H)."""

import os
import json
import pickle
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QThread
from frontend.app import DeployAIApplication
from frontend.pages.training_page import TrainingPage
from frontend.pages.training_execution import TrainingExecutionPage
from frontend.pages.workers.training_worker import TrainingWorker
from backend.app.ml_plan.schemas import MLPlan
from backend.app.ml_execution.evaluation_engine import EvaluationResult


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """Setup and clean up the QApplication reuse state."""
    DeployAIApplication._allow_qapp_reuse = True
    DeployAIApplication._reset()
    yield
    DeployAIApplication._reset()
    DeployAIApplication._allow_qapp_reuse = False


def _make_mock_metadata(name: str, location: str, size: int = 100) -> dict:
    return {
        "name": name,
        "file_name": os.path.basename(location),
        "file_type": "CSV",
        "rows": 15,
        "columns": 3,
        "size_bytes": size,
        "imported_time": "2026-07-15T12:00:00Z",
        "location": location,
        "missing_values": 0,
        "duplicate_rows": 0,
        "target": "target_col",
    }


def _wait_for_worker(worker: QThread) -> None:
    from PySide6.QtWidgets import QApplication
    while worker.isRunning():
        QApplication.processEvents()
        QThread.msleep(10)
    for _ in range(10):
        QApplication.processEvents()
        QThread.msleep(1)


def _write_valid_plan(filepath: Path) -> None:
    plan_dict = {
        "plan_id": "p_01",
        "dataset_id": "d_01",
        "request_id": "r_01",
        "problem_definition_id": "pd_01",
        "compute_capability_id": "c_01",
        "problem_type": "classification",
        "target_column": "target_col",
        "feature_columns": ["col1"],
        "preprocessing_steps": [],
        "feature_engineering_steps": [],
        "feature_selection": {
            "method": "none",
            "candidate_columns": ["col1"],
            "max_features": None,
            "parameters": {},
            "reason": "None required."
        },
        "split_plan": {
            "strategy": "random",
            "test_size": 0.2,
            "validation_size": 0.1,
            "shuffle": True,
            "random_state": 42,
            "reason": "Standard split."
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
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(plan_dict, f)


def test_page_creation_and_widgets(tmp_path: Path) -> None:
    """Verify that widgets exist and are correctly configured initially."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = None

    page = TrainingExecutionPage(orchestrator=mock_orch)
    page.refresh_page()
    assert page.lbl_project.text() == "No active project"
    assert page.lbl_dataset.text() == "-"
    assert page.lbl_plan.text() == "-"
    
    assert page.btn_start.isEnabled() is True
    assert page.btn_cancel.isEnabled() is False
    assert page.btn_open_results.isEnabled() is False


def test_validation_rules_start_training(tmp_path: Path) -> None:
    """Verify Start Training fails with QMessageBox warning when context is missing."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = None

    page = TrainingExecutionPage(orchestrator=mock_orch)

    # 1. No workspace path
    with patch("frontend.pages.training_execution.QMessageBox.warning") as mock_warn:
        page._on_start_training()
        mock_warn.assert_called_with(page, "Workspace Closed", "Please open or create a project workspace first.")

    # 2. Plan missing
    mock_orch.active_workspace_path = str(tmp_path)
    page.refresh_page()
    with patch("frontend.pages.training_execution.QMessageBox.warning") as mock_warn:
        page._on_start_training()
        mock_warn.assert_called_with(page, "Plan Missing", "Please generate an ML Plan before starting training.")

    # 3. Dataset missing
    _write_valid_plan(tmp_path / "configs" / "ml_plan.json")
    # No datasets folder or no json file
    page.refresh_page()
    with patch("frontend.pages.training_execution.QMessageBox.warning") as mock_warn:
        page._on_start_training()
        mock_warn.assert_called_with(page, "Dataset Missing", "No imported dataset found inside this workspace.")


def test_training_page_stack_integration(tmp_path: Path) -> None:
    """Verify training summary page proceeding navigates to execution index."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    # Write configs
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    _write_valid_plan(configs_dir / "ml_plan.json")

    # Launch page
    parent_page = TrainingPage(orchestrator=mock_orch)
    parent_page.refresh_page()
    assert parent_page.stack.currentIndex() == 2

    # Click Proceed button
    parent_page.btn_proceed_training.click()
    assert parent_page.stack.currentIndex() == 3

    # Refresh while on index 3 maintains index 3
    parent_page.refresh_page()
    assert parent_page.stack.currentIndex() == 3


def test_training_worker_success_flow(tmp_path: Path) -> None:
    """Verify training thread signals update progress labels and dump files on success."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    _write_valid_plan(configs_dir / "ml_plan.json")

    # Touch mock dataset CSV
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    csv_file = datasets_dir / "ds1.csv"
    df = pd.DataFrame({"col1": list(range(15)), "target_col": [0, 1] * 7 + [0]})
    df.to_csv(csv_file, index=False)

    meta = _make_mock_metadata("d_01", str(csv_file))
    with open(datasets_dir / "ds1.csv.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    page = TrainingExecutionPage(orchestrator=mock_orch)
    page.refresh_page()
    assert page.lbl_dataset.text() == "d_01 (CSV)"

    # Patch backend orchestrator to run quickly and return custom mock result
    from backend.app.ml_plan.schemas import ModelFamily
    from backend.app.ml_execution.orchestrator import MLExecutionResult
    from backend.app.ml_execution.evaluation_engine import EvaluationResult

    eval_res = EvaluationResult(
        candidate_id="m_01",
        model_family=ModelFamily.RANDOM_FOREST,
        predictions=[],
        primary_metric="accuracy",
        primary_metric_value=0.85,
        all_metrics={"accuracy": 0.85},
        confusion_matrix=[[1, 2], [3, 4]],
        classification_report={},
        feature_importance={},
        cross_validation_scores=[0.8, 0.85, 0.9],
        evaluation_duration_seconds=0.1,
        train_score=0.9,
        test_score=0.85,
        prediction_count=10,
        training_duration=0.5,
        evaluation_duration=0.1,
        model_parameters={},
        warnings=[],
        execution_summary={}
    )
    
    mock_exec_res = MLExecutionResult(
        plan_id="p_01",
        problem_definition_id="pd_01",
        candidate_results={"m_01": eval_res},
        best_candidate_id="m_01",
        best_model=MagicMock(),
        best_evaluation=eval_res,
        execution_duration_seconds=1.2,
        execution_summary={
            "best_score": 0.85,
            "plan_id": "p_01",
            "candidates_evaluated": 1,
            "best_candidate_id": "m_01",
            "primary_metric": "accuracy",
            "duration_seconds": 1.2
        }
    )

    with patch("backend.app.ml_execution.orchestrator.MLExecutionOrchestrator.execute", return_value=mock_exec_res) as mock_execute:
        # Patch pickle.dump to avoid pickling MagicMock best_model
        with patch("pickle.dump") as mock_pickle:
            with patch("frontend.pages.training_execution.QMessageBox.information") as mock_info, \
                 patch("frontend.pages.training_execution.QMessageBox.critical") as mock_crit:
                page._on_start_training()
                assert page.worker is not None
                _wait_for_worker(page.worker)
                
                # Check results
                mock_info.assert_called_once()
                mock_crit.assert_not_called()
                assert page.btn_start.isEnabled() is True
                assert page.btn_cancel.isEnabled() is False
                assert page.btn_open_results.isEnabled() is True

            # Verify files are successfully dumped
            report_file = tmp_path / "reports" / "execution_report.json"
            assert report_file.exists()

            # Verify runs folder timestamped structure
            runs_dir = tmp_path / "runs"
            assert runs_dir.exists()
            run_folders = list(runs_dir.glob("run_*"))
            assert len(run_folders) == 1
            run_dir = run_folders[0]
            assert (run_dir / "ml_plan.json").exists()
            assert (run_dir / "execution_report.json").exists()
            assert (run_dir / "model.pkl").exists()
            assert (run_dir / "training.log").exists()
            assert (run_dir / "metrics.json").exists()


def test_training_worker_failure_flow(tmp_path: Path) -> None:
    """Verify training worker failure transitions UI buttons and raisesMessageBox critical dialog."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    _write_valid_plan(configs_dir / "ml_plan.json")

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    csv_file = datasets_dir / "ds1.csv"
    df = pd.DataFrame({"col1": list(range(15)), "target_col": [0, 1] * 7 + [0]})
    df.to_csv(csv_file, index=False)

    meta = _make_mock_metadata("d_01", str(csv_file))
    with open(datasets_dir / "ds1.csv.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    page = TrainingExecutionPage(orchestrator=mock_orch)
    page.refresh_page()

    with patch("backend.app.ml_execution.orchestrator.MLExecutionOrchestrator.execute", side_effect=ValueError("pipeline error")):
        with patch("frontend.pages.training_execution.QMessageBox.critical") as mock_crit:
            page._on_start_training()
            assert page.worker is not None
            _wait_for_worker(page.worker)
            
            mock_crit.assert_called_once()
            assert page.btn_start.isEnabled() is True
            assert page.btn_cancel.isEnabled() is False
            assert page.btn_open_results.isEnabled() is False


def test_training_cancellation_flow(tmp_path: Path) -> None:
    """Verify Cancel button terminates training worker cleanly and shows failure dialog."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    mock_orch.active_workspace_path = str(tmp_path)

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    _write_valid_plan(configs_dir / "ml_plan.json")

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    csv_file = datasets_dir / "ds1.csv"
    df = pd.DataFrame({"col1": list(range(15)), "target_col": [0, 1] * 7 + [0]})
    df.to_csv(csv_file, index=False)

    meta = _make_mock_metadata("d_01", str(csv_file))
    with open(datasets_dir / "ds1.csv.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    page = TrainingExecutionPage(orchestrator=mock_orch)
    page.refresh_page()

    # Simulate cancellation right when orchestrator runs
    def mock_run_cancel(*args, **kwargs):
        page.worker.cancel()
        raise RuntimeError("cancelled")

    with patch("backend.app.ml_execution.orchestrator.MLExecutionOrchestrator.execute", side_effect=mock_run_cancel):
        with patch("frontend.pages.training_execution.QMessageBox.critical") as mock_crit:
            page._on_start_training()
            _wait_for_worker(page.worker)
            
            mock_crit.assert_called_once()
            assert page.btn_start.isEnabled() is True
            assert page.btn_cancel.isEnabled() is False
            assert page.btn_open_results.isEnabled() is False


def test_training_open_results_navigation(tmp_path: Path) -> None:
    """Verify Open Results triggers page change via NavigationController."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    page = TrainingExecutionPage(orchestrator=mock_orch)

    # Mock window shell nav controller
    mock_nav = MagicMock()
    mock_win = MagicMock()
    mock_win.shell.nav_controller = mock_nav
    
    with patch.object(page, "window", return_value=mock_win):
        page._on_open_results()
        mock_nav.switch_to_page.assert_called_with("Reports")


def test_training_worker_missing_plan_file(tmp_path: Path) -> None:
    """Verify worker raises FileNotFoundError when plan is missing."""
    app_obj = DeployAIApplication()
    meta = _make_mock_metadata("d_01", str(tmp_path / "ds1.csv"))
    worker = TrainingWorker(workspace_path=str(tmp_path), dataset_metadata=meta)
    
    with patch.object(worker, "failed") as mock_failed:
        worker.run()
        mock_failed.emit.assert_called_with("ML Plan configuration file configs/ml_plan.json not found.")


def test_training_worker_missing_csv_file(tmp_path: Path) -> None:
    """Verify worker raises FileNotFoundError when csv file is missing."""
    app_obj = DeployAIApplication()
    _write_valid_plan(tmp_path / "configs" / "ml_plan.json")
    meta = _make_mock_metadata("d_01", str(tmp_path / "ds1_missing.csv"))
    worker = TrainingWorker(workspace_path=str(tmp_path), dataset_metadata=meta)

    with patch.object(worker, "failed") as mock_failed:
        worker.run()
        # The exact message contains 'Dataset source file not found'
        args, kwargs = mock_failed.emit.call_args
        assert "Dataset source file not found" in args[0]


def test_timer_tick(tmp_path: Path) -> None:
    """Verify UI timer tick calculates remaining values properly."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    page = TrainingExecutionPage(orchestrator=mock_orch)

    page.start_time = time.time() - 10 # 10 seconds elapsed
    page.progress_bar.setValue(20) # 20% progress
    page._on_timer_tick()
    assert "Elapsed Time: 00:10" in page.lbl_elapsed.text()
    # 10s * (100-20)/20 = 40s remaining = 00:40
    assert "Estimated Remaining Time: 00:40" in page.lbl_remaining.text()


def test_candidate_ui_updates(tmp_path: Path) -> None:
    """Verify active model candidate diagnostic fields update text values."""
    app_obj = DeployAIApplication()
    mock_orch = MagicMock()
    page = TrainingExecutionPage(orchestrator=mock_orch)

    page._on_candidate_changed("m_01", "Evaluating", "0.9123")
    assert page.lbl_model_name.text() == "m_01"
    assert page.lbl_model_status.text() == "Evaluating"
    assert page.lbl_model_metric.text() == "0.9123"
