"""Unit and integration tests for the ML Planning Wizard (Stage 12G)."""

import os
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from PySide6.QtWidgets import QMessageBox, QDialog, QWizard
from PySide6.QtCore import Qt
from frontend.app import DeployAIApplication
from frontend.pages.planning_wizard import (
    MLPlanningWizard,
    DatasetSelectionPage,
    ProblemDefinitionPage,
    ComputeCapabilityPage,
    PlanningPage,
    AIPlanningPage,
    SummaryPage,
    MockAIProvider,
)
from frontend.pages.training_page import TrainingPage
from backend.app.ml_plan.schemas import MLPlan


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
        "rows": 120,
        "columns": 3,
        "size_bytes": size,
        "imported_time": "2026-07-15T12:00:00Z",
        "location": location,
        "missing_values": 0,
        "duplicate_rows": 0,
        "target": "target_col",
        "status": "Active",
        "column_names": ["col1", "col2", "target_col"]
    }


class TestMLPlanningWizard:

    def test_wizard_creation_and_flow(self, tmp_path: Path) -> None:
        """Verify the wizard holds 6 pages and initializes values."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))

        assert wizard.pageIds() == [0, 1, 2, 3, 4, 5]
        assert wizard.workspace_path == str(tmp_path)
        assert wizard.selected_dataset is None

    def test_step1_dataset_selection(self, tmp_path: Path) -> None:
        """Verify DatasetSelectionPage loads datasets list and updates selection state."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        page = wizard.page(0)

        # Before workspace exists, table is empty
        page.initializePage()
        assert page.table.rowCount() == 0
        assert page.isComplete() is False

        # Setup mock datasets
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        
        csv_file = datasets_dir / "ds1.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "target_col": [0, 1]})
        df.to_csv(csv_file, index=False)

        meta = _make_mock_metadata("Dataset One", str(csv_file), 256)
        with open(datasets_dir / "ds1.csv.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        page.initializePage()
        assert page.table.rowCount() == 1
        assert page.table.item(0, 0).text() == "Dataset One"
        assert wizard.selected_dataset == meta
        assert page.isComplete() is True

        # Deselect row
        page.table.clearSelection()
        assert wizard.selected_dataset is None
        assert page.isComplete() is False

    def test_step2_problem_definition_classification(self, tmp_path: Path) -> None:
        """Verify target column dropdown pre-selects default and resolves classification objective."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        csv_file = datasets_dir / "ds1.csv"
        # Discrete classes for target_col (classification classification task)
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
            "target_col": ["A", "B", "A", "B", "A"]
        })
        df.to_csv(csv_file, index=False)

        meta = _make_mock_metadata("Dataset One", str(csv_file), 256)
        wizard.selected_dataset = meta

        page = wizard.page(1)
        page.initializePage()

        assert page.combo_target.count() == 3
        assert page.combo_target.currentText() == "target_col"
        assert page.lbl_prob_type.text() == "Classification"
        assert "Accuracy" in page.lbl_goal.text()
        assert wizard.resolved_problem is not None
        assert page.isComplete() is True

    def test_step2_problem_definition_regression(self, tmp_path: Path) -> None:
        """Verify target column resolves regression objective for continuous variable."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        csv_file = datasets_dir / "ds1.csv"
        # Continuous numeric values (regression task)
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0],
            "col2": [10.0, 20.0, 30.0],
            "target_col": [0.55, 1.89, 4.31]
        })
        df.to_csv(csv_file, index=False)

        meta = _make_mock_metadata("Dataset One", str(csv_file), 256)
        wizard.selected_dataset = meta

        page = wizard.page(1)
        page.initializePage()

        assert page.lbl_prob_type.text() == "Regression"
        assert "RMSE" in page.lbl_goal.text()

    def test_step2_problem_definition_empty_or_errors(self, tmp_path: Path) -> None:
        """Verify problem definition fallback behavior on resolution errors or empty targets."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        page = wizard.page(1)

        # 1. No target current text
        page.combo_target.setCurrentText("")
        page._on_target_changed()
        assert wizard.resolved_problem is None
        assert page.isComplete() is False

        # 2. Location file doesn't exist
        meta = _make_mock_metadata("Bad", str(tmp_path / "missing.csv"))
        wizard.selected_dataset = meta
        page.combo_target.addItem("target_col")
        page.combo_target.setCurrentText("target_col")
        page._on_target_changed()
        assert wizard.resolved_problem is None

        # 3. Pandas load crashes
        csv_file = tmp_path / "corrupt.csv"
        csv_file.touch()
        meta = _make_mock_metadata("Bad", str(csv_file))
        wizard.selected_dataset = meta
        with patch("pandas.read_csv", side_effect=ValueError("pandas crash")):
            with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
                page._on_target_changed()
                mock_warn.assert_called_once()
                assert wizard.resolved_problem is None

    def test_step3_compute_capability(self, tmp_path: Path) -> None:
        """Verify compute specs are scanned and labels updated."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        page = wizard.page(2)

        page.initializePage()
        assert page.lbl_cpu.text() != "-"
        assert page.lbl_tier.text() in ["MINIMAL", "STANDARD", "PREMIUM", "BASIC"]

        # Scanner exception fallback
        with patch("backend.app.hardware.detector.HardwareDetector.detect", side_effect=RuntimeError("hardware error")):
            page.initializePage()
            assert page.lbl_tier.text() == "Error: failed hardware scan"

    def test_step4_planning_baseline(self, tmp_path: Path) -> None:
        """Verify baseline ML planning generates steps and populates tree widget."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        csv_file = datasets_dir / "ds1.csv"
        df = pd.DataFrame({"col1": list(range(15)), "col2": list(range(15)), "target_col": [0, 1] * 7 + [0]})
        df.to_csv(csv_file, index=False)

        wizard.selected_dataset = _make_mock_metadata("Ds", str(csv_file))
        
        # Populate dependencies on wizard
        wizard.page(1).initializePage() # Resolves resolved_problem & dataset_context
        wizard.page(2).initializePage() # Resolves compute_capabilities

        page = wizard.page(3)
        page.initializePage()
        assert page.isComplete() is False

        # Run generate plan click
        page._on_generate_plan()
        assert page.isComplete() is True
        assert wizard.baseline_plan is not None
        assert page.tree.topLevelItemCount() > 0

        # Trigger failure path
        wizard.dataset_context = None
        page.initializePage()
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            page._on_generate_plan()
            mock_warn.assert_called_with(page, "Orchestration Error", "Dependencies are incomplete. Return to previous steps.")

    def test_step5_ai_planning(self, tmp_path: Path) -> None:
        """Verify AI assisted planning checkbox toggles proposals, merging mock suggestions."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        csv_file = datasets_dir / "ds1.csv"
        df = pd.DataFrame({"col1": list(range(15)), "col2": list(range(15)), "target_col": [0, 1] * 7 + [0]})
        df.to_csv(csv_file, index=False)

        wizard.selected_dataset = _make_mock_metadata("Ds", str(csv_file))
        wizard.page(1).initializePage()
        wizard.page(2).initializePage()
        wizard.page(3)._on_generate_plan()

        page = wizard.page(4)
        page.initializePage()
        assert page.info_group.isEnabled() is False

        # Check enable AI checkbox
        page.chk_ai.setChecked(True)
        assert page.info_group.isEnabled() is True
        assert page.lbl_suggestions.text() != "-"
        assert page.lbl_applied.text() == "Yes"
        assert page.lbl_confidence.text() == "HIGH"

        # Check disable AI checkbox reverts plan
        page.chk_ai.setChecked(False)
        assert page.info_group.isEnabled() is False
        assert page.lbl_suggestions.text() == "-"

        # AI Planning error handling
        page.chk_ai.setChecked(True)
        with patch("backend.app.ml_plan.orchestrator.MLPlanningOrchestrator.create_plan", side_effect=RuntimeError("AI crash")):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                page._on_ai_toggle()
                mock_crit.assert_called_with(page, "AI Assist Failed", "AI planning execution failed: AI crash")
                assert page.chk_ai.isChecked() is False

    def test_step6_summary_and_finish_stores_plan(self, tmp_path: Path) -> None:
        """Verify summary calculation is correct, and Finish action serializes final plan to disk."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        csv_file = datasets_dir / "ds1.csv"
        df = pd.DataFrame({"col1": list(range(15)), "col2": list(range(15)), "target_col": [0, 1] * 7 + [0]})
        df.to_csv(csv_file, index=False)

        wizard.selected_dataset = _make_mock_metadata("Ds", str(csv_file), 15000000000000000000) # Huge file
        wizard.page(1).initializePage()
        wizard.page(2).initializePage()
        wizard.page(3)._on_generate_plan()

        page = wizard.page(5)
        page.initializePage()
        assert page.lbl_status.text() == "Ready for Training"

        # Click Accept (Finish)
        wizard.accept()
        
        plan_file = tmp_path / "configs" / "ml_plan.json"
        assert plan_file.exists()
        
        with open(plan_file, "r", encoding="utf-8") as f:
            plan_json = json.load(f)
            assert plan_json["target_column"] == "target_col"

    def test_wizard_accept_failures(self, tmp_path: Path) -> None:
        """Verify accept does not save if plan is incomplete, and surfaces storage errors."""
        app_obj = DeployAIApplication()
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        wizard.final_plan = MagicMock()
        
        # Test directory write error wraps in QMessageBox.critical
        with patch("os.makedirs", side_effect=OSError("write block")):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                wizard.accept()
                mock_crit.assert_called_once()

    def test_mock_ai_provider_generate(self) -> None:
        """Verify MockAIProvider return format."""
        plan = MagicMock()
        plan.plan_id = "p_01"
        plan.dataset_id = "d_01"
        plan.request_id = "r_01"
        plan.problem_definition_id = "pd_01"
        plan.compute_capability_id = "c_01"
        plan.problem_type.value = "classification"

        provider = MockAIProvider(plan)
        res = provider.generate(system_prompt="", user_prompt="")
        data = json.loads(res)
        assert data["proposal_set_id"] == "ps_mock_01"
        assert data["evaluation_proposal"]["primary_metric"] is None


class TestTrainingPage:

    def test_page_states(self, tmp_path: Path) -> None:
        """Verify training page displays correct stack indexes for different states."""
        app_obj = DeployAIApplication()
        
        # State 1: No active orchestrator or workspace path
        page = TrainingPage(orchestrator=None)
        page.refresh_page()
        assert page.stack.currentIndex() == 0

        mock_orch = MagicMock()
        mock_orch.active_workspace_path = None
        page = TrainingPage(orchestrator=mock_orch)
        page.refresh_page()
        assert page.stack.currentIndex() == 0

        # State 2: Active workspace but no plan
        mock_orch.active_workspace_path = str(tmp_path)
        page.refresh_page()
        assert page.stack.currentIndex() == 1

        # State 3: Active workspace and active plan exists
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir(parents=True)
        
        # Build and serialize a valid MLPlan using baseline planner
        wizard = MLPlanningWizard(workspace_path=str(tmp_path))
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        csv_file = datasets_dir / "ds_state.csv"
        df = pd.DataFrame({"col1": list(range(15)), "col2": list(range(15)), "target_col": [0, 1] * 7 + [0]})
        df.to_csv(csv_file, index=False)
        wizard.selected_dataset = _make_mock_metadata("DsState", str(csv_file))
        wizard.page(1).initializePage()
        wizard.page(2).initializePage()
        wizard.page(3)._on_generate_plan()

        with open(configs_dir / "ml_plan.json", "w", encoding="utf-8") as f:
            f.write(wizard.final_plan.model_dump_json())

        page.refresh_page()
        assert page.stack.currentIndex() == 2
        assert page.lbl_target.text() == "target_col"
        assert page.lbl_problem_type.text() == "Classification"

        # Corrupt plan file handling
        with open(configs_dir / "ml_plan.json", "w", encoding="utf-8") as f:
            f.write("corrupt_plan")
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            page.refresh_page()
            mock_warn.assert_called_once()
            assert page.stack.currentIndex() == 1

    def test_launch_wizard_triggers(self, tmp_path: Path) -> None:
        """Verify launching wizard triggers creation and accepts updates."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.active_workspace_path = str(tmp_path)
        page = TrainingPage(orchestrator=mock_orch)

        # Setup mock files for wizard initial dataset scan
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        csv_file = datasets_dir / "ds1.csv"
        df = pd.DataFrame({"col1": list(range(15)), "col2": list(range(15)), "target_col": [0, 1] * 7 + [0]})
        df.to_csv(csv_file, index=False)
        meta = _make_mock_metadata("Dataset One", str(csv_file), 256)
        with open(datasets_dir / "ds1.csv.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Wizard launches and accepts
        with patch("PySide6.QtWidgets.QWizard.exec", return_value=QDialog.DialogCode.Accepted):
            page._launch_wizard()
            
        # Launching wizard warning when no active path
        mock_orch.active_workspace_path = None
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            page._launch_wizard()
            mock_warn.assert_called_with(page, "Workspace Closed", "Please open a project workspace first.")


def test_wizard_additional_branches(tmp_path: Path) -> None:
    """Trigger remaining branch lines in planning_wizard.py to achieve 100% coverage."""
    app_obj = DeployAIApplication()
    
    # 1. DatasetSelectionPage initializePage empty workspace path
    wiz = MLPlanningWizard(workspace_path="")
    page0 = wiz.page(0)
    page0.initializePage()
    assert wiz.selected_dataset is None

    # 2. DatasetSelectionPage initializePage json load exception
    wiz2 = MLPlanningWizard(workspace_path=str(tmp_path))
    page0_2 = wiz2.page(0)
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    corrupt_json = datasets_dir / "corrupt.json"
    with open(corrupt_json, "w", encoding="utf-8") as f:
        f.write("{invalid_json}")
    page0_2.initializePage()
    assert wiz2.selected_dataset is None

    # 3. Selection change with no QTableWidgetItem
    page0_2.table.setRowCount(1)
    page0_2.table.setItem(0, 0, None)
    # Trigger selection change
    page0_2.table.selectRow(0)
    assert wiz2.selected_dataset is None

    # 4. ProblemDefinitionPage Excel read branch
    page1 = wiz2.page(1)
    meta = _make_mock_metadata("DsExcel", str(tmp_path / "ds.xlsx"))
    Path(meta["location"]).touch()
    wiz2.selected_dataset = meta
    page1.combo_target.blockSignals(True)
    page1.combo_target.addItem("target_col")
    page1.combo_target.setCurrentText("target_col")
    page1.combo_target.blockSignals(False)
    with patch("pandas.read_excel") as mock_excel:
        mock_excel.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "target_col": [0, 1]})
        with patch("backend.app.analysis.analysis_service.AnalysisService.analyze") as mock_analyze:
            with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
                page1._on_target_changed()
                mock_excel.assert_called_once()

    # 5. PlanningPage generate plan exception paths
    page3 = wiz2.page(3)
    # Dependencies are complete, but create_plan raises exception
    wiz2.dataset_context = MagicMock()
    wiz2.user_request = MagicMock()
    wiz2.compute_capabilities = MagicMock()
    with patch("backend.app.ml_plan.orchestrator.MLPlanningOrchestrator.create_plan", side_effect=ValueError("create plan failed")):
        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            page3._on_generate_plan()
            mock_crit.assert_called_once()
            assert page3.plan_generated is False

    # 6. PlanningPage _populate_tree with feature engineering steps
    plan = MagicMock()
    plan.preprocessing_steps = []
    
    fe_step = MagicMock()
    fe_step.step_id = "fe_01"
    fe_step.operation.value = "interaction"
    plan.feature_engineering_steps = [fe_step]
    
    plan.feature_selection.method.value = "none"
    plan.feature_selection.max_features = 5
    plan.model_candidates = []
    plan.split_plan.strategy.value = "random"
    plan.evaluation_plan.primary_metric = "accuracy"
    plan.evaluation_plan.cross_validation_folds = 5
    page3._populate_tree(plan)
    assert page3.tree.topLevelItemCount() > 0

    # 7. AIPlanningPage toggle empty warnings block
    page4 = wiz2.page(4)
    res = MagicMock()
    res.ai_result.proposal.summary = "summary text"
    res.ai_result.proposal.evaluation_proposal = None
    res.ai_result.proposal.warnings = [] # empty warnings
    res.ai_changes_applied = False
    res.final_plan = MagicMock()
    with patch("backend.app.ml_plan.orchestrator.MLPlanningOrchestrator.create_plan", return_value=res):
        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            page4.chk_ai.setChecked(True)
            assert page4.txt_warnings.toPlainText() == "No suggestions warnings."

    # 8. SummaryPage incomplete plan branch
    page5 = wiz2.page(5)
    wiz2.final_plan = None
    page5.initializePage()
    assert page5.lbl_status.text() == "Incomplete plan"

    # 9. MLPlanningWizard accept with missing plan
    wiz2.final_plan = None
    wiz2.accept()

