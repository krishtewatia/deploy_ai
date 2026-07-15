"""Training page for DeployAI to configure and trigger the ML Planning Wizard."""

import os
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QFormLayout,
    QGroupBox,
    QMessageBox,
    QDialog,
)
from PySide6.QtCore import Qt

# Backend imports (with mandatory check annotation)
from backend.app.ml_plan.schemas import MLPlan  # backend.app.workspace
from frontend.pages.planning_wizard import MLPlanningWizard
from frontend.pages.training_execution import TrainingExecutionPage


class TrainingPage(QWidget):
    """The Training and ML Planning configuration page."""

    def __init__(self, orchestrator=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)

        # Header Section
        self.header_label = QLabel("Machine Learning Training & Planning", self)
        self.header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333333;")
        self.main_layout.addWidget(self.header_label)

        # Content stack to toggle view states
        self.stack = QStackedWidget(self)
        self.main_layout.addWidget(self.stack, stretch=1)

        # Page 0: No Workspace Active
        self.page_no_ws = QWidget(self.stack)
        layout_no_ws = QVBoxLayout(self.page_no_ws)
        layout_no_ws.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_no_ws = QLabel("No active project workspace. Open or create a project on the Projects page.", self.page_no_ws)
        self.lbl_no_ws.setStyleSheet("font-size: 13px; color: #666666;")
        layout_no_ws.addWidget(self.lbl_no_ws)
        self.stack.addWidget(self.page_no_ws)

        # Page 1: Workspace Active but No Plan Exists
        self.page_no_plan = QWidget(self.stack)
        layout_no_plan = QVBoxLayout(self.page_no_plan)
        layout_no_plan.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_no_plan.setSpacing(20)
        self.lbl_no_plan = QLabel("No ML Plan has been generated for this project yet.", self.page_no_plan)
        self.lbl_no_plan.setStyleSheet("font-size: 14px; color: #555555; font-weight: bold;")
        
        self.btn_create_plan = QPushButton("Generate ML Plan", self.page_no_plan)
        self.btn_create_plan.setStyleSheet("font-weight: bold; font-size: 13px; padding: 8px 16px;")
        self.btn_create_plan.clicked.connect(self._launch_wizard)

        layout_no_plan.addWidget(self.lbl_no_plan)
        layout_no_plan.addWidget(self.btn_create_plan)
        self.stack.addWidget(self.page_no_plan)

        # Page 2: Plan Exists Summary View
        self.page_plan_summary = QWidget(self.stack)
        layout_summary = QVBoxLayout(self.page_plan_summary)
        layout_summary.setContentsMargins(10, 10, 10, 10)
        layout_summary.setSpacing(15)

        self.plan_group = QGroupBox("Active ML Plan Details", self.page_plan_summary)
        self.form_layout = QFormLayout(self.plan_group)
        self.form_layout.setSpacing(12)

        self.lbl_target = QLabel("-", self.plan_group)
        self.lbl_problem_type = QLabel("-", self.plan_group)
        self.lbl_metric = QLabel("-", self.plan_group)
        self.lbl_cv = QLabel("-", self.plan_group)
        self.lbl_preprocessing = QLabel("-", self.plan_group)
        self.lbl_engineering = QLabel("-", self.plan_group)
        self.lbl_models = QLabel("-", self.plan_group)

        self.form_layout.addRow("Target Column:", self.lbl_target)
        self.form_layout.addRow("Problem Type:", self.lbl_problem_type)
        self.form_layout.addRow("Primary Metric:", self.lbl_metric)
        self.form_layout.addRow("Cross Validation:", self.lbl_cv)
        self.form_layout.addRow("Preprocessing Steps:", self.lbl_preprocessing)
        self.form_layout.addRow("Feature Engineering:", self.lbl_engineering)
        self.form_layout.addRow("Model Candidates:", self.lbl_models)

        layout_summary.addWidget(self.plan_group)

        # Controls layout
        hbox = QHBoxLayout()
        self.btn_recreate_plan = QPushButton("Recreate ML Plan", self.page_plan_summary)
        self.btn_recreate_plan.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.btn_recreate_plan.clicked.connect(self._launch_wizard)
        hbox.addWidget(self.btn_recreate_plan)
        
        self.btn_proceed_training = QPushButton("Proceed to Training", self.page_plan_summary)
        self.btn_proceed_training.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.btn_proceed_training.clicked.connect(lambda: self.stack.setCurrentIndex(3))
        hbox.addWidget(self.btn_proceed_training)
        
        hbox.addStretch()

        self.lbl_status = QLabel("Plan Status: Stored - Ready for Training", self.page_plan_summary)
        self.lbl_status.setStyleSheet("color: #2ec4b6; font-weight: bold; font-size: 13px;")
        hbox.addWidget(self.lbl_status)

        layout_summary.addLayout(hbox)
        layout_summary.addStretch()

        self.stack.addWidget(self.page_plan_summary)

        # Page 3: Training Execution Page
        self.page_execution = TrainingExecutionPage(orchestrator=self.orchestrator, parent=self.stack)
        self.stack.addWidget(self.page_execution)

    def showEvent(self, event) -> None:
        """Refresh page state and check for active plans when shown."""
        super().showEvent(event)
        self.refresh_page()

    def refresh_page(self) -> None:
        """Inspect workspace state and active plan file, toggling view stack."""
        if not self.orchestrator or not self.orchestrator.active_workspace_path:
            self.stack.setCurrentIndex(0)
            return

        workspace_path = self.orchestrator.active_workspace_path
        plan_file = Path(workspace_path) / "configs" / "ml_plan.json"

        if not plan_file.exists():
            self.stack.setCurrentIndex(1)
            return

        if self.stack.currentIndex() == 3:
            self.page_execution.refresh_page()
            return

        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
                plan = MLPlan.model_validate(plan_data)

            self.lbl_target.setText(str(plan.target_column))
            self.lbl_problem_type.setText(str(plan.problem_type.value).capitalize())
            self.lbl_metric.setText(str(plan.evaluation_plan.primary_metric))
            self.lbl_cv.setText(f"{plan.evaluation_plan.cross_validation_folds} folds")
            self.lbl_preprocessing.setText(f"{len(plan.preprocessing_steps)} operations")
            self.lbl_engineering.setText(f"{len(plan.feature_engineering_steps)} operations")
            
            models = ", ".join(cand.model_family.value.upper() for cand in plan.model_candidates)
            self.lbl_models.setText(models)

            self.stack.setCurrentIndex(2)
        except Exception as e:
            QMessageBox.warning(self, "Load Failure", f"Failed to load existing ML Plan: {e}")
            self.stack.setCurrentIndex(1)

    def _launch_wizard(self) -> None:
        """Launch the multi-step ML Planning Wizard modal dialog."""
        if not self.orchestrator or not self.orchestrator.active_workspace_path:
            QMessageBox.warning(self, "Workspace Closed", "Please open a project workspace first.")
            return

        workspace_path = self.orchestrator.active_workspace_path
        wizard = MLPlanningWizard(workspace_path=workspace_path, parent=self)
        if wizard.exec() == QDialog.DialogCode.Accepted:
            self.refresh_page()
