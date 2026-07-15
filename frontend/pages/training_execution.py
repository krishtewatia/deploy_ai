"""Training execution page displaying progress and logs from background training worker."""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QGroupBox,
    QFormLayout,
    QMessageBox,
)
from PySide6.QtCore import Qt, QTimer

# Backend imports (with mandatory check annotation)
from backend.app.ml_plan.schemas import MLPlan  # backend.app.workspace
from frontend.pages.workers.training_worker import TrainingWorker


class TrainingExecutionPage(QWidget):
    """The page executing ML plan and monitoring training execution progress."""

    def __init__(self, orchestrator=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.worker: TrainingWorker | None = None
        self.start_time: float = 0.0

        # UI Timer to update elapsed and remaining time labels
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._on_timer_tick)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)

        # 1. Header Title
        self.header_label = QLabel("Training Execution", self)
        self.header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333333;")
        self.main_layout.addWidget(self.header_label)

        # 2. Project Details Group
        self.details_group = QGroupBox("Project Details Context", self)
        self.details_form = QFormLayout(self.details_group)
        self.details_form.setSpacing(8)

        self.lbl_project = QLabel("-", self.details_group)
        self.lbl_dataset = QLabel("-", self.details_group)
        self.lbl_plan = QLabel("-", self.details_group)

        self.details_form.addRow("Current Project:", self.lbl_project)
        self.details_form.addRow("Current Dataset:", self.lbl_dataset)
        self.details_form.addRow("Current ML Plan:", self.lbl_plan)
        self.main_layout.addWidget(self.details_group)

        # 3. Execution Progress Group
        self.progress_group = QGroupBox("Execution Progress", self)
        self.progress_layout = QVBoxLayout(self.progress_group)
        self.progress_layout.setSpacing(10)

        self.progress_bar = QProgressBar(self.progress_group)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_layout.addWidget(self.progress_bar)

        self.lbl_stage = QLabel("Stage: Idle", self.progress_group)
        self.lbl_stage.setStyleSheet("font-weight: bold; color: #555555;")
        self.progress_layout.addWidget(self.lbl_stage)

        time_layout = QHBoxLayout()
        self.lbl_elapsed = QLabel("Elapsed Time: 00:00", self.progress_group)
        self.lbl_remaining = QLabel("Estimated Remaining Time: -", self.progress_group)
        time_layout.addWidget(self.lbl_elapsed)
        time_layout.addWidget(self.lbl_remaining)
        self.progress_layout.addLayout(time_layout)
        self.main_layout.addWidget(self.progress_group)

        # 4. Current Candidate Group
        self.candidate_group = QGroupBox("Current Model Candidate Details", self)
        self.candidate_form = QFormLayout(self.candidate_group)
        self.candidate_form.setSpacing(8)

        self.lbl_model_name = QLabel("-", self.candidate_group)
        self.lbl_model_status = QLabel("-", self.candidate_group)
        self.lbl_model_metric = QLabel("-", self.candidate_group)

        self.candidate_form.addRow("Model Name:", self.lbl_model_name)
        self.candidate_form.addRow("Status:", self.lbl_model_status)
        self.candidate_form.addRow("Metric:", self.lbl_model_metric)
        self.main_layout.addWidget(self.candidate_group)

        # 5. Execution Log Terminal Console
        self.log_group = QGroupBox("Execution Logs", self)
        self.log_layout = QVBoxLayout(self.log_group)
        self.txt_log = QPlainTextEdit(self.log_group)
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; font-family: Consolas, monospace; font-size: 11px;"
        )
        self.log_layout.addWidget(self.txt_log)
        self.main_layout.addWidget(self.log_group, stretch=1)

        # 6. Bottom Buttons Layout
        self.btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Training", self)
        self.btn_start.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.btn_start.clicked.connect(self._on_start_training)
        self.btn_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet("padding: 6px 16px;")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_training)
        self.btn_layout.addWidget(self.btn_cancel)

        self.btn_layout.addStretch()

        self.btn_open_results = QPushButton("Open Results", self)
        self.btn_open_results.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.btn_open_results.setEnabled(False)
        self.btn_open_results.clicked.connect(self._on_open_results)
        self.btn_layout.addWidget(self.btn_open_results)

        self.main_layout.addLayout(self.btn_layout)

        # Active workspace state configuration metadata
        self.active_plan: MLPlan | None = None
        self.active_dataset_metadata: dict[str, Any] | None = None

    def showEvent(self, event) -> None:
        """Scan active plan details and refresh form values when view displays."""
        super().showEvent(event)
        self.refresh_page()

    def refresh_page(self) -> None:
        """Read and validate current project plan configuration schemas."""
        if not self.orchestrator or not self.orchestrator.active_workspace_path:
            self.lbl_project.setText("No active project")
            self.lbl_dataset.setText("-")
            self.lbl_plan.setText("-")
            self.active_plan = None
            self.active_dataset_metadata = None
            return

        workspace_path = self.orchestrator.active_workspace_path
        self.lbl_project.setText(os.path.basename(workspace_path))

        plan_file = Path(workspace_path) / "configs" / "ml_plan.json"
        if not plan_file.exists():
            self.lbl_dataset.setText("-")
            self.lbl_plan.setText("-")
            self.active_plan = None
            self.active_dataset_metadata = None
            return

        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
                plan = MLPlan.model_validate(plan_data)

            self.active_plan = plan
            self.lbl_plan.setText(str(plan.plan_id))

            # Retrieve dataset metadata matching MLPlan context ID
            datasets_dir = Path(workspace_path) / "datasets"
            selected_meta = None
            if datasets_dir.exists():
                for meta_file in datasets_dir.glob("*.json"):
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            if meta.get("name") == plan.dataset_id:
                                selected_meta = meta
                                break
                    except Exception:
                        pass
                
                if not selected_meta:
                    for meta_file in sorted(datasets_dir.glob("*.json")):
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                selected_meta = json.load(f)
                                break
                        except Exception:
                            pass

            self.active_dataset_metadata = selected_meta
            if selected_meta:
                self.lbl_dataset.setText(f"{selected_meta.get('name')} ({selected_meta.get('file_type')})")
            else:
                self.lbl_dataset.setText("Unknown")

        except Exception as e:
            QMessageBox.warning(self, "Load Failure", f"Failed to load execution planning contexts: {e}")
            self.active_plan = None
            self.active_dataset_metadata = None

    def _on_start_training(self) -> None:
        """Validate input configurations and launch background executor thread."""
        if not self.orchestrator or not self.orchestrator.active_workspace_path:
            QMessageBox.warning(self, "Workspace Closed", "Please open or create a project workspace first.")
            return

        if not self.active_plan:
            QMessageBox.warning(self, "Plan Missing", "Please generate an ML Plan before starting training.")
            return

        if not self.active_dataset_metadata:
            QMessageBox.warning(self, "Dataset Missing", "No imported dataset found inside this workspace.")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Execution Busy", "Another training job is currently running.")
            return

        # Initialize progress panel states
        self.progress_bar.setValue(0)
        self.lbl_stage.setText("Stage: Preparing Dataset")
        self.lbl_elapsed.setText("Elapsed Time: 00:00")
        self.lbl_remaining.setText("Estimated Remaining Time: -")
        
        self.lbl_model_name.setText("-")
        self.lbl_model_status.setText("-")
        self.lbl_model_metric.setText("-")

        self.txt_log.clear()

        # Update button enable state transitions
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_open_results.setEnabled(False)

        # Trigger timer monitoring elapsed intervals
        self.start_time = time.time()
        self.timer.start()

        # Instantiate worker thread
        self.worker = TrainingWorker(
            workspace_path=self.orchestrator.active_workspace_path,
            dataset_metadata=self.active_dataset_metadata,
            parent=self,
        )

        self.worker.progressChanged.connect(self.progress_bar.setValue)
        self.worker.stageChanged.connect(lambda stage: self.lbl_stage.setText(f"Stage: {stage}"))
        self.worker.candidateChanged.connect(self._on_candidate_changed)
        self.worker.logAppended.connect(self.txt_log.appendPlainText)
        self.worker.finished.connect(self._on_training_finished)
        self.worker.failed.connect(self._on_training_failed)

        self.worker.start()

    def _on_cancel_training(self) -> None:
        """Trigger clean cancel sequence on background thread worker."""
        if self.worker:
            self.worker.cancel()
            self.txt_log.appendPlainText("\nCancellation requested by user. Aborting...")
            self.btn_cancel.setEnabled(False)

    def _on_timer_tick(self) -> None:
        """Increment execution interval statistics labels."""
        elapsed = int(time.time() - self.start_time)
        m, s = divmod(elapsed, 60)
        self.lbl_elapsed.setText(f"Elapsed Time: {m:02d}:{s:02d}")

        prog = self.progress_bar.value()
        if prog > 0 and prog < 100:
            est_total = elapsed * (100.0 / prog)
            rem = max(0, int(est_total - elapsed))
            rm, rs = divmod(rem, 60)
            self.lbl_remaining.setText(f"Estimated Remaining Time: {rm:02d}:{rs:02d}")
        else:
            self.lbl_remaining.setText("Estimated Remaining Time: -")

    def _on_candidate_changed(self, cid: str, status: str, metric: str) -> None:
        """Update active model candidate diagnostic fields."""
        self.lbl_model_name.setText(cid)
        self.lbl_model_status.setText(status)
        self.lbl_model_metric.setText(metric)

    def _on_training_finished(self) -> None:
        """Handle training completed workflow cleanup actions."""
        if self.lbl_stage.text() in ("Stage: Failed", "Stage: Cancelled"):
            return
        if self.lbl_stage.text() != "Stage: Completed":
            # Guard against early built-in finished triggers before stage completes
            self.lbl_stage.setText("Stage: Completed")
        if self.btn_open_results.isEnabled():
            return
            
        self.timer.stop()
        self.progress_bar.setValue(100)
        self.lbl_stage.setText("Stage: Completed")
        
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_open_results.setEnabled(True)
        
        QMessageBox.information(self, "Training Successful", "The model training execution pipeline completed successfully.")

    def _on_training_failed(self, error_msg: str) -> None:
        """Handle execution pipeline thread crashes cleanly."""
        if self.lbl_stage.text() == "Stage: Failed":
            return
        self.timer.stop()
        self.lbl_stage.setText("Stage: Failed")
        
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_open_results.setEnabled(False)
        
        QMessageBox.critical(self, "Training Failed", f"The training pipeline execution failed:\n{error_msg}")

    def _on_open_results(self) -> None:
        """Navigate shell navigation controller to Reports display page."""
        main_win = self.window()
        if main_win and hasattr(main_win, "shell") and hasattr(main_win.shell, "nav_controller"):
            nav = main_win.shell.nav_controller
            reports_page = nav.pages.get("Reports")
            if hasattr(reports_page, "_results_widget") and reports_page._results_widget:
                reports_page._results_widget.refresh_results()
            nav.switch_to_page("Reports")
        else:
            QMessageBox.information(self, "Reports Closed", "Training execution completed. Reports are ready inside the workspace.")
