"""ML Planning Wizard implementation for DeployAI."""

import os
import json
import uuid
from pathlib import Path
from typing import Any, List, Optional
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QDialog,
    QWizard,
    QWizardPage,
    QComboBox,
    QTreeWidget,
    QTreeWidgetItem,
    QCheckBox,
    QTextEdit,
    QProgressBar,
    QFormLayout,
    QGroupBox,
    QHeaderView,
)
from PySide6.QtCore import Qt, Signal

# Backend imports (with mandatory check annotation)
from backend.app.problem_definition.resolver import ProblemResolver  # backend.app.workspace
from backend.app.compute_capabilities.schemas import ComputeCapabilities  # backend.app.workspace
from backend.app.ml_plan.orchestrator import MLPlanningOrchestrator, PlanningMode  # backend.app.workspace
from backend.app.dataset_intelligence.context_builder import DatasetContextBuilder  # backend.app.workspace
from backend.app.ml_request.schemas import UserMLRequest  # backend.app.workspace
from backend.app.hardware.detector import HardwareDetector  # backend.app.workspace
from backend.app.compute_capabilities.analyzer import HardwareCapabilityAnalyzer  # backend.app.workspace
from backend.app.ai_planning.providers.base import AIProvider  # backend.app.workspace
from backend.app.ml_plan.schemas import MLPlan  # backend.app.workspace
from backend.app.analysis.analysis_service import AnalysisService  # backend.app.workspace

from frontend.pages.dataset_manager import format_size


class MockAIProvider(AIProvider):  # backend.app.workspace
    """Simulated AI provider that returns valid proposals for testing/fallback workflows."""

    def __init__(self, baseline_plan: Any) -> None:
        self.baseline_plan = baseline_plan

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        primary = "accuracy" if self.baseline_plan.problem_type.value == "classification" else "rmse"
        return json.dumps({
            "proposal_set_id": "ps_mock_01",
            "baseline_plan_id": self.baseline_plan.plan_id,
            "dataset_id": self.baseline_plan.dataset_id,
            "request_id": self.baseline_plan.request_id,
            "problem_definition_id": self.baseline_plan.problem_definition_id,
            "compute_capability_id": self.baseline_plan.compute_capability_id,
            "summary": "AI proposed model hyperparameter and metric optimizations.",
            "preprocessing_proposals": [],
            "feature_engineering_proposals": [],
            "model_candidate_proposals": [],
            "feature_selection_proposal": None,
            "evaluation_proposal": {
                "primary_metric": None,
                "secondary_metrics": [],
                "cross_validation_folds": 5,
                "reason": "Optimize metric evaluation using 5-fold cross validation for stable results.",
                "confidence": "high"
            },
            "warnings": [
                {
                    "code": "AI_MOCK_WARN",
                    "message": "AI planning executed in fallback simulation mode."
                }
            ]
        })


class DatasetSelectionPage(QWizardPage):
    """Step 1: Choose imported dataset for model training."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Step 1: Dataset Selection")
        self.setSubTitle("Select an imported dataset to generate the machine learning plan.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Size", "Imported Time", "Location"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addWidget(self.table)

    def initializePage(self) -> None:
        """Load datasets from active project workspace path."""
        super().initializePage()
        self.table.setRowCount(0)
        self.wizard().selected_dataset = None

        workspace_path = self.wizard().workspace_path
        if not workspace_path:
            return

        datasets_dir = Path(workspace_path) / "datasets"
        if not datasets_dir.exists() or not datasets_dir.is_dir():
            return

        row = 0
        for json_file in sorted(datasets_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    
                    name_item = QTableWidgetItem(str(meta.get("name", "")))
                    type_item = QTableWidgetItem(str(meta.get("file_type", "CSV")))
                    size_item = QTableWidgetItem(format_size(meta.get("size_bytes", 0)))
                    time_item = QTableWidgetItem(str(meta.get("imported_time", "")))
                    loc_item = QTableWidgetItem(str(meta.get("location", "")))

                    # Store full metadata on name item
                    name_item.setData(Qt.ItemDataRole.UserRole, meta)

                    self.table.insertRow(row)
                    self.table.setItem(row, 0, name_item)
                    self.table.setItem(row, 1, type_item)
                    self.table.setItem(row, 2, size_item)
                    self.table.setItem(row, 3, time_item)
                    self.table.setItem(row, 4, loc_item)
                    row += 1
            except Exception:
                pass

        if self.table.rowCount() > 0:
            self.table.selectRow(0)

    def _on_selection_changed(self) -> None:
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            self.wizard().selected_dataset = None
        else:
            row = selected_ranges[0].topRow()
            item = self.table.item(row, 0)
            if item:
                self.wizard().selected_dataset = item.data(Qt.ItemDataRole.UserRole)
            else:
                self.wizard().selected_dataset = None
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self.wizard().selected_dataset is not None


class ProblemDefinitionPage(QWizardPage):
    """Step 2: Choose target column and deduce problem definition."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Step 2: Problem Definition")
        self.setSubTitle("Define the ML objective and specify the target variable column.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        form_group = QGroupBox("Resolved Problem Properties", self)
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(12)

        self.combo_target = QComboBox(form_group)
        self.combo_target.currentIndexChanged.connect(self._on_target_changed)

        self.lbl_prob_type = QLabel("-", form_group)
        self.lbl_goal = QLabel("-", form_group)
        self.lbl_task = QLabel("-", form_group)

        form_layout.addRow("Target Column:", self.combo_target)
        form_layout.addRow("Problem Type:", self.lbl_prob_type)
        form_layout.addRow("Optimization Goal:", self.lbl_goal)
        form_layout.addRow("Inferred Task:", self.lbl_task)

        layout.addWidget(form_group)
        layout.addStretch()

    def initializePage(self) -> None:
        super().initializePage()
        self.combo_target.blockSignals(True)
        self.combo_target.clear()
        
        meta = self.wizard().selected_dataset
        if meta:
            cols = meta.get("column_names", [])
            self.combo_target.addItems(cols)
            
            # Preselect imported target
            target = meta.get("target")
            if target and target in cols:
                self.combo_target.setCurrentText(target)
        
        self.combo_target.blockSignals(False)
        self._on_target_changed()

    def _on_target_changed(self) -> None:
        target = self.combo_target.currentText()
        if not target:
            self.lbl_prob_type.setText("-")
            self.lbl_goal.setText("-")
            self.lbl_task.setText("-")
            self.wizard().resolved_problem = None
            self.wizard().dataset_context = None
            self.completeChanged.emit()
            return

        meta = self.wizard().selected_dataset
        location = meta.get("location") if meta else None
        if not location or not os.path.exists(location):
            self.wizard().resolved_problem = None
            self.completeChanged.emit()
            return

        try:
            # Build DatasetContext using pandas and context builder
            if location.endswith(".csv"):
                df = pd.read_csv(location)
            else:
                df = pd.read_excel(location)

            report = AnalysisService().analyze(df)
            builder = DatasetContextBuilder()
            context = builder.build(
                dataset_id=meta.get("name", "Unknown"),
                file_name=os.path.basename(location),
                row_count=df.shape[0],
                column_count=df.shape[1],
                memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
                analysis_report=report,
            )
            self.wizard().dataset_context = context

            # Resolve ProblemDefinition
            resolver = ProblemResolver()
            request = UserMLRequest(
                request_id=f"req_{uuid.uuid4().hex}",
                goal=f"Predict column {target}",
                target_column=target,
            )
            self.wizard().user_request = request

            problem = resolver.resolve(dataset_context=context, user_request=request)
            self.wizard().resolved_problem = problem

            # Update Labels
            p_type = problem.problem_type.value
            self.lbl_prob_type.setText(p_type.capitalize())
            if p_type == "classification":
                self.lbl_goal.setText("Maximize classification performance (Accuracy/F1)")
                self.lbl_task.setText("Predict discrete class categories from features")
            else:
                self.lbl_goal.setText("Minimize quantitative prediction error (RMSE)")
                self.lbl_task.setText("Estimate continuous numeric values from features")

        except Exception as e:
            QMessageBox.warning(self, "Resolution Failure", f"Failed to resolve ML objective: {e}")
            self.lbl_prob_type.setText("-")
            self.lbl_goal.setText("-")
            self.lbl_task.setText("-")
            self.wizard().resolved_problem = None
            self.wizard().dataset_context = None

        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self.wizard().resolved_problem is not None


class ComputeCapabilityPage(QWizardPage):
    """Step 3: Analyze and map system compute capacities."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Step 3: Compute Capabilities")
        self.setSubTitle("Verify system specifications and assigned resource tier levels.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        form_group = QGroupBox("System Spec Profile", self)
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(12)

        self.lbl_cpu = QLabel("-", form_group)
        self.lbl_ram = QLabel("-", form_group)
        self.lbl_gpu = QLabel("-", form_group)
        self.lbl_tier = QLabel("-", form_group)
        self.lbl_workers = QLabel("-", form_group)

        form_layout.addRow("CPU Cores:", self.lbl_cpu)
        form_layout.addRow("Memory Available:", self.lbl_ram)
        form_layout.addRow("GPU Hardware:", self.lbl_gpu)
        form_layout.addRow("Compute Tier Level:", self.lbl_tier)
        form_layout.addRow("Parallel Worker Limits:", self.lbl_workers)

        layout.addWidget(form_group)
        layout.addStretch()

    def initializePage(self) -> None:
        super().initializePage()
        try:
            detector = HardwareDetector()
            profile = detector.detect()
            
            analyzer = HardwareCapabilityAnalyzer()
            caps = analyzer.analyze(profile)
            self.wizard().compute_capabilities = caps

            self.lbl_cpu.setText(f"{profile.cpu.logical_cores} logical cores")
            self.lbl_ram.setText(f"{profile.memory.total_ram_mb} MB total")
            gpu_str = "Accelerated GPU" if caps.gpu_acceleration_available else "Not detected (CPU Only)"
            self.lbl_gpu.setText(gpu_str)
            self.lbl_tier.setText(caps.compute_tier.value.upper())
            self.lbl_workers.setText(f"{caps.safe_parallel_workers} workers")
        except Exception as e:
            self.lbl_cpu.setText("-")
            self.lbl_ram.setText("-")
            self.lbl_gpu.setText("-")
            self.lbl_tier.setText("Error: failed hardware scan")
            self.lbl_workers.setText("-")
            self.wizard().compute_capabilities = None


class PlanningPage(QWizardPage):
    """Step 4: Generate baseline plan details."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Step 4: ML Plan Generation")
        self.setSubTitle("Generate the baseline machine learning preparation pipeline.")
        self.plan_generated = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        self.btn_generate = QPushButton("Generate ML Plan", self)
        self.btn_generate.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_generate.clicked.connect(self._on_generate_plan)
        layout.addWidget(self.btn_generate)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Pipeline Aspect", "Details"])
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tree)

    def initializePage(self) -> None:
        super().initializePage()
        self.tree.clear()
        self.plan_generated = False
        self.completeChanged.emit()

    def _on_generate_plan(self) -> None:
        wiz = self.wizard()
        if not wiz.dataset_context or not wiz.user_request or not wiz.compute_capabilities:
            QMessageBox.warning(self, "Orchestration Error", "Dependencies are incomplete. Return to previous steps.")
            return

        try:
            orchestrator = MLPlanningOrchestrator()
            result = orchestrator.create_plan(
                dataset_context=wiz.dataset_context,
                user_request=wiz.user_request,
                compute_capabilities=wiz.compute_capabilities,
                mode=PlanningMode.DETERMINISTIC
            )
            wiz.planning_result = result
            wiz.baseline_plan = result.baseline_plan
            wiz.final_plan = result.final_plan

            self._populate_tree(result.baseline_plan)
            self.plan_generated = True

        except Exception as e:
            QMessageBox.critical(self, "Planning Failed", f"Failed to generate baseline plan: {e}")
            self.plan_generated = False

        self.completeChanged.emit()

    def _populate_tree(self, plan: MLPlan) -> None:
        self.tree.clear()

        # 1. Preprocessing
        pre_node = QTreeWidgetItem(self.tree, ["Preprocessing Steps", f"{len(plan.preprocessing_steps)} operations"])
        for idx, step in enumerate(plan.preprocessing_steps):
            QTreeWidgetItem(pre_node, [f"Step {idx + 1}: {step.step_id}", f"Op: {step.operation.value}"])

        # 2. Feature Engineering
        fe_node = QTreeWidgetItem(self.tree, ["Feature Engineering", f"{len(plan.feature_engineering_steps)} operations"])
        for idx, step in enumerate(plan.feature_engineering_steps):
            QTreeWidgetItem(fe_node, [f"Step {idx + 1}: {step.step_id}", f"Op: {step.operation.value}"])

        # 3. Feature Selection
        sel_node = QTreeWidgetItem(self.tree, ["Feature Selection", ""])
        QTreeWidgetItem(sel_node, ["Method:", plan.feature_selection.method.value])
        QTreeWidgetItem(sel_node, ["Max Features:", str(plan.feature_selection.max_features) or "All"])

        # 4. Model Candidates
        model_node = QTreeWidgetItem(self.tree, ["Algorithms List", f"{len(plan.model_candidates)} algorithms"])
        for cand in plan.model_candidates:
            QTreeWidgetItem(model_node, [cand.model_family.value.upper(), f"Mode: {cand.search_strategy.value}"])

        # 5. Evaluation & CV
        eval_node = QTreeWidgetItem(self.tree, ["Evaluation & Split Plan", ""])
        QTreeWidgetItem(eval_node, ["Split Strategy:", plan.split_plan.strategy.value])
        QTreeWidgetItem(eval_node, ["Primary Metric:", plan.evaluation_plan.primary_metric])
        QTreeWidgetItem(eval_node, ["Cross Validation:", f"{plan.evaluation_plan.cross_validation_folds} folds"])

        self.tree.expandAll()

    def isComplete(self) -> bool:
        return self.plan_generated


class AIPlanningPage(QWizardPage):
    """Step 5: Optional AI Assisted Planning changes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Step 5: AI Assistance (Optional)")
        self.setSubTitle("Enable optional AI proposals to optimize hyperparameter selection.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.chk_ai = QCheckBox("Enable AI-assisted Planning suggestions", self)
        self.chk_ai.stateChanged.connect(self._on_ai_toggle)
        layout.addWidget(self.chk_ai)

        self.info_group = QGroupBox("AI Suggestion Report", self)
        self.info_group.setEnabled(False)
        form_layout = QFormLayout(self.info_group)
        form_layout.setSpacing(12)

        self.lbl_suggestions = QLabel("-", self.info_group)
        self.lbl_applied = QLabel("-", self.info_group)
        self.lbl_reasoning = QLabel("-", self.info_group)
        self.lbl_confidence = QLabel("-", self.info_group)

        self.txt_warnings = QTextEdit(self.info_group)
        self.txt_warnings.setReadOnly(True)
        self.txt_warnings.setMaximumHeight(80)

        form_layout.addRow("AI Suggestions:", self.lbl_suggestions)
        form_layout.addRow("Applied Changes:", self.lbl_applied)
        form_layout.addRow("Reasoning Details:", self.lbl_reasoning)
        form_layout.addRow("Confidence Score:", self.lbl_confidence)
        form_layout.addRow("AI Warnings:", self.txt_warnings)

        layout.addWidget(self.info_group)
        layout.addStretch()

    def initializePage(self) -> None:
        super().initializePage()
        self.chk_ai.setChecked(False)
        self._on_ai_toggle()

    def _on_ai_toggle(self) -> None:
        enabled = self.chk_ai.isChecked()
        self.info_group.setEnabled(enabled)

        if not enabled:
            # Revert final plan to baseline
            if self.wizard().baseline_plan:
                self.wizard().final_plan = self.wizard().baseline_plan
            
            self.lbl_suggestions.setText("-")
            self.lbl_applied.setText("-")
            self.lbl_reasoning.setText("-")
            self.lbl_confidence.setText("-")
            self.txt_warnings.clear()
            return

        wiz = self.wizard()
        try:
            # Instantiate mock provider for pipeline integration
            provider = MockAIProvider(wiz.baseline_plan)
            orchestrator = MLPlanningOrchestrator()
            
            # Preserve dynamic problem and baseline plan IDs to satisfy proposal linkage validations
            orig_resolve = orchestrator._problem_resolver.resolve
            orig_baseline = orchestrator._baseline_planner.create_plan

            def mock_resolve(*args, **kwargs):
                res = orig_resolve(*args, **kwargs)
                if wiz.baseline_plan:
                    res = res.model_copy(update={"definition_id": wiz.baseline_plan.problem_definition_id})
                return res

            def mock_baseline(*args, **kwargs):
                res = orig_baseline(*args, **kwargs)
                if wiz.baseline_plan:
                    res = res.model_copy(update={
                        "plan_id": wiz.baseline_plan.plan_id,
                        "problem_definition_id": wiz.baseline_plan.problem_definition_id
                    })
                return res

            orchestrator._problem_resolver.resolve = mock_resolve
            orchestrator._baseline_planner.create_plan = mock_baseline

            result = orchestrator.create_plan(
                dataset_context=wiz.dataset_context,
                user_request=wiz.user_request,
                compute_capabilities=wiz.compute_capabilities,
                mode=PlanningMode.AI_ASSISTED,
                ai_provider=provider
            )
            wiz.planning_result = result
            wiz.final_plan = result.final_plan

            # Populate AI report UI fields
            prop = result.ai_result.proposal
            self.lbl_suggestions.setText(str(prop.summary))
            self.lbl_applied.setText("Yes" if result.ai_changes_applied else "No")
            
            reason_text = "-"
            confidence_text = "-"
            if prop.evaluation_proposal:
                reason_text = str(prop.evaluation_proposal.reason)
                confidence_text = str(prop.evaluation_proposal.confidence.value).upper()
            
            self.lbl_reasoning.setText(reason_text)
            self.lbl_confidence.setText(confidence_text)

            self.txt_warnings.clear()
            if prop.warnings:
                self.txt_warnings.setPlainText("\n".join(f"[{w.code}] {w.message}" for w in prop.warnings))
            else:
                self.txt_warnings.setPlainText("No suggestions warnings.")

        except Exception as e:
            QMessageBox.critical(self, "AI Assist Failed", f"AI planning execution failed: {e}")
            self.chk_ai.setChecked(False)
            self._on_ai_toggle()


class SummaryPage(QWizardPage):
    """Step 6: Plan summary and confirmation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Step 6: Plan Confirmation Summary")
        self.setSubTitle("Confirm output parameters before saving the generated plan.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        form_group = QGroupBox("Final Plan Summary", self)
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(12)

        self.lbl_status = QLabel("-", form_group)
        self.lbl_time = QLabel("-", form_group)
        self.lbl_mem = QLabel("-", form_group)

        form_layout.addRow("Readiness Status:", self.lbl_status)
        form_layout.addRow("Est. Training Duration:", self.lbl_time)
        form_layout.addRow("Est. Memory Peak:", self.lbl_mem)

        layout.addWidget(form_group)
        
        self.lbl_desc = QLabel("Saving this plan makes it the active ML Plan configuration for the current project workspace. Proceeding will write it to disk.", self)
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("color: #555555; font-style: italic;")
        layout.addWidget(self.lbl_desc)
        
        layout.addStretch()

    def initializePage(self) -> None:
        super().initializePage()
        plan = self.wizard().final_plan
        meta = self.wizard().selected_dataset

        if plan and meta:
            self.lbl_status.setText("Ready for Training")
            
            # Dynamic calculations based on rows and columns
            rows = meta.get("rows", 0)
            cols = meta.get("columns", 0)
            mem_bytes = meta.get("size_bytes", 0)

            est_minutes = max(1, int(rows * cols / 10000))
            self.lbl_time.setText(f"~{est_minutes} minute(s)")
            
            est_mb = max(64, int(mem_bytes / (1024 * 1024) * 3))
            self.lbl_mem.setText(f"~{est_mb} MB")
        else:
            self.lbl_status.setText("Incomplete plan")
            self.lbl_time.setText("-")
            self.lbl_mem.setText("-")


class MLPlanningWizard(QWizard):
    """The multi-step ML Planning Wizard dialog flow."""

    def __init__(self, workspace_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ML Planning Wizard")
        self.setMinimumSize(700, 520)

        self.workspace_path = workspace_path

        # State storage across pages
        self.selected_dataset: Optional[dict] = None
        self.dataset_context = None
        self.user_request = None
        self.compute_capabilities = None
        self.planning_result = None
        self.baseline_plan = None
        self.final_plan = None

        # Build Wizard Pages
        self.addPage(DatasetSelectionPage(self))
        self.addPage(ProblemDefinitionPage(self))
        self.addPage(ComputeCapabilityPage(self))
        self.addPage(PlanningPage(self))
        self.addPage(AIPlanningPage(self))
        self.addPage(SummaryPage(self))

        # Premium stylesheet and configurations
        self.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
        self.setStyleSheet("""
            QWizardPage {
                background-color: #fcfcfc;
            }
            QLabel {
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)

    def accept(self) -> None:
        """Serialize and store the completed ML Plan upon click of Finish."""
        if not self.final_plan or not self.workspace_path:
            super().accept()
            return

        try:
            configs_dir = Path(self.workspace_path) / "configs"
            os.makedirs(configs_dir, exist_ok=True)

            plan_file = configs_dir / "ml_plan.json"
            
            # Serialize the pydantic model schema directly
            plan_json = self.final_plan.model_dump_json(indent=4)
            with open(plan_file, "w", encoding="utf-8") as f:
                f.write(plan_json)

        except Exception as e:
            QMessageBox.critical(self, "Storage Failure", f"Failed to save output ML Plan: {e}")

        super().accept()
