"""Training worker thread executing ML plan pipelines in background."""

from __future__ import annotations

import os
import json
import time
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from PySide6.QtCore import QThread, Signal

# Backend imports (with mandatory check annotation)
from backend.app.dataset_intelligence.schemas import DatasetContext  # backend.app.workspace
from backend.app.dataset_intelligence.context_builder import DatasetContextBuilder  # backend.app.workspace
from backend.app.analysis.analysis_service import AnalysisService  # backend.app.workspace
from backend.app.ml_plan.schemas import MLPlan  # backend.app.workspace
from backend.app.problem_definition.schemas import (  # backend.app.workspace
    ProblemDefinition,
    ResolutionStatus,
    TargetSource,
)
from backend.app.ml_execution.orchestrator import MLExecutionOrchestrator  # backend.app.workspace
from backend.app.ml_execution.split_executor import SplitExecutor  # backend.app.workspace
from backend.app.ml_execution.preprocessing_builder import PreprocessingPipelineBuilder  # backend.app.workspace
from backend.app.ml_execution.feature_engineering_executor import FeatureEngineeringExecutor  # backend.app.workspace
from backend.app.ml_execution.feature_selection_executor import FeatureSelectionExecutor  # backend.app.workspace
from backend.app.ml_execution.hyperparameter_optimizer import HyperparameterOptimizer  # backend.app.workspace
from backend.app.ml_execution.training_executor import TrainingExecutor  # backend.app.workspace
from backend.app.ml_execution.evaluation_engine import EvaluationEngine  # backend.app.workspace
from backend.app.ml_execution.execution_report import ExecutionReportBuilder  # backend.app.workspace


class TrainingWorker(QThread):
    """Background worker executing the MLPlan and emitting live progress updates."""

    progressChanged = Signal(int)
    stageChanged = Signal(str)
    candidateChanged = Signal(str, str, str)  # candidate_id, status, metric
    logAppended = Signal(str)
    finished = Signal()
    failed = Signal(str)

    _active_worker: TrainingWorker | None = None

    def __init__(self, workspace_path: str, dataset_metadata: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.workspace_path = workspace_path
        self.dataset_metadata = dataset_metadata
        self._is_cancelled = False
        self._log_history: list[str] = []

    def cancel(self) -> None:
        """Cancel execution request."""
        self._is_cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancel was requested."""
        return self._is_cancelled

    def _log(self, text: str) -> None:
        """Internal helper to record log statements."""
        self._log_history.append(text)
        self.logAppended.emit(text)

    def run(self) -> None:
        """Execute the pipeline in a background thread."""
        TrainingWorker._active_worker = self

        self.stageChanged.emit("Preparing Dataset")
        self.progressChanged.emit(5)
        self._log("Preparing dataset...")

        if self.is_cancelled():
            self.failed.emit("Training cancelled by user")
            return

        try:
            # 1. Load MLPlan
            plan_file = Path(self.workspace_path) / "configs" / "ml_plan.json"
            if not plan_file.exists():
                raise FileNotFoundError("ML Plan configuration file configs/ml_plan.json not found.")

            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
                plan = MLPlan.model_validate(plan_data)

            if self.is_cancelled():
                raise RuntimeError("Training cancelled by user")

            # 2. Load dataset Dataframe
            location = self.dataset_metadata.get("location")
            if not location or not os.path.exists(location):
                raise FileNotFoundError(f"Dataset source file not found at {location}")

            if location.endswith(".csv"):
                df = pd.read_csv(location)
            else:
                df = pd.read_excel(location)

            if self.is_cancelled():
                raise RuntimeError("Training cancelled by user")

            # 3. Build DatasetContext
            analysis_report = AnalysisService().analyze(df)
            builder = DatasetContextBuilder()
            dataset_context = builder.build(
                dataset_id=self.dataset_metadata.get("name", "Unknown"),
                file_name=os.path.basename(location),
                row_count=df.shape[0],
                column_count=df.shape[1],
                memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
                analysis_report=analysis_report,
            )

            if self.is_cancelled():
                raise RuntimeError("Training cancelled by user")

            # Setup wrapping patches to intercept pipeline execution stages dynamically
            orig_split = SplitExecutor.execute
            orig_prep = PreprocessingPipelineBuilder.build
            orig_fe = FeatureEngineeringExecutor.execute
            orig_fs = FeatureSelectionExecutor.execute
            orig_opt = HyperparameterOptimizer.optimize
            orig_train = TrainingExecutor.train
            orig_eval = EvaluationEngine.evaluate

            def wrap_split(self_exec, *args, **kwargs):
                if TrainingWorker._active_worker and TrainingWorker._active_worker.is_cancelled():
                    raise RuntimeError("Training cancelled by user")
                TrainingWorker._active_worker.stageChanged.emit("Splitting Dataset")
                TrainingWorker._active_worker.progressChanged.emit(15)
                TrainingWorker._active_worker._log("Splitting dataset...")
                return orig_split(self_exec, *args, **kwargs)

            def wrap_prep(self_exec, *args, **kwargs):
                if TrainingWorker._active_worker and TrainingWorker._active_worker.is_cancelled():
                    raise RuntimeError("Training cancelled by user")
                TrainingWorker._active_worker.stageChanged.emit("Building Preprocessing Pipeline")
                TrainingWorker._active_worker.progressChanged.emit(30)
                TrainingWorker._active_worker._log("Building preprocessing pipeline...")
                return orig_prep(self_exec, *args, **kwargs)

            def wrap_fe(self_exec, *args, **kwargs):
                if TrainingWorker._active_worker and TrainingWorker._active_worker.is_cancelled():
                    raise RuntimeError("Training cancelled by user")
                TrainingWorker._active_worker.stageChanged.emit("Feature Engineering")
                TrainingWorker._active_worker.progressChanged.emit(45)
                TrainingWorker._active_worker._log("Running feature engineering...")
                return orig_fe(self_exec, *args, **kwargs)

            def wrap_fs(self_exec, *args, **kwargs):
                if TrainingWorker._active_worker and TrainingWorker._active_worker.is_cancelled():
                    raise RuntimeError("Training cancelled by user")
                TrainingWorker._active_worker.stageChanged.emit("Feature Selection")
                TrainingWorker._active_worker.progressChanged.emit(60)
                TrainingWorker._active_worker._log("Running feature selection...")
                return orig_fs(self_exec, *args, **kwargs)

            def wrap_opt(self_exec, *args, **kwargs):
                if TrainingWorker._active_worker and TrainingWorker._active_worker.is_cancelled():
                    raise RuntimeError("Training cancelled by user")
                candidate = kwargs.get("candidate") or args[1]
                cid = candidate.candidate_id
                TrainingWorker._active_worker.stageChanged.emit("Hyperparameter Search")
                TrainingWorker._active_worker.progressChanged.emit(70)
                TrainingWorker._active_worker._log(f"Optimizing hyperparameters for model {cid}...")
                TrainingWorker._active_worker.candidateChanged.emit(cid, "Optimizing", "-")
                return orig_opt(self_exec, *args, **kwargs)

            def wrap_train(self_exec, *args, **kwargs):
                if TrainingWorker._active_worker and TrainingWorker._active_worker.is_cancelled():
                    raise RuntimeError("Training cancelled by user")
                candidate = kwargs.get("candidate") or args[3]
                cid = candidate.candidate_id
                TrainingWorker._active_worker.stageChanged.emit("Training Models")
                TrainingWorker._active_worker.progressChanged.emit(80)
                TrainingWorker._active_worker._log(f"Training candidate model {cid}...")
                TrainingWorker._active_worker.candidateChanged.emit(cid, "Training", "-")
                return orig_train(self_exec, *args, **kwargs)

            def wrap_eval(self_exec, *args, **kwargs):
                if TrainingWorker._active_worker and TrainingWorker._active_worker.is_cancelled():
                    raise RuntimeError("Training cancelled by user")
                training_res = kwargs.get("training_result") or args[3]
                cid = training_res.candidate.candidate_id
                TrainingWorker._active_worker.stageChanged.emit("Evaluating Models")
                TrainingWorker._active_worker.progressChanged.emit(90)
                TrainingWorker._active_worker._log(f"Evaluating candidate model {cid}...")
                TrainingWorker._active_worker.candidateChanged.emit(cid, "Evaluating", "-")
                
                res = orig_eval(self_exec, *args, **kwargs)
                
                metric_val = f"{res.primary_metric_value:.4f}"
                TrainingWorker._active_worker._log(f"Finished evaluating {cid}. Score: {metric_val}")
                TrainingWorker._active_worker.candidateChanged.emit(cid, "Completed", metric_val)
                return res

            with patch.object(SplitExecutor, "execute", wrap_split), \
                 patch.object(PreprocessingPipelineBuilder, "build", wrap_prep), \
                 patch.object(FeatureEngineeringExecutor, "execute", wrap_fe), \
                 patch.object(FeatureSelectionExecutor, "execute", wrap_fs), \
                 patch.object(HyperparameterOptimizer, "optimize", wrap_opt), \
                 patch.object(TrainingExecutor, "train", wrap_train), \
                 patch.object(EvaluationEngine, "evaluate", wrap_eval):
                
                # Execute baseline ML orchestrator pipeline
                orchestrator = MLExecutionOrchestrator()
                execution_result = orchestrator.execute(
                    dataframe=df,
                    dataset_context=dataset_context,
                    plan=plan,
                )

            if self.is_cancelled():
                raise RuntimeError("Training cancelled by user")

            # 4. Selecting Champion
            self.stageChanged.emit("Selecting Champion")
            self.progressChanged.emit(95)
            self._log(f"Selecting best model champion: {execution_result.best_candidate_id}")

            # 5. Generating Report
            self.stageChanged.emit("Generating Report")
            self.progressChanged.emit(98)
            self._log("Generating execution report...")

            problem_def = ProblemDefinition(
                definition_id=plan.problem_definition_id,
                request_id=plan.request_id,
                dataset_id=plan.dataset_id,
                goal="Orchestrated ML execution",
                problem_type=plan.problem_type,
                target_column=plan.target_column,
                target_source=TargetSource.USER,
                feature_columns=list(plan.feature_columns),
                excluded_columns=[],
                primary_metric=plan.evaluation_plan.primary_metric,
                status=ResolutionStatus.RESOLVED,
            )

            report_builder = ExecutionReportBuilder()
            report = report_builder.build(
                dataset_context=dataset_context,
                problem_definition=problem_def,
                plan=plan,
                execution_result=execution_result,
            )

            # Global report save
            reports_dir = Path(self.workspace_path) / "reports"
            os.makedirs(reports_dir, exist_ok=True)
            report_file = reports_dir / "execution_report.json"
            
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=4))

            # Runs-based report directory save
            runs_dir = Path(self.workspace_path) / "runs"
            run_name = time.strftime("run_%Y_%m_%d_%H%M%S")
            active_run_dir = runs_dir / run_name
            os.makedirs(active_run_dir, exist_ok=True)

            with open(active_run_dir / "ml_plan.json", "w", encoding="utf-8") as f:
                f.write(plan.model_dump_json(indent=4))
            with open(active_run_dir / "execution_report.json", "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=4))

            with open(active_run_dir / "model.pkl", "wb") as f:
                pickle.dump(execution_result.best_model, f)

            with open(active_run_dir / "training.log", "w", encoding="utf-8") as f:
                f.write("\n".join(self._log_history))

            metrics_summary = {
                cid: float(res.primary_metric_value)
                for cid, res in execution_result.candidate_results.items()
            }
            with open(active_run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=4)

            self.stageChanged.emit("Completed")
            self.progressChanged.emit(100)
            self._log("Execution report and run folder saved successfully.")
            self.finished.emit()

        except Exception as e:
            self._log(f"Execution Error: {e}")
            self.failed.emit(str(e))
        finally:
            TrainingWorker._active_worker = None
