"""DeployAIEngine — Central pipeline orchestrator.

Orchestrates the 11 core stages of the DeployAI workflow:
Upload -> Validation -> Dataset Intelligence -> Problem Definition -> AI Configuration ->
Planning -> Execution -> Champion Selection -> AI Explanation -> PDF Generation -> Export
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from backend.app.upload.upload_service import UploadService
from backend.app.upload.csv_loader import CSVLoader
from backend.app.upload.excel_loader import ExcelLoader
from backend.app.analysis.analysis_service import AnalysisService
from backend.app.dataset_intelligence.context_builder import DatasetContextBuilder
from backend.app.problem_definition.resolver import ProblemResolver
from backend.app.ml_request.schemas import UserMLRequest
from backend.app.hardware.detector import HardwareDetector
from backend.app.compute_capabilities.analyzer import HardwareCapabilityAnalyzer
from backend.app.ml_plan.orchestrator import MLPlanningOrchestrator, PlanningMode
from backend.app.ml_execution.orchestrator import MLExecutionOrchestrator
from backend.app.ml_execution.execution_report import ExecutionReportBuilder
from backend.app.model_governance.comparator import ChampionComparator
from backend.app.model_governance.schemas import ChampionDecision, Winner
from backend.app.ai_model_critic.critic_service import AIModelCritic
from backend.app.reporting.schemas import ExecutiveReport
from backend.app.ai_planning.providers.base import AIProvider
from backend.app.pipeline.schemas import (
    AIExplanation,
    ExportResult,
    PipelineContext,
    PipelineStage,
    PipelineStatus,
)

logger = logging.getLogger(__name__)


class DeployAIEngineError(Exception):
    """Raised when an unrecoverable pipeline execution error occurs."""


class DeployAIEngine:
    """Central orchestrator managing pipeline execution across all backend packages.

    Operates purely over in-memory domain models and the unified :class:`PipelineContext`.
    Does NOT depend on web servers or HTTP APIs.
    """

    def __init__(self) -> None:
        self._contexts: Dict[str, PipelineContext] = {}

    def get_context(self, pipeline_id: str) -> Optional[PipelineContext]:
        """Retrieve an active pipeline context by ID."""
        return self._contexts.get(pipeline_id)

    def run_pipeline(
        self,
        dataset_path: str,
        target_column: Optional[str] = None,
        user_request: Optional[UserMLRequest] = None,
        planning_mode: PlanningMode = PlanningMode.AI_ASSISTED,
        ai_provider: Optional[AIProvider] = None,
        export_format: str = "PICKLE",
        progress_callback: Optional[Callable[[PipelineStage, float], None]] = None,
    ) -> PipelineContext:
        """Run the complete end-to-end ML pipeline sequentially.

        Parameters
        ----------
        dataset_path:
            Path to the local dataset file (.csv, .xlsx, .xls).
        target_column:
            Optional target column name specified by the user.
        user_request:
            Optional user ML requirements and preferences.
        planning_mode:
            Planning mode (BASELINE_ONLY, AI_ENHANCED, or HYBRID).
        ai_provider:
            Optional custom AI provider to inject.
        export_format:
            Desired export format (e.g. PICKLE, JOBLIB, ONNX).
        progress_callback:
            Optional callback invoked at each stage with (stage, progress_percentage).

        Returns
        -------
        PipelineContext
            The fully populated pipeline context upon completion or failure.
        """
        context = PipelineContext(dataset_path=dataset_path, export_format=export_format)
        self._contexts[context.pipeline_id] = context
        context.status = PipelineStatus.RUNNING

        stages = [
            (PipelineStage.UPLOAD, 0.1),
            (PipelineStage.VALIDATION, 0.2),
            (PipelineStage.DATASET_INTELLIGENCE, 0.3),
            (PipelineStage.PROBLEM_DEFINITION, 0.4),
            (PipelineStage.AI_CONFIGURATION, 0.5),
            (PipelineStage.PLANNING, 0.6),
            (PipelineStage.EXECUTION, 0.7),
            (PipelineStage.CHAMPION_SELECTION, 0.8),
            (PipelineStage.AI_EXPLANATION, 0.85),
            (PipelineStage.PDF_GENERATION, 0.9),
            (PipelineStage.EXPORT, 1.0),
        ]

        logger.info("Starting pipeline %s for dataset %r", context.pipeline_id, dataset_path)

        if planning_mode == PlanningMode.AI_ASSISTED and ai_provider is None:
            from backend.app.ai_providers.factory import detect_ai_provider
            ai_provider = detect_ai_provider()

        try:
            for stage, progress in stages:
                context.current_stage = stage
                if progress_callback:
                    progress_callback(stage, progress)

                context = self.run_stage(
                    stage=stage,
                    context=context,
                    target_column=target_column,
                    user_request=user_request,
                    planning_mode=planning_mode,
                    ai_provider=ai_provider,
                )

            context.status = PipelineStatus.COMPLETED
            logger.info("Pipeline %s completed successfully.", context.pipeline_id)

        except Exception as exc:
            logger.exception("Pipeline %s failed at stage %s: %s", context.pipeline_id, context.current_stage, exc)
            context.status = PipelineStatus.FAILED
            context.error_message = str(exc)

        return context

    def run_stage(
        self,
        stage: PipelineStage,
        context: PipelineContext,
        target_column: Optional[str] = None,
        user_request: Optional[UserMLRequest] = None,
        planning_mode: PlanningMode = PlanningMode.AI_ASSISTED,
        ai_provider: Optional[AIProvider] = None,
    ) -> PipelineContext:
        """Run a single specific pipeline stage on a PipelineContext."""
        logger.info("Executing stage %s for pipeline %s", stage.value, context.pipeline_id)

        if stage == PipelineStage.UPLOAD:
            self._execute_upload(context)
        elif stage == PipelineStage.VALIDATION:
            self._execute_validation(context, target_column=target_column)
        elif stage == PipelineStage.DATASET_INTELLIGENCE:
            self._execute_dataset_intelligence(context)
        elif stage == PipelineStage.PROBLEM_DEFINITION:
            self._execute_problem_definition(context, target_column=target_column, user_request=user_request)
        elif stage == PipelineStage.AI_CONFIGURATION:
            self._execute_ai_configuration(context)
        elif stage == PipelineStage.PLANNING:
            self._execute_planning(context, user_request=user_request, planning_mode=planning_mode, ai_provider=ai_provider)
        elif stage == PipelineStage.EXECUTION:
            self._execute_execution(context)
        elif stage == PipelineStage.CHAMPION_SELECTION:
            self._execute_champion_selection(context)
        elif stage == PipelineStage.AI_EXPLANATION:
            self._execute_ai_explanation(context, ai_provider=ai_provider)
        elif stage == PipelineStage.PDF_GENERATION:
            self._execute_pdf_generation(context)
        elif stage == PipelineStage.EXPORT:
            self._execute_export(context)

        context.mark_stage_completed(stage)
        return context

    # ── Stage Executions ────────────────────────────────────────────────

    def _execute_upload(self, context: PipelineContext) -> None:
        if not context.dataset_path:
            raise DeployAIEngineError("dataset_path is required for UPLOAD stage.")

        import pandas as pd
        
        # Check for multiple files separated by commas
        paths_raw = [p.strip() for p in context.dataset_path.split(",")]
        dfs = []
        last_metadata = None
        total_rows = 0

        upload_service = UploadService()

        for path_str in paths_raw:
            path = Path(path_str).resolve()
            if not path.exists():
                raise DeployAIEngineError(f"Dataset file not found: {path_str}")

            # Ingest and validate file metadata
            upload_result = upload_service.process(str(path), original_filename=path.name)
            last_metadata = upload_result["metadata"]

            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = CSVLoader().load(str(path))
            elif suffix in [".xlsx", ".xls"]:
                df = ExcelLoader().load(str(path))
            else:
                raise DeployAIEngineError(f"Unsupported dataset format: {suffix}")

            dfs.append(df)
            total_rows += len(df)

        if not dfs:
            raise DeployAIEngineError("No datasets were successfully loaded.")

        # If multiple datasets are provided, verify they have matching columns (same context) and merge them
        if len(dfs) > 1:
            first_cols = set(dfs[0].columns)
            for idx, df in enumerate(dfs[1:], 1):
                if set(df.columns) != first_cols:
                    raise DeployAIEngineError(
                        f"Cannot merge datasets: schema mismatch. "
                        f"Dataset {paths_raw[idx]} columns do not match the first dataset's columns."
                    )
            # Concatenate along rows
            df_final = pd.concat(dfs, ignore_index=True)
            logger.info("Merged %d datasets with matching context into a single DataFrame of %d rows.", len(dfs), len(df_final))
        else:
            df_final = dfs[0]

        # Inconsistency Correction: Deduplicate identical columns (same content/data values)
        columns_to_drop = []
        for i in range(len(df_final.columns)):
            col1 = df_final.columns[i]
            if col1 in columns_to_drop:
                continue
            for j in range(i + 1, len(df_final.columns)):
                col2 = df_final.columns[j]
                if col2 in columns_to_drop:
                    continue
                # Compare series content, handling NaN values cleanly
                if df_final[col1].equals(df_final[col2]):
                    columns_to_drop.append(col2)

        if columns_to_drop:
            logger.info("Auto-deduplicated matching columns: %s", columns_to_drop)
            df_final = df_final.drop(columns=columns_to_drop)

        # Update metadata with final counts
        last_metadata.rows = len(df_final)
        last_metadata.columns = len(df_final.columns)

        context.dataset_metadata = last_metadata
        context.raw_dataframe = df_final

    def _execute_validation(self, context: PipelineContext, target_column: Optional[str] = None) -> None:
        if context.raw_dataframe is None:
            raise DeployAIEngineError("raw_dataframe missing. Run UPLOAD stage first.")

        analysis_service = AnalysisService()
        context.analysis_report = analysis_service.analyze(
            context.raw_dataframe,
            target_column=target_column,
        )

    def _execute_dataset_intelligence(self, context: PipelineContext) -> None:
        if not context.analysis_report:
            raise DeployAIEngineError("analysis_report missing. Run VALIDATION stage first.")

        meta = context.dataset_metadata
        builder = DatasetContextBuilder()
        context.dataset_context = builder.build(
            dataset_id=context.pipeline_id,
            file_name=meta.file_name if meta else "dataset",
            row_count=meta.rows if meta else len(context.raw_dataframe),
            column_count=meta.columns if meta else len(context.raw_dataframe.columns),
            memory_usage_bytes=context.raw_dataframe.memory_usage(deep=True).sum() if context.raw_dataframe is not None else 0,
            analysis_report=context.analysis_report,
        )

    def _execute_problem_definition(
        self,
        context: PipelineContext,
        target_column: Optional[str] = None,
        user_request: Optional[UserMLRequest] = None,
    ) -> None:
        if not context.dataset_context:
            raise DeployAIEngineError("dataset_context missing. Run DATASET_INTELLIGENCE stage first.")

        if user_request is None:
            user_request = UserMLRequest(
                request_id=f"req-{context.pipeline_id}",
                goal="Automated model training",
                target_column=target_column,
            )

        resolver = ProblemResolver()
        context.problem_definition = resolver.resolve(
            dataset_context=context.dataset_context,
            user_request=user_request,
        )

    def _execute_ai_configuration(self, context: PipelineContext) -> None:
        detector = HardwareDetector()
        context.hardware_profile = detector.detect()

        analyzer = HardwareCapabilityAnalyzer()
        context.compute_capabilities = analyzer.analyze(context.hardware_profile)

    def _execute_planning(
        self,
        context: PipelineContext,
        user_request: Optional[UserMLRequest] = None,
        planning_mode: PlanningMode = PlanningMode.AI_ASSISTED,
        ai_provider: Optional[AIProvider] = None,
    ) -> None:
        if not context.dataset_context or not context.problem_definition or not context.compute_capabilities:
            raise DeployAIEngineError("Context unfulfilled for PLANNING stage.")

        if user_request is None:
            target_col = context.problem_definition.target_column if context.problem_definition else None
            user_request = UserMLRequest(
                request_id=f"req-{context.pipeline_id}",
                goal="Automated model training",
                target_column=target_col,
            )

        orchestrator = MLPlanningOrchestrator()
        result = orchestrator.create_plan(
            dataset_context=context.dataset_context,
            user_request=user_request,
            compute_capabilities=context.compute_capabilities,
            mode=planning_mode,
            ai_provider=ai_provider,
        )

        context.problem_definition = result.problem_definition
        context.ml_plan = result.final_plan
        context.plan_validation = result.validation_result

    def _execute_execution(self, context: PipelineContext) -> None:
        if context.raw_dataframe is None or not context.ml_plan or not context.dataset_context:
            raise DeployAIEngineError("Context unfulfilled for EXECUTION stage.")

        orchestrator = MLExecutionOrchestrator()
        context.execution_result = orchestrator.execute(
            dataframe=context.raw_dataframe,
            dataset_context=context.dataset_context,
            plan=context.ml_plan,
        )
        builder = ExecutionReportBuilder()
        context.execution_report = builder.build(
            dataset_context=context.dataset_context,
            problem_definition=context.problem_definition,
            plan=context.ml_plan,
            execution_result=context.execution_result,
        )

    def _execute_champion_selection(self, context: PipelineContext) -> None:
        if not context.execution_report:
            raise DeployAIEngineError("execution_report missing. Run EXECUTION stage first.")

        comparator = ChampionComparator()
        context.champion_decision = comparator.compare(
            baseline_report=context.execution_report,
            retrained_report=context.execution_report,
        )

    def _execute_ai_explanation(
        self,
        context: PipelineContext,
        ai_provider: Optional[AIProvider] = None,
    ) -> None:
        if not context.champion_decision or not context.execution_report:
            raise DeployAIEngineError("Context unfulfilled for AI_EXPLANATION stage.")

        champ = context.execution_report.champion_summary
        champ_id = champ.candidate_id
        feat_importances = champ.feature_importance or {}

        critique = None
        if ai_provider is not None:
            try:
                critic = AIModelCritic(provider=ai_provider)
                critique = critic.review(execution_report=context.execution_report)
            except Exception as exc:
                logger.warning("AIModelCritic optional evaluation skipped: %s", exc)

        context.ai_explanation = AIExplanation(
            champion_model_id=champ_id,
            feature_importances=feat_importances,
            ai_critique=critique,
            explanation_text=context.champion_decision.decision_reason,
        )

    def _execute_pdf_generation(self, context: PipelineContext) -> None:
        if not context.execution_report or not context.champion_decision:
            raise DeployAIEngineError("Context unfulfilled for PDF_GENERATION stage.")

        exp_rep = context.execution_report
        champ = exp_rep.champion_summary

        report = ExecutiveReport(
            report_id=f"exec-{uuid.uuid4().hex[:8]}",
            title=f"DeployAI Executive Model Report: {champ.model_family}",
            generated_timestamp=datetime.now(timezone.utc).isoformat(),
            problem_summary={
                "problem_type": str(exp_rep.problem_type),
                "target_column": exp_rep.target_column,
                "feature_columns": exp_rep.feature_columns,
            },
            dataset_summary={
                "dataset_id": exp_rep.dataset_id,
                "rows": context.dataset_metadata.rows if context.dataset_metadata else 0,
                "columns": context.dataset_metadata.columns if context.dataset_metadata else 0,
            },
            pipeline_summary={
                "plan_id": exp_rep.plan_id,
                "execution_duration_seconds": exp_rep.execution_duration,
            },
            models_summary=[
                {
                    "candidate_id": c.candidate_id,
                    "algorithm": str(c.model_family),
                    "primary_metric": c.primary_metric,
                    "metric_value": c.primary_metric_value,
                }
                for c in exp_rep.candidate_summaries
            ],
            champion_summary={
                "candidate_id": champ.candidate_id,
                "algorithm": str(champ.model_family),
                "primary_metric": champ.primary_metric,
                "primary_metric_value": champ.primary_metric_value,
            },
            optimization_summary={"actions": []},
            ai_review={"critique": context.ai_explanation.explanation_text if context.ai_explanation else "N/A"},
            governance_summary={
                "winner": context.champion_decision.winner,
                "decision_reason": context.champion_decision.decision_reason,
            },
            deployment_summary={
                "production_ready": context.champion_decision.production_ready,
            },
            warnings=exp_rep.warnings,
            recommendations=["Deploy model to production endpoint"],
            executive_summary=context.champion_decision.decision_reason,
        )
        context.executive_report = report

        # Write actual PDF report to disk
        from backend.app.reporting.pdf_writer import PDFReportWriter
        pdf_path = f"reports/exec_report_{context.pipeline_id}.pdf"
        writer = PDFReportWriter()
        writer.write_pdf(report=report, output_path=pdf_path)
        context.pdf_report_path = pdf_path

    def _execute_export(self, context: PipelineContext) -> None:
        if not context.champion_decision or not context.execution_report:
            raise DeployAIEngineError("Champion model missing for EXPORT stage.")

        champ = context.execution_report.champion_summary
        fmt = context.export_format.upper()
        ext = "pkl"
        if fmt == "JOBLIB":
            ext = "joblib"
        elif fmt == "ONNX":
            ext = "onnx"

        # Resolve output file path
        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{champ.candidate_id}_{context.pipeline_id}.{ext}"

        # Get actual model object from the execution result
        model = context.execution_result.best_model if context.execution_result else None
        
        # Serialize model if it exists
        if model is not None:
            if fmt == "JOBLIB":
                import joblib
                joblib.dump(model, str(filepath))
            elif fmt == "ONNX":
                try:
                    import skl2onnx
                    from skl2onnx.common.data_types import FloatTensorType
                    
                    # Estimate features count
                    feat_cols = context.execution_report.feature_columns
                    initial_type = [('float_input', FloatTensorType([None, len(feat_cols)]))]
                    onx = skl2onnx.convert_sklearn(model, initial_types=initial_type)
                    with open(filepath, "wb") as f:
                        f.write(onx.SerializeToString())
                except Exception as exc:
                    logger.warning("ONNX conversion failed or skl2onnx not installed. Falling back to PICKLE for ONNX output: %s", exc)
                    import pickle
                    with open(filepath, "wb") as f:
                        pickle.dump(model, f)
            else:
                import pickle
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)
            logger.info("Successfully serialized champion model to %s (%s format).", filepath, fmt)

        context.export_result = ExportResult(
            champion_model_id=champ.candidate_id,
            export_format=fmt,
            artifact_path=str(filepath.resolve()),
        )