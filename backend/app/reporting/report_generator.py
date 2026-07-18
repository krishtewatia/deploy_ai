"""Report generator consolidating execution, review, optimization, and governance outputs."""

from __future__ import annotations

import datetime
import uuid
from backend.app.ml_execution.execution_report import ExecutionReport
from backend.app.ai_model_critic.schemas import ModelCritique
from backend.app.ai_model_optimizer.schemas import OptimizationResult
from backend.app.model_governance.schemas import ChampionDecision
from backend.app.reporting.schemas import ExecutiveReport


class ExecutiveReportGeneratorError(Exception):
    """Raised when ExecutiveReport generation fails."""

    pass


class ExecutiveReportGenerator:
    """Generates the final comprehensive ExecutiveReport presenting model training results."""

    def generate(
        self,
        *,
        execution_report: ExecutionReport,
        critique: ModelCritique,
        optimization: OptimizationResult,
        governance: ChampionDecision,
    ) -> ExecutiveReport:
        """Combine reports, critique, and decision data into one consolidated ExecutiveReport.

        Args:
            execution_report: Baseline run report.
            critique: Critique review of the baseline run.
            optimization: Mapped actions and optimized plan outputs.
            governance: Deterministic champion comparison decision.

        Returns:
            A consolidated ExecutiveReport schema instance.

        Raises:
            ExecutiveReportGeneratorError: For validation failures or identity mismatches.
        """
        # 1. Reject None inputs
        if execution_report is None:
            raise ExecutiveReportGeneratorError("execution_report cannot be None")
        if critique is None:
            raise ExecutiveReportGeneratorError("critique cannot be None")
        if optimization is None:
            raise ExecutiveReportGeneratorError("optimization cannot be None")
        if governance is None:
            raise ExecutiveReportGeneratorError("governance cannot be None")

        # 2. Reject wrong parameter types
        if not isinstance(execution_report, ExecutionReport):
            raise ExecutiveReportGeneratorError("execution_report must be an ExecutionReport instance")
        if not isinstance(critique, ModelCritique):
            raise ExecutiveReportGeneratorError("critique must be a ModelCritique instance")
        if not isinstance(optimization, OptimizationResult):
            raise ExecutiveReportGeneratorError("optimization must be an OptimizationResult instance")
        if not isinstance(governance, ChampionDecision):
            raise ExecutiveReportGeneratorError("governance must be a ChampionDecision instance")

        # 3. Identity validation checks
        if execution_report.report_id != critique.report_id:
            raise ExecutiveReportGeneratorError(
                f"Identity mismatch: critique links to report '{critique.report_id}' "
                f"but execution_report has ID '{execution_report.report_id}'"
            )
        if execution_report.plan_id != optimization.baseline_plan_id:
            raise ExecutiveReportGeneratorError(
                f"Identity mismatch: optimization baseline plan '{optimization.baseline_plan_id}' "
                f"does not match execution_report plan '{execution_report.plan_id}'"
            )
        if governance.baseline_report_id != execution_report.report_id:
            raise ExecutiveReportGeneratorError(
                f"Identity mismatch: governance baseline report '{governance.baseline_report_id}' "
                f"does not match execution_report ID '{execution_report.report_id}'"
            )

        # 4. Formulate summary sections
        prob_sum = {
            "problem_type": execution_report.problem_type,
            "target_column": execution_report.target_column,
            "goal": execution_report.training_summary.get("goal") or "Optimize ML Model Performance",
        }

        ds_sum = {
            "dataset_id": execution_report.dataset_id,
            "features": execution_report.feature_columns,
            "feature_count": len(execution_report.feature_columns),
        }

        opt_plan = optimization.optimized_plan
        pipeline_sum = {
            "preprocessing": [step.operation for step in opt_plan.preprocessing_steps],
            "feature_engineering": [step.operation for step in opt_plan.feature_engineering_steps],
            "feature_selection": opt_plan.feature_selection.method,
            "search_strategy": opt_plan.model_candidates[0].search_strategy if opt_plan.model_candidates else "none",
            "cross_validation": opt_plan.evaluation_plan.cross_validation_folds,
            "training_duration": governance.winner_report.execution_duration,
            "evaluation_duration": sum(c.evaluation_duration for c in governance.winner_report.candidate_summaries),
        }

        models_sum = [
            {
                "family": c.model_family,
                "metric": c.primary_metric,
                "score": c.primary_metric_value,
                "training_time": c.training_duration,
                "winner": (c.candidate_id == governance.winner_report.champion_summary.candidate_id),
            }
            for c in governance.winner_report.candidate_summaries
        ]

        champ_sum = {
            "candidate_id": governance.winner_report.champion_summary.candidate_id,
            "model_family": governance.winner_report.champion_summary.model_family,
            "metric_name": governance.metric_name,
            "metric_value": governance.winner_report.champion_summary.primary_metric_value,
            "relative_improvement": governance.relative_improvement,
            "production_readiness": governance.production_ready,
            "winner_configuration": governance.winner,
            "reason": governance.decision_reason,
        }

        opt_sum = {
            "optimization_id": optimization.optimization_id,
            "actions_applied": [
                {
                    "type": act.action_type,
                    "target": act.target,
                    "replacement": act.replacement,
                    "reason": act.reason,
                }
                for act in optimization.actions
                if act.action_type != "NO_ACTION"
            ],
            "summary": optimization.summary,
        }

        ai_rev = {
            "overall_grade": critique.overall_grade,
            "confidence": critique.confidence,
            "strengths": critique.strengths,
            "weaknesses": critique.weaknesses,
            "risks": critique.risks,
            "recommendations": critique.recommendations,
            "summary": critique.summary,
        }

        gov_sum = {
            "winner": governance.winner,
            "improvement_detected": governance.improvement_detected,
            "baseline_metric": governance.baseline_metric,
            "retrained_metric": governance.retrained_metric,
            "decision_reason": governance.decision_reason,
        }

        deploy_sum = {
            "production_ready": governance.production_ready,
            "recommended_monitoring": [
                "Monitor data drift on feature columns",
                "Log prediction confidence distributions",
            ],
            "recommended_retraining": [
                "Retrain if primary metric performance drops below 5%",
                "Scheduled retraining on monthly basis",
            ],
            "confidence": critique.confidence,
        }

        # Combine all warning messages
        all_warnings = sorted(list(set(
            execution_report.warnings +
            critique.warnings +
            governance.winner_report.warnings
        )))

        exec_summary_text = (
            f"Executive summary of AutoML run for dataset {execution_report.dataset_id}. "
            f"The model governance engine evaluated the retrained vs baseline configurations. "
            f"Resulting winner: {governance.winner}. Grade card rating: {critique.overall_grade}."
        )

        return ExecutiveReport(
            report_id=f"exec_report_{uuid.uuid4().hex[:8]}",
            title="DeployAI Executive Model Training & Governance Report",
            generated_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            problem_summary=prob_sum,
            dataset_summary=ds_sum,
            pipeline_summary=pipeline_sum,
            models_summary=models_sum,
            champion_summary=champ_sum,
            optimization_summary=opt_sum,
            ai_review=ai_rev,
            governance_summary=gov_sum,
            deployment_summary=deploy_sum,
            warnings=all_warnings,
            recommendations=critique.recommendations,
            executive_summary=exec_summary_text,
        )
