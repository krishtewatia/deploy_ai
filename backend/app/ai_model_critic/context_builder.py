"""Context builder for extracting a compact JSON description of an ExecutionReport."""

from __future__ import annotations

import json
from backend.app.ml_execution.execution_report import ExecutionReport


class AIModelCriticContextBuilder:
    """Builds a compact context dictionary from an ExecutionReport and serializes it to JSON."""

    def build(self, execution_report: ExecutionReport) -> str:
        """Convert ExecutionReport into a compact context dictionary serialized to JSON.

        Args:
            execution_report: Consolidated ExecutionReport from execution.

        Returns:
            A compact JSON string representing the execution metadata and results.
        """
        # Construct compact context mapping
        context_data = {
            "report_id": execution_report.report_id,
            "plan_id": execution_report.plan_id,
            "problem": {
                "problem_type": execution_report.problem_type,
                "goal": execution_report.training_summary.get("goal") or "AutoML model training workload",
            },
            "target": execution_report.target_column,
            "feature_count": len(execution_report.feature_columns),
            "candidate_summaries": [
                {
                    "candidate_id": c.candidate_id,
                    "model_family": c.model_family,
                    "primary_metric": c.primary_metric,
                    "primary_metric_value": c.primary_metric_value,
                    "training_duration": c.training_duration,
                    "evaluation_duration": c.evaluation_duration,
                    "search_strategy": c.search_strategy,
                    "best_parameters": c.best_parameters,
                }
                for c in execution_report.candidate_summaries
            ],
            "champion": {
                "candidate_id": execution_report.champion_summary.candidate_id,
                "model_family": execution_report.champion_summary.model_family,
                "metrics": {
                    "primary": execution_report.champion_summary.primary_metric,
                    "value": execution_report.champion_summary.primary_metric_value,
                },
                "feature_importance": execution_report.champion_summary.feature_importance,
                "training_duration": execution_report.champion_summary.training_duration,
                "evaluation_duration": execution_report.champion_summary.evaluation_duration,
            },
            "metrics": execution_report.evaluation_summary,
            "training_time": execution_report.execution_duration,
            "evaluation_time": sum(c.evaluation_duration for c in execution_report.candidate_summaries),
            "warnings": execution_report.warnings,
        }

        return json.dumps(context_data, indent=2)
