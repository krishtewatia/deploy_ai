"""Champion Comparator executing deterministic report evaluations and model governance selection."""

from __future__ import annotations

import datetime
import uuid
from backend.app.ml_execution.execution_report import ExecutionReport
from backend.app.model_governance.schemas import ChampionDecision, Winner


class ChampionComparatorError(Exception):
    """Raised when champion model comparison fails."""

    pass


class ChampionComparator:
    """Compares baseline and retrained execution reports deterministically."""

    # Define metric directions
    HIGHER_IS_BETTER = {"accuracy", "precision", "recall", "f1", "roc_auc", "r2"}
    LOWER_IS_BETTER = {"mae", "mse", "rmse"}

    def compare(
        self,
        baseline_report: ExecutionReport,
        retrained_report: ExecutionReport,
    ) -> ChampionDecision:
        """Compare baseline and retrained reports and select the champion.

        Args:
            baseline_report: Execution report of the baseline run.
            retrained_report: Execution report of the retrained run.

        Returns:
            A ChampionDecision record.

        Raises:
            ChampionComparatorError: On validation, identity, or metric mismatches.
        """
        # 1. Validation for None
        if baseline_report is None:
            raise ChampionComparatorError("baseline_report cannot be None")
        if retrained_report is None:
            raise ChampionComparatorError("retrained_report cannot be None")

        # 2. Validation for types
        if not isinstance(baseline_report, ExecutionReport):
            raise ChampionComparatorError("baseline_report must be an ExecutionReport instance")
        if not isinstance(retrained_report, ExecutionReport):
            raise ChampionComparatorError("retrained_report must be an ExecutionReport instance")

        # 3. Identity validations
        if baseline_report.dataset_id != retrained_report.dataset_id:
            raise ChampionComparatorError(
                f"Identity mismatch: baseline dataset_id '{baseline_report.dataset_id}' "
                f"does not match retrained dataset_id '{retrained_report.dataset_id}'"
            )
        if baseline_report.problem_definition_id != retrained_report.problem_definition_id:
            raise ChampionComparatorError(
                f"Identity mismatch: baseline problem_definition_id '{baseline_report.problem_definition_id}' "
                f"does not match retrained problem_definition_id '{retrained_report.problem_definition_id}'"
            )
        if baseline_report.target_column != retrained_report.target_column:
            raise ChampionComparatorError(
                f"Identity mismatch: baseline target_column '{baseline_report.target_column}' "
                f"does not match retrained target_column '{retrained_report.target_column}'"
            )

        # 4. Primary metric validations
        b_metric_name = baseline_report.champion_summary.primary_metric
        r_metric_name = retrained_report.champion_summary.primary_metric
        if b_metric_name != r_metric_name:
            raise ChampionComparatorError(
                f"Metric mismatch: baseline primary_metric '{b_metric_name}' "
                f"does not match retrained primary_metric '{r_metric_name}'"
            )

        metric_name = b_metric_name.lower()
        if metric_name not in self.HIGHER_IS_BETTER and metric_name not in self.LOWER_IS_BETTER:
            raise ChampionComparatorError(f"Unsupported primary metric for comparison: '{b_metric_name}'")

        # Extract metric values
        b_val = baseline_report.champion_summary.primary_metric_value
        r_val = retrained_report.champion_summary.primary_metric_value

        # Calculate absolute difference
        diff = r_val - b_val
        abs_diff = abs(diff)

        # Determine winner
        if abs_diff < 1e-9:
            winner = Winner.TIE
            winner_report = baseline_report
            improvement_detected = False
            reason = f"Baseline and retrained models are tied on {b_metric_name} with difference {diff:.12f} (< 1e-9)"
        elif metric_name in self.HIGHER_IS_BETTER:
            if diff > 0:
                winner = Winner.RETRAINED
                winner_report = retrained_report
                improvement_detected = True
                reason = f"Retrained model won on {b_metric_name} ({r_val:.6f} vs baseline {b_val:.6f})"
            else:
                winner = Winner.BASELINE
                winner_report = baseline_report
                improvement_detected = False
                reason = f"Baseline model won on {b_metric_name} ({b_val:.6f} vs retrained {r_val:.6f})"
        else:  # LOWER_IS_BETTER
            if diff < 0:
                winner = Winner.RETRAINED
                winner_report = retrained_report
                improvement_detected = True
                reason = f"Retrained model won on {b_metric_name} ({r_val:.6f} vs baseline {b_val:.6f})"
            else:
                winner = Winner.BASELINE
                winner_report = baseline_report
                improvement_detected = False
                reason = f"Baseline model won on {b_metric_name} ({b_val:.6f} vs retrained {r_val:.6f})"

        # Calculate relative improvement
        # Higher is better: (retrained - baseline) / baseline
        # Lower is better: (baseline - retrained) / baseline
        if b_val != 0:
            if metric_name in self.HIGHER_IS_BETTER:
                rel_improvement = (r_val - b_val) / b_val
            else:
                rel_improvement = (b_val - r_val) / b_val
        else:
            rel_improvement = 0.0

        # Production ready check (from training_summary or evaluation_summary according to critic review)
        is_prod_ready = bool(
            winner_report.training_summary.get("production_ready", False)
            or winner_report.evaluation_summary.get("production_ready", False)
        )

        return ChampionDecision(
            decision_id=f"decision_{uuid.uuid4().hex[:8]}",
            baseline_report_id=baseline_report.report_id,
            retrained_report_id=retrained_report.report_id,
            winner=winner,
            winner_report=winner_report,
            improvement_detected=improvement_detected,
            metric_name=b_metric_name,
            baseline_metric=b_val,
            retrained_metric=r_val,
            relative_improvement=rel_improvement,
            decision_reason=reason,
            production_ready=is_prod_ready,
            comparison_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
