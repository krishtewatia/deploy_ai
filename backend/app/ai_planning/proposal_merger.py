"""Controlled Proposal Merger.

Applies validated AIDecisionProposal changes to a copy of the deterministic
baseline MLPlan.  The merger enforces safety boundaries, never mutates the
baseline, and validates the candidate plan through MLPlanValidator.
"""

from __future__ import annotations

import copy
import uuid
from typing import Optional

from backend.app.ai_planning.schemas import (
    AIDecisionProposal,
    AIEvaluationProposal,
    AIFeatureEngineeringProposal,
    AIFeatureSelectionProposal,
    AIModelCandidateProposal,
    AIPreprocessingProposal,
    ProposalAction,
)
from backend.app.compute_capabilities.schemas import ComputeCapabilities
from backend.app.dataset_intelligence.schemas import DatasetContext
from backend.app.ml_plan.schemas import (
    EvaluationPlan,
    FeatureEngineeringStep,
    FeatureSelectionPlan,
    MLPlan,
    ModelCandidate,
    PreprocessingStep,
    SearchStrategy,
)
from backend.app.ml_plan.validator import MLPlanValidator
from backend.app.problem_definition.schemas import ProblemDefinition


class ProposalMergerError(Exception):
    """Raised when the merger encounters an error during proposal application."""


class ProposalMerger:
    """Applies AI proposals to a baseline plan with safety guards.

    Safety boundaries:
    - Never mutates the baseline plan.
    - Validates artifact linkage before merging.
    - AI cannot modify: dataset/request/problem identity, target column,
      problem type, compute tier, safe worker limits, accelerator, split strategy.
    - The candidate plan is validated through MLPlanValidator.
    - Invalid candidates are rejected with a ProposalMergerError.
    """

    def __init__(self, validator: Optional[MLPlanValidator] = None) -> None:
        self._validator = validator or MLPlanValidator()

    def merge(
        self,
        *,
        baseline_plan: MLPlan,
        ai_proposal: AIDecisionProposal,
        dataset_context: DatasetContext,
        problem_definition: ProblemDefinition,
        compute_capabilities: ComputeCapabilities,
    ) -> MLPlan:
        """Merge AI proposals into the baseline plan and validate.

        Args:
            baseline_plan: The deterministic baseline MLPlan (never mutated).
            ai_proposal: The validated AI decision proposal.
            dataset_context: Dataset metadata for validation.
            problem_definition: Resolved problem for validation.
            compute_capabilities: Compute constraints for validation.

        Returns:
            A validated candidate MLPlan with AI proposals applied.

        Raises:
            ProposalMergerError: On linkage mismatch, merge failure, or
                validation errors in the candidate plan.
        """
        self._verify_linkage(baseline_plan, ai_proposal)

        # Deep copy the baseline to create the candidate
        candidate_data = copy.deepcopy(baseline_plan.model_dump())

        # Apply proposals
        self._apply_preprocessing_proposals(
            candidate_data, ai_proposal.preprocessing_proposals
        )
        self._apply_feature_engineering_proposals(
            candidate_data, ai_proposal.feature_engineering_proposals
        )
        self._apply_model_candidate_proposals(
            candidate_data, ai_proposal.model_candidate_proposals
        )
        if ai_proposal.feature_selection_proposal is not None:
            self._apply_feature_selection_proposal(
                candidate_data, ai_proposal.feature_selection_proposal
            )
        if ai_proposal.evaluation_proposal is not None:
            self._apply_evaluation_proposal(
                candidate_data, ai_proposal.evaluation_proposal
            )

        # Reconstruct and validate
        try:
            candidate_plan = MLPlan.model_validate(candidate_data)
        except Exception as exc:
            raise ProposalMergerError(
                f"Failed to construct candidate MLPlan after merge: {exc}"
            ) from exc

        # Run MLPlanValidator
        result = self._validator.validate(
            plan=candidate_plan,
            dataset_context=dataset_context,
            problem_definition=problem_definition,
            compute_capabilities=compute_capabilities,
        )

        if not result.is_valid:
            error_messages = "; ".join(
                f"[{e.code}] {e.message}" for e in result.errors
            )
            raise ProposalMergerError(
                f"AI-assisted candidate plan failed validation: {error_messages}"
            )

        return candidate_plan

    def _verify_linkage(
        self, plan: MLPlan, proposal: AIDecisionProposal
    ) -> None:
        """Verify artifact identity linkage between plan and proposal."""
        mismatches = []
        if proposal.baseline_plan_id != plan.plan_id:
            mismatches.append(
                f"baseline_plan_id: proposal='{proposal.baseline_plan_id}' "
                f"vs plan='{plan.plan_id}'"
            )
        if proposal.dataset_id != plan.dataset_id:
            mismatches.append(
                f"dataset_id: proposal='{proposal.dataset_id}' "
                f"vs plan='{plan.dataset_id}'"
            )
        if proposal.request_id != plan.request_id:
            mismatches.append(
                f"request_id: proposal='{proposal.request_id}' "
                f"vs plan='{plan.request_id}'"
            )
        if proposal.problem_definition_id != plan.problem_definition_id:
            mismatches.append(
                f"problem_definition_id: proposal='{proposal.problem_definition_id}' "
                f"vs plan='{plan.problem_definition_id}'"
            )
        if proposal.compute_capability_id != plan.compute_capability_id:
            mismatches.append(
                f"compute_capability_id: proposal='{proposal.compute_capability_id}' "
                f"vs plan='{plan.compute_capability_id}'"
            )

        if mismatches:
            raise ProposalMergerError(
                f"Artifact linkage mismatch: {'; '.join(mismatches)}"
            )

    # ── Preprocessing Proposals ────────────────────────────────────────

    def _apply_preprocessing_proposals(
        self,
        candidate: dict,
        proposals: list[AIPreprocessingProposal],
    ) -> None:
        """Apply preprocessing proposals to the candidate data dict."""
        steps: list[dict] = candidate.get("preprocessing_steps", [])

        for proposal in proposals:
            if proposal.action == ProposalAction.ADD:
                new_step = {
                    "step_id": f"ai_{proposal.proposal_id}",
                    "operation": proposal.operation.value,
                    "columns": proposal.columns,
                    "parameters": proposal.parameters,
                    "reason": f"[AI] {proposal.reason}",
                }
                steps.append(new_step)
            elif proposal.action == ProposalAction.REMOVE:
                match_idx = self._find_step_by_operation_and_columns(
                    steps, proposal.operation.value, proposal.columns
                )
                if match_idx is None:
                    raise ProposalMergerError(
                        f"Cannot REMOVE preprocessing step: no matching step found "
                        f"for operation='{proposal.operation.value}' "
                        f"columns={proposal.columns}"
                    )
                steps.pop(match_idx)
            elif proposal.action == ProposalAction.REPLACE:
                match_idx = self._find_step_by_operation_and_columns(
                    steps, proposal.operation.value, proposal.columns
                )
                if match_idx is None:
                    raise ProposalMergerError(
                        f"Cannot REPLACE preprocessing step: no matching step found "
                        f"for operation='{proposal.operation.value}' "
                        f"columns={proposal.columns}"
                    )
                steps[match_idx] = {
                    "step_id": f"ai_{proposal.proposal_id}",
                    "operation": proposal.operation.value,
                    "columns": proposal.columns,
                    "parameters": proposal.parameters,
                    "reason": f"[AI] {proposal.reason}",
                }

        candidate["preprocessing_steps"] = steps

    def _find_step_by_operation_and_columns(
        self, steps: list[dict], operation: str, columns: list[str]
    ) -> int | None:
        """Find the first step matching operation and columns exactly."""
        for idx, step in enumerate(steps):
            if step.get("operation") == operation:
                if sorted(step.get("columns", [])) == sorted(columns):
                    return idx
        return None

    # ── Feature Engineering Proposals ──────────────────────────────────

    def _apply_feature_engineering_proposals(
        self,
        candidate: dict,
        proposals: list[AIFeatureEngineeringProposal],
    ) -> None:
        """Apply feature engineering proposals to the candidate data dict."""
        steps: list[dict] = candidate.get("feature_engineering_steps", [])

        for proposal in proposals:
            if proposal.action == ProposalAction.ADD:
                new_step = {
                    "step_id": f"ai_{proposal.proposal_id}",
                    "operation": proposal.operation.value,
                    "input_columns": proposal.input_columns,
                    "output_columns": proposal.output_columns,
                    "parameters": proposal.parameters,
                    "reason": f"[AI] {proposal.reason}",
                }
                steps.append(new_step)
            elif proposal.action == ProposalAction.REMOVE:
                match_idx = self._find_fe_step_by_operation_and_io(
                    steps, proposal.operation.value,
                    proposal.input_columns, proposal.output_columns,
                )
                if match_idx is None:
                    raise ProposalMergerError(
                        f"Cannot REMOVE feature engineering step: no matching step "
                        f"for operation='{proposal.operation.value}'"
                    )
                steps.pop(match_idx)
            elif proposal.action == ProposalAction.REPLACE:
                match_idx = self._find_fe_step_by_operation_and_io(
                    steps, proposal.operation.value,
                    proposal.input_columns, proposal.output_columns,
                )
                if match_idx is None:
                    raise ProposalMergerError(
                        f"Cannot REPLACE feature engineering step: no matching step "
                        f"for operation='{proposal.operation.value}'"
                    )
                steps[match_idx] = {
                    "step_id": f"ai_{proposal.proposal_id}",
                    "operation": proposal.operation.value,
                    "input_columns": proposal.input_columns,
                    "output_columns": proposal.output_columns,
                    "parameters": proposal.parameters,
                    "reason": f"[AI] {proposal.reason}",
                }

        candidate["feature_engineering_steps"] = steps

    def _find_fe_step_by_operation_and_io(
        self, steps: list[dict], operation: str,
        input_columns: list[str], output_columns: list[str],
    ) -> int | None:
        """Find feature engineering step by operation, input, and output columns."""
        for idx, step in enumerate(steps):
            if step.get("operation") == operation:
                if (sorted(step.get("input_columns", [])) == sorted(input_columns)
                        and sorted(step.get("output_columns", [])) == sorted(output_columns)):
                    return idx
        return None

    # ── Model Candidate Proposals ──────────────────────────────────────

    def _apply_model_candidate_proposals(
        self,
        candidate: dict,
        proposals: list[AIModelCandidateProposal],
    ) -> None:
        """Apply model candidate proposals to the candidate data dict."""
        candidates: list[dict] = candidate.get("model_candidates", [])

        for proposal in proposals:
            if proposal.action == ProposalAction.ADD:
                new_candidate = {
                    "candidate_id": f"ai_{proposal.proposal_id}",
                    "model_family": proposal.model_family.value,
                    "parameters": proposal.parameters,
                    "search_strategy": proposal.search_strategy.value,
                    "search_space": proposal.search_space,
                    "reason": f"[AI] {proposal.reason}",
                }
                candidates.append(new_candidate)
            elif proposal.action == ProposalAction.REMOVE:
                match_idx = self._find_model_by_family(
                    candidates, proposal.model_family.value
                )
                if match_idx is None:
                    raise ProposalMergerError(
                        f"Cannot REMOVE model candidate: no matching candidate "
                        f"for model_family='{proposal.model_family.value}'"
                    )
                candidates.pop(match_idx)
            elif proposal.action == ProposalAction.REPLACE:
                match_idx = self._find_model_by_family(
                    candidates, proposal.model_family.value
                )
                if match_idx is None:
                    raise ProposalMergerError(
                        f"Cannot REPLACE model candidate: no matching candidate "
                        f"for model_family='{proposal.model_family.value}'"
                    )
                candidates[match_idx] = {
                    "candidate_id": f"ai_{proposal.proposal_id}",
                    "model_family": proposal.model_family.value,
                    "parameters": proposal.parameters,
                    "search_strategy": proposal.search_strategy.value,
                    "search_space": proposal.search_space,
                    "reason": f"[AI] {proposal.reason}",
                }

        candidate["model_candidates"] = candidates

    def _find_model_by_family(
        self, candidates: list[dict], model_family: str
    ) -> int | None:
        """Find first model candidate matching the model family."""
        for idx, cand in enumerate(candidates):
            if cand.get("model_family") == model_family:
                return idx
        return None

    # ── Feature Selection Proposal ─────────────────────────────────────

    def _apply_feature_selection_proposal(
        self,
        candidate: dict,
        proposal: AIFeatureSelectionProposal,
    ) -> None:
        """Apply feature selection proposal."""
        fs = candidate.get("feature_selection", {})
        fs["method"] = proposal.method.value
        if proposal.candidate_columns:
            fs["candidate_columns"] = proposal.candidate_columns
        if proposal.max_features is not None:
            fs["max_features"] = proposal.max_features
        fs["parameters"] = proposal.parameters
        fs["reason"] = f"[AI] {proposal.reason}"
        candidate["feature_selection"] = fs

    # ── Evaluation Proposal ────────────────────────────────────────────

    def _apply_evaluation_proposal(
        self,
        candidate: dict,
        proposal: AIEvaluationProposal,
    ) -> None:
        """Apply evaluation proposal."""
        ep = candidate.get("evaluation_plan", {})
        if proposal.primary_metric is not None:
            ep["primary_metric"] = proposal.primary_metric
        if proposal.secondary_metrics:
            ep["secondary_metrics"] = proposal.secondary_metrics
        if proposal.cross_validation_folds is not None:
            ep["cross_validation_folds"] = proposal.cross_validation_folds
        candidate["evaluation_plan"] = ep
