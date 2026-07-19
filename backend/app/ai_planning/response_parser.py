"""AI Response Parser.

Converts raw AI provider text output into a validated AIDecisionProposal.
Handles common edge cases like markdown fences around JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from backend.app.ai_planning.schemas import AIDecisionProposal


class AIResponseParseError(Exception):
    """Raised when the AI response cannot be parsed into a valid proposal."""


class AIResponseParser:
    """Parses and validates raw AI text responses into AIDecisionProposal.

    Security constraints:
    - No eval() or exec()
    - No dynamic imports
    - No execution of returned content
    - No arbitrary Python repair
    """

    def parse(self, raw_response: str) -> AIDecisionProposal:
        """Parse the raw AI response into a validated AIDecisionProposal.

        Args:
            raw_response: Raw text output from an AI provider.

        Returns:
            A validated AIDecisionProposal instance.

        Raises:
            AIResponseParseError: If the response cannot be parsed.
        """
        if not raw_response or not raw_response.strip():
            raise AIResponseParseError("AI response is empty.")

        cleaned = self._strip_markdown_fences(raw_response.strip())

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Fallback: find first '{' and last '}' to extract raw JSON block
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = cleaned[start:end+1]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as exc2:
                    raise AIResponseParseError(
                        f"AI response is not valid JSON block: {exc2}. Raw: {cleaned[:300]}"
                    ) from exc2
            else:
                raise AIResponseParseError(
                    f"AI response is not valid JSON and contains no JSON block: {exc}. Raw: {cleaned[:300]}"
                ) from exc

        data = self._sanitize_data(data)

        try:
            proposal = AIDecisionProposal.model_validate(data)
        except ValidationError as exc:
            raise AIResponseParseError(
                f"AI response does not match AIDecisionProposal schema: {exc}"
            ) from exc

        return proposal

    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize JSON dictionary elements before Pydantic validation."""
        if not isinstance(data, dict):
            return data

        valid_confidences = {"low", "medium", "high"}

        # Handle misplaced model candidates placed inside feature_engineering_proposals
        fe_list = data.get("feature_engineering_proposals")
        mc_list = data.get("model_candidate_proposals")
        if not isinstance(mc_list, list) and "model_candidate_proposals" in data:
            mc_list = []
            data["model_candidate_proposals"] = mc_list

        valid_fe_ops = {
            "interaction", "polynomial", "ratio", "difference", "datetime_parts", "log_transform", "custom"
        }
        valid_prep_ops = {
            "drop_column", "impute_mean", "impute_median", "impute_mode", "impute_constant",
            "one_hot_encode", "ordinal_encode", "standard_scale", "minmax_scale", "robust_scale",
            "passthrough"
        }

        # Preprocessing proposals
        if isinstance(data.get("preprocessing_proposals"), list):
            clean_preprocessing = []
            for idx, p in enumerate(data["preprocessing_proposals"]):
                if isinstance(p, dict):
                    # A concrete column list is required to create a valid MLPlan
                    # step.  Ignore incomplete model suggestions rather than
                    # allowing them to terminate an otherwise valid training run.
                    if not self._has_named_columns(p.get("columns")):
                        continue
                    if not p.get("proposal_id") or not str(p.get("proposal_id")).strip():
                        p["proposal_id"] = f"prep_{idx+1}"
                    if isinstance(p.get("action"), str) and "|" in p["action"]:
                        p["action"] = "add"
                    if isinstance(p.get("confidence"), str) and "|" in p["confidence"]:
                        p["confidence"] = "medium"
                    if p.get("operation") not in valid_prep_ops and isinstance(p.get("operation"), str) and "|" in p["operation"]:
                        p["operation"] = "standard_scale"
                    if p.get("operation") not in valid_prep_ops:
                        continue
                    if not p.get("reason"):
                        p["reason"] = "AI preprocessing adjustment"
                    clean_preprocessing.append(p)
            data["preprocessing_proposals"] = clean_preprocessing

        # Feature engineering proposals
        if isinstance(fe_list, list):
            clean_fe = []
            for idx, p in enumerate(fe_list):
                if isinstance(p, dict):
                    if "model_family" in p:
                        if isinstance(mc_list, list):
                            mc_list.append(p)
                    else:
                        # Feature engineering requires named inputs and outputs;
                        # an empty list cannot be executed by the training engine.
                        if not (
                            self._has_named_columns(p.get("input_columns"))
                            and self._has_named_columns(p.get("output_columns"))
                        ):
                            continue
                        if not p.get("proposal_id") or not str(p.get("proposal_id")).strip():
                            p["proposal_id"] = f"fe_{idx+1}"
                        if isinstance(p.get("action"), str) and "|" in p["action"]:
                            p["action"] = "add"
                        if isinstance(p.get("confidence"), str) and "|" in p["confidence"]:
                            p["confidence"] = "medium"
                        if p.get("operation") not in valid_fe_ops and isinstance(p.get("operation"), str) and "|" in p["operation"]:
                            p["operation"] = "interaction"
                        if not p.get("reason"):
                            p["reason"] = "AI feature engineering adjustment"
                        clean_fe.append(p)
            data["feature_engineering_proposals"] = clean_fe

        # Model candidate proposals
        valid_model_families = {
            "linear_regression", "logistic_regression", "ridge", "lasso",
            "decision_tree", "random_forest", "gradient_boosting", "extra_trees", "knn", "svm"
        }
        if isinstance(mc_list, list):
            for idx, p in enumerate(mc_list):
                if isinstance(p, dict):
                    if not p.get("proposal_id") or not str(p.get("proposal_id")).strip():
                        p["proposal_id"] = f"mc_{idx+1}"
                    if isinstance(p.get("action"), str) and "|" in p["action"]:
                        p["action"] = "add"
                    if isinstance(p.get("confidence"), str) and "|" in p["confidence"]:
                        p["confidence"] = "medium"
                    if p.get("model_family") not in valid_model_families and isinstance(p.get("model_family"), str) and "|" in p["model_family"]:
                        p["model_family"] = "random_forest"
                    if not p.get("reason"):
                        p["reason"] = "AI model candidate addition"

        # Feature selection proposal
        valid_fs_methods = {"none", "variance_threshold", "correlation_filter", "mutual_information", "model_based"}
        fs = data.get("feature_selection_proposal")
        if isinstance(fs, dict):
            # This section is optional.  If the provider emits only a rationale,
            # make it an explicit no-op instead of failing the entire training
            # pipeline on missing optional fields.
            if fs.get("method") not in valid_fs_methods:
                fs["method"] = "none"
            if fs.get("confidence") not in valid_confidences:
                fs["confidence"] = "medium"
            if not fs.get("reason"):
                fs["reason"] = "Feature selection evaluation"

        # Evaluation advice is optional for the final plan.  Keep a partial
        # provider response valid by supplying the required confidence/reason.
        evaluation = data.get("evaluation_proposal")
        if isinstance(evaluation, dict):
            if evaluation.get("confidence") not in valid_confidences:
                evaluation["confidence"] = "medium"
            if not evaluation.get("reason"):
                evaluation["reason"] = "Evaluation plan review"

        return data

    @staticmethod
    def _has_named_columns(value: Any) -> bool:
        """Return whether a JSON value is a non-empty list of named columns."""
        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(column, str) and column.strip() for column in value)
        )

    def _strip_markdown_fences(self, text: str) -> str:
        """Strip common markdown JSON fences from the response.

        Handles patterns like:
        - ```json\\n...\\n```
        - ```\\n...\\n```
        """
        # Match ```json ... ``` or ``` ... ```
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
        match = re.match(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
