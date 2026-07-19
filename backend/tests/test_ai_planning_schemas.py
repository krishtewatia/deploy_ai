"""Tests for AI planning decision schemas (Stage 7D1)."""

import json
import pytest

from backend.app.ai_planning.schemas import (
    AIDecisionConfidence,
    AIDecisionProposal,
    AIDecisionWarning,
    AIEvaluationProposal,
    AIFeatureEngineeringProposal,
    AIFeatureSelectionProposal,
    AIModelCandidateProposal,
    AIPreprocessingProposal,
    ProposalAction,
)
from backend.app.ml_plan.schemas import (
    FeatureEngineeringOperation,
    FeatureSelectionMethod,
    ModelFamily,
    PreprocessingOperation,
    SearchStrategy,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _minimal_proposal_kwargs() -> dict:
    """Kwargs for a valid empty AIDecisionProposal."""
    return dict(
        proposal_set_id="ps_01",
        baseline_plan_id="bp_01",
        dataset_id="ds_01",
        request_id="req_01",
        problem_definition_id="pd_01",
        compute_capability_id="cap_01",
        summary="No changes proposed.",
    )


# ── Enum Tests ─────────────────────────────────────────────────────────


class TestEnums:
    def test_confidence_values(self):
        assert AIDecisionConfidence.LOW.value == "low"
        assert AIDecisionConfidence.MEDIUM.value == "medium"
        assert AIDecisionConfidence.HIGH.value == "high"

    def test_proposal_action_values(self):
        assert ProposalAction.ADD.value == "add"
        assert ProposalAction.REMOVE.value == "remove"
        assert ProposalAction.REPLACE.value == "replace"


# ── AIPreprocessingProposal Tests ──────────────────────────────────────


class TestAIPreprocessingProposal:
    def test_valid_creation(self):
        p = AIPreprocessingProposal(
            proposal_id="prep_01",
            action=ProposalAction.ADD,
            operation=PreprocessingOperation.IMPUTE_MEDIAN,
            columns=["age"],
            parameters={},
            reason="Impute missing age values.",
            confidence=AIDecisionConfidence.HIGH,
        )
        assert p.proposal_id == "prep_01"
        assert p.action == ProposalAction.ADD
        assert p.operation == PreprocessingOperation.IMPUTE_MEDIAN

    def test_empty_proposal_id_rejected(self):
        with pytest.raises(Exception):
            AIPreprocessingProposal(
                proposal_id="  ",
                action=ProposalAction.ADD,
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["age"],
                reason="Test.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_empty_reason_rejected(self):
        with pytest.raises(Exception):
            AIPreprocessingProposal(
                proposal_id="p1",
                action=ProposalAction.ADD,
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["age"],
                reason="  ",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_non_serializable_parameters_rejected(self):
        with pytest.raises(Exception):
            AIPreprocessingProposal(
                proposal_id="p1",
                action=ProposalAction.ADD,
                operation=PreprocessingOperation.IMPUTE_MEDIAN,
                columns=["age"],
                parameters={"bad": object()},
                reason="Test.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_empty_columns_rejected(self):
        with pytest.raises(Exception, match="at least one column"):
            AIPreprocessingProposal(
                proposal_id="p1",
                action=ProposalAction.REMOVE,
                operation=PreprocessingOperation.DROP_COLUMN,
                columns=[],
                reason="Remove step.",
                confidence=AIDecisionConfidence.MEDIUM,
            )

    def test_serialization(self):
        p = AIPreprocessingProposal(
            proposal_id="prep_01",
            action=ProposalAction.ADD,
            operation=PreprocessingOperation.STANDARD_SCALE,
            columns=["salary"],
            parameters={"with_mean": True},
            reason="Scale numeric feature.",
            confidence=AIDecisionConfidence.MEDIUM,
        )
        d = p.model_dump()
        assert isinstance(d, dict)
        j = p.model_dump_json()
        assert isinstance(json.loads(j), dict)


# ── AIFeatureEngineeringProposal Tests ─────────────────────────────────


class TestAIFeatureEngineeringProposal:
    def test_valid_creation(self):
        p = AIFeatureEngineeringProposal(
            proposal_id="fe_01",
            action=ProposalAction.ADD,
            operation=FeatureEngineeringOperation.RATIO,
            input_columns=["salary", "age"],
            output_columns=["salary_age_ratio"],
            parameters={},
            reason="Create ratio feature.",
            confidence=AIDecisionConfidence.HIGH,
        )
        assert p.operation == FeatureEngineeringOperation.RATIO
        assert p.input_columns == ["salary", "age"]

    def test_empty_id_rejected(self):
        with pytest.raises(Exception):
            AIFeatureEngineeringProposal(
                proposal_id="",
                action=ProposalAction.ADD,
                operation=FeatureEngineeringOperation.RATIO,
                input_columns=["a"],
                output_columns=["b"],
                reason="Test.",
                confidence=AIDecisionConfidence.LOW,
            )

    @pytest.mark.parametrize(
        ("input_columns", "output_columns"),
        [([], ["ratio"]), (["income"], [])],
    )
    def test_empty_feature_columns_rejected(self, input_columns, output_columns):
        with pytest.raises(Exception, match="at least one column"):
            AIFeatureEngineeringProposal(
                proposal_id="fe_empty",
                action=ProposalAction.ADD,
                operation=FeatureEngineeringOperation.RATIO,
                input_columns=input_columns,
                output_columns=output_columns,
                reason="Incomplete feature proposal.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_serialization(self):
        p = AIFeatureEngineeringProposal(
            proposal_id="fe_01",
            action=ProposalAction.ADD,
            operation=FeatureEngineeringOperation.POLYNOMIAL,
            input_columns=["x"],
            output_columns=["x_poly"],
            reason="Polynomial feature.",
            confidence=AIDecisionConfidence.MEDIUM,
        )
        d = p.model_dump()
        assert d["operation"] == "polynomial"


# ── AIModelCandidateProposal Tests ─────────────────────────────────────


class TestAIModelCandidateProposal:
    def test_valid_creation(self):
        p = AIModelCandidateProposal(
            proposal_id="mc_01",
            action=ProposalAction.ADD,
            model_family=ModelFamily.KNN,
            parameters={"n_neighbors": 5},
            search_strategy=SearchStrategy.NONE,
            search_space={},
            reason="Add KNN for comparison.",
            confidence=AIDecisionConfidence.MEDIUM,
        )
        assert p.model_family == ModelFamily.KNN
        assert p.search_strategy == SearchStrategy.NONE

    def test_with_search_strategy(self):
        p = AIModelCandidateProposal(
            proposal_id="mc_02",
            action=ProposalAction.ADD,
            model_family=ModelFamily.SVM,
            parameters={},
            search_strategy=SearchStrategy.GRID,
            search_space={"C": [0.1, 1.0, 10.0]},
            reason="Add SVM with grid search.",
            confidence=AIDecisionConfidence.LOW,
        )
        assert p.search_strategy == SearchStrategy.GRID

    def test_non_serializable_search_space_rejected(self):
        with pytest.raises(Exception):
            AIModelCandidateProposal(
                proposal_id="mc_03",
                action=ProposalAction.ADD,
                model_family=ModelFamily.KNN,
                search_space={"bad": object()},
                reason="Bad.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_serialization(self):
        p = AIModelCandidateProposal(
            proposal_id="mc_01",
            action=ProposalAction.REPLACE,
            model_family=ModelFamily.EXTRA_TREES,
            reason="Replace with extra trees.",
            confidence=AIDecisionConfidence.HIGH,
        )
        j = json.loads(p.model_dump_json())
        assert j["model_family"] == "extra_trees"


# ── AIFeatureSelectionProposal Tests ───────────────────────────────────


class TestAIFeatureSelectionProposal:
    def test_valid_creation(self):
        p = AIFeatureSelectionProposal(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            candidate_columns=["age", "salary"],
            max_features=2,
            parameters={},
            reason="Select best features.",
            confidence=AIDecisionConfidence.HIGH,
        )
        assert p.method == FeatureSelectionMethod.MUTUAL_INFORMATION

    def test_empty_candidates_allowed(self):
        p = AIFeatureSelectionProposal(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=[],
            reason="No selection needed.",
            confidence=AIDecisionConfidence.HIGH,
        )
        assert p.candidate_columns == []

    def test_max_features_invalid(self):
        with pytest.raises(Exception):
            AIFeatureSelectionProposal(
                method=FeatureSelectionMethod.MUTUAL_INFORMATION,
                candidate_columns=["a"],
                max_features=0,
                reason="Bad.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_duplicate_candidates_rejected(self):
        with pytest.raises(Exception):
            AIFeatureSelectionProposal(
                method=FeatureSelectionMethod.MUTUAL_INFORMATION,
                candidate_columns=["age", "age"],
                reason="Bad.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_serialization(self):
        p = AIFeatureSelectionProposal(
            method=FeatureSelectionMethod.VARIANCE_THRESHOLD,
            candidate_columns=["a", "b"],
            reason="Variance filter.",
            confidence=AIDecisionConfidence.MEDIUM,
        )
        d = p.model_dump()
        assert d["method"] == "variance_threshold"


# ── AIEvaluationProposal Tests ─────────────────────────────────────────


class TestAIEvaluationProposal:
    def test_valid_creation(self):
        p = AIEvaluationProposal(
            primary_metric="roc_auc",
            secondary_metrics=["f1", "precision"],
            cross_validation_folds=10,
            reason="Prefer AUC for imbalanced data.",
            confidence=AIDecisionConfidence.HIGH,
        )
        assert p.primary_metric == "roc_auc"
        assert p.cross_validation_folds == 10

    def test_no_change_proposal(self):
        p = AIEvaluationProposal(
            reason="No evaluation changes needed.",
            confidence=AIDecisionConfidence.HIGH,
        )
        assert p.primary_metric is None
        assert p.secondary_metrics == []
        assert p.cross_validation_folds is None

    def test_invalid_cv_folds(self):
        with pytest.raises(Exception):
            AIEvaluationProposal(
                cross_validation_folds=1,
                reason="Bad.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_empty_primary_metric_rejected(self):
        with pytest.raises(Exception):
            AIEvaluationProposal(
                primary_metric="  ",
                reason="Bad.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_serialization(self):
        p = AIEvaluationProposal(
            primary_metric="accuracy",
            reason="Change metric.",
            confidence=AIDecisionConfidence.MEDIUM,
        )
        j = json.loads(p.model_dump_json())
        assert j["primary_metric"] == "accuracy"


# ── AIDecisionWarning Tests ────────────────────────────────────────────


class TestAIDecisionWarning:
    def test_valid_creation(self):
        w = AIDecisionWarning(code="AI_LOW_CONFIDENCE", message="Some proposals have low confidence.")
        assert w.code == "AI_LOW_CONFIDENCE"

    def test_empty_code_rejected(self):
        with pytest.raises(Exception):
            AIDecisionWarning(code="  ", message="msg")

    def test_empty_message_rejected(self):
        with pytest.raises(Exception):
            AIDecisionWarning(code="CODE", message="  ")


# ── AIDecisionProposal Tests ──────────────────────────────────────────


class TestAIDecisionProposal:
    def test_empty_no_change_proposal_valid(self):
        p = AIDecisionProposal(**_minimal_proposal_kwargs())
        assert p.preprocessing_proposals == []
        assert p.feature_engineering_proposals == []
        assert p.model_candidate_proposals == []
        assert p.feature_selection_proposal is None
        assert p.evaluation_proposal is None
        assert p.warnings == []

    def test_full_proposal(self):
        p = AIDecisionProposal(
            **_minimal_proposal_kwargs(),
            preprocessing_proposals=[
                AIPreprocessingProposal(
                    proposal_id="prep_01",
                    action=ProposalAction.ADD,
                    operation=PreprocessingOperation.ROBUST_SCALE,
                    columns=["salary"],
                    reason="Robust scaling.",
                    confidence=AIDecisionConfidence.HIGH,
                ),
            ],
            feature_engineering_proposals=[
                AIFeatureEngineeringProposal(
                    proposal_id="fe_01",
                    action=ProposalAction.ADD,
                    operation=FeatureEngineeringOperation.LOG_TRANSFORM,
                    input_columns=["salary"],
                    output_columns=["log_salary"],
                    reason="Log transform.",
                    confidence=AIDecisionConfidence.MEDIUM,
                ),
            ],
            model_candidate_proposals=[
                AIModelCandidateProposal(
                    proposal_id="mc_01",
                    action=ProposalAction.ADD,
                    model_family=ModelFamily.DECISION_TREE,
                    reason="Add decision tree.",
                    confidence=AIDecisionConfidence.LOW,
                ),
            ],
            feature_selection_proposal=AIFeatureSelectionProposal(
                method=FeatureSelectionMethod.CORRELATION_FILTER,
                candidate_columns=["salary", "age"],
                reason="Filter correlated features.",
                confidence=AIDecisionConfidence.MEDIUM,
            ),
            evaluation_proposal=AIEvaluationProposal(
                primary_metric="roc_auc",
                reason="Switch to AUC.",
                confidence=AIDecisionConfidence.HIGH,
            ),
            warnings=[
                AIDecisionWarning(code="AI_NOTE", message="Some notes."),
            ],
        )
        assert len(p.preprocessing_proposals) == 1
        assert len(p.feature_engineering_proposals) == 1
        assert len(p.model_candidate_proposals) == 1
        assert p.feature_selection_proposal is not None
        assert p.evaluation_proposal is not None
        assert len(p.warnings) == 1

    def test_empty_ids_rejected(self):
        for field in ["proposal_set_id", "baseline_plan_id", "dataset_id",
                       "request_id", "problem_definition_id", "compute_capability_id", "summary"]:
            kwargs = _minimal_proposal_kwargs()
            kwargs[field] = "  "
            with pytest.raises(Exception):
                AIDecisionProposal(**kwargs)

    def test_duplicate_proposal_ids_across_categories_rejected(self):
        with pytest.raises(Exception, match="Duplicate proposal_id"):
            AIDecisionProposal(
                **_minimal_proposal_kwargs(),
                preprocessing_proposals=[
                    AIPreprocessingProposal(
                        proposal_id="dup_01",
                        action=ProposalAction.ADD,
                        operation=PreprocessingOperation.IMPUTE_MEAN,
                        columns=["x"],
                        reason="R.",
                        confidence=AIDecisionConfidence.LOW,
                    ),
                ],
                model_candidate_proposals=[
                    AIModelCandidateProposal(
                        proposal_id="dup_01",
                        action=ProposalAction.ADD,
                        model_family=ModelFamily.KNN,
                        reason="R.",
                        confidence=AIDecisionConfidence.LOW,
                    ),
                ],
            )

    def test_duplicate_proposal_ids_within_same_category_rejected(self):
        with pytest.raises(Exception, match="Duplicate proposal_id"):
            AIDecisionProposal(
                **_minimal_proposal_kwargs(),
                preprocessing_proposals=[
                    AIPreprocessingProposal(
                        proposal_id="dup_01",
                        action=ProposalAction.ADD,
                        operation=PreprocessingOperation.IMPUTE_MEAN,
                        columns=["x"],
                        reason="R.",
                        confidence=AIDecisionConfidence.LOW,
                    ),
                    AIPreprocessingProposal(
                        proposal_id="dup_01",
                        action=ProposalAction.ADD,
                        operation=PreprocessingOperation.IMPUTE_MEDIAN,
                        columns=["y"],
                        reason="R.",
                        confidence=AIDecisionConfidence.LOW,
                    ),
                ],
            )

    def test_unique_proposal_ids_across_categories_accepted(self):
        p = AIDecisionProposal(
            **_minimal_proposal_kwargs(),
            preprocessing_proposals=[
                AIPreprocessingProposal(
                    proposal_id="prep_01",
                    action=ProposalAction.ADD,
                    operation=PreprocessingOperation.IMPUTE_MEAN,
                    columns=["x"],
                    reason="R.",
                    confidence=AIDecisionConfidence.LOW,
                ),
            ],
            feature_engineering_proposals=[
                AIFeatureEngineeringProposal(
                    proposal_id="fe_01",
                    action=ProposalAction.ADD,
                    operation=FeatureEngineeringOperation.RATIO,
                    input_columns=["a"],
                    output_columns=["b"],
                    reason="R.",
                    confidence=AIDecisionConfidence.LOW,
                ),
            ],
            model_candidate_proposals=[
                AIModelCandidateProposal(
                    proposal_id="mc_01",
                    action=ProposalAction.ADD,
                    model_family=ModelFamily.KNN,
                    reason="R.",
                    confidence=AIDecisionConfidence.LOW,
                ),
            ],
        )
        assert len(p.preprocessing_proposals) == 1

    def test_json_serialization(self):
        p = AIDecisionProposal(**_minimal_proposal_kwargs())
        j = p.model_dump_json()
        parsed = json.loads(j)
        assert parsed["proposal_set_id"] == "ps_01"
        assert parsed["summary"] == "No changes proposed."

    def test_model_dump(self):
        p = AIDecisionProposal(**_minimal_proposal_kwargs())
        d = p.model_dump()
        assert isinstance(d, dict)
        assert d["baseline_plan_id"] == "bp_01"

    def test_json_schema_generation(self):
        schema = AIDecisionProposal.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema


# ── Helper Validator Edge Cases ────────────────────────────────────────


class TestHelperValidatorEdgeCases:
    """Tests to exercise missed branches in helper validators."""

    def test_non_string_proposal_id(self):
        """Line 62: _strip_nonempty receives non-string."""
        with pytest.raises(Exception):
            AIPreprocessingProposal(
                proposal_id=123,
                action=ProposalAction.ADD,
                operation=PreprocessingOperation.IMPUTE_MEAN,
                columns=[],
                reason="Test.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_feature_selection_non_list_candidates(self):
        """Line 72: _validate_columns_list receives non-list."""
        with pytest.raises(Exception):
            AIFeatureSelectionProposal(
                method=FeatureSelectionMethod.MUTUAL_INFORMATION,
                candidate_columns="not_a_list",
                reason="Test.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_feature_selection_non_string_item(self):
        """Line 77: _validate_columns_list item is not a string."""
        with pytest.raises(Exception):
            AIFeatureSelectionProposal(
                method=FeatureSelectionMethod.MUTUAL_INFORMATION,
                candidate_columns=[123],
                reason="Test.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_feature_selection_empty_string_item(self):
        """Line 80: _validate_columns_list item is empty string."""
        with pytest.raises(Exception):
            AIFeatureSelectionProposal(
                method=FeatureSelectionMethod.MUTUAL_INFORMATION,
                candidate_columns=["  "],
                reason="Test.",
                confidence=AIDecisionConfidence.LOW,
            )

    def test_evaluation_proposal_none_primary_metric(self):
        """Line 325: primary_metric None passthrough."""
        p = AIEvaluationProposal(
            primary_metric=None,
            reason="No metric change.",
            confidence=AIDecisionConfidence.HIGH,
        )
        assert p.primary_metric is None

