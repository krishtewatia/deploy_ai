"""Tests for backend.app.problem_definition.schemas.

Covers all 43+ specified test cases for ProblemDefinition and supporting schemas.
"""

import json
import pytest
from pydantic import ValidationError

from backend.app.problem_definition.schemas import (
    ConfirmationItem,
    ProblemDefinition,
    ProblemType,
    ProblemWarning,
    ResolutionStatus,
    TargetSource,
)


def _make_valid_definition_dict(**overrides) -> dict:
    """Return a minimal valid ProblemDefinition dict."""
    base = {
        "definition_id": "def-123",
        "request_id": "req-123",
        "dataset_id": "ds-123",
        "goal": "Predict customer churn",
        "problem_type": ProblemType.CLASSIFICATION,
        "target_column": "churn",
        "target_source": TargetSource.USER,
        "feature_columns": ["age", "tenure"],
        "primary_metric": "f1",
    }

    base.update(overrides)
    return base


class TestProblemDefinitionSchemas:
    """Comprehensive tests for problem_definition enums and schemas."""

    # 1. Minimal valid resolved ProblemDefinition
    def test_minimal_valid_resolved_definition(self):
        defn = ProblemDefinition(**_make_valid_definition_dict())
        assert defn.definition_id == "def-123"
        assert defn.status == ResolutionStatus.RESOLVED
        assert defn.warnings == []
        assert defn.confirmation_items == []

    # 2. Fully populated resolved definition
    def test_fully_populated_resolved_definition(self):
        defn = ProblemDefinition(
            definition_id="def-456",
            request_id="req-456",
            dataset_id="ds-456",
            goal="Estimate home value",
            problem_type=ProblemType.REGRESSION,
            target_column="price",
            target_source=TargetSource.USER,
            feature_columns=["sqft", "bedrooms", "bathrooms"],
            excluded_columns=["address"],
            primary_metric="rmse",
            status=ResolutionStatus.RESOLVED,
            warnings=[
                ProblemWarning(
                    code="MISSING_TARGET_VALUES",
                    message="Target column has missing values",
                    column_name="price",
                )
            ],
            confirmation_items=[],
        )
        assert defn.definition_id == "def-456"
        assert len(defn.warnings) == 1
        assert defn.warnings[0].code == "MISSING_TARGET_VALUES"

    # 3. Classification problem works
    def test_classification_problem_works(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(problem_type=ProblemType.CLASSIFICATION)
        )
        assert defn.problem_type == ProblemType.CLASSIFICATION

    # 4. Regression problem works
    def test_regression_problem_works(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(problem_type=ProblemType.REGRESSION)
        )
        assert defn.problem_type == ProblemType.REGRESSION

    # 5. Target source user works
    def test_target_source_user_works(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(target_source=TargetSource.USER)
        )
        assert defn.target_source == TargetSource.USER

    # 6. Target source inferred works
    def test_target_source_inferred_works(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(target_source=TargetSource.INFERRED)
        )
        assert defn.target_source == TargetSource.INFERRED

    # 7. Default status is resolved
    def test_default_status_is_resolved(self):
        defn = ProblemDefinition(**_make_valid_definition_dict())
        assert defn.status == ResolutionStatus.RESOLVED

    # 8-13. String fields validation checks (empty rejected)
    @pytest.mark.parametrize(
        "field",
        [
            "definition_id",
            "request_id",
            "dataset_id",
            "goal",
            "target_column",
            "primary_metric",
        ],
    )
    def test_empty_string_fields_rejected(self, field):
        kwargs = _make_valid_definition_dict()
        kwargs[field] = ""
        with pytest.raises(ValidationError, match=field):
            ProblemDefinition(**kwargs)

        kwargs[field] = "   "
        with pytest.raises(ValidationError, match=field):
            ProblemDefinition(**kwargs)

    # 14. Empty feature_columns rejected
    def test_empty_feature_columns_rejected(self):
        with pytest.raises(ValidationError, match="feature_columns"):
            ProblemDefinition(**_make_valid_definition_dict(feature_columns=[]))

    # 15. Duplicate feature columns rejected
    def test_duplicate_feature_columns_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate column name found"):
            ProblemDefinition(
                **_make_valid_definition_dict(feature_columns=["age", "tenure", "age"])
            )

    # 16. Invalid empty feature name rejected
    def test_invalid_empty_feature_name_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            ProblemDefinition(
                **_make_valid_definition_dict(feature_columns=["age", "", "tenure"])
            )
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            ProblemDefinition(
                **_make_valid_definition_dict(feature_columns=["age", "   ", "tenure"])
            )

    # 17. Duplicate excluded columns rejected
    def test_duplicate_excluded_columns_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate column name found"):
            ProblemDefinition(
                **_make_valid_definition_dict(
                    excluded_columns=["zip_code", "zip_code"]
                )
            )

    # 18. Invalid empty excluded column rejected
    def test_invalid_empty_excluded_column_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            ProblemDefinition(
                **_make_valid_definition_dict(excluded_columns=[""])
            )
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            ProblemDefinition(
                **_make_valid_definition_dict(excluded_columns=["   "])
            )

    # 19. Target in feature_columns rejected
    def test_target_in_feature_columns_rejected(self):
        with pytest.raises(ValidationError, match="cannot be in feature_columns"):
            ProblemDefinition(
                **_make_valid_definition_dict(
                    target_column="churn", feature_columns=["age", "churn"]
                )
            )

    # 20. Target in excluded_columns rejected
    def test_target_in_excluded_columns_rejected(self):
        with pytest.raises(ValidationError, match="cannot be in excluded_columns"):
            ProblemDefinition(
                **_make_valid_definition_dict(
                    target_column="churn", excluded_columns=["zip", "churn"]
                )
            )

    # 21. Same column in feature_columns and excluded_columns rejected
    def test_overlap_feature_excluded_rejected(self):
        with pytest.raises(
            ValidationError, match="Columns cannot be in both feature_columns and excluded_columns"
        ):
            ProblemDefinition(
                **_make_valid_definition_dict(
                    feature_columns=["age", "tenure"],
                    excluded_columns=["zip", "age"],
                )
            )

    # 22. Feature column order preserved
    def test_feature_column_order_preserved(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(feature_columns=["z_col", "a_col"])
        )
        assert defn.feature_columns == ["z_col", "a_col"]

    # 23. Excluded column order preserved
    def test_excluded_column_order_preserved(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(excluded_columns=["z_col", "a_col"])
        )
        assert defn.excluded_columns == ["z_col", "a_col"]

    # 24. Valid ProblemWarning works
    def test_valid_warning_works(self):
        warning = ProblemWarning(
            code="IMBALANCE", message="Severe class imbalance", column_name="churn"
        )
        assert warning.code == "IMBALANCE"
        assert warning.message == "Severe class imbalance"
        assert warning.column_name == "churn"

    # 25-27. Invalid ProblemWarning fields rejected
    def test_invalid_warning_fields_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ProblemWarning(code="", message="msg")
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ProblemWarning(code="code", message="  ")
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ProblemWarning(code="code", message="msg", column_name="  ")

    # 28. Valid ConfirmationItem works
    def test_valid_confirmation_item_works(self):
        item = ConfirmationItem(
            key="key", question="Use churn?", reason="It was inferred"
        )
        assert item.key == "key"
        assert item.question == "Use churn?"
        assert item.reason == "It was inferred"

    # 29-31. Invalid ConfirmationItem fields rejected
    def test_invalid_confirmation_item_fields_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ConfirmationItem(key="", question="Q?", reason="R")
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ConfirmationItem(key="K", question="   ", reason="R")
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ConfirmationItem(key="K", question="Q?", reason="")

    # 32. resolved + empty confirmation_items works
    def test_resolved_empty_confirmations_works(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(
                status=ResolutionStatus.RESOLVED, confirmation_items=[]
            )
        )
        assert defn.status == ResolutionStatus.RESOLVED
        assert defn.confirmation_items == []

    # 33. resolved + confirmation_items is rejected
    def test_resolved_with_confirmations_rejected(self):
        with pytest.raises(
            ValidationError, match="confirmation_items must be empty when status is 'resolved'"
        ):
            ProblemDefinition(
                **_make_valid_definition_dict(
                    status=ResolutionStatus.RESOLVED,
                    confirmation_items=[
                        ConfirmationItem(key="K", question="Q?", reason="R")
                    ],
                )
            )

    # 34. needs_confirmation + confirmation item works
    def test_needs_confirmation_with_item_works(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(
                status=ResolutionStatus.NEEDS_CONFIRMATION,
                confirmation_items=[
                    ConfirmationItem(key="K", question="Q?", reason="R")
                ],
            )
        )
        assert defn.status == ResolutionStatus.NEEDS_CONFIRMATION
        assert len(defn.confirmation_items) == 1

    # 35. needs_confirmation + empty confirmation_items is rejected
    def test_needs_confirmation_empty_confirmations_rejected(self):
        with pytest.raises(
            ValidationError,
            match="confirmation_items must contain at least one item when status is 'needs_confirmation'",
        ):
            ProblemDefinition(
                **_make_valid_definition_dict(
                    status=ResolutionStatus.NEEDS_CONFIRMATION,
                    confirmation_items=[],
                )
            )

    # 36. blocked status works without confirmation items
    def test_blocked_status_works_without_confirmations(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(
                status=ResolutionStatus.BLOCKED,
                confirmation_items=[],
                warnings=[
                    ProblemWarning(code="BLOCKED_ERR", message="Severe error")
                ],
            )
        )
        assert defn.status == ResolutionStatus.BLOCKED
        assert defn.confirmation_items == []
        assert len(defn.warnings) == 1

    # 37-38. warnings and confirmation_items use independent list instances
    def test_independent_list_instances(self):
        defn1 = ProblemDefinition(**_make_valid_definition_dict())
        defn2 = ProblemDefinition(**_make_valid_definition_dict())
        defn1.warnings.append(ProblemWarning(code="W", message="M"))
        assert len(defn2.warnings) == 0

        defn3 = ProblemDefinition(
            **_make_valid_definition_dict(
                status=ResolutionStatus.NEEDS_CONFIRMATION,
                confirmation_items=[
                    ConfirmationItem(key="K1", question="Q?", reason="R")
                ],
            )
        )
        defn4 = ProblemDefinition(
            **_make_valid_definition_dict(
                status=ResolutionStatus.NEEDS_CONFIRMATION,
                confirmation_items=[
                    ConfirmationItem(key="K2", question="Q?", reason="R")
                ],
            )
        )
        assert defn3.confirmation_items[0].key == "K1"
        assert defn4.confirmation_items[0].key == "K2"

    # 39. model_dump() works
    def test_model_dump(self):
        defn = ProblemDefinition(**_make_valid_definition_dict())
        dumped = defn.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["definition_id"] == "def-123"
        assert dumped["status"] == "resolved"

    # 40. model_dump_json() produces valid JSON
    def test_model_dump_json(self):
        defn = ProblemDefinition(**_make_valid_definition_dict())
        json_str = defn.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["definition_id"] == "def-123"

    # 41. Enum values serialize as strings
    def test_enums_serialize_as_strings(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(
                problem_type=ProblemType.REGRESSION,
                target_source=TargetSource.INFERRED,
                status=ResolutionStatus.RESOLVED,
            )
        )
        json_str = defn.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["problem_type"] == "regression"
        assert parsed["target_source"] == "inferred"
        assert parsed["status"] == "resolved"

    # 42-43. Nested warnings and confirmations serialize correctly
    def test_nested_elements_serialization(self):
        defn = ProblemDefinition(
            **_make_valid_definition_dict(
                status=ResolutionStatus.NEEDS_CONFIRMATION,
                warnings=[ProblemWarning(code="WARN_CODE", message="Warn msg")],
                confirmation_items=[
                    ConfirmationItem(key="key_1", question="Q?", reason="R")
                ],
            )
        )
        json_str = defn.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["warnings"][0]["code"] == "WARN_CODE"
        assert parsed["confirmation_items"][0]["key"] == "key_1"

    # Additional coverage tests for validator type checking
    def test_invalid_types_rejected(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            ProblemDefinition(**_make_valid_definition_dict(definition_id=123))

        with pytest.raises(ValidationError, match="feature_columns must be a list"):
            ProblemDefinition(
                **_make_valid_definition_dict(feature_columns="not-a-list")
            )

        with pytest.raises(ValidationError, match="All column names must be strings"):
            ProblemDefinition(
                **_make_valid_definition_dict(feature_columns=[123])
            )

        with pytest.raises(ValidationError, match="excluded_columns must be a list"):
            ProblemDefinition(
                **_make_valid_definition_dict(excluded_columns="not-a-list")
            )

        with pytest.raises(ValidationError, match="All column names must be strings"):
            ProblemDefinition(
                **_make_valid_definition_dict(excluded_columns=[123])
            )

        with pytest.raises(ValidationError, match="Field must be a string"):
            ProblemWarning(code=12, message="msg")

        with pytest.raises(ValidationError, match="Field must be a string"):
            ConfirmationItem(key="K", question=True, reason="R")

    def test_explicit_none_values_accepted(self):
        warning = ProblemWarning(code="CODE", message="message", column_name=None)
        assert warning.column_name is None

