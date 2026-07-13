"""Tests for backend.app.ml_request.schemas.

Covers all 30 specified testing requirements for enums and UserMLRequest.
"""

import json
import pytest
from pydantic import ValidationError

from backend.app.ml_request.schemas import (
    AutomationLevel,
    ComputePreference,
    ProblemTypePreference,
    UserMLRequest,
)


def _make_valid_request_dict(**overrides) -> dict:
    """Return a minimal valid UserMLRequest dictionary."""
    base = {
        "request_id": "req-123",
        "goal": "Predict customer churn",
    }
    base.update(overrides)
    return base


class TestUserMLRequestSchemas:
    """Comprehensive tests covering UserMLRequest schemas and enums."""

    # 1. Minimal valid request
    def test_minimal_valid_request(self):
        req = UserMLRequest(**_make_valid_request_dict())
        assert req.request_id == "req-123"
        assert req.goal == "Predict customer churn"

    # 2. Fully populated request
    def test_fully_populated_request(self):
        req = UserMLRequest(
            request_id="req-456",
            goal="Predict housing prices",
            target_column="price",
            problem_type=ProblemTypePreference.REGRESSION,
            primary_metric="rmse",
            automation_level=AutomationLevel.GUIDED,
            compute_preference=ComputePreference.THOROUGH,
            excluded_columns=["id", "zip_code"],
            additional_context="We need regression with 10% tolerance",
        )
        assert req.request_id == "req-456"
        assert req.goal == "Predict housing prices"
        assert req.target_column == "price"
        assert req.problem_type == ProblemTypePreference.REGRESSION
        assert req.primary_metric == "rmse"
        assert req.automation_level == AutomationLevel.GUIDED
        assert req.compute_preference == ComputePreference.THOROUGH
        assert req.excluded_columns == ["id", "zip_code"]
        assert req.additional_context == "We need regression with 10% tolerance"

    # 3. Default problem_type is auto
    def test_default_problem_type_is_auto(self):
        req = UserMLRequest(**_make_valid_request_dict())
        assert req.problem_type == ProblemTypePreference.AUTO

    # 4. Default automation_level is automatic
    def test_default_automation_level_is_automatic(self):
        req = UserMLRequest(**_make_valid_request_dict())
        assert req.automation_level == AutomationLevel.AUTOMATIC

    # 5. Default compute_preference is balanced
    def test_default_compute_preference_is_balanced(self):
        req = UserMLRequest(**_make_valid_request_dict())
        assert req.compute_preference == ComputePreference.BALANCED

    # 6. Default excluded_columns is an empty list
    def test_default_excluded_columns_is_empty_list(self):
        req = UserMLRequest(**_make_valid_request_dict())
        assert req.excluded_columns == []

    # 7. Empty request_id rejected
    def test_empty_request_id_rejected(self):
        with pytest.raises(ValidationError, match="request_id"):
            UserMLRequest(**_make_valid_request_dict(request_id=""))

    # 8. Whitespace-only request_id rejected
    def test_whitespace_only_request_id_rejected(self):
        with pytest.raises(ValidationError, match="request_id"):
            UserMLRequest(**_make_valid_request_dict(request_id="   "))

    # 9. Empty goal rejected
    def test_empty_goal_rejected(self):
        with pytest.raises(ValidationError, match="goal"):
            UserMLRequest(**_make_valid_request_dict(goal=""))

    # 10. Whitespace-only goal rejected
    def test_whitespace_only_goal_rejected(self):
        with pytest.raises(ValidationError, match="goal"):
            UserMLRequest(**_make_valid_request_dict(goal="  \n\t  "))

    # 11. Valid target_column accepted
    def test_valid_target_column_accepted(self):
        req = UserMLRequest(**_make_valid_request_dict(target_column="churn_label"))
        assert req.target_column == "churn_label"

    # 12. Empty target_column rejected
    def test_empty_target_column_rejected(self):
        with pytest.raises(ValidationError, match="target_column"):
            UserMLRequest(**_make_valid_request_dict(target_column=""))

    # 13. Whitespace-only target_column rejected
    def test_whitespace_only_target_column_rejected(self):
        with pytest.raises(ValidationError, match="target_column"):
            UserMLRequest(**_make_valid_request_dict(target_column="   "))

    # 14. Valid primary_metric accepted
    def test_valid_primary_metric_accepted(self):
        req = UserMLRequest(**_make_valid_request_dict(primary_metric="accuracy"))
        assert req.primary_metric == "accuracy"

    # 15. Empty primary_metric rejected
    def test_empty_primary_metric_rejected(self):
        with pytest.raises(ValidationError, match="primary_metric"):
            UserMLRequest(**_make_valid_request_dict(primary_metric=""))
        with pytest.raises(ValidationError, match="primary_metric"):
            UserMLRequest(**_make_valid_request_dict(primary_metric="   "))

    # 16. Valid additional_context accepted
    def test_valid_additional_context_accepted(self):
        req = UserMLRequest(**_make_valid_request_dict(additional_context="Precision is critical"))
        assert req.additional_context == "Precision is critical"

    # 17. Invalid whitespace-only additional_context handled according to the chosen rule (rejected)
    def test_whitespace_only_additional_context_rejected(self):
        with pytest.raises(ValidationError, match="additional_context"):
            UserMLRequest(**_make_valid_request_dict(additional_context="   "))
        with pytest.raises(ValidationError, match="additional_context"):
            UserMLRequest(**_make_valid_request_dict(additional_context=""))

    # 18. Valid excluded_columns accepted
    def test_valid_excluded_columns_accepted(self):
        req = UserMLRequest(**_make_valid_request_dict(excluded_columns=["col1", "col2"]))
        assert req.excluded_columns == ["col1", "col2"]

    # 19. Duplicate excluded_columns rejected
    def test_duplicate_excluded_columns_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate excluded column"):
            UserMLRequest(**_make_valid_request_dict(excluded_columns=["col1", "col1"]))

    # 20. Empty excluded column name rejected
    def test_empty_excluded_column_name_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            UserMLRequest(**_make_valid_request_dict(excluded_columns=["col1", ""]))

    # 21. Whitespace-only excluded column rejected
    def test_whitespace_only_excluded_column_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty or whitespace-only"):
            UserMLRequest(**_make_valid_request_dict(excluded_columns=["  ", "col2"]))

    # 22. target_column in excluded_columns rejected
    def test_target_column_in_excluded_columns_rejected(self):
        with pytest.raises(ValidationError, match="cannot be in the list of excluded columns"):
            UserMLRequest(
                **_make_valid_request_dict(
                    target_column="churn",
                    excluded_columns=["col1", "churn"]
                )
            )

    # 23. Excluded column order preserved
    def test_excluded_column_order_preserved(self):
        req = UserMLRequest(
            **_make_valid_request_dict(excluded_columns=["z_col", "a_col", "m_col"])
        )
        assert req.excluded_columns == ["z_col", "a_col", "m_col"]

    # 24. All ProblemTypePreference enum values work
    @pytest.mark.parametrize("val", ["auto", "classification", "regression"])
    def test_problem_type_values(self, val):
        req = UserMLRequest(**_make_valid_request_dict(problem_type=val))
        assert req.problem_type == val

    # 25. All AutomationLevel enum values work
    @pytest.mark.parametrize("val", ["automatic", "guided"])
    def test_automation_level_values(self, val):
        req = UserMLRequest(**_make_valid_request_dict(automation_level=val))
        assert req.automation_level == val

    # 26. All ComputePreference enum values work
    @pytest.mark.parametrize("val", ["balanced", "fast", "thorough"])
    def test_compute_preference_values(self, val):
        req = UserMLRequest(**_make_valid_request_dict(compute_preference=val))
        assert req.compute_preference == val

    # 27. model_dump() works
    def test_model_dump(self):
        req = UserMLRequest(
            request_id="req-dump",
            goal="Test dumping",
            target_column="y",
            problem_type=ProblemTypePreference.CLASSIFICATION,
            excluded_columns=["x1", "x2"],
        )
        dumped = req.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["request_id"] == "req-dump"
        assert dumped["goal"] == "Test dumping"
        assert dumped["target_column"] == "y"
        assert dumped["problem_type"] == "classification"
        assert dumped["excluded_columns"] == ["x1", "x2"]

    # 28. model_dump_json() produces valid JSON
    def test_model_dump_json(self):
        req = UserMLRequest(**_make_valid_request_dict())
        json_str = req.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["request_id"] == "req-123"
        assert parsed["goal"] == "Predict customer churn"

    # 29. Enum values serialize as strings
    def test_enum_values_serialize_as_strings(self):
        req = UserMLRequest(
            request_id="req-enum",
            goal="Test serialization",
            problem_type=ProblemTypePreference.REGRESSION,
            automation_level=AutomationLevel.GUIDED,
            compute_preference=ComputePreference.FAST,
        )
        json_str = req.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["problem_type"] == "regression"
        assert parsed["automation_level"] == "guided"
        assert parsed["compute_preference"] == "fast"

    # 30. Two instances do not share the same excluded_columns list
    def test_excluded_columns_not_shared(self):
        req1 = UserMLRequest(**_make_valid_request_dict())
        req2 = UserMLRequest(**_make_valid_request_dict())
        req1.excluded_columns.append("col")
        assert len(req2.excluded_columns) == 0
        assert req1.excluded_columns != req2.excluded_columns

    # Additional coverage tests for validator type checking and Nones
    def test_invalid_types_rejected(self):
        with pytest.raises(ValidationError, match="request_id must be a string"):
            UserMLRequest(**_make_valid_request_dict(request_id=123))

        with pytest.raises(ValidationError, match="goal must be a string"):
            UserMLRequest(**_make_valid_request_dict(goal=True))

        with pytest.raises(ValidationError, match="target_column must be a string"):
            UserMLRequest(**_make_valid_request_dict(target_column=1.5))

        with pytest.raises(ValidationError, match="primary_metric must be a string"):
            UserMLRequest(**_make_valid_request_dict(primary_metric=123))

        with pytest.raises(ValidationError, match="additional_context must be a string"):
            UserMLRequest(**_make_valid_request_dict(additional_context=123))

        with pytest.raises(ValidationError, match="excluded_columns must be a list"):
            UserMLRequest(**_make_valid_request_dict(excluded_columns="not-a-list"))

        with pytest.raises(ValidationError, match="All excluded column names must be strings"):
            UserMLRequest(**_make_valid_request_dict(excluded_columns=[123]))

    def test_explicit_none_values_accepted(self):
        req = UserMLRequest(
            request_id="req-none",
            goal="Predict churn",
            target_column=None,
            primary_metric=None,
            additional_context=None
        )
        assert req.target_column is None
        assert req.primary_metric is None
        assert req.additional_context is None

