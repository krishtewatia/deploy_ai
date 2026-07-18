"""Tests for backend.app.problem_definition.resolver.

Covers all 36 specified testing requirements for ProblemResolver.
"""

import json
import pytest

from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)
from backend.app.ml_request.schemas import (
    AutomationLevel,
    ComputePreference,
    ProblemTypePreference,
    UserMLRequest,
)
from backend.app.problem_definition.resolver import (
    ProblemResolver,
    ProblemResolverError,
)
from backend.app.problem_definition.schemas import (
    ProblemDefinition,
    ProblemType,
    ResolutionStatus,
    TargetSource,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_column(
    name: str,
    *,
    dtype: str = "float64",
    is_numeric: bool = True,
    is_categorical: bool = False,
    is_datetime: bool = False,
    unique_count: int = 100,
    unique_percentage: float = 10.0,
) -> ColumnContext:
    """Helper to construct a ColumnContext."""
    return ColumnContext(
        name=name,
        dtype=dtype,
        is_numeric=is_numeric,
        is_categorical=is_categorical,
        is_datetime=is_datetime,
        missing_count=0,
        missing_percentage=0.0,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        sample_values=[],
        statistics=None,
    )


def _make_dataset_context(columns: list[ColumnContext]) -> DatasetContext:
    """Helper to construct a DatasetContext."""
    return DatasetContext(
        basic_info=DatasetBasicInfo(
            dataset_id="ds-test",
            file_name="test.csv",
            row_count=1000,
            column_count=len(columns),
            memory_usage_bytes=5000,
        ),
        columns=columns,
        missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
        target_candidates=[],
    )


def _make_user_request(**overrides) -> UserMLRequest:
    """Helper to construct a UserMLRequest."""
    base = {
        "request_id": "req-test",
        "goal": "Build an ML model",
        "target_column": "target",
        "problem_type": ProblemTypePreference.AUTO,
        "primary_metric": None,
        "automation_level": AutomationLevel.AUTOMATIC,
        "compute_preference": ComputePreference.BALANCED,
        "excluded_columns": [],
        "additional_context": None,
    }
    base.update(overrides)
    return UserMLRequest(**base)


# ── Tests ──────────────────────────────────────────────────────────────


class TestProblemResolver:
    """Comprehensive unit tests for ProblemResolver."""

    # Helper for a base dataset context setup
    @pytest.fixture
    def base_dataset(self) -> DatasetContext:
        cols = [
            _make_column("id", dtype="int64", unique_count=1000, unique_percentage=100.0),
            _make_column("age", unique_count=50, unique_percentage=5.0),
            _make_column("salary", unique_count=200, unique_percentage=20.0),
            _make_column("department", dtype="object", is_numeric=False, is_categorical=True, unique_count=5),
            _make_column("target", unique_count=2, unique_percentage=0.2),
        ]
        return _make_dataset_context(cols)

    # 1. Explicit classification request resolves successfully
    def test_explicit_classification_request(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(problem_type=ProblemTypePreference.CLASSIFICATION)
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.problem_type == ProblemType.CLASSIFICATION

    # 2. Explicit regression request resolves successfully
    def test_explicit_regression_request(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(problem_type=ProblemTypePreference.REGRESSION)
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.problem_type == ProblemType.REGRESSION

    # 3. Explicit user problem type is respected
    def test_explicit_problem_type_is_respected(self, base_dataset):
        resolver = ProblemResolver()
        # Even though target is binary (normally classification), user request is regression
        req = _make_user_request(problem_type=ProblemTypePreference.REGRESSION)
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.problem_type == ProblemType.REGRESSION

    # 4. target_source is always USER
    def test_target_source_is_always_user(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.target_source == TargetSource.USER

    # 5. target_column=None raises ProblemResolverError
    def test_target_column_none_raises_error(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(target_column=None)
        with pytest.raises(ProblemResolverError, match="target_column is None"):
            resolver.resolve(dataset_context=base_dataset, user_request=req)

    # 6. Missing target column raises ProblemResolverError
    def test_missing_target_column_raises_error(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(target_column="non_existent")
        with pytest.raises(ProblemResolverError, match="does not exist in the dataset"):
            resolver.resolve(dataset_context=base_dataset, user_request=req)

    # 7. Invalid excluded column raises ProblemResolverError
    def test_invalid_excluded_column_raises_error(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(excluded_columns=["non_existent"])
        with pytest.raises(ProblemResolverError, match="Excluded column.*do not exist"):
            resolver.resolve(dataset_context=base_dataset, user_request=req)

    # 8. Multiple invalid excluded columns are reported clearly
    def test_multiple_invalid_excluded_columns_reported(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(excluded_columns=["fake1", "fake2"])
        with pytest.raises(ProblemResolverError, match="fake1.*fake2"):
            resolver.resolve(dataset_context=base_dataset, user_request=req)

    # 9. Feature columns exclude target
    def test_feature_columns_exclude_target(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(target_column="target")
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert "target" not in defn.feature_columns

    # 10. Feature columns exclude user-excluded columns
    def test_feature_columns_exclude_user_exclusions(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(excluded_columns=["id", "department"])
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert "id" not in defn.feature_columns
        assert "department" not in defn.feature_columns

    # 11. Feature column order follows DatasetContext order
    def test_feature_column_order_preserved(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(excluded_columns=["id"])
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        # Expected: id is excluded, target is target_column. Remaining features: age, salary, department.
        assert defn.feature_columns == ["age", "salary", "department"]

    # 12. Excluded column order is preserved
    def test_excluded_column_order_preserved(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(excluded_columns=["salary", "id"])
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.excluded_columns == ["salary", "id"]

    # 13. No remaining feature columns raises ProblemResolverError
    def test_no_remaining_features_raises_error(self):
        cols = [_make_column("target", unique_count=2)]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request(target_column="target")
        with pytest.raises(ProblemResolverError, match="No feature columns remain"):
            resolver.resolve(dataset_context=dataset, user_request=req)

    # 14. AUTO + categorical target → classification
    def test_auto_categorical_target_inferred_classification(self):
        cols = [
            _make_column("feat", unique_count=10),
            _make_column("target", dtype="object", is_numeric=False, is_categorical=True, unique_count=5),
        ]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=dataset, user_request=req)
        assert defn.problem_type == ProblemType.CLASSIFICATION

    # 15. AUTO + binary numeric target → classification
    def test_auto_binary_numeric_target_inferred_classification(self):
        cols = [
            _make_column("feat", unique_count=10),
            _make_column("target", dtype="int64", is_numeric=True, unique_count=2),
        ]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=dataset, user_request=req)
        assert defn.problem_type == ProblemType.CLASSIFICATION

    # 16. AUTO + numeric target with unique_count <= 20 and unique_percentage <= 5.0 → classification
    def test_auto_numeric_low_cardinality_inferred_classification(self):
        cols = [
            _make_column("feat", unique_count=10),
            # unique_count = 15 (<= 20), unique_percentage = 1.5% (<= 5.0)
            _make_column("target", dtype="float64", is_numeric=True, unique_count=15, unique_percentage=1.5),
        ]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=dataset, user_request=req)
        assert defn.problem_type == ProblemType.CLASSIFICATION

    # 17. AUTO + high-cardinality numeric target → regression
    def test_auto_numeric_high_cardinality_inferred_regression(self):
        cols = [
            _make_column("feat", unique_count=10),
            _make_column("target", dtype="float64", is_numeric=True, unique_count=100, unique_percentage=10.0),
        ]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=dataset, user_request=req)
        assert defn.problem_type == ProblemType.REGRESSION

    # 18. AUTO + unsupported/ambiguous target type raises ProblemResolverError
    def test_auto_unsupported_target_type_raises_error(self):
        cols = [
            _make_column("feat", unique_count=10),
            _make_column("target", dtype="datetime64[ns]", is_numeric=False, is_categorical=False, is_datetime=True),
        ]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request()
        with pytest.raises(ProblemResolverError, match="Problem type cannot be inferred"):
            resolver.resolve(dataset_context=dataset, user_request=req)

    # 19. User-provided primary_metric is preserved
    def test_user_provided_metric_preserved(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(primary_metric="auc")
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.primary_metric == "auc"

    # 20. Classification without primary_metric defaults to "f1"
    def test_classification_metric_default(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(problem_type=ProblemTypePreference.CLASSIFICATION)
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.primary_metric == "f1"

    # 21. Regression without primary_metric defaults to "rmse"
    def test_regression_metric_default(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(problem_type=ProblemTypePreference.REGRESSION)
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.primary_metric == "rmse"

    # 22. Goal is preserved exactly
    def test_goal_preserved(self, base_dataset):
        resolver = ProblemResolver()
        goal_text = "Predict stock returns using daily pricing details"
        req = _make_user_request(goal=goal_text)
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.goal == goal_text

    # 23. request_id is copied correctly
    def test_request_id_copied(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(request_id="req-churn-v1")
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.request_id == "req-churn-v1"

    # 24. dataset_id is copied correctly
    def test_dataset_id_copied(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.dataset_id == base_dataset.basic_info.dataset_id

    # 25. definition_id is generated and non-empty
    def test_definition_id_generated(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.definition_id.startswith("definition_")
        assert len(defn.definition_id) > len("definition_")

    # 26. Two resolutions produce different definition IDs
    def test_multiple_resolutions_different_ids(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn1 = resolver.resolve(dataset_context=base_dataset, user_request=req)
        defn2 = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn1.definition_id != defn2.definition_id

    # 27. Successful result status is RESOLVED
    def test_result_status_resolved(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.status == ResolutionStatus.RESOLVED

    # 28. Successful result warnings is empty
    def test_result_warnings_empty(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.warnings == []

    # 29. Successful result confirmation_items is empty
    def test_result_confirmation_items_empty(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert defn.confirmation_items == []

    # 30. Result is a valid ProblemDefinition
    def test_result_is_valid_pydantic_model(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert isinstance(defn, ProblemDefinition)

    # 31. Result supports model_dump_json()
    def test_result_json_serialization(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=base_dataset, user_request=req)
        json_str = defn.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["definition_id"] == defn.definition_id
        assert parsed["problem_type"] == "classification"

    # 32. Resolver does not mutate DatasetContext
    def test_no_mutation_dataset_context(self, base_dataset):
        resolver = ProblemResolver()
        original_cols = list(base_dataset.columns)
        req = _make_user_request()
        resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert base_dataset.columns == original_cols

    # 33. Resolver does not mutate UserMLRequest
    def test_no_mutation_user_request(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request(excluded_columns=["id"])
        original_exclusions = list(req.excluded_columns)
        resolver.resolve(dataset_context=base_dataset, user_request=req)
        assert req.excluded_columns == original_exclusions

    # 34. Numeric low-cardinality boundary: unique_count == 20 and unique_percentage == 5.0 → classification
    def test_auto_numeric_low_cardinality_boundary_classification(self):
        cols = [
            _make_column("feat", unique_count=10),
            _make_column("target", dtype="float64", is_numeric=True, unique_count=20, unique_percentage=5.0),
        ]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=dataset, user_request=req)
        assert defn.problem_type == ProblemType.CLASSIFICATION

    # 35. Numeric target with unique_count > 20 → regression
    def test_auto_numeric_boundary_unique_count_exceeded_regression(self):
        cols = [
            _make_column("feat", unique_count=10),
            # unique_count is 21 (> 20), even though unique_percentage is low (2.1%)
            _make_column("target", dtype="float64", is_numeric=True, unique_count=21, unique_percentage=2.1),
        ]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=dataset, user_request=req)
        assert defn.problem_type == ProblemType.REGRESSION

    # 36. Numeric target with unique_percentage > 5.0 and not binary → regression
    def test_auto_numeric_boundary_percentage_exceeded_regression(self):
        cols = [
            _make_column("feat", unique_count=10),
            # unique_percentage is 5.1 (> 5.0), even though unique_count is low (10)
            _make_column("target", dtype="float64", is_numeric=True, unique_count=10, unique_percentage=5.1),
        ]
        dataset = _make_dataset_context(cols)
        resolver = ProblemResolver()
        req = _make_user_request()
        defn = resolver.resolve(dataset_context=dataset, user_request=req)
        assert defn.problem_type == ProblemType.REGRESSION

    def test_unsupported_problem_type_preference_raises_error(self, base_dataset):
        resolver = ProblemResolver()
        req = _make_user_request()
        # Override problem_type directly to bypass validation for test coverage
        req.problem_type = "unsupported_type"
        with pytest.raises(ProblemResolverError, match="Unsupported problem type preference"):
            resolver.resolve(dataset_context=base_dataset, user_request=req)

