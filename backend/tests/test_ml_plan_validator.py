"""Unit tests for MLPlanValidator.

Covers all 96 specified test scenarios for cross-artifact validation.
"""

import copy
import json
import pytest
from pydantic import ValidationError

from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
)
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
)
from backend.app.ml_plan.schemas import (
    DatasetSplitPlan,
    EvaluationPlan,
    ExecutionConstraints,
    FeatureEngineeringOperation,
    FeatureEngineeringStep,
    FeatureSelectionMethod,
    FeatureSelectionPlan,
    MLPlan,
    MLPlanConfirmationItem,
    MLPlanStatus,
    MLPlanWarning,
    ModelCandidate,
    ModelFamily,
    PreprocessingOperation,
    PreprocessingStep,
    SearchStrategy,
    SplitStrategy,
)
from backend.app.ml_plan.validator import (
    MLPlanValidationError,
    MLPlanValidationIssue,
    MLPlanValidationResult,
    MLPlanValidator,
    ValidationSeverity,
)
from backend.app.problem_definition.schemas import (
    ConfirmationItem,
    ProblemDefinition,
    ProblemType,
    ResolutionStatus,
    TargetSource,
)


# ── Fixtures & Helper Builders ─────────────────────────────────────────


def _make_dataset_context(
    dataset_id: str = "ds_01",
    columns: list[ColumnContext] = None,
) -> DatasetContext:
    if columns is None:
        columns = [
            ColumnContext(
                name="age",
                dtype="float64",
                is_numeric=True,
                is_categorical=False,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=50,
                unique_percentage=5.0,
                sample_values=[25.0, 30.0],
            ),
            ColumnContext(
                name="department",
                dtype="object",
                is_numeric=False,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=5,
                unique_percentage=0.5,
                sample_values=["sales", "engineering"],
            ),
            ColumnContext(
                name="signup_date",
                dtype="datetime64",
                is_numeric=False,
                is_categorical=False,
                is_datetime=True,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=100,
                unique_percentage=10.0,
                sample_values=["2026-01-01"],
            ),
            ColumnContext(
                name="churn",
                dtype="int64",
                is_numeric=True,
                is_categorical=True,
                is_datetime=False,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=2,
                unique_percentage=0.2,
                sample_values=[0, 1],
            ),
        ]

    basic_info = DatasetBasicInfo(
        dataset_id=dataset_id,
        file_name="data.csv",
        row_count=1000,
        column_count=len(columns),
        memory_usage_bytes=50000,
    )
    return DatasetContext(
        basic_info=basic_info,
        columns=columns,
        missing_data=MissingDataSummary(total_missing_cells=0, columns_with_missing=[]),
        duplicates=DuplicateSummary(duplicate_rows=0, duplicate_percentage=0.0),
    )


def _make_problem_definition(
    dataset_id: str = "ds_01",
    request_id: str = "req_01",
    definition_id: str = "pd_01",
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    target_column: str = "churn",
    feature_columns: list[str] = None,
    status: ResolutionStatus = ResolutionStatus.RESOLVED,
    confirmation_items: list[ConfirmationItem] = None,
) -> ProblemDefinition:
    if feature_columns is None:
        feature_columns = ["age", "department"]

    if confirmation_items is None:
        if status == ResolutionStatus.NEEDS_CONFIRMATION:
            confirmation_items = [ConfirmationItem(key="k", question="q", reason="r")]
        else:
            confirmation_items = []

    primary_metric = "f1" if problem_type == ProblemType.CLASSIFICATION else "mae"

    return ProblemDefinition(
        definition_id=definition_id,
        request_id=request_id,
        dataset_id=dataset_id,
        goal="Predict customer churn",
        problem_type=problem_type,
        target_column=target_column,
        target_source=TargetSource.USER,
        feature_columns=feature_columns,
        excluded_columns=[],
        primary_metric=primary_metric,
        status=status,
        confirmation_items=confirmation_items,
    )


def _make_compute_capabilities(
    capability_id: str = "cap_01",
    hardware_profile_id: str = "hw_01",
    safe_parallel_workers: int = 4,
    gpu_acceleration_available: bool = False,
    accelerator_type: AcceleratorType = AcceleratorType.NONE,
) -> ComputeCapabilities:
    return ComputeCapabilities(
        capability_id=capability_id,
        hardware_profile_id=hardware_profile_id,
        compute_tier=ComputeTier.STANDARD,
        memory_constraint=MemoryConstraintLevel.MODERATE,
        cpu_training_available=True,
        gpu_acceleration_available=gpu_acceleration_available,
        accelerator_type=accelerator_type,
        safe_parallel_workers=safe_parallel_workers,
        max_parallel_workers=8,
        available_ram_mb_snapshot=4096,
        total_ram_mb=8192,
        warnings=[],
    )


def _make_valid_ml_plan(
    plan_id: str = "plan_01",
    dataset_id: str = "ds_01",
    request_id: str = "req_01",
    problem_definition_id: str = "pd_01",
    compute_capability_id: str = "cap_01",
    target_column: str = "churn",
    feature_columns: list[str] = None,
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    preprocessing_steps: list[PreprocessingStep] = None,
    feature_engineering_steps: list[FeatureEngineeringStep] = None,
    feature_selection: FeatureSelectionPlan = None,
    split_plan: DatasetSplitPlan = None,
    model_candidates: list[ModelCandidate] = None,
    evaluation_plan: EvaluationPlan = None,
    execution_constraints: ExecutionConstraints = None,
    status: MLPlanStatus = MLPlanStatus.DRAFT,
    confirmation_items: list[MLPlanConfirmationItem] = None,
) -> MLPlan:
    if feature_columns is None:
        feature_columns = ["age", "department"]

    if preprocessing_steps is None:
        preprocessing_steps = []

    if feature_engineering_steps is None:
        feature_engineering_steps = []

    if feature_selection is None:
        feature_selection = FeatureSelectionPlan(
            method=FeatureSelectionMethod.NONE,
            candidate_columns=feature_columns,
            reason="No selection needed.",
        )

    if split_plan is None:
        if strategy := (SplitStrategy.STRATIFIED if problem_type == ProblemType.CLASSIFICATION else SplitStrategy.RANDOM):
            split_plan = DatasetSplitPlan(
                strategy=strategy,
                test_size=0.2,
                stratify_column=target_column if strategy == SplitStrategy.STRATIFIED else None,
            )

    if model_candidates is None:
        model_family = ModelFamily.LOGISTIC_REGRESSION if problem_type == ProblemType.CLASSIFICATION else ModelFamily.LINEAR_REGRESSION
        model_candidates = [
            ModelCandidate(
                candidate_id="model_01",
                model_family=model_family,
                parameters={},
                reason="Baseline algorithm",
            )
        ]

    if evaluation_plan is None:
        primary_metric = "f1" if problem_type == ProblemType.CLASSIFICATION else "mae"
        evaluation_plan = EvaluationPlan(
            primary_metric=primary_metric,
            secondary_metrics=[],
            cross_validation_folds=5,
        )

    if execution_constraints is None:
        execution_constraints = ExecutionConstraints(
            parallel_workers=2,
            use_gpu_acceleration=False,
            accelerator_type=AcceleratorType.NONE,
            compute_tier=ComputeTier.STANDARD,
        )

    return MLPlan(
        plan_id=plan_id,
        dataset_id=dataset_id,
        request_id=request_id,
        problem_definition_id=problem_definition_id,
        compute_capability_id=compute_capability_id,
        problem_type=problem_type,
        target_column=target_column,
        feature_columns=feature_columns,
        preprocessing_steps=preprocessing_steps,
        feature_engineering_steps=feature_engineering_steps,
        feature_selection=feature_selection,
        split_plan=split_plan,
        model_candidates=model_candidates,
        evaluation_plan=evaluation_plan,
        execution_constraints=execution_constraints,
        status=status,
        confirmation_items=confirmation_items or [],
    )


# ── Tests ──────────────────────────────────────────────────────────────


class TestValidationResultSchemas:
    """Scenarios 1-14: MLPlanValidationIssue & MLPlanValidationResult validations."""

    # 1-2. Valid error / warning issues
    def test_valid_issues(self):
        issue_err = MLPlanValidationIssue(
            code="CODE_A", message="msg A", severity=ValidationSeverity.ERROR, location="field"
        )
        issue_warn = MLPlanValidationIssue(
            code="CODE_B", message="msg B", severity=ValidationSeverity.WARNING, location=None
        )
        assert issue_err.severity == ValidationSeverity.ERROR
        assert issue_warn.severity == ValidationSeverity.WARNING

    # 3-5. Empty code, message, or location rejected
    def test_invalid_issue_strings_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            MLPlanValidationIssue(code=" ", message="msg", severity=ValidationSeverity.ERROR)
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            MLPlanValidationIssue(code="C", message="", severity=ValidationSeverity.ERROR)
        with pytest.raises(ValidationError, match="location cannot be empty"):
            MLPlanValidationIssue(code="C", message="m", severity=ValidationSeverity.ERROR, location=" ")

    # 6-7. Valid result with/without errors
    def test_valid_results(self):
        issue = MLPlanValidationIssue(code="E", message="m", severity=ValidationSeverity.ERROR)
        res_valid = MLPlanValidationResult(plan_id="p", is_valid=True, errors=[], warnings=[])
        res_invalid = MLPlanValidationResult(plan_id="p", is_valid=False, errors=[issue], warnings=[])
        assert res_valid.is_valid is True
        assert res_invalid.is_valid is False

    # 8-11. Severity and consistency validations
    def test_validation_result_consistency_rules(self):
        err = MLPlanValidationIssue(code="E", message="m", severity=ValidationSeverity.ERROR)
        warn = MLPlanValidationIssue(code="W", message="m", severity=ValidationSeverity.WARNING)

        # 8. Error list containing warning severity rejected
        with pytest.raises(ValidationError, match="must have severity 'error'"):
            MLPlanValidationResult(plan_id="p", is_valid=False, errors=[warn])

        # 9. Warning list containing error severity rejected
        with pytest.raises(ValidationError, match="must have severity 'warning'"):
            MLPlanValidationResult(plan_id="p", is_valid=True, errors=[], warnings=[err])

        # 10. is_valid=True with errors rejected
        with pytest.raises(ValidationError, match="is_valid must be False"):
            MLPlanValidationResult(plan_id="p", is_valid=True, errors=[err])

        # 11. is_valid=False without errors rejected
        with pytest.raises(ValidationError, match="is_valid must be True"):
            MLPlanValidationResult(plan_id="p", is_valid=False, errors=[])

    # 12. Mutable defaults are independent
    def test_mutable_defaults_independent(self):
        r1 = MLPlanValidationResult(plan_id="p1", is_valid=True)
        r2 = MLPlanValidationResult(plan_id="p2", is_valid=True)
        r1.warnings.append(MLPlanValidationIssue(code="W", message="m", severity=ValidationSeverity.WARNING))
        assert len(r1.warnings) == 1
        assert len(r2.warnings) == 0

    # 13-14. JSON & Enum serialization
    def test_serialization(self):
        res = MLPlanValidationResult(
            plan_id="p",
            is_valid=True,
            errors=[],
            warnings=[MLPlanValidationIssue(code="W", message="m", severity=ValidationSeverity.WARNING)],
        )
        json_str = res.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["warnings"][0]["severity"] == "warning"


class TestValidatorInputValidation:
    """Scenarios 15-18: Validator input type validations."""

    def test_invalid_types_raise_validation_error(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan()
        ds = _make_dataset_context()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()

        # 15. Non-MLPlan
        with pytest.raises(MLPlanValidationError, match="plan must be an instance"):
            val.validate("not-a-plan", ds, pd, cc)

        # 16. Non-DatasetContext
        with pytest.raises(MLPlanValidationError, match="dataset_context must be an instance"):
            val.validate(plan, "not-a-ds", pd, cc)

        # 17. Non-ProblemDefinition
        with pytest.raises(MLPlanValidationError, match="problem_definition must be an instance"):
            val.validate(plan, ds, "not-a-pd", cc)

        # 18. Non-ComputeCapabilities
        with pytest.raises(MLPlanValidationError, match="compute_capabilities must be an instance"):
            val.validate(plan, ds, pd, "not-a-cc")


class TestValidatorOutputs:
    """Scenarios 19-22: Valid classification/regression plans checks."""

    # 19. Valid classification plan
    def test_valid_classification_plan(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(status=MLPlanStatus.READY)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert res.is_valid is True
        assert len(res.errors) == 0
        assert len(res.warnings) == 0

    # 20. Valid regression plan
    def test_valid_regression_plan(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn", "signup_date"]
        )
        ds = _make_dataset_context()
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn", "signup_date"]
        )
        cc = _make_compute_capabilities()
        res = val.validate(plan, ds, pd, cc)
        assert res.is_valid is True

    # 21-22. Zero errors and status warnings check
    def test_zero_errors_and_warnings_when_ready(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(status=MLPlanStatus.READY)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0
        assert len(res.warnings) == 0


class TestValidatorArtifactIds:
    """Scenarios 23-27: Upstream ID mismatch errors."""

    # 23. Dataset ID mismatch
    def test_dataset_id_mismatch(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(dataset_id="ds_wrong")
        res = val.validate(plan, _make_dataset_context(dataset_id="ds_real"), _make_problem_definition(), _make_compute_capabilities())
        assert res.is_valid is False
        assert any(e.code == "DATASET_ID_MISMATCH" for e in res.errors)

    # 24. Problem dataset ID mismatch
    def test_problem_dataset_id_mismatch(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(dataset_id="ds_01")
        pd = _make_problem_definition(dataset_id="ds_wrong")
        res = val.validate(plan, _make_dataset_context(dataset_id="ds_01"), pd, _make_compute_capabilities())
        assert res.is_valid is False
        assert any(e.code == "PROBLEM_DATASET_ID_MISMATCH" for e in res.errors)

    # 25. Request ID mismatch
    def test_request_id_mismatch(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(request_id="req_wrong")
        pd = _make_problem_definition(request_id="req_real")
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert res.is_valid is False
        assert any(e.code == "REQUEST_ID_MISMATCH" for e in res.errors)

    # 26. Problem definition ID mismatch
    def test_problem_definition_id_mismatch(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(problem_definition_id="pd_wrong")
        pd = _make_problem_definition(definition_id="pd_real")
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert res.is_valid is False
        assert any(e.code == "PROBLEM_DEFINITION_ID_MISMATCH" for e in res.errors)

    # 27. Compute capability ID mismatch
    def test_compute_capability_id_mismatch(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(compute_capability_id="cap_wrong")
        cc = _make_compute_capabilities(capability_id="cap_real")
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), cc)
        assert res.is_valid is False
        assert any(e.code == "COMPUTE_CAPABILITY_ID_MISMATCH" for e in res.errors)


class TestValidatorProblemDefinition:
    """Scenarios 28-32: ProblemDefinition consistency validations."""

    # 28. Problem type mismatch
    def test_problem_type_mismatch(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(problem_type=ProblemType.CLASSIFICATION)
        pd = _make_problem_definition(problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"])
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "PROBLEM_TYPE_MISMATCH" for e in res.errors)

    # 29. Target mismatch
    def test_target_mismatch(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(target_column="churn")
        pd = _make_problem_definition(target_column="age", feature_columns=["churn"])
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "TARGET_COLUMN_MISMATCH" for e in res.errors)

    # 30. Feature column mismatch
    def test_feature_columns_mismatch(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(feature_columns=["age", "department"])
        pd = _make_problem_definition(feature_columns=["department", "age"])  # order mismatch
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "FEATURE_COLUMNS_MISMATCH" for e in res.errors)

    # 31. Blocked ProblemDefinition
    def test_blocked_problem_definition(self):
        val = MLPlanValidator()
        pd = _make_problem_definition(status=ResolutionStatus.BLOCKED)
        res = val.validate(_make_valid_ml_plan(), _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "PROBLEM_DEFINITION_BLOCKED" for e in res.errors)

    # 32. Needs-confirmation ProblemDefinition
    def test_needs_confirmation_problem_definition(self):
        val = MLPlanValidator()
        pd = _make_problem_definition(status=ResolutionStatus.NEEDS_CONFIRMATION)
        # Mock confirmation item to satisfy Pydantic schema validation for status
        pd.confirmation_items.append(MLPlanConfirmationItem(key="a", question="b", reason="c"))
        res = val.validate(_make_valid_ml_plan(), _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "PROBLEM_DEFINITION_NEEDS_CONFIRMATION" for e in res.errors)


class TestValidatorDatasetColumns:
    """Scenarios 33-35: Column existence validations."""

    # 33. Missing target
    def test_missing_target_column(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(target_column="non_existent")
        pd = _make_problem_definition(target_column="non_existent")
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "TARGET_COLUMN_NOT_FOUND" for e in res.errors)

    # 34-35. Missing features validation order
    def test_missing_features(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(feature_columns=["missing_a", "missing_b"])
        pd = _make_problem_definition(feature_columns=["missing_a", "missing_b"])
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        missing_errors = [e for e in res.errors if e.code == "FEATURE_COLUMN_NOT_FOUND"]
        assert len(missing_errors) == 2
        assert "missing_a" in missing_errors[0].message
        assert "missing_b" in missing_errors[1].message


class TestValidatorPreprocessing:
    """Scenarios 36-44: Preprocessing step type validations."""

    # 36. Valid numeric preprocessing
    def test_valid_numeric_preprocessing(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.STANDARD_SCALE,
            columns=["age"],
            reason="Numeric scaling",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 37. Valid categorical preprocessing
    def test_valid_categorical_preprocessing(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.ONE_HOT_ENCODE,
            columns=["department"],
            reason="Categorical encoding",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 38. Valid datetime preprocessing
    def test_valid_datetime_preprocessing(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.DATETIME_EXTRACT,
            columns=["signup_date"],
            reason="Date extracting",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 39. Missing preprocessing column
    def test_missing_preprocessing_column(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.STANDARD_SCALE,
            columns=["missing_col"],
            reason="err",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "PREPROCESSING_COLUMN_NOT_FOUND" for e in res.errors)

    # 40. Numeric operation on categorical column
    def test_numeric_op_on_categorical_rejected(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.STANDARD_SCALE,
            columns=["department"],
            reason="err",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "PREPROCESSING_REQUIRES_NUMERIC" for e in res.errors)

    # 41. Categorical operation on numeric column
    def test_categorical_op_on_numeric_rejected(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.ONE_HOT_ENCODE,
            columns=["age"],
            reason="err",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "PREPROCESSING_REQUIRES_CATEGORICAL" for e in res.errors)

    # 42. Datetime operation on non-datetime column
    def test_datetime_op_on_non_datetime_rejected(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.DATETIME_EXTRACT,
            columns=["age"],
            reason="err",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "PREPROCESSING_REQUIRES_DATETIME" for e in res.errors)

    # 43. Target in preprocessing step
    def test_target_in_preprocessing_step_rejected(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.STANDARD_SCALE,
            columns=["churn"],
            reason="err",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "TARGET_IN_PREPROCESSING_STEP" for e in res.errors)

    # 44. General preprocessing operation accepted
    def test_general_preprocessing_operation_accepted(self):
        val = MLPlanValidator()
        step = PreprocessingStep(
            step_id="p1",
            operation=PreprocessingOperation.DROP_COLUMN,
            columns=["department"],
            reason="dropping",
        )
        plan = _make_valid_ml_plan(preprocessing_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0


class TestValidatorFeatureEngineering:
    """Scenarios 45-51: FeatureEngineering step validations."""

    # 45. Valid feature engineering
    def test_valid_feature_engineering(self):
        val = MLPlanValidator()
        step = FeatureEngineeringStep(
            step_id="fe1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["age", "department"],
            output_columns=["age_dept_interaction"],
            reason="Valid",
        )
        plan = _make_valid_ml_plan(feature_engineering_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 46. Missing feature-engineering input
    def test_missing_fe_input_column_rejected(self):
        val = MLPlanValidator()
        step = FeatureEngineeringStep(
            step_id="fe1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["missing_col"],
            output_columns=["out_col"],
            reason="err",
        )
        plan = _make_valid_ml_plan(feature_engineering_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "FEATURE_ENGINEERING_INPUT_NOT_FOUND" for e in res.errors)

    # 47. Output collision
    def test_fe_output_collision_rejected(self):
        val = MLPlanValidator()
        step = FeatureEngineeringStep(
            step_id="fe1",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["age"],
            output_columns=["department"],  # department already exists
            reason="err",
        )
        plan = _make_valid_ml_plan(feature_engineering_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "FEATURE_ENGINEERING_OUTPUT_COLLISION" for e in res.errors)

    # 48. Later step can consume earlier output
    def test_later_fe_step_can_consume_earlier_output(self):
        val = MLPlanValidator()
        s1 = FeatureEngineeringStep(
            step_id="fe1",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["age"],
            output_columns=["log_age"],
            reason="step1",
        )
        s2 = FeatureEngineeringStep(
            step_id="fe2",
            operation=FeatureEngineeringOperation.POLYNOMIAL,
            input_columns=["log_age"],  # Created in step 1
            output_columns=["poly_log_age"],
            reason="step2",
        )
        plan = _make_valid_ml_plan(feature_engineering_steps=[s1, s2])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 49. Target used as feature-engineering input
    def test_target_in_fe_input_rejected(self):
        val = MLPlanValidator()
        step = FeatureEngineeringStep(
            step_id="fe1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["churn", "age"],
            output_columns=["churn_age"],
            reason="err",
        )
        plan = _make_valid_ml_plan(feature_engineering_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "TARGET_IN_FEATURE_ENGINEERING_INPUT" for e in res.errors)

    # 50. Target used as feature-engineering output
    def test_target_in_fe_output_rejected(self):
        val = MLPlanValidator()
        step = FeatureEngineeringStep(
            step_id="fe1",
            operation=FeatureEngineeringOperation.INTERACTION,
            input_columns=["age", "department"],
            output_columns=["churn"],
            reason="err",
        )
        plan = _make_valid_ml_plan(feature_engineering_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "TARGET_IN_FEATURE_ENGINEERING_OUTPUT" for e in res.errors)

    # 51. Custom feature engineering accepted
    def test_custom_feature_engineering_accepted(self):
        val = MLPlanValidator()
        step = FeatureEngineeringStep(
            step_id="fe1",
            operation=FeatureEngineeringOperation.CUSTOM,
            input_columns=["age"],
            output_columns=["custom_out"],
            reason="custom calculation",
        )
        plan = _make_valid_ml_plan(feature_engineering_steps=[step])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0


class TestValidatorFeatureSelection:
    """Scenarios 52-55: FeatureSelectionPlan validations."""

    # 52. Valid original feature candidate
    def test_valid_original_feature_candidate_accepted(self):
        val = MLPlanValidator()
        sel = FeatureSelectionPlan(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            candidate_columns=["age"],
            reason="age",
        )
        plan = _make_valid_ml_plan(feature_selection=sel)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 53. Valid engineered feature candidate
    def test_valid_engineered_feature_candidate_accepted(self):
        val = MLPlanValidator()
        step = FeatureEngineeringStep(
            step_id="fe1",
            operation=FeatureEngineeringOperation.LOG_TRANSFORM,
            input_columns=["age"],
            output_columns=["log_age"],
            reason="log",
        )
        sel = FeatureSelectionPlan(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            candidate_columns=["log_age"],
            reason="selection",
        )
        plan = _make_valid_ml_plan(feature_engineering_steps=[step], feature_selection=sel)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 54. Unavailable candidate rejected
    def test_unavailable_feature_selection_candidate_rejected(self):
        val = MLPlanValidator()
        sel = FeatureSelectionPlan(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            candidate_columns=["missing_col"],
            reason="selection",
        )
        plan = _make_valid_ml_plan(feature_selection=sel)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "FEATURE_SELECTION_COLUMN_NOT_AVAILABLE" for e in res.errors)

    # 55. Target candidate rejected
    def test_target_in_feature_selection_rejected(self):
        val = MLPlanValidator()
        sel = FeatureSelectionPlan(
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            candidate_columns=["churn"],
            reason="target selected",
        )
        plan = _make_valid_ml_plan(feature_selection=sel)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "TARGET_IN_FEATURE_SELECTION" for e in res.errors)


class TestValidatorSplitting:
    """Scenarios 56-61: DatasetSplitPlan validations."""

    # 56. Valid classification stratification
    def test_valid_classification_stratification(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(status=MLPlanStatus.READY)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 57. Wrong stratification column
    def test_wrong_stratification_column_rejected(self):
        val = MLPlanValidator()
        split = DatasetSplitPlan(
            strategy=SplitStrategy.STRATIFIED,
            test_size=0.2,
            stratify_column="age",  # Mismatch with target column 'churn'
        )
        plan = _make_valid_ml_plan(split_plan=split)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "INVALID_STRATIFY_COLUMN" for e in res.errors)

    # 58. Stratified regression rejected
    def test_stratified_regression_rejected(self):
        val = MLPlanValidator()
        split = DatasetSplitPlan(
            strategy=SplitStrategy.STRATIFIED,
            test_size=0.2,
            stratify_column="age",
        )
        plan = _make_valid_ml_plan(
            problem_type=ProblemType.REGRESSION,
            target_column="age",
            feature_columns=["churn"],
            split_plan=split,
        )
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "STRATIFIED_SPLIT_FOR_REGRESSION" for e in res.errors)

    # 59. Valid datetime time-based split
    def test_valid_time_based_split(self):
        val = MLPlanValidator()
        split = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.2,
            time_column="signup_date",
            shuffle=False,
        )
        plan = _make_valid_ml_plan(split_plan=split)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 60. Missing time column
    def test_missing_time_column_rejected(self):
        val = MLPlanValidator()
        split = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.2,
            time_column="missing_date",
            shuffle=False,
        )
        plan = _make_valid_ml_plan(split_plan=split)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "TIME_COLUMN_NOT_FOUND" for e in res.errors)

    # 61. Non-datetime time column
    def test_non_datetime_time_column_rejected(self):
        val = MLPlanValidator()
        split = DatasetSplitPlan(
            strategy=SplitStrategy.TIME_BASED,
            test_size=0.2,
            time_column="age",  # age is numeric, not datetime
            shuffle=False,
        )
        plan = _make_valid_ml_plan(split_plan=split)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "TIME_COLUMN_NOT_DATETIME" for e in res.errors)


class TestValidatorModelCompatibility:
    """Scenarios 62-68: Model family checks."""

    # 62. Every classification-compatible family accepted
    @pytest.mark.parametrize("family", [
        ModelFamily.LOGISTIC_REGRESSION, ModelFamily.DECISION_TREE,
        ModelFamily.RANDOM_FOREST, ModelFamily.GRADIENT_BOOSTING,
        ModelFamily.EXTRA_TREES, ModelFamily.KNN, ModelFamily.SVM
    ])
    def test_classification_models_accepted(self, family):
        val = MLPlanValidator()
        cand = ModelCandidate(candidate_id="m", model_family=family, reason="ok")
        plan = _make_valid_ml_plan(model_candidates=[cand])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 63-65. Linear, Ridge, Lasso regression models rejected for classification
    @pytest.mark.parametrize("family", [
        ModelFamily.LINEAR_REGRESSION, ModelFamily.RIDGE, ModelFamily.LASSO
    ])
    def test_regression_models_rejected_for_classification(self, family):
        val = MLPlanValidator()
        cand = ModelCandidate(candidate_id="m", model_family=family, reason="err")
        plan = _make_valid_ml_plan(model_candidates=[cand])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "MODEL_FAMILY_INCOMPATIBLE" for e in res.errors)

    # 66. Every regression-compatible family accepted
    @pytest.mark.parametrize("family", [
        ModelFamily.LINEAR_REGRESSION, ModelFamily.RIDGE, ModelFamily.LASSO,
        ModelFamily.DECISION_TREE, ModelFamily.RANDOM_FOREST,
        ModelFamily.GRADIENT_BOOSTING, ModelFamily.EXTRA_TREES,
        ModelFamily.KNN, ModelFamily.SVM
    ])
    def test_regression_models_accepted(self, family):
        val = MLPlanValidator()
        cand = ModelCandidate(candidate_id="m", model_family=family, reason="ok")
        plan = _make_valid_ml_plan(
            problem_type=ProblemType.REGRESSION,
            target_column="age",
            feature_columns=["churn"],
            model_candidates=[cand],
        )
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert len(res.errors) == 0

    # 67. Logistic regression rejected for regression
    def test_logistic_regression_rejected_for_regression(self):
        val = MLPlanValidator()
        cand = ModelCandidate(candidate_id="m", model_family=ModelFamily.LOGISTIC_REGRESSION, reason="err")
        plan = _make_valid_ml_plan(
            problem_type=ProblemType.REGRESSION,
            target_column="age",
            feature_columns=["churn"],
            model_candidates=[cand],
        )
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "MODEL_FAMILY_INCOMPATIBLE" for e in res.errors)

    # 68. Multiple incompatible models preserve candidate order
    def test_incompatible_models_preserve_order(self):
        val = MLPlanValidator()
        cand1 = ModelCandidate(candidate_id="m1", model_family=ModelFamily.LINEAR_REGRESSION, reason="err")
        cand2 = ModelCandidate(candidate_id="m2", model_family=ModelFamily.RIDGE, reason="err")
        plan = _make_valid_ml_plan(model_candidates=[cand1, cand2])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        compat_errors = [e for e in res.errors if e.code == "MODEL_FAMILY_INCOMPATIBLE"]
        assert len(compat_errors) == 2
        assert "model_candidates[0]" in compat_errors[0].location
        assert "model_candidates[1]" in compat_errors[1].location


class TestValidatorMetrics:
    """Scenarios 69-75: Metric compatibility checks."""

    # 69. Classification metrics accepted
    @pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1", "roc_auc"])
    def test_classification_metrics_accepted(self, metric):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(evaluation_plan=EvaluationPlan(primary_metric=metric))
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 70. Regression metrics accepted
    @pytest.mark.parametrize("metric", ["mae", "mse", "rmse", "r2"])
    def test_regression_metrics_accepted(self, metric):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            problem_type=ProblemType.REGRESSION,
            target_column="age",
            feature_columns=["churn"],
            evaluation_plan=EvaluationPlan(primary_metric=metric),
        )
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert len(res.errors) == 0

    # 71. Invalid classification primary metric
    def test_invalid_classification_primary_metric(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(evaluation_plan=EvaluationPlan(primary_metric="mae"))
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "METRIC_INCOMPATIBLE" and e.location == "evaluation_plan.primary_metric" for e in res.errors)

    # 72. Invalid regression primary metric
    def test_invalid_regression_primary_metric(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            problem_type=ProblemType.REGRESSION,
            target_column="age",
            feature_columns=["churn"],
            evaluation_plan=EvaluationPlan(primary_metric="accuracy"),
        )
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "METRIC_INCOMPATIBLE" and e.location == "evaluation_plan.primary_metric" for e in res.errors)

    # 73. Invalid secondary metric
    def test_invalid_secondary_metric_rejected(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            evaluation_plan=EvaluationPlan(primary_metric="f1", secondary_metrics=["accuracy", "mae"])
        )
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "METRIC_INCOMPATIBLE" and e.location == "evaluation_plan.secondary_metrics[1]" for e in res.errors)

    # 74. Metric comparison is case-normalized without mutating plan
    def test_metric_comparison_case_normalized_and_no_mutation(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            evaluation_plan=EvaluationPlan(primary_metric="  F1_aUc  ")  # Wait, F1_aUc is not a valid classification metric, let's use "  F1  " or "  AcCuRaCy  "
        )
        plan.evaluation_plan.primary_metric = "  AcCuRaCy  "
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0
        assert plan.evaluation_plan.primary_metric == "  AcCuRaCy  "  # Unmutated

    # 75. Multiple invalid metrics preserve order
    def test_multiple_invalid_metrics_preserve_order(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            evaluation_plan=EvaluationPlan(primary_metric="mae", secondary_metrics=["rmse"])
        )
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        metric_errors = [e for e in res.errors if e.code == "METRIC_INCOMPATIBLE"]
        assert len(metric_errors) == 2
        assert "primary_metric" in metric_errors[0].location
        assert "secondary_metrics[0]" in metric_errors[1].location


class TestValidatorExecutionConstraints:
    """Scenarios 76-84: Hardware constraint consistency checks."""

    # 76. Matching compute tier accepted
    def test_matching_compute_tier(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan()
        cc = _make_compute_capabilities()
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), cc)
        assert len(res.errors) == 0

    # 77. Compute tier mismatch
    def test_compute_tier_mismatch_rejected(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            execution_constraints=ExecutionConstraints(
                parallel_workers=2,
                use_gpu_acceleration=False,
                accelerator_type=AcceleratorType.NONE,
                compute_tier=ComputeTier.HIGH,  # Mismatch with capability tier 'STANDARD'
            )
        )
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(e.code == "COMPUTE_TIER_MISMATCH" for e in res.errors)

    # 78. Safe worker count accepted
    def test_safe_worker_count(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            execution_constraints=ExecutionConstraints(
                parallel_workers=4,
                use_gpu_acceleration=False,
                accelerator_type=AcceleratorType.NONE,
                compute_tier=ComputeTier.STANDARD,
            )
        )
        cc = _make_compute_capabilities(safe_parallel_workers=4)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), cc)
        assert len(res.errors) == 0

    # 79. Worker count exceeding safe limit rejected
    def test_worker_count_exceeding_safe_limit_rejected(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            execution_constraints=ExecutionConstraints(
                parallel_workers=5,
                use_gpu_acceleration=False,
                accelerator_type=AcceleratorType.NONE,
                compute_tier=ComputeTier.STANDARD,
            )
        )
        cc = _make_compute_capabilities(safe_parallel_workers=4)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), cc)
        assert any(e.code == "PARALLEL_WORKERS_EXCEED_SAFE_LIMIT" for e in res.errors)

    # 80. Valid CPU execution
    def test_valid_cpu_execution(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan()
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 0

    # 81. Valid CUDA execution
    def test_valid_cuda_execution(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            execution_constraints=ExecutionConstraints(
                parallel_workers=2,
                use_gpu_acceleration=True,
                accelerator_type=AcceleratorType.CUDA,
                compute_tier=ComputeTier.STANDARD,
            )
        )
        cc = _make_compute_capabilities(gpu_acceleration_available=True, accelerator_type=AcceleratorType.CUDA)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), cc)
        assert len(res.errors) == 0

    # 82. GPU requested when unavailable
    def test_gpu_requested_when_unavailable_rejected(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            execution_constraints=ExecutionConstraints(
                parallel_workers=2,
                use_gpu_acceleration=True,
                accelerator_type=AcceleratorType.CUDA,
                compute_tier=ComputeTier.STANDARD,
            )
        )
        cc = _make_compute_capabilities(gpu_acceleration_available=False)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), cc)
        assert any(e.code == "GPU_ACCELERATION_UNAVAILABLE" for e in res.errors)

    # 83. Accelerator mismatch
    def test_accelerator_mismatch_rejected(self):
        val = MLPlanValidator()
        # CUDA requested but compute capabilities says another type or NONE
        # Note: Since Pydantic execution constraints requires consistency too, we construct a valid CUDA model
        plan = _make_valid_ml_plan(
            execution_constraints=ExecutionConstraints(
                parallel_workers=2,
                use_gpu_acceleration=True,
                accelerator_type=AcceleratorType.CUDA,
                compute_tier=ComputeTier.STANDARD,
            )
        )
        # Compute capabilities says gpu available but type is NONE or something else
        # Wait, ComputeCapabilities has model validator too! If gpu available is True, accelerator_type must NOT be NONE.
        # But wait, there is no other accelerator type than CUDA and NONE in Stage 6A.
        # How do we trigger ACCELERATOR_TYPE_MISMATCH without failing ComputeCapabilities schema validators?
        # Ah! ComputeCapabilities validation: if gpu_acceleration_available is True, accelerator_type must NOT be NONE.
        # What if we patch `compute_capabilities.accelerator_type` to NONE after construction?
        # Yes! Patching is very simple and doesn't trigger validators.
        cc = _make_compute_capabilities(gpu_acceleration_available=True, accelerator_type=AcceleratorType.CUDA)
        from unittest.mock import patch
        with patch.object(cc, "accelerator_type", AcceleratorType.NONE):
            res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), cc)
            assert any(e.code == "ACCELERATOR_TYPE_MISMATCH" for e in res.errors)

    # 84. CPU execution accepted even when GPU is available
    def test_cpu_execution_accepted_on_gpu_machine(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            execution_constraints=ExecutionConstraints(
                parallel_workers=2,
                use_gpu_acceleration=False,
                accelerator_type=AcceleratorType.NONE,
                compute_tier=ComputeTier.STANDARD,
            )
        )
        cc = _make_compute_capabilities(gpu_acceleration_available=True, accelerator_type=AcceleratorType.CUDA)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), cc)
        assert len(res.errors) == 0


class TestValidatorStatus:
    """Scenarios 85-89: Plan status warning rules."""

    # 85. Ready produces no status warning
    def test_ready_status_produces_no_warning(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(status=MLPlanStatus.READY)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.warnings) == 0

    # 86. Draft produces PLAN_STATUS_DRAFT
    def test_draft_status_warning(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(status=MLPlanStatus.DRAFT)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(w.code == "PLAN_STATUS_DRAFT" for w in res.warnings)

    # 87. Needs-confirmation produces PLAN_REQUIRES_CONFIRMATION
    def test_needs_confirmation_status_warning(self):
        val = MLPlanValidator()
        # Mock confirmation item to satisfy Pydantic schema validation for status
        item = MLPlanConfirmationItem(key="k", question="q", reason="r")
        plan = _make_valid_ml_plan(status=MLPlanStatus.NEEDS_CONFIRMATION, confirmation_items=[item])
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(w.code == "PLAN_REQUIRES_CONFIRMATION" for w in res.warnings)

    # 88. Blocked produces PLAN_STATUS_BLOCKED
    def test_blocked_status_warning(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(status=MLPlanStatus.BLOCKED)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert any(w.code == "PLAN_STATUS_BLOCKED" for w in res.warnings)

    # 89. Status warnings do not independently make is_valid=False
    def test_status_warnings_do_not_affect_is_valid(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(status=MLPlanStatus.DRAFT)
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.warnings) > 0
        assert res.is_valid is True  # No errors


class TestValidatorBehavior:
    """Scenarios 90-96: Multi-issue, deterministic order, non-mutation, repeatability."""

    # 90. Multiple semantic errors are collected
    def test_multiple_errors_collected(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(dataset_id="wrong_ds", request_id="wrong_req")
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) >= 2

    # 91. Validation order is deterministic
    def test_validation_order_is_deterministic(self):
        val = MLPlanValidator()
        # Trigger dataset ID mismatch (identity category) and metric incompatible (metric category)
        plan = _make_valid_ml_plan(
            dataset_id="wrong_ds",
            evaluation_plan=EvaluationPlan(primary_metric="mae")  # incompatible metric for classification
        )
        res = val.validate(plan, _make_dataset_context(), _make_problem_definition(), _make_compute_capabilities())
        assert len(res.errors) == 3
        # Identity category error must come first
        assert res.errors[0].code == "DATASET_ID_MISMATCH"
        assert res.errors[1].code == "PROBLEM_DATASET_ID_MISMATCH"
        assert res.errors[2].code == "METRIC_INCOMPATIBLE"

    # 92-95. Non-mutation of upstream artifacts
    def test_non_mutation_of_inputs(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan()
        ds = _make_dataset_context()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()

        orig_plan = copy.deepcopy(plan)
        orig_ds = copy.deepcopy(ds)
        orig_pd = copy.deepcopy(pd)
        orig_cc = copy.deepcopy(cc)

        val.validate(plan, ds, pd, cc)

        assert plan == orig_plan
        assert ds == orig_ds
        assert pd == orig_pd
        assert cc == orig_cc

    # 96. Repeated validation produces equivalent results
    def test_repeated_validation_equivalent(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(dataset_id="wrong_ds")
        ds = _make_dataset_context()
        pd = _make_problem_definition()
        cc = _make_compute_capabilities()

        res1 = val.validate(plan, ds, pd, cc)
        res2 = val.validate(plan, ds, pd, cc)

        assert res1.model_dump() == res2.model_dump()

    def test_invalid_types_issue_validation(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            MLPlanValidationIssue(code=123, message="msg", severity=ValidationSeverity.ERROR)
        with pytest.raises(ValidationError, match="location must be a string"):
            MLPlanValidationIssue(code="C", message="msg", severity=ValidationSeverity.ERROR, location=123)
        with pytest.raises(ValidationError, match="location cannot be empty"):
            MLPlanValidationIssue(code="C", message="msg", severity=ValidationSeverity.ERROR, location="   ")

    def test_invalid_types_result_validation(self):
        with pytest.raises(ValidationError, match="plan_id must be a string"):
            MLPlanValidationResult(plan_id=123, is_valid=True)
        with pytest.raises(ValidationError, match="plan_id cannot be empty"):
            MLPlanValidationResult(plan_id="   ", is_valid=True)

    def test_invalid_regression_secondary_metric(self):
        val = MLPlanValidator()
        plan = _make_valid_ml_plan(
            problem_type=ProblemType.REGRESSION,
            target_column="age",
            feature_columns=["churn"],
            evaluation_plan=EvaluationPlan(primary_metric="mae", secondary_metrics=["rmse", "accuracy"])
        )
        pd = _make_problem_definition(
            problem_type=ProblemType.REGRESSION, target_column="age", feature_columns=["churn"]
        )
        res = val.validate(plan, _make_dataset_context(), pd, _make_compute_capabilities())
        assert any(e.code == "METRIC_INCOMPATIBLE" and e.location == "evaluation_plan.secondary_metrics[1]" for e in res.errors)

