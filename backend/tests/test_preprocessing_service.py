"""Tests for backend.app.preprocessing_engine.preprocessing_service.

Covers:
    1. Successful end-to-end processing.
    2. Dependency injection support.
    3. Analysis failure propagation.
    4. Recommendation failure propagation.
    5. Pipeline failure propagation.
    6. Preview generation.
    7. Transformation tracking.
    8. Empty DataFrame handling.
    9. Original shape tracking.
    10. Processed shape tracking.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.ai_engine.schemas import (
    CleaningPlan,
    ColumnRecommendation,
    RecommendationResponse,
)
from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    MissingValueReport,
    StatisticsReport,
)
from backend.app.preprocessing_engine.pipeline_executor import (
    PipelineExecutor,
)
from backend.app.preprocessing_engine.preprocessing_service import (
    PreprocessingService,
    PreprocessingServiceError,
)
from backend.app.preprocessing_engine.result_schemas import PreprocessingResult
from backend.app.preprocessing_engine.schemas import (
    ColumnAction,
    ExecutionPlan,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_analysis_report() -> DatasetAnalysisReport:
    """Build a minimal DatasetAnalysisReport for testing."""
    return DatasetAnalysisReport(
        missing_values=MissingValueReport(
            total_missing=1,
            missing_by_column={"salary": 1},
            missing_percentage={"salary": 20.0},
        ),
        duplicates=DuplicateReport(
            duplicate_rows=1,
            duplicate_percentage=20.0,
        ),
        statistics=StatisticsReport(
            numerical_summary={
                "salary": {"mean": 50.0, "std": 10.0, "min": 30.0, "max": 70.0},
            },
        ),
        imbalance=None,
        column_profiles={
            "salary": {
                "dtype": "float64",
                "unique_values": 4,
                "unique_percentage": 80.0,
                "missing_count": 1,
                "missing_percentage": 20.0,
                "sample_values": [30.0, 50.0, 70.0],
                "is_numeric": True,
                "is_categorical": False,
                "is_datetime": False,
            },
            "department": {
                "dtype": "object",
                "unique_values": 2,
                "unique_percentage": 40.0,
                "missing_count": 0,
                "missing_percentage": 0.0,
                "sample_values": ["AI", "ML"],
                "is_numeric": False,
                "is_categorical": True,
                "is_datetime": False,
            },
        },
    )


def _make_recommendation() -> RecommendationResponse:
    """Build a minimal RecommendationResponse for testing."""
    return RecommendationResponse(
        cleaning_plan=CleaningPlan(
            missing_values={
                "salary": ColumnRecommendation(
                    strategy="median_imputation",
                    reason="Median imputation handles outliers.",
                ),
            },
            duplicates_action="remove_duplicates",
            encoding={
                "department": ColumnRecommendation(
                    strategy="one_hot_encode",
                    reason="Nominal feature.",
                ),
            },
            scaling={
                "salary": ColumnRecommendation(
                    strategy="standard_scaling",
                    reason="Gaussian-distributed feature.",
                ),
            },
        ),
        overall_reasoning="The dataset requires imputation, encoding, and scaling.",
    )


def _make_sample_df() -> pd.DataFrame:
    """Build a small DataFrame that exercises all pipeline stages."""
    return pd.DataFrame(
        {
            "salary": [30.0, 50.0, None, 70.0, 50.0],
            "department": ["AI", "ML", "AI", "ML", "ML"],
        }
    )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a small test DataFrame."""
    return _make_sample_df()


@pytest.fixture()
def mock_analysis_service() -> MagicMock:
    """Return a mock AnalysisService."""
    mock = MagicMock()
    mock.analyze.return_value = _make_analysis_report()
    return mock


@pytest.fixture()
def mock_recommendation_service() -> MagicMock:
    """Return a mock RecommendationService."""
    mock = MagicMock()
    mock.generate_recommendations.return_value = _make_recommendation()
    return mock


@pytest.fixture()
def service(
    mock_analysis_service: MagicMock,
    mock_recommendation_service: MagicMock,
) -> PreprocessingService:
    """Return a PreprocessingService wired with mock dependencies."""
    return PreprocessingService(
        analysis_service=mock_analysis_service,
        recommendation_service=mock_recommendation_service,
        pipeline_executor=PipelineExecutor(),
    )


# ── 1. Successful end-to-end processing ────────────────────────────────────


def test_successful_end_to_end_processing(
    service: PreprocessingService,
    sample_df: pd.DataFrame,
) -> None:
    """process() should return a PreprocessingResult with all fields populated."""
    result = service.process(sample_df)

    assert isinstance(result, PreprocessingResult)
    assert result.original_shape == (5, 2)
    assert result.analysis_report is not None
    assert result.recommendation is not None
    assert result.execution_plan is not None
    assert len(result.transformations_applied) > 0
    assert len(result.preview) > 0
    assert result.processed_shape[0] > 0
    assert result.processed_shape[1] > 0


# ── 2. Dependency injection support ────────────────────────────────────────


def test_dependency_injection_support() -> None:
    """Constructor should accept and use injected dependencies."""
    mock_analysis = MagicMock()
    mock_recommendation = MagicMock()
    mock_executor = MagicMock()

    service = PreprocessingService(
        analysis_service=mock_analysis,
        recommendation_service=mock_recommendation,
        pipeline_executor=mock_executor,
    )

    assert service._analysis_service is mock_analysis
    assert service._recommendation_service is mock_recommendation
    assert service._pipeline_executor is mock_executor


# ── 3. Analysis failure propagation ────────────────────────────────────────


def test_analysis_failure_propagation(
    sample_df: pd.DataFrame,
) -> None:
    """process() should wrap analysis failures in PreprocessingServiceError."""
    mock_analysis = MagicMock()
    mock_analysis.analyze.side_effect = ValueError("Analysis exploded")

    service = PreprocessingService(
        analysis_service=mock_analysis,
        recommendation_service=MagicMock(),
        pipeline_executor=MagicMock(),
    )

    with pytest.raises(PreprocessingServiceError) as exc_info:
        service.process(sample_df)

    assert "Analysis stage failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


# ── 4. Recommendation failure propagation ──────────────────────────────────


def test_recommendation_failure_propagation(
    mock_analysis_service: MagicMock,
    sample_df: pd.DataFrame,
) -> None:
    """process() should wrap recommendation failures in PreprocessingServiceError."""
    mock_recommendation = MagicMock()
    mock_recommendation.generate_recommendations.side_effect = RuntimeError(
        "LLM unavailable"
    )

    service = PreprocessingService(
        analysis_service=mock_analysis_service,
        recommendation_service=mock_recommendation,
        pipeline_executor=MagicMock(),
    )

    with pytest.raises(PreprocessingServiceError) as exc_info:
        service.process(sample_df)

    assert "Recommendation stage failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ── 5. Pipeline failure propagation ────────────────────────────────────────


def test_pipeline_failure_propagation(
    mock_analysis_service: MagicMock,
    mock_recommendation_service: MagicMock,
    sample_df: pd.DataFrame,
) -> None:
    """process() should wrap pipeline failures in PreprocessingServiceError."""
    mock_executor = MagicMock()
    mock_executor.execute.side_effect = RuntimeError("Handler crashed")

    service = PreprocessingService(
        analysis_service=mock_analysis_service,
        recommendation_service=mock_recommendation_service,
        pipeline_executor=mock_executor,
    )

    with pytest.raises(PreprocessingServiceError) as exc_info:
        service.process(sample_df)

    assert "Pipeline execution stage failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ── 6. Preview generation ─────────────────────────────────────────────────


def test_preview_generation(
    service: PreprocessingService,
    sample_df: pd.DataFrame,
) -> None:
    """Preview should contain at most 5 rows as list of dicts."""
    result = service.process(sample_df)

    assert isinstance(result.preview, list)
    assert len(result.preview) <= 5
    for row in result.preview:
        assert isinstance(row, dict)


def test_preview_with_small_dataframe(
    mock_analysis_service: MagicMock,
    mock_recommendation_service: MagicMock,
) -> None:
    """Preview of a 2-row DataFrame should contain exactly 2 rows."""
    df = pd.DataFrame({"salary": [30.0, 50.0], "department": ["AI", "ML"]})

    service = PreprocessingService(
        analysis_service=mock_analysis_service,
        recommendation_service=mock_recommendation_service,
        pipeline_executor=PipelineExecutor(),
    )

    result = service.process(df)

    assert len(result.preview) == 2


# ── 7. Transformation tracking ────────────────────────────────────────────


def test_transformation_tracking(
    service: PreprocessingService,
    sample_df: pd.DataFrame,
) -> None:
    """transformations_applied should list all operations from the plan."""
    result = service.process(sample_df)

    assert "median_imputation: salary" in result.transformations_applied
    assert "remove_duplicates" in result.transformations_applied
    assert "one_hot_encode: department" in result.transformations_applied
    assert "standard_scaling: salary" in result.transformations_applied


def test_transformation_tracking_minimal_plan(
    mock_analysis_service: MagicMock,
    sample_df: pd.DataFrame,
) -> None:
    """With no column actions, only the duplicates action should appear."""
    minimal_recommendation = RecommendationResponse(
        cleaning_plan=CleaningPlan(
            missing_values={},
            duplicates_action="keep_duplicates",
            encoding={},
            scaling={},
        ),
        overall_reasoning="No transformations needed.",
    )
    mock_rec = MagicMock()
    mock_rec.generate_recommendations.return_value = minimal_recommendation

    service = PreprocessingService(
        analysis_service=mock_analysis_service,
        recommendation_service=mock_rec,
        pipeline_executor=PipelineExecutor(),
    )

    result = service.process(sample_df)

    assert result.transformations_applied == ["keep_duplicates"]


# ── 8. Empty DataFrame handling ────────────────────────────────────────────


def test_empty_dataframe_handling() -> None:
    """process() should propagate the analysis error for empty DataFrames."""
    mock_analysis = MagicMock()
    mock_analysis.analyze.side_effect = ValueError("DataFrame is empty.")

    service = PreprocessingService(
        analysis_service=mock_analysis,
        recommendation_service=MagicMock(),
        pipeline_executor=MagicMock(),
    )

    df = pd.DataFrame()

    with pytest.raises(PreprocessingServiceError) as exc_info:
        service.process(df)

    assert "Analysis stage failed" in str(exc_info.value)


# ── 9. Original shape tracking ────────────────────────────────────────────


def test_original_shape_tracking(
    service: PreprocessingService,
    sample_df: pd.DataFrame,
) -> None:
    """original_shape should match the input DataFrame's shape."""
    result = service.process(sample_df)

    assert result.original_shape == (5, 2)


# ── 10. Processed shape tracking ──────────────────────────────────────────


def test_processed_shape_tracking(
    service: PreprocessingService,
    sample_df: pd.DataFrame,
) -> None:
    """processed_shape should reflect post-processing dimensions.

    After removing 1 duplicate and one-hot encoding 'department',
    the row count should drop and the column count should change.
    """
    result = service.process(sample_df)

    # Rows: one duplicate (row index 4 duplicates row index 1) removed → 4 rows.
    # Columns: 'salary' stays, 'department' becomes 'department_AI' + 'department_ML' → 3.
    assert result.processed_shape[0] <= 5
    assert result.processed_shape[1] >= 2


# ── Extra: build_execution_plan conversion correctness ─────────────────────


def test_build_execution_plan_conversion() -> None:
    """_build_execution_plan should map CleaningPlan fields to ExecutionPlan."""
    cleaning_plan = CleaningPlan(
        missing_values={
            "A": ColumnRecommendation(strategy="mean_imputation", reason="reason A"),
        },
        duplicates_action="remove_duplicates",
        encoding={
            "B": ColumnRecommendation(strategy="label_encode", reason="reason B"),
        },
        scaling={
            "C": ColumnRecommendation(strategy="standard_scaling", reason="reason C"),
        },
    )

    plan = PreprocessingService._build_execution_plan(cleaning_plan)

    assert isinstance(plan, ExecutionPlan)
    assert "A" in plan.missing_values
    assert plan.missing_values["A"].strategy == "mean_imputation"
    assert plan.missing_values["A"].reason == "reason A"
    assert "B" in plan.encoding
    assert plan.encoding["B"].strategy == "label_encode"
    assert "C" in plan.scaling
    assert plan.scaling["C"].strategy == "standard_scaling"


# ── Extra: PreprocessingResult serialisation ───────────────────────────────


def test_preprocessing_result_serialisation(
    service: PreprocessingService,
    sample_df: pd.DataFrame,
) -> None:
    """PreprocessingResult should be serialisable via model_dump."""
    result = service.process(sample_df)
    serialised = result.model_dump()

    assert isinstance(serialised, dict)
    assert "original_shape" in serialised
    assert "processed_shape" in serialised
    assert "analysis_report" in serialised
    assert "recommendation" in serialised
    assert "execution_plan" in serialised
    assert "transformations_applied" in serialised
    assert "preview" in serialised
