"""Tests for backend.app.ai_engine.recommendation_service — 100 % coverage.

All tests use mocks; **no real Groq API calls are made**.

Covers:
* Successful end-to-end recommendation generation
* PromptBuilder failure → RecommendationServiceError
* GroqClient failure   → RecommendationServiceError
* PlanParser failure   → RecommendationServiceError
* Unexpected exception → RecommendationServiceError
* Dependency injection of all three collaborators
* Default construction (via patching)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.ai_engine.groq_client import GroqClientError
from backend.app.ai_engine.plan_parser import InvalidJSONError
from backend.app.ai_engine.recommendation_service import (
    RecommendationService,
    RecommendationServiceError,
)
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


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def analysis_report() -> DatasetAnalysisReport:
    return DatasetAnalysisReport(
        missing_values=MissingValueReport(
            total_missing=1,
            missing_by_column={"age": 1},
            missing_percentage={"age": 5.0},
        ),
        duplicates=DuplicateReport(duplicate_rows=0, duplicate_percentage=0.0),
        statistics=StatisticsReport(
            numerical_summary={"age": {"mean": 30.0, "std": 5.0}}
        ),
        imbalance=None,
        column_profiles={},
    )


@pytest.fixture()
def mock_recommendation() -> RecommendationResponse:
    return RecommendationResponse(
        cleaning_plan=CleaningPlan(
            missing_values={
                "age": ColumnRecommendation(
                    strategy="median_imputation",
                    reason="Contains outliers.",
                )
            },
            duplicates_action="remove_duplicates",
            encoding={},
            scaling={},
        ),
        overall_reasoning="Recommended preprocessing plan.",
    )


@pytest.fixture()
def mock_prompt_builder() -> MagicMock:
    builder = MagicMock()
    builder.build_prompt.return_value = "mock prompt"
    return builder


@pytest.fixture()
def mock_groq_client() -> MagicMock:
    client = MagicMock()
    client.generate_recommendations.return_value = '{"mock": "json"}'
    return client


@pytest.fixture()
def mock_plan_parser(mock_recommendation: RecommendationResponse) -> MagicMock:
    parser = MagicMock()
    parser.parse.return_value = mock_recommendation
    return parser


# ── Successful generation ──────────────────────────────────────────────────


class TestSuccessfulGeneration:
    """End-to-end happy path with injected mocks."""

    def test_returns_recommendation_response(
        self,
        analysis_report: DatasetAnalysisReport,
        mock_prompt_builder: MagicMock,
        mock_groq_client: MagicMock,
        mock_plan_parser: MagicMock,
        mock_recommendation: RecommendationResponse,
    ) -> None:
        service = RecommendationService(
            prompt_builder=mock_prompt_builder,
            groq_client=mock_groq_client,
            plan_parser=mock_plan_parser,
        )

        result = service.generate_recommendations(analysis_report)

        assert result is mock_recommendation
        mock_prompt_builder.build_prompt.assert_called_once_with(analysis_report)
        mock_groq_client.generate_recommendations.assert_called_once_with("mock prompt")
        mock_plan_parser.parse.assert_called_once_with('{"mock": "json"}')


# ── PromptBuilder failure ──────────────────────────────────────────────────


class TestPromptBuilderFailure:
    """Errors in PromptBuilder are wrapped in RecommendationServiceError."""

    def test_prompt_builder_exception(
        self,
        analysis_report: DatasetAnalysisReport,
        mock_groq_client: MagicMock,
        mock_plan_parser: MagicMock,
    ) -> None:
        builder = MagicMock()
        builder.build_prompt.side_effect = RuntimeError("prompt build failed")

        service = RecommendationService(
            prompt_builder=builder,
            groq_client=mock_groq_client,
            plan_parser=mock_plan_parser,
        )

        with pytest.raises(RecommendationServiceError, match="Unexpected error"):
            service.generate_recommendations(analysis_report)


# ── GroqClient failure ─────────────────────────────────────────────────────


class TestGroqClientFailure:
    """GroqClientError is wrapped in RecommendationServiceError."""

    def test_groq_client_error(
        self,
        analysis_report: DatasetAnalysisReport,
        mock_prompt_builder: MagicMock,
        mock_plan_parser: MagicMock,
    ) -> None:
        client = MagicMock()
        client.generate_recommendations.side_effect = GroqClientError("API down")

        service = RecommendationService(
            prompt_builder=mock_prompt_builder,
            groq_client=client,
            plan_parser=mock_plan_parser,
        )

        with pytest.raises(RecommendationServiceError, match="Failed to generate"):
            service.generate_recommendations(analysis_report)


# ── PlanParser failure ─────────────────────────────────────────────────────


class TestPlanParserFailure:
    """PlanParserError is wrapped in RecommendationServiceError."""

    def test_plan_parser_error(
        self,
        analysis_report: DatasetAnalysisReport,
        mock_prompt_builder: MagicMock,
        mock_groq_client: MagicMock,
    ) -> None:
        parser = MagicMock()
        parser.parse.side_effect = InvalidJSONError("bad json")

        service = RecommendationService(
            prompt_builder=mock_prompt_builder,
            groq_client=mock_groq_client,
            plan_parser=parser,
        )

        with pytest.raises(RecommendationServiceError, match="Failed to generate"):
            service.generate_recommendations(analysis_report)


# ── Dependency injection ───────────────────────────────────────────────────


class TestDependencyInjection:
    """Verify that injected dependencies are actually used."""

    def test_custom_dependencies_are_stored(
        self,
        mock_prompt_builder: MagicMock,
        mock_groq_client: MagicMock,
        mock_plan_parser: MagicMock,
    ) -> None:
        service = RecommendationService(
            prompt_builder=mock_prompt_builder,
            groq_client=mock_groq_client,
            plan_parser=mock_plan_parser,
        )
        assert service._prompt_builder is mock_prompt_builder
        assert service._groq_client is mock_groq_client
        assert service._plan_parser is mock_plan_parser

    @patch("backend.app.ai_engine.recommendation_service.PlanParser")
    @patch("backend.app.ai_engine.recommendation_service.GroqClient")
    @patch("backend.app.ai_engine.recommendation_service.PromptBuilder")
    def test_defaults_created_when_none(
        self,
        mock_pb_cls: MagicMock,
        mock_gc_cls: MagicMock,
        mock_pp_cls: MagicMock,
    ) -> None:
        service = RecommendationService()
        mock_pb_cls.assert_called_once()
        mock_gc_cls.assert_called_once()
        mock_pp_cls.assert_called_once()
        assert service._prompt_builder is mock_pb_cls.return_value
        assert service._groq_client is mock_gc_cls.return_value
        assert service._plan_parser is mock_pp_cls.return_value
