"""Tests for backend.app.ai_engine.prompt_builder — targeting 100 % coverage.

Covers:
* Full report (all sections populated)
* Report without imbalance section
* Prompt contains every allowed strategy option
* Prompt contains the JSON response template
* Edge cases: empty missing values, empty numerical summary
"""

from __future__ import annotations

import json

import pytest

from backend.app.ai_engine.prompt_builder import (
    DUPLICATE_STRATEGIES,
    ENCODING_STRATEGIES,
    MISSING_VALUE_STRATEGIES,
    RESPONSE_TEMPLATE_JSON,
    SCALING_STRATEGIES,
    PromptBuilder,
)
from backend.app.analysis.schemas import (
    DatasetAnalysisReport,
    DuplicateReport,
    ImbalanceReport,
    MissingValueReport,
    StatisticsReport,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def missing_values_report() -> MissingValueReport:
    return MissingValueReport(
        total_missing=5,
        missing_by_column={"age": 3, "salary": 2},
        missing_percentage={"age": 10.0, "salary": 6.7},
    )


@pytest.fixture()
def duplicate_report() -> DuplicateReport:
    return DuplicateReport(duplicate_rows=4, duplicate_percentage=8.0)


@pytest.fixture()
def statistics_report() -> StatisticsReport:
    return StatisticsReport(
        numerical_summary={
            "age": {"mean": 30.5, "std": 5.2, "min": 20.0, "max": 45.0},
            "salary": {"mean": 55000.0, "std": 12000.0, "min": 30000.0, "max": 90000.0},
        }
    )


@pytest.fixture()
def imbalance_report() -> ImbalanceReport:
    return ImbalanceReport(
        imbalanced=True,
        distribution={"yes": 150, "no": 850},
    )


@pytest.fixture()
def full_report(
    missing_values_report: MissingValueReport,
    duplicate_report: DuplicateReport,
    statistics_report: StatisticsReport,
    imbalance_report: ImbalanceReport,
) -> DatasetAnalysisReport:
    """Report with every section populated, including imbalance."""
    return DatasetAnalysisReport(
        missing_values=missing_values_report,
        duplicates=duplicate_report,
        statistics=statistics_report,
        imbalance=imbalance_report,
        column_profiles={},
    )


@pytest.fixture()
def report_without_imbalance(
    missing_values_report: MissingValueReport,
    duplicate_report: DuplicateReport,
    statistics_report: StatisticsReport,
) -> DatasetAnalysisReport:
    """Report without the optional imbalance section."""
    return DatasetAnalysisReport(
        missing_values=missing_values_report,
        duplicates=duplicate_report,
        statistics=statistics_report,
        imbalance=None,
        column_profiles={},
    )


@pytest.fixture()
def report_empty_missing_and_stats(
    duplicate_report: DuplicateReport,
) -> DatasetAnalysisReport:
    """Report where missing_by_column and numerical_summary are empty."""
    return DatasetAnalysisReport(
        missing_values=MissingValueReport(
            total_missing=0,
            missing_by_column={},
            missing_percentage={},
        ),
        duplicates=duplicate_report,
        statistics=StatisticsReport(numerical_summary={}),
        imbalance=None,
        column_profiles={},
    )


@pytest.fixture()
def builder() -> PromptBuilder:
    return PromptBuilder()


# ── Tests: full report ──────────────────────────────────────────────────────


class TestFullReport:
    """Prompt built from a report with all sections populated."""

    def test_contains_missing_values_section(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "--- Missing Values ---" in prompt
        assert "age: 3 missing (10.0%)" in prompt
        assert "salary: 2 missing (6.7%)" in prompt

    def test_contains_duplicates_section(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "--- Duplicates ---" in prompt
        assert "Duplicate rows: 4 (8.0%)" in prompt

    def test_contains_numerical_statistics(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "--- Numerical Statistics ---" in prompt
        assert "age:" in prompt
        assert "mean=30.5000" in prompt

    def test_contains_imbalance_section(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "--- Class Imbalance ---" in prompt
        assert "Imbalanced: True" in prompt
        assert "yes: 150" in prompt
        assert "no: 850" in prompt


# ── Tests: report without imbalance ─────────────────────────────────────────


class TestReportWithoutImbalance:
    """Prompt built from a report with ``imbalance=None``."""

    def test_no_imbalance_section(
        self, builder: PromptBuilder, report_without_imbalance: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(report_without_imbalance)
        assert "--- Class Imbalance ---" not in prompt

    def test_other_sections_still_present(
        self, builder: PromptBuilder, report_without_imbalance: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(report_without_imbalance)
        assert "--- Missing Values ---" in prompt
        assert "--- Duplicates ---" in prompt
        assert "--- Numerical Statistics ---" in prompt


# ── Tests: empty edge cases ────────────────────────────────────────────────


class TestEmptyEdgeCases:
    """Cover the fallback messages for empty maps."""

    def test_no_missing_values_message(
        self, builder: PromptBuilder, report_empty_missing_and_stats: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(report_empty_missing_and_stats)
        assert "No missing values detected." in prompt

    def test_no_numerical_columns_message(
        self, builder: PromptBuilder, report_empty_missing_and_stats: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(report_empty_missing_and_stats)
        assert "No numerical columns detected." in prompt


# ── Tests: strategy options in prompt ───────────────────────────────────────


class TestStrategyOptionsPresent:
    """Every allowed strategy must appear verbatim in the prompt."""

    def test_missing_value_strategies(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        for strategy in MISSING_VALUE_STRATEGIES:
            assert strategy in prompt, f"Missing strategy not found: {strategy}"

    def test_duplicate_strategies(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        for strategy in DUPLICATE_STRATEGIES:
            assert strategy in prompt, f"Duplicate strategy not found: {strategy}"

    def test_encoding_strategies(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        for strategy in ENCODING_STRATEGIES:
            assert strategy in prompt, f"Encoding strategy not found: {strategy}"

    def test_scaling_strategies(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        for strategy in SCALING_STRATEGIES:
            assert strategy in prompt, f"Scaling strategy not found: {strategy}"


# ── Tests: JSON response template ──────────────────────────────────────────


class TestResponseTemplate:
    """The prompt must include the JSON response template."""

    def test_contains_response_template(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert RESPONSE_TEMPLATE_JSON in prompt

    def test_template_is_valid_json(self) -> None:
        parsed = json.loads(RESPONSE_TEMPLATE_JSON)
        assert "cleaning_plan" in parsed
        assert "overall_reasoning" in parsed

    def test_contains_json_only_instructions(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "Return ONLY valid JSON" in prompt
        assert "Do NOT include markdown" in prompt
        assert "Do NOT include any explanations" in prompt


# ── Tests: role preamble ───────────────────────────────────────────────────


class TestRolePreamble:
    """The prompt should start with a role-setting instruction."""

    def test_contains_role(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "senior data scientist" in prompt
        assert "preprocessing plan" in prompt
