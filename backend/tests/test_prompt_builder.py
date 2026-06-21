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
    REQUIRED_RESPONSE_SCHEMA_JSON,
    RESPONSE_EXAMPLE_EMPTY_ARRAYS_JSON,
    RESPONSE_EXAMPLE_POPULATED_JSON,
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
        column_profiles={
            "age": {
                "dtype": "int64",
                "unique_values": 20,
                "unique_percentage": 66.67,
                "missing_count": 3,
                "missing_percentage": 10.0,
                "sample_values": [25, 30, 45],
                "is_numeric": True,
                "is_categorical": False,
                "is_datetime": False,
            },
            "department": {
                "dtype": "object",
                "unique_values": 3,
                "unique_percentage": 10.0,
                "missing_count": 0,
                "missing_percentage": 0.0,
                "sample_values": ["Sales", "Engineering", "HR"],
                "is_numeric": False,
                "is_categorical": True,
                "is_datetime": False,
            },
        },
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
        column_profiles={
            "age": {
                "dtype": "int64",
                "unique_values": 20,
                "unique_percentage": 66.67,
                "missing_count": 3,
                "missing_percentage": 10.0,
                "sample_values": [25, 30, 45],
                "is_numeric": True,
                "is_categorical": False,
                "is_datetime": False,
            },
            "department": {
                "dtype": "object",
                "unique_values": 3,
                "unique_percentage": 10.0,
                "missing_count": 0,
                "missing_percentage": 0.0,
                "sample_values": ["Sales", "Engineering", "HR"],
                "is_numeric": False,
                "is_categorical": True,
                "is_datetime": False,
            },
        },
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

    def test_contains_dataset_overview(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "DATASET OVERVIEW" in prompt
        assert "* File name: Unknown" in prompt
        assert "* Number of rows: 30" in prompt
        assert "* Number of columns: 2" in prompt
        assert '* Column names: ["age", "department"]' in prompt
        assert "* Dataset shape: [30, 2]" in prompt

    def test_contains_column_profiles(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "COLUMN PROFILES" in prompt
        assert "* Column name: age" in prompt
        assert "  * Data type: int64" in prompt
        assert "  * Missing count: 3" in prompt
        assert "  * Missing percentage: 10.0" in prompt
        assert "  * Unique values count: 20" in prompt
        assert "  * Unique percentage: 66.67" in prompt
        assert "  * Sample values: [25, 30, 45]" in prompt
        assert "  * is_numeric: True" in prompt
        assert "  * is_categorical: False" in prompt
        assert "  * is_datetime: False" in prompt

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

    def test_imbalance_section_notes_missing_report(
        self, builder: PromptBuilder, report_without_imbalance: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(report_without_imbalance)
        assert "--- Class Imbalance ---" in prompt
        assert "No class imbalance report was provided." in prompt

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
        assert "potential_identifier_columns" in parsed
        assert "potential_target_columns" in parsed
        assert "potential_leakage_columns" in parsed
        assert "drop_recommendations" in parsed
        assert "overall_reasoning" in parsed

    def test_contains_json_only_instructions(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "Return ONLY valid JSON" in prompt
        assert "Do NOT include markdown" in prompt
        assert "Do NOT include any explanations" in prompt

    def test_contains_mandatory_field_instructions(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "Every field shown in the schema is mandatory." in prompt
        assert "If no values exist for an array field, return an empty array." in prompt
        assert "Never omit a field." in prompt
        assert "The response must contain all schema keys exactly as defined." in prompt

    def test_required_top_level_schema_is_in_prompt(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        parsed = json.loads(REQUIRED_RESPONSE_SCHEMA_JSON)
        expected_keys = [
            "cleaning_plan",
            "potential_identifier_columns",
            "potential_target_columns",
            "potential_leakage_columns",
            "drop_recommendations",
            "overall_reasoning",
        ]
        assert list(parsed) == expected_keys
        assert REQUIRED_RESPONSE_SCHEMA_JSON in prompt

    def test_response_examples_include_all_required_fields(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        required_keys = set(json.loads(REQUIRED_RESPONSE_SCHEMA_JSON))

        populated = json.loads(RESPONSE_EXAMPLE_POPULATED_JSON)
        empty_arrays = json.loads(RESPONSE_EXAMPLE_EMPTY_ARRAYS_JSON)

        assert set(populated) == required_keys
        assert set(empty_arrays) == required_keys
        assert populated["potential_identifier_columns"]
        assert empty_arrays["potential_identifier_columns"] == []
        assert empty_arrays["potential_target_columns"] == []
        assert empty_arrays["potential_leakage_columns"] == []
        assert empty_arrays["drop_recommendations"] == []
        assert RESPONSE_EXAMPLE_POPULATED_JSON in prompt
        assert RESPONSE_EXAMPLE_EMPTY_ARRAYS_JSON in prompt


# ── Tests: role preamble ───────────────────────────────────────────────────


class TestRolePreamble:
    """The prompt should start with a role-setting instruction."""

    def test_contains_role(
        self, builder: PromptBuilder, full_report: DatasetAnalysisReport
    ) -> None:
        prompt = builder.build_prompt(full_report)
        assert "senior data scientist" in prompt
        assert "preprocessing plan" in prompt
