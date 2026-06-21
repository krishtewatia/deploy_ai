"""Tests for backend.app.eda.insight_generator — targeting 100% coverage.

Covers:
    1.  Prompt generation.
    2.  Successful insight generation.
    3.  Report file creation.
    4.  Dataset-specific directory creation.
    5.  Empty Groq response.
    6.  Invalid JSON response.
    7.  Missing fields.
    8.  Exception propagation.
    9.  Correct InsightReport creation.
    10. Serialization compatibility.
    11. Report directory property.
    12. Multiple dataset isolation.
    13. Field type validation (non-list field).
    14. Non-dict JSON response.
    15. Markdown code-fence stripping.
    16. Invalid groq_client type.
    17. Output directory creation failure.
    18. InsightGenerationError re-raise.
    19. Unexpected exception wrapping.

All tests mock the Groq API — no network calls, no API key required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.eda.insight_generator import InsightGenerator, InsightGenerationError
from backend.app.eda.insight_schemas import InsightReport
from backend.app.eda.schemas import (
    ChartType,
    DatasetSummary,
    DescriptiveAnalytics,
    DiagnosticAnalytics,
    Insight,
    Severity,
    VisualizationRecommendation,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _valid_llm_response() -> str:
    """Return a deterministic, valid JSON response matching the expected LLM format."""
    return json.dumps(
        {
            "descriptive_insights": [
                "The dataset has 1000 rows and 10 columns.",
                "Salary column is right-skewed.",
            ],
            "diagnostic_insights": [
                "Age and salary are positively correlated (r=0.85).",
            ],
            "predictive_observations": [
                "High cardinality in department may cause model overfitting.",
            ],
            "prescriptive_recommendations": [
                "Apply log transformation to salary.",
                "Remove duplicate rows before modelling.",
            ],
        }
    )


def _build_descriptive() -> DescriptiveAnalytics:
    """Build a sample DescriptiveAnalytics object."""
    return DescriptiveAnalytics(
        dataset_summary=DatasetSummary(
            rows=1000,
            columns=10,
            numerical_columns=["age", "salary"],
            categorical_columns=["department"],
            datetime_columns=["hire_date"],
            missing_cells=42,
            duplicate_rows=5,
        ),
        key_findings=[
            Insight(
                title="Salary Skewness",
                description="Salary column is right-skewed with skewness of 2.1.",
                severity=Severity.WARNING,
            ),
            Insight(
                title="Complete Dataset",
                description="No critical data quality issues.",
                severity=Severity.INFO,
            ),
        ],
    )


def _build_diagnostic() -> DiagnosticAnalytics:
    """Build a sample DiagnosticAnalytics object."""
    return DiagnosticAnalytics(
        correlation_findings=[
            Insight(
                title="Strong Correlation",
                description="Age and salary have a Pearson coefficient of 0.85.",
                severity=Severity.WARNING,
            ),
        ],
        anomaly_findings=[
            Insight(
                title="Outlier Detected",
                description="Salary contains 3 extreme outliers (> 3 std).",
                severity=Severity.CRITICAL,
            ),
        ],
    )


def _build_visualizations() -> list[VisualizationRecommendation]:
    """Build a sample list of visualization recommendations."""
    return [
        VisualizationRecommendation(
            chart_type=ChartType.HISTOGRAM,
            column_names=["salary"],
            reason="Visualize distribution and skewness.",
        ),
        VisualizationRecommendation(
            chart_type=ChartType.CORRELATION_HEATMAP,
            column_names=["age", "salary"],
            reason="Analyze feature relationships.",
        ),
    ]


def _mock_groq_client(return_value: str | None = None, side_effect: Exception | None = None) -> MagicMock:
    """Create a MagicMock that passes isinstance(_, GroqClient) checks."""
    from backend.app.ai_engine.groq_client import GroqClient

    mock = MagicMock(spec=GroqClient)
    if side_effect is not None:
        mock.generate_recommendations.side_effect = side_effect
    else:
        mock.generate_recommendations.return_value = (
            _valid_llm_response() if return_value is None else return_value
        )
    return mock


# ── 1. Prompt Generation ───────────────────────────────────────────────────


def test_prompt_contains_dataset_summary(tmp_path: Path) -> None:
    """The generated prompt should contain all dataset summary fields."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    descriptive = _build_descriptive()
    diagnostic = _build_diagnostic()
    visualizations = _build_visualizations()

    prompt = gen._build_prompt(descriptive, diagnostic, visualizations)

    assert "Rows: 1000" in prompt
    assert "Columns: 10" in prompt
    assert "age" in prompt
    assert "salary" in prompt
    assert "department" in prompt
    assert "hire_date" in prompt
    assert "Missing cells: 42" in prompt
    assert "Duplicate rows: 5" in prompt


def test_prompt_contains_descriptive_findings(tmp_path: Path) -> None:
    """The prompt should include descriptive key findings."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    prompt = gen._build_prompt(_build_descriptive(), _build_diagnostic(), _build_visualizations())

    assert "Salary Skewness" in prompt
    assert "right-skewed" in prompt


def test_prompt_contains_diagnostic_findings(tmp_path: Path) -> None:
    """The prompt should include diagnostic correlation and anomaly findings."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    prompt = gen._build_prompt(_build_descriptive(), _build_diagnostic(), _build_visualizations())

    assert "Strong Correlation" in prompt
    assert "Outlier Detected" in prompt
    assert "Correlation Findings" in prompt
    assert "Anomaly Findings" in prompt


def test_prompt_contains_visualizations(tmp_path: Path) -> None:
    """The prompt should include visualization recommendations."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    prompt = gen._build_prompt(_build_descriptive(), _build_diagnostic(), _build_visualizations())

    assert "histogram" in prompt
    assert "correlation_heatmap" in prompt
    assert "VISUALIZATION RECOMMENDATIONS" in prompt


def test_prompt_contains_response_format_instruction(tmp_path: Path) -> None:
    """The prompt should instruct the LLM to return JSON only."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    prompt = gen._build_prompt(_build_descriptive(), _build_diagnostic(), _build_visualizations())

    assert "JSON only" in prompt
    assert "descriptive_insights" in prompt
    assert "diagnostic_insights" in prompt
    assert "predictive_observations" in prompt
    assert "prescriptive_recommendations" in prompt


def test_prompt_with_empty_findings(tmp_path: Path) -> None:
    """The prompt should handle empty findings gracefully."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    descriptive = DescriptiveAnalytics(
        dataset_summary=DatasetSummary(
            rows=0, columns=0, missing_cells=0, duplicate_rows=0,
        ),
        key_findings=[],
    )
    diagnostic = DiagnosticAnalytics(
        correlation_findings=[],
        anomaly_findings=[],
    )

    prompt = gen._build_prompt(descriptive, diagnostic, [])

    assert "No descriptive findings." in prompt
    assert "No correlation findings." in prompt
    assert "No anomaly findings." in prompt
    assert "No visualization recommendations." in prompt


def test_prompt_role_instruction(tmp_path: Path) -> None:
    """The prompt should contain the LLM role instruction."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    prompt = gen._build_prompt(_build_descriptive(), _build_diagnostic(), _build_visualizations())

    assert "Senior Data Analyst" in prompt
    assert "Business Intelligence Consultant" in prompt
    assert "Machine Learning Engineer" in prompt


# ── 2. Successful Insight Generation ────────────────────────────────────────


def test_successful_insight_generation(tmp_path: Path) -> None:
    """generate_insights should return a valid InsightReport from a well-formed LLM response."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    report = gen.generate_insights(
        dataset_id="test_ds",
        descriptive=_build_descriptive(),
        diagnostic=_build_diagnostic(),
        visualizations=_build_visualizations(),
    )

    assert isinstance(report, InsightReport)
    assert report.dataset_id == "test_ds"
    assert len(report.descriptive_insights) == 2
    assert len(report.diagnostic_insights) == 1
    assert len(report.predictive_observations) == 1
    assert len(report.prescriptive_recommendations) == 2
    assert report.generated_at is not None

    # Verify the mock was called exactly once
    client.generate_recommendations.assert_called_once()


# ── 3. Report File Creation ────────────────────────────────────────────────


def test_report_file_creation(tmp_path: Path) -> None:
    """generate_insights should persist an insight_report.json file."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    gen.generate_insights(
        dataset_id="file_test",
        descriptive=_build_descriptive(),
        diagnostic=_build_diagnostic(),
        visualizations=_build_visualizations(),
    )

    report_path = tmp_path / "file_test" / "insight_report.json"
    assert report_path.exists()
    assert report_path.is_file()

    # Verify it contains valid JSON that can be deserialized
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "descriptive_insights" in data
    assert "dataset_id" in data
    assert data["dataset_id"] == "file_test"


# ── 4. Dataset-Specific Directory Creation ──────────────────────────────────


def test_dataset_directory_creation(tmp_path: Path) -> None:
    """A per-dataset subdirectory should be created under the base output dir."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    gen.generate_insights(
        dataset_id="customer_churn",
        descriptive=_build_descriptive(),
        diagnostic=_build_diagnostic(),
        visualizations=_build_visualizations(),
    )

    dataset_dir = tmp_path / "customer_churn"
    assert dataset_dir.exists()
    assert dataset_dir.is_dir()


# ── 5. Empty Groq Response ─────────────────────────────────────────────────


def test_empty_groq_response(tmp_path: Path) -> None:
    """An empty string from Groq should raise InsightGenerationError."""
    client = _mock_groq_client(return_value="")
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    with pytest.raises(InsightGenerationError, match="empty response"):
        gen.generate_insights(
            dataset_id="empty_test",
            descriptive=_build_descriptive(),
            diagnostic=_build_diagnostic(),
            visualizations=_build_visualizations(),
        )


def test_whitespace_only_groq_response(tmp_path: Path) -> None:
    """A whitespace-only string from Groq should raise InsightGenerationError."""
    client = _mock_groq_client(return_value="   \n  ")
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    with pytest.raises(InsightGenerationError, match="empty response"):
        gen.generate_insights(
            dataset_id="ws_test",
            descriptive=_build_descriptive(),
            diagnostic=_build_diagnostic(),
            visualizations=_build_visualizations(),
        )


# ── 6. Invalid JSON Response ──────────────────────────────────────────────


def test_invalid_json_response(tmp_path: Path) -> None:
    """Non-JSON text from Groq should raise InsightGenerationError."""
    client = _mock_groq_client(return_value="This is not JSON at all.")
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    with pytest.raises(InsightGenerationError, match="Failed to parse Groq response as JSON"):
        gen.generate_insights(
            dataset_id="bad_json",
            descriptive=_build_descriptive(),
            diagnostic=_build_diagnostic(),
            visualizations=_build_visualizations(),
        )


def test_non_dict_json_response(tmp_path: Path) -> None:
    """A JSON array (not object) from Groq should raise InsightGenerationError."""
    client = _mock_groq_client(return_value='["item1", "item2"]')
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    with pytest.raises(InsightGenerationError, match="Expected a JSON object"):
        gen.generate_insights(
            dataset_id="array_json",
            descriptive=_build_descriptive(),
            diagnostic=_build_diagnostic(),
            visualizations=_build_visualizations(),
        )


# ── 7. Missing Fields ─────────────────────────────────────────────────────


def test_missing_required_field(tmp_path: Path) -> None:
    """A JSON response missing a required field should raise InsightGenerationError."""
    incomplete = json.dumps({
        "descriptive_insights": ["ok"],
        # diagnostic_insights is missing
        "predictive_observations": ["ok"],
        "prescriptive_recommendations": ["ok"],
    })
    client = _mock_groq_client(return_value=incomplete)
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    with pytest.raises(InsightGenerationError, match="Missing required field.*diagnostic_insights"):
        gen.generate_insights(
            dataset_id="missing_field",
            descriptive=_build_descriptive(),
            diagnostic=_build_diagnostic(),
            visualizations=_build_visualizations(),
        )


def test_field_type_not_list(tmp_path: Path) -> None:
    """A field that is not a list should raise InsightGenerationError."""
    bad_type = json.dumps({
        "descriptive_insights": "not a list",
        "diagnostic_insights": ["ok"],
        "predictive_observations": ["ok"],
        "prescriptive_recommendations": ["ok"],
    })
    client = _mock_groq_client(return_value=bad_type)
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    with pytest.raises(InsightGenerationError, match="must be a list"):
        gen.generate_insights(
            dataset_id="bad_type",
            descriptive=_build_descriptive(),
            diagnostic=_build_diagnostic(),
            visualizations=_build_visualizations(),
        )


# ── 8. Exception Propagation ──────────────────────────────────────────────


def test_groq_exception_propagation(tmp_path: Path) -> None:
    """An exception from groq_client.generate_recommendations should be wrapped."""
    client = _mock_groq_client(side_effect=RuntimeError("API went boom"))
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    with pytest.raises(InsightGenerationError, match="Failed to generate insights"):
        gen.generate_insights(
            dataset_id="error_test",
            descriptive=_build_descriptive(),
            diagnostic=_build_diagnostic(),
            visualizations=_build_visualizations(),
        )


def test_insight_generation_error_reraise(tmp_path: Path) -> None:
    """An InsightGenerationError raised internally should be re-raised directly."""
    client = _mock_groq_client(return_value="")
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    with pytest.raises(InsightGenerationError, match="empty response"):
        gen.generate_insights(
            dataset_id="reraise_test",
            descriptive=_build_descriptive(),
            diagnostic=_build_diagnostic(),
            visualizations=_build_visualizations(),
        )


# ── 9. Correct InsightReport Creation ──────────────────────────────────────


def test_insight_report_fields(tmp_path: Path) -> None:
    """The InsightReport should have correct field values from the LLM response."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    report = gen.generate_insights(
        dataset_id="fields_test",
        descriptive=_build_descriptive(),
        diagnostic=_build_diagnostic(),
        visualizations=_build_visualizations(),
    )

    assert report.descriptive_insights[0] == "The dataset has 1000 rows and 10 columns."
    assert report.descriptive_insights[1] == "Salary column is right-skewed."
    assert report.diagnostic_insights[0] == "Age and salary are positively correlated (r=0.85)."
    assert report.predictive_observations[0] == "High cardinality in department may cause model overfitting."
    assert report.prescriptive_recommendations[0] == "Apply log transformation to salary."
    assert report.prescriptive_recommendations[1] == "Remove duplicate rows before modelling."


# ── 10. Serialization Compatibility ────────────────────────────────────────


def test_serialization_compatibility(tmp_path: Path) -> None:
    """InsightReport should serialize and deserialize cleanly."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    report = gen.generate_insights(
        dataset_id="serial_test",
        descriptive=_build_descriptive(),
        diagnostic=_build_diagnostic(),
        visualizations=_build_visualizations(),
    )

    # model_dump round-trip
    data = report.model_dump()
    reconstructed = InsightReport.model_validate(data)
    assert reconstructed.dataset_id == report.dataset_id
    assert reconstructed.descriptive_insights == report.descriptive_insights

    # JSON round-trip
    json_str = report.model_dump_json()
    from_json = InsightReport.model_validate_json(json_str)
    assert from_json.dataset_id == report.dataset_id
    assert from_json.descriptive_insights == report.descriptive_insights


# ── 11. Report Directory Property ──────────────────────────────────────────


def test_report_directory_property(tmp_path: Path) -> None:
    """The report_directory property should return the resolved base output directory."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    assert isinstance(gen.report_directory, Path)
    assert gen.report_directory == tmp_path.resolve()


# ── 12. Multiple Dataset Isolation ─────────────────────────────────────────


def test_multiple_dataset_isolation(tmp_path: Path) -> None:
    """Reports from different datasets should be stored in separate directories."""
    client = _mock_groq_client()
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    gen.generate_insights(
        dataset_id="dataset_A",
        descriptive=_build_descriptive(),
        diagnostic=_build_diagnostic(),
        visualizations=_build_visualizations(),
    )
    gen.generate_insights(
        dataset_id="dataset_B",
        descriptive=_build_descriptive(),
        diagnostic=_build_diagnostic(),
        visualizations=_build_visualizations(),
    )

    report_a = tmp_path / "dataset_A" / "insight_report.json"
    report_b = tmp_path / "dataset_B" / "insight_report.json"

    assert report_a.exists()
    assert report_b.exists()

    data_a = json.loads(report_a.read_text(encoding="utf-8"))
    data_b = json.loads(report_b.read_text(encoding="utf-8"))

    assert data_a["dataset_id"] == "dataset_A"
    assert data_b["dataset_id"] == "dataset_B"


# ── 13. Markdown Code-Fence Stripping ──────────────────────────────────────


def test_markdown_code_fence_stripping(tmp_path: Path) -> None:
    """The parser should strip markdown code fences if the LLM wraps JSON in them."""
    wrapped = "```json\n" + _valid_llm_response() + "\n```"
    client = _mock_groq_client(return_value=wrapped)
    gen = InsightGenerator(groq_client=client, output_dir=str(tmp_path))

    report = gen.generate_insights(
        dataset_id="fence_test",
        descriptive=_build_descriptive(),
        diagnostic=_build_diagnostic(),
        visualizations=_build_visualizations(),
    )

    assert isinstance(report, InsightReport)
    assert len(report.descriptive_insights) == 2


# ── 14. Invalid GroqClient Type ────────────────────────────────────────────


def test_invalid_groq_client_type(tmp_path: Path) -> None:
    """Passing a non-GroqClient should raise InsightGenerationError."""
    with pytest.raises(InsightGenerationError, match="Expected a GroqClient instance"):
        InsightGenerator(groq_client="not a client", output_dir=str(tmp_path))  # type: ignore


# ── 15. Output Directory Creation Failure ──────────────────────────────────


def test_output_directory_creation_failure() -> None:
    """An OSError during directory creation should raise InsightGenerationError."""
    client = _mock_groq_client()
    with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(InsightGenerationError, match="Failed to create output directory"):
            InsightGenerator(groq_client=client, output_dir="/invalid/path")
