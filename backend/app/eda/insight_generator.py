"""AI-powered insight generation engine for the EDA module.

Transforms deterministic EDA findings into professional, human-readable
analytics by sending structured prompts to the Groq LLM.  The generator
produces four categories of insights:

1. **Descriptive** — patterns, data quality, distributions.
2. **Diagnostic** — causes, correlations, anomalies.
3. **Predictive** — risks, trends, modelling concerns.
4. **Prescriptive** — cleaning, engineering, business recommendations.

Reports are persisted as JSON under a dataset-specific directory so that
multiple datasets never collide.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.ai_engine.groq_client import GroqClient
from backend.app.eda.insight_schemas import InsightReport
from backend.app.eda.schemas import (
    DescriptiveAnalytics,
    DiagnosticAnalytics,
    VisualizationRecommendation,
)

logger = logging.getLogger(__name__)


# ── Custom exception ───────────────────────────────────────────────────────


class InsightGenerationError(Exception):
    """Raised when insight generation, parsing, or persistence fails."""


# ── Generator ──────────────────────────────────────────────────────────────


class InsightGenerator:
    """Generate AI-powered insight reports from EDA findings.

    Parameters
    ----------
    groq_client : GroqClient
        A pre-configured :class:`GroqClient` instance used for LLM
        inference.
    output_dir : str
        Base directory for persisted insight reports.
    """

    _REQUIRED_FIELDS: tuple[str, ...] = (
        "descriptive_insights",
        "diagnostic_insights",
        "predictive_observations",
        "prescriptive_recommendations",
    )

    def __init__(
        self,
        groq_client: GroqClient,
        output_dir: str = "reports/insights",
    ) -> None:
        """Initialize the InsightGenerator and create the base output directory.

        Parameters
        ----------
        groq_client : GroqClient
            The Groq API client used for LLM inference.
        output_dir : str
            The base directory path where insight reports will be saved.
        """
        if not isinstance(groq_client, GroqClient):
            raise InsightGenerationError(
                f"Expected a GroqClient instance, got {type(groq_client).__name__}."
            )
        self._groq_client = groq_client
        self._base_output_dir = Path(output_dir).resolve()
        try:
            self._base_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Insight output directory ready: %s", self._base_output_dir)
        except Exception as exc:
            raise InsightGenerationError(
                f"Failed to create output directory '{output_dir}': {exc}"
            ) from exc

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def report_directory(self) -> Path:
        """Get the base directory path where insight reports are saved."""
        return self._base_output_dir

    # ── Public API ──────────────────────────────────────────────────────

    def generate_insights(
        self,
        dataset_id: str,
        descriptive: DescriptiveAnalytics,
        diagnostic: DiagnosticAnalytics,
        visualizations: list[VisualizationRecommendation],
    ) -> InsightReport:
        """Generate an AI-powered insight report for the given dataset findings.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset.
        descriptive : DescriptiveAnalytics
            Descriptive analytics (summary + key findings).
        diagnostic : DiagnosticAnalytics
            Diagnostic analytics (correlations + anomalies).
        visualizations : list[VisualizationRecommendation]
            Recommended visualizations.

        Returns
        -------
        InsightReport
            A structured report with four insight categories.

        Raises
        ------
        InsightGenerationError
            If prompt building, LLM inference, parsing, or persistence
            fails.
        """
        logger.info("Starting insight generation for dataset '%s'.", dataset_id)
        try:
            prompt = self._build_prompt(descriptive, diagnostic, visualizations)
            raw_response = self._groq_client.generate_recommendations(prompt)
            parsed = self._parse_response(raw_response)
            report = InsightReport(
                descriptive_insights=parsed["descriptive_insights"],
                diagnostic_insights=parsed["diagnostic_insights"],
                predictive_observations=parsed["predictive_observations"],
                prescriptive_recommendations=parsed["prescriptive_recommendations"],
                generated_at=datetime.now(timezone.utc),
                dataset_id=dataset_id,
            )
            self._save_report(report, dataset_id)
        except InsightGenerationError:
            raise
        except Exception as exc:
            logger.exception("Insight generation failed for dataset '%s'.", dataset_id)
            raise InsightGenerationError(
                f"Failed to generate insights for dataset '{dataset_id}': {exc}"
            ) from exc

        logger.info("Insight generation complete for dataset '%s'.", dataset_id)
        return report

    # ── Prompt construction ─────────────────────────────────────────────

    def _build_prompt(
        self,
        descriptive: DescriptiveAnalytics,
        diagnostic: DiagnosticAnalytics,
        visualizations: list[VisualizationRecommendation],
    ) -> str:
        """Assemble a structured prompt from EDA findings.

        The prompt instructs the LLM to act as a Senior Data Analyst,
        BI Consultant, and ML Engineer, then feeds all deterministic
        findings and asks for a JSON-only response.
        """
        summary = descriptive.dataset_summary

        # ── Role instruction
        sections: list[str] = [
            "You are a Senior Data Analyst, Business Intelligence Consultant, "
            "and Machine Learning Engineer.",
            "",
            "Analyze the following dataset findings and generate insights.",
            "",
        ]

        # ── Dataset summary
        sections.append("## DATASET SUMMARY")
        sections.append(f"Rows: {summary.rows}")
        sections.append(f"Columns: {summary.columns}")
        sections.append(f"Numerical columns: {', '.join(summary.numerical_columns) or 'None'}")
        sections.append(f"Categorical columns: {', '.join(summary.categorical_columns) or 'None'}")
        sections.append(f"Datetime columns: {', '.join(summary.datetime_columns) or 'None'}")
        sections.append(f"Missing cells: {summary.missing_cells}")
        sections.append(f"Duplicate rows: {summary.duplicate_rows}")
        sections.append("")

        # ── Descriptive findings
        sections.append("## DESCRIPTIVE FINDINGS")
        if descriptive.key_findings:
            for finding in descriptive.key_findings:
                sections.append(f"- [{finding.severity}] {finding.title}: {finding.description}")
        else:
            sections.append("No descriptive findings.")
        sections.append("")

        # ── Diagnostic findings
        sections.append("## DIAGNOSTIC FINDINGS")
        sections.append("### Correlation Findings")
        if diagnostic.correlation_findings:
            for finding in diagnostic.correlation_findings:
                sections.append(f"- [{finding.severity}] {finding.title}: {finding.description}")
        else:
            sections.append("No correlation findings.")
        sections.append("### Anomaly Findings")
        if diagnostic.anomaly_findings:
            for finding in diagnostic.anomaly_findings:
                sections.append(f"- [{finding.severity}] {finding.title}: {finding.description}")
        else:
            sections.append("No anomaly findings.")
        sections.append("")

        # ── Visualization recommendations
        sections.append("## VISUALIZATION RECOMMENDATIONS")
        if visualizations:
            for viz in visualizations:
                sections.append(f"- {viz.chart_type}: {', '.join(viz.column_names)} — {viz.reason}")
        else:
            sections.append("No visualization recommendations.")
        sections.append("")

        # ── Response format instruction
        sections.append("Based on these findings, generate:")
        sections.append("")
        sections.append("## DESCRIPTIVE INSIGHTS")
        sections.append("Explain important patterns, data quality observations, and important distributions.")
        sections.append("")
        sections.append("## DIAGNOSTIC INSIGHTS")
        sections.append("Explain possible causes, relationships, correlations, and anomalies.")
        sections.append("")
        sections.append("## PREDICTIVE OBSERVATIONS")
        sections.append("Explain potential risks, potential trends, modeling concerns, and future behavior indicators.")
        sections.append("")
        sections.append("## PRESCRIPTIVE RECOMMENDATIONS")
        sections.append("Explain data cleaning improvements, feature engineering suggestions, modeling suggestions, and business recommendations.")
        sections.append("")
        sections.append("You MUST return JSON only. No markdown, no explanation, no code fences.")
        sections.append("Expected structure:")
        sections.append(json.dumps(
            {
                "descriptive_insights": ["..."],
                "diagnostic_insights": ["..."],
                "predictive_observations": ["..."],
                "prescriptive_recommendations": ["..."],
            },
            indent=2,
        ))

        return "\n".join(sections)

    # ── Response parsing ────────────────────────────────────────────────

    def _parse_response(self, raw_response: str) -> dict[str, list[str]]:
        """Parse and validate the raw LLM JSON response.

        Parameters
        ----------
        raw_response : str
            Raw text returned by the LLM.

        Returns
        -------
        dict[str, list[str]]
            Validated dictionary with the four insight categories.

        Raises
        ------
        InsightGenerationError
            If parsing or validation fails.
        """
        if not raw_response or not raw_response.strip():
            raise InsightGenerationError("Groq returned an empty response.")

        # Strip markdown code fences if the model wrapped the JSON
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[: -3]
            cleaned = cleaned.strip()

        try:
            data: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise InsightGenerationError(
                f"Failed to parse Groq response as JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise InsightGenerationError(
                f"Expected a JSON object, got {type(data).__name__}."
            )

        for field in self._REQUIRED_FIELDS:
            if field not in data:
                raise InsightGenerationError(
                    f"Missing required field in LLM response: '{field}'."
                )
            if not isinstance(data[field], list):
                raise InsightGenerationError(
                    f"Field '{field}' must be a list, got {type(data[field]).__name__}."
                )

        return {field: data[field] for field in self._REQUIRED_FIELDS}

    # ── Report persistence ──────────────────────────────────────────────

    def _save_report(self, report: InsightReport, dataset_id: str) -> Path:
        """Serialize the report to JSON and save under the dataset directory.

        Directory structure::

            reports/
            └── insights/
                └── <dataset_id>/
                    └── insight_report.json

        Returns
        -------
        Path
            Absolute path to the saved JSON file.
        """
        dataset_dir = self._base_output_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        filepath = dataset_dir / "insight_report.json"
        filepath.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Insight report saved to %s", filepath)
        return filepath
