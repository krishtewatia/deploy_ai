"""Build structured LLM prompts from dataset analysis reports.

This module converts a :class:`DatasetAnalysisReport` into a deterministic,
self-contained prompt that instructs a Groq-hosted LLM to return a valid
JSON preprocessing plan — and nothing else.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from backend.app.analysis.schemas import DatasetAnalysisReport

logger = logging.getLogger(__name__)

# ── Allowed strategy catalogues (single source of truth) ────────────────────

MISSING_VALUE_STRATEGIES = [
    "mean_imputation",
    "median_imputation",
    "mode_imputation",
    "drop_column",
]

DUPLICATE_STRATEGIES = [
    "remove_duplicates",
    "keep_duplicates",
]

ENCODING_STRATEGIES = [
    "one_hot_encode",
    "label_encode",
]

SCALING_STRATEGIES = [
    "standard_scaling",
    "minmax_scaling",
    "no_scaling",
]

# ── JSON response template ──────────────────────────────────────────────────

_RESPONSE_TEMPLATE: Dict[str, Any] = {
    "cleaning_plan": {
        "missing_values": {
            "<column_name>": {
                "strategy": "<one of: mean_imputation | median_imputation | mode_imputation | drop_column>",
                "reason": "<short explanation>",
            }
        },
        "duplicates_action": "<one of: remove_duplicates | keep_duplicates>",
        "encoding": {
            "<column_name>": {
                "strategy": "<one of: one_hot_encode | label_encode>",
                "reason": "<short explanation>",
            }
        },
        "scaling": {
            "<column_name>": {
                "strategy": "<one of: standard_scaling | minmax_scaling | no_scaling>",
                "reason": "<short explanation>",
            }
        },
    },
    "overall_reasoning": "<brief summary of your recommendations>",
}

RESPONSE_TEMPLATE_JSON: str = json.dumps(_RESPONSE_TEMPLATE, indent=2)


class PromptBuilder:
    """Convert a :class:`DatasetAnalysisReport` into an LLM-ready prompt.

    The generated prompt contains:

    * A human-readable dataset summary (missing values, duplicates,
      numerical statistics, and optional class-imbalance info).
    * Explicit instructions constraining the LLM to return **only** valid
      JSON — no markdown fences, no explanatory prose.
    * The exhaustive list of allowed preprocessing strategies.
    * A JSON response template the LLM must follow.

    Usage::

        builder = PromptBuilder()
        prompt  = builder.build_prompt(analysis_report)
    """

    # ── public API ──────────────────────────────────────────────────────

    def build_prompt(self, analysis_report: DatasetAnalysisReport) -> str:
        """Build the full LLM prompt from *analysis_report*.

        Parameters
        ----------
        analysis_report:
            The aggregated dataset analysis produced by the analysis service.

        Returns
        -------
        str
            A multi-section plain-text prompt ready for submission to the
            Groq API.
        """
        logger.info("Building LLM prompt from analysis report.")

        sections = [
            self._build_role_section(),
            self._build_dataset_summary(analysis_report),
            self._build_strategy_catalogue(),
            self._build_response_format_section(),
        ]

        prompt = "\n\n".join(sections)
        logger.debug("Prompt built successfully (%d characters).", len(prompt))
        return prompt

    # ── private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_role_section() -> str:
        """Return the system-role preamble."""
        return (
            "You are a senior data scientist specializing in data "
            "preprocessing and feature engineering.\n"
            "Your task is to analyze the dataset summary below and provide "
            "a complete preprocessing plan."
        )

    @staticmethod
    def _build_dataset_summary(report: DatasetAnalysisReport) -> str:
        """Render *report* as a human-readable dataset summary."""
        lines: list[str] = ["=== DATASET SUMMARY ==="]

        # -- Missing values -----------------------------------------------
        mv = report.missing_values
        lines.append("\n--- Missing Values ---")
        lines.append(f"Total missing cells: {mv.total_missing}")
        if mv.missing_by_column:
            for col, count in mv.missing_by_column.items():
                pct = mv.missing_percentage.get(col, 0.0)
                lines.append(f"  • {col}: {count} missing ({pct:.1f}%)")
        else:
            lines.append("  No missing values detected.")

        # -- Duplicates ---------------------------------------------------
        dup = report.duplicates
        lines.append("\n--- Duplicates ---")
        lines.append(
            f"Duplicate rows: {dup.duplicate_rows} "
            f"({dup.duplicate_percentage:.1f}%)"
        )

        # -- Numerical statistics -----------------------------------------
        stats = report.statistics
        lines.append("\n--- Numerical Statistics ---")
        if stats.numerical_summary:
            for col, metrics in stats.numerical_summary.items():
                formatted = ", ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                )
                lines.append(f"  • {col}: {formatted}")
        else:
            lines.append("  No numerical columns detected.")

        # -- Class imbalance (optional) -----------------------------------
        if report.imbalance is not None:
            imb = report.imbalance
            lines.append("\n--- Class Imbalance ---")
            lines.append(f"Imbalanced: {imb.imbalanced}")
            if imb.distribution:
                for label, count in imb.distribution.items():
                    lines.append(f"  • {label}: {count}")

        return "\n".join(lines)

    @staticmethod
    def _build_strategy_catalogue() -> str:
        """List every allowed strategy by category."""
        return (
            "=== ALLOWED PREPROCESSING STRATEGIES ===\n"
            "\n"
            "Missing value strategies:\n"
            + "\n".join(f"  - {s}" for s in MISSING_VALUE_STRATEGIES)
            + "\n\n"
            "Duplicate strategies:\n"
            + "\n".join(f"  - {s}" for s in DUPLICATE_STRATEGIES)
            + "\n\n"
            "Encoding strategies:\n"
            + "\n".join(f"  - {s}" for s in ENCODING_STRATEGIES)
            + "\n\n"
            "Scaling strategies:\n"
            + "\n".join(f"  - {s}" for s in SCALING_STRATEGIES)
        )

    @staticmethod
    def _build_response_format_section() -> str:
        """Emit strict JSON-only output instructions + template."""
        return (
            "=== RESPONSE FORMAT ===\n"
            "\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Return ONLY valid JSON.\n"
            "2. Do NOT include markdown code fences, backticks, or "
            "language identifiers.\n"
            "3. Do NOT include any explanations, comments, or extra text "
            "before or after the JSON.\n"
            "4. Use ONLY the strategies listed above — no custom strategy "
            "names.\n"
            "5. Every column with missing values MUST appear in the "
            "\"missing_values\" section.\n"
            "6. Every categorical column MUST appear in the \"encoding\" "
            "section.\n"
            "7. Every numerical column MUST appear in the \"scaling\" "
            "section.\n"
            "\n"
            "Your response must follow this exact JSON structure:\n"
            f"{RESPONSE_TEMPLATE_JSON}"
        )
