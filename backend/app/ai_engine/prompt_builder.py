"""Build dataset-intelligence prompts from analysis reports.

This module converts a :class:`DatasetAnalysisReport` into a deterministic,
self-contained prompt that gives an LLM enough dataset context to recommend
preprocessing steps like an experienced data scientist while still returning
only a valid JSON preprocessing plan.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from backend.app.analysis.schemas import DatasetAnalysisReport

logger = logging.getLogger(__name__)

MISSING_VALUE_STRATEGIES: list[str] = [
    "mean_imputation",
    "median_imputation",
    "mode_imputation",
    "drop_column",
]

DUPLICATE_STRATEGIES: list[str] = [
    "remove_duplicates",
    "keep_duplicates",
]

ENCODING_STRATEGIES: list[str] = [
    "one_hot_encode",
    "label_encode",
]

SCALING_STRATEGIES: list[str] = [
    "standard_scaling",
    "minmax_scaling",
    "no_scaling",
]

_RESPONSE_TEMPLATE: dict[str, Any] = {
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
    "potential_identifier_columns": [],
    "potential_target_columns": [],
    "potential_leakage_columns": [],
    "drop_recommendations": [],
    "overall_reasoning": "<brief summary of your recommendations>",
}

RESPONSE_TEMPLATE_JSON: str = json.dumps(_RESPONSE_TEMPLATE, indent=2)

_REQUIRED_RESPONSE_SCHEMA: dict[str, Any] = {
    "cleaning_plan": {},
    "potential_identifier_columns": [],
    "potential_target_columns": [],
    "potential_leakage_columns": [],
    "drop_recommendations": [],
    "overall_reasoning": "",
}

_RESPONSE_EXAMPLE_POPULATED: dict[str, Any] = {
    "cleaning_plan": {
        "missing_values": {
            "age": {
                "strategy": "median_imputation",
                "reason": "Age is numeric and may be skewed.",
            }
        },
        "duplicates_action": "remove_duplicates",
        "encoding": {
            "department": {
                "strategy": "one_hot_encode",
                "reason": "Department is a low-cardinality categorical feature.",
            }
        },
        "scaling": {
            "salary": {
                "strategy": "standard_scaling",
                "reason": "Salary is numeric and scale-sensitive for many models.",
            }
        },
    },
    "potential_identifier_columns": ["customer_id"],
    "potential_target_columns": ["churn"],
    "potential_leakage_columns": ["churn_date"],
    "drop_recommendations": ["email"],
    "overall_reasoning": "Identifiers and leakage-prone columns should not be used as model features.",
}

_RESPONSE_EXAMPLE_EMPTY_ARRAYS: dict[str, Any] = {
    "cleaning_plan": {
        "missing_values": {},
        "duplicates_action": "keep_duplicates",
        "encoding": {},
        "scaling": {},
    },
    "potential_identifier_columns": [],
    "potential_target_columns": [],
    "potential_leakage_columns": [],
    "drop_recommendations": [],
    "overall_reasoning": "No identifier, target, leakage, or drop columns were detected.",
}

REQUIRED_RESPONSE_SCHEMA_JSON: str = json.dumps(_REQUIRED_RESPONSE_SCHEMA, indent=2)
RESPONSE_EXAMPLE_POPULATED_JSON: str = json.dumps(
    _RESPONSE_EXAMPLE_POPULATED,
    indent=2,
)
RESPONSE_EXAMPLE_EMPTY_ARRAYS_JSON: str = json.dumps(
    _RESPONSE_EXAMPLE_EMPTY_ARRAYS,
    indent=2,
)

_UNKNOWN: str = "Unknown"


class PromptBuilder:
    """Convert a :class:`DatasetAnalysisReport` into an LLM-ready prompt.

    The generated prompt includes dataset overview, per-column profiles, data
    quality analysis, semantic inspection instructions, the recommendation
    task, allowed strategy catalogues, and a strict JSON response contract.
    """

    def build_prompt(self, analysis_report: DatasetAnalysisReport) -> str:
        """Build the full LLM prompt from *analysis_report*.

        Args:
            analysis_report: Aggregated dataset analysis from the analysis
                service.

        Returns:
            A multi-section plain-text prompt ready for LLM submission.
        """
        logger.info("Building dataset intelligence prompt from analysis report.")

        metadata = self._extract_dataset_metadata(analysis_report)
        sections = [
            self._build_role_section(),
            self._build_dataset_overview(metadata),
            self._build_column_profiles_section(analysis_report),
            self._build_data_quality_report(analysis_report),
            self._build_semantic_understanding_section(),
            self._build_recommendation_task_section(),
            self._build_strategy_catalogue(),
            self._build_response_format_section(),
        ]

        prompt = "\n\n".join(sections)
        logger.debug("Dataset intelligence prompt built (%d characters).", len(prompt))
        return prompt

    @staticmethod
    def _build_role_section() -> str:
        """Return the system-role preamble."""
        return (
            "You are a senior data scientist specializing in data "
            "preprocessing, feature engineering, data quality assessment, "
            "and dataset semantics.\n"
            "Your task is to analyze the dataset intelligence report below "
            "and provide a complete preprocessing plan."
        )

    def _build_dataset_overview(self, metadata: dict[str, Any]) -> str:
        """Render dataset-level metadata."""
        return "\n".join(
            [
                "==================================================",
                "DATASET OVERVIEW",
                "================",
                "",
                f"* File name: {metadata['file_name']}",
                f"* Number of rows: {metadata['rows']}",
                f"* Number of columns: {metadata['columns']}",
                f"* Column names: {self._format_value(metadata['column_names'])}",
                f"* Dataset shape: {self._format_value(metadata['shape'])}",
            ]
        )

    def _build_column_profiles_section(self, report: DatasetAnalysisReport) -> str:
        """Render detailed profile information for every known column."""
        lines = [
            "==================================================",
            "COLUMN PROFILES",
            "===============",
        ]

        column_profiles = report.column_profiles or {}
        if not column_profiles:
            lines.append("")
            lines.append("No column profile information was provided.")
            return "\n".join(lines)

        for column_name, profile in column_profiles.items():
            normalized = self._normalize_profile(column_name, profile)
            lines.extend(
                [
                    "",
                    f"* Column name: {normalized['column_name']}",
                    f"  * Data type: {normalized['dtype']}",
                    f"  * Missing count: {normalized['missing_count']}",
                    f"  * Missing percentage: {normalized['missing_percentage']}",
                    f"  * Unique values count: {normalized['unique_values_count']}",
                    f"  * Unique percentage: {normalized['unique_percentage']}",
                    f"  * Sample values: {self._format_value(normalized['sample_values'])}",
                    f"  * is_numeric: {normalized['is_numeric']}",
                    f"  * is_categorical: {normalized['is_categorical']}",
                    f"  * is_datetime: {normalized['is_datetime']}",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _build_data_quality_report(report: DatasetAnalysisReport) -> str:
        """Render missingness, duplicates, statistics, and class balance."""
        lines: list[str] = [
            "==================================================",
            "DATA QUALITY REPORT",
            "===================",
            "",
            "--- Missing Values ---",
            "Missing value analysis:",
        ]

        mv = report.missing_values
        lines.append(f"Total missing cells: {mv.total_missing}")
        if mv.missing_by_column:
            for col, count in mv.missing_by_column.items():
                pct = mv.missing_percentage.get(col, 0.0)
                lines.append(f"  * {col}: {count} missing ({pct:.1f}%)")
        else:
            lines.append("  No missing values detected.")

        dup = report.duplicates
        lines.extend(
            [
                "",
                "--- Duplicates ---",
                "Duplicate analysis:",
                f"Duplicate rows: {dup.duplicate_rows} ({dup.duplicate_percentage:.1f}%)",
            ]
        )

        stats = report.statistics
        lines.extend(["", "--- Numerical Statistics ---", "Statistics analysis:"])
        if stats.numerical_summary:
            for col, metrics in stats.numerical_summary.items():
                formatted = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                lines.append(f"  * {col}: {formatted}")
        else:
            lines.append("  No numerical columns detected.")

        lines.extend(["", "--- Class Imbalance ---", "Class imbalance analysis:"])
        if report.imbalance is None:
            lines.append("  No class imbalance report was provided.")
            return "\n".join(lines)

        imb = report.imbalance
        lines.append(f"Imbalanced: {imb.imbalanced}")
        if imb.distribution:
            for label, count in imb.distribution.items():
                lines.append(f"  * {label}: {count}")
        else:
            lines.append("  No class distribution values were provided.")

        return "\n".join(lines)

    @staticmethod
    def _build_semantic_understanding_section() -> str:
        """Instruct the LLM to infer business meaning from dataset semantics."""
        return "\n".join(
            [
                "==================================================",
                "SEMANTIC DATASET UNDERSTANDING",
                "==============================",
                "",
                "Inspect the dataset name, file name, column names, column relationships,",
                "sample values, and potential business meaning of columns before making",
                "recommendations.",
                "",
                "Use semantic understanding for columns such as:",
                "customer_id, employee_id, salary, department, city, country,",
                "email, date, timestamp, target, and label.",
                "",
                "Infer whether columns represent identifiers, targets, labels,",
                "timestamps, geographic attributes, contact information, business",
                "categories, monetary values, or possible leakage signals.",
            ]
        )

    @staticmethod
    def _build_recommendation_task_section() -> str:
        """Describe the preprocessing recommendation task."""
        return "\n".join(
            [
                "==================================================",
                "PREPROCESSING RECOMMENDATION TASK",
                "=================================",
                "",
                "Determine all of the following:",
                "1. Missing value handling strategy",
                "2. Duplicate handling strategy",
                "3. Encoding strategy",
                "4. Scaling strategy",
                "5. Potential identifier columns",
                "6. Potential target columns",
                "7. Potential leakage columns",
                "8. High-cardinality columns",
                "9. Datetime handling suggestions",
                "10. Columns that should be dropped",
            ]
        )

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
        """Emit strict JSON-only output instructions and template."""
        return (
            "==================================================\n"
            "RESPONSE FORMAT\n"
            "===============\n"
            "\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Return ONLY valid JSON.\n"
            "2. Do NOT include markdown code fences, backticks, or language identifiers.\n"
            "3. Do NOT include any explanations, comments, or extra text before or after the JSON.\n"
            "4. Use ONLY the strategies listed above; no custom strategy names.\n"
            "5. Every column with missing values MUST appear in the \"missing_values\" section.\n"
            "6. Every categorical column MUST appear in the \"encoding\" section.\n"
            "7. Every numerical column MUST appear in the \"scaling\" section.\n"
            "8. Every field shown in the schema is mandatory.\n"
            "9. Never omit a field.\n"
            "10. If no values exist for an array field, return an empty array.\n"
            "11. The response must contain all schema keys exactly as defined.\n"
            "12. The fields \"potential_identifier_columns\", \"potential_target_columns\", "
            "\"potential_leakage_columns\", and \"drop_recommendations\" are mandatory "
            "in every response, even when empty.\n"
            "\n"
            "Required top-level JSON structure:\n"
            f"{REQUIRED_RESPONSE_SCHEMA_JSON}\n"
            "\n"
            "Your response must follow this JSON structure:\n"
            f"{RESPONSE_TEMPLATE_JSON}\n"
            "\n"
            "Example response with populated arrays:\n"
            f"{RESPONSE_EXAMPLE_POPULATED_JSON}\n"
            "\n"
            "Example response with empty arrays:\n"
            f"{RESPONSE_EXAMPLE_EMPTY_ARRAYS_JSON}"
        )

    def _extract_dataset_metadata(self, report: DatasetAnalysisReport) -> dict[str, Any]:
        """Extract or infer dataset-level metadata without breaking old reports."""
        report_data = self._model_to_dict(report)
        nested_metadata = self._first_mapping(
            report_data.get("metadata"),
            report_data.get("dataset_metadata"),
            report_data.get("upload_metadata"),
        )
        metadata = {**nested_metadata, **report_data}

        column_profiles = report.column_profiles or {}
        column_names = self._get_first_present(
            metadata,
            "column_names",
            "columns_names",
            "columns_list",
        )
        if not column_names:
            column_names = list(column_profiles.keys())

        column_names = [str(column) for column in column_names]
        inferred_rows = self._infer_row_count(column_profiles)
        rows = self._get_first_present(metadata, "rows", "row_count", "num_rows", "n_rows")
        columns = self._get_first_present(
            metadata,
            "columns",
            "column_count",
            "num_columns",
            "n_columns",
        )

        rows = rows if rows is not None else inferred_rows
        columns = columns if columns is not None else len(column_names) or None
        shape = self._get_first_present(metadata, "shape", "dataset_shape")
        if shape is None and rows is not None and columns is not None:
            shape = [rows, columns]

        return {
            "file_name": self._get_first_present(
                metadata,
                "file_name",
                "filename",
                "dataset_name",
                "name",
            )
            or _UNKNOWN,
            "rows": rows if rows is not None else _UNKNOWN,
            "columns": columns if columns is not None else _UNKNOWN,
            "column_names": column_names,
            "shape": shape if shape is not None else _UNKNOWN,
        }

    @staticmethod
    def _normalize_profile(column_name: str, profile: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize profile keys from current and legacy profiler outputs."""
        return {
            "column_name": column_name,
            "dtype": profile.get("dtype", profile.get("data_type", _UNKNOWN)),
            "missing_count": profile.get("missing_count", _UNKNOWN),
            "missing_percentage": profile.get("missing_percentage", _UNKNOWN),
            "unique_values_count": profile.get(
                "unique_values_count",
                profile.get("unique_values", profile.get("unique_count", _UNKNOWN)),
            ),
            "unique_percentage": profile.get("unique_percentage", _UNKNOWN),
            "sample_values": profile.get("sample_values", []),
            "is_numeric": profile.get("is_numeric", _UNKNOWN),
            "is_categorical": profile.get("is_categorical", _UNKNOWN),
            "is_datetime": profile.get("is_datetime", _UNKNOWN),
        }

    @staticmethod
    def _infer_row_count(column_profiles: Mapping[str, Mapping[str, Any]]) -> int | None:
        """Infer row count from percentage/count pairs when metadata is absent."""
        candidates: list[int] = []
        for profile in column_profiles.values():
            for count_key, pct_key in (
                ("missing_count", "missing_percentage"),
                ("unique_values", "unique_percentage"),
                ("unique_values_count", "unique_percentage"),
                ("unique_count", "unique_percentage"),
            ):
                count = profile.get(count_key)
                percentage = profile.get(pct_key)
                if not isinstance(count, (int, float)) or not isinstance(
                    percentage, (int, float)
                ):
                    continue
                if count <= 0 or percentage <= 0:
                    continue
                candidates.append(round(count * 100 / percentage))

        return candidates[0] if candidates else None

    @staticmethod
    def _model_to_dict(value: Any) -> dict[str, Any]:
        """Return a dict for Pydantic models, mappings, or unknown objects."""
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, Mapping):
            return dict(value)
        return {
            key: getattr(value, key)
            for key in dir(value)
            if not key.startswith("_") and not callable(getattr(value, key))
        }

    @staticmethod
    def _first_mapping(*values: Any) -> dict[str, Any]:
        """Return the first mapping-like value as a plain dict."""
        for value in values:
            if isinstance(value, BaseModel):
                return value.model_dump()
            if isinstance(value, Mapping):
                return dict(value)
        return {}

    @staticmethod
    def _get_first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
        """Return the first non-empty value for *keys* from *mapping*."""
        for key in keys:
            value = mapping.get(key)
            if value not in (None, "", []):
                return value
        return None

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format values deterministically for prompt readability."""
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return json.dumps(list(value), ensure_ascii=True)
        if isinstance(value, Mapping):
            return json.dumps(dict(value), ensure_ascii=True, sort_keys=True)
        return str(value)
