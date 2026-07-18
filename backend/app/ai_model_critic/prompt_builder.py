"""Prompt builder constructing instructions and schema guidance for the AI model critic LLM."""

from __future__ import annotations


class AIModelCriticPromptBuilder:
    """Constructs system and user prompts to guide LLM completion behavior."""

    def build_system_prompt(self) -> str:
        """Construct the system prompt for the senior ML engineer persona."""
        return (
            "You are a senior machine learning engineer and expert model critic.\n"
            "Your task is to analyze the execution report of an AutoML run and "
            "provide a structured review and critique.\n\n"
            "Guidelines and Constraints:\n"
            "1. Review the performance, metrics, diagnostics, and configurations details.\n"
            "2. Never invent or hallucinate metrics. Rely only on the facts in the report.\n"
            "3. Never recommend unsupported model algorithms.\n"
            "4. Return ONLY a valid JSON object matching the schema below. "
            "Do not wrap your response in markdown code blocks like ```json ... ``` or include any extra text.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "report_id": "string",\n'
            '  "overall_grade": "A+" | "A" | "A-" | "B+" | "B" | "B-" | "C" | "D" | "F",\n'
            '  "production_ready": true | false,\n'
            '  "confidence": float (between 0.0 and 1.0),\n'
            '  "strengths": ["string"],\n'
            '  "weaknesses": ["string"],\n'
            '  "risks": ["string"],\n'
            '  "recommendations": ["string"],\n'
            '  "warnings": ["string"],\n'
            '  "summary": "string"\n'
            "}"
        )

    def build_user_prompt(self, report_json: str) -> str:
        """Construct the user prompt containing the ExecutionReport JSON context."""
        return (
            f"Here is the compact JSON context of the ML execution run:\n\n"
            f"{report_json}\n\n"
            f"Analyze this run and output your review critique matching the required JSON structure exactly."
        )
