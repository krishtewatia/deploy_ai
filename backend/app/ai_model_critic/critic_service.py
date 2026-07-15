"""AI Model Critic service orchestrating report review completions."""

from __future__ import annotations

from backend.app.ai_planning.providers import AIProvider
from backend.app.ml_execution.execution_report import ExecutionReport
from backend.app.ai_model_critic.schemas import ModelCritique
from backend.app.ai_model_critic.context_builder import AIModelCriticContextBuilder
from backend.app.ai_model_critic.prompt_builder import AIModelCriticPromptBuilder
from backend.app.ai_model_critic.response_parser import AIModelCriticResponseParser


class AIModelCriticError(Exception):
    """Raised when the AI Model Critic review process fails."""

    pass


class AIModelCritic:
    """Analyzes a completed ExecutionReport and provides an expert review critique."""

    def __init__(self, provider: AIProvider) -> None:
        """Initialize the critic service with an AI provider.

        Args:
            provider: Authorized LLM completion provider.

        Raises:
            AIModelCriticError: If provider is invalid or None.
        """
        if provider is None:
            raise AIModelCriticError("provider cannot be None")
        if not hasattr(provider, "generate"):
            raise AIModelCriticError("provider must implement 'generate' method")
        self.provider = provider
        self.context_builder = AIModelCriticContextBuilder()
        self.prompt_builder = AIModelCriticPromptBuilder()
        self.response_parser = AIModelCriticResponseParser()

    def review(self, execution_report: ExecutionReport) -> ModelCritique:
        """Run the full review pipeline for an ExecutionReport.

        Args:
            execution_report: Consolidated execution report contract.

        Returns:
            A validated ModelCritique containing overall grades and weaknesses.

        Raises:
            AIModelCriticError: For validation failures, LLM provider errors,
                                json parsing errors, or schema validation violations.
        """
        # 1. Validation checks
        if execution_report is None:
            raise AIModelCriticError("execution_report cannot be None")
        if not isinstance(execution_report, ExecutionReport):
            raise AIModelCriticError("execution_report must be an ExecutionReport instance")
        
        # Reject empty report (e.g. missing candidate summaries)
        if not execution_report.candidate_summaries:
            raise AIModelCriticError("execution_report candidate_summaries list cannot be empty")

        # 2. Context Builder
        try:
            report_json = self.context_builder.build(execution_report)
        except Exception as e:
            raise AIModelCriticError(f"Context builder failed: {e}") from e

        # 3. Prompt Builder
        sys_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(report_json)

        # 4. Invoke LLM Provider
        try:
            raw_response = self.provider.generate(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            raise AIModelCriticError(f"AI Provider generation failed: {e}") from e

        # 5. Response Parser
        try:
            critique = self.response_parser.parse(
                text=raw_response,
                report_id=execution_report.report_id,
            )
        except Exception as e:
            raise AIModelCriticError(f"Critique parsing or validation failed: {e}") from e

        return critique
