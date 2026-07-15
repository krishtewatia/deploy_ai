"""Unit tests for AIModelCriticPromptBuilder."""

from __future__ import annotations

from backend.app.ai_model_critic.prompt_builder import AIModelCriticPromptBuilder


class TestAIModelCriticPromptBuilder:
    """Tests covering system and user prompt string builds."""

    def test_prompt_builder_strings(self):
        """Verify prompt strings contain essential persona and rules constraints."""
        builder = AIModelCriticPromptBuilder()

        sys = builder.build_system_prompt()
        assert "senior machine learning engineer" in sys.lower()
        assert "valid json" in sys.lower()
        assert "never invent" in sys.lower()

        user = builder.build_user_prompt('{"report_id": "r_1"}')
        assert "r_1" in user
        assert "compact json" in user.lower()
