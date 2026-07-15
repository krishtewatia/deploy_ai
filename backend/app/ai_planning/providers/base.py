"""Abstract AI provider contract.

Defines a provider-agnostic interface for generating text completions
from system and user prompts.  Concrete implementations (Groq, OpenAI,
Ollama, etc.) will subclass this contract in future stages.
"""

from __future__ import annotations

import abc


class AIProviderError(Exception):
    """Raised when an AI provider encounters an error during generation."""


class AIProvider(abc.ABC):
    """Abstract base class for AI text completion providers.

    Concrete subclasses must implement :meth:`generate` to accept a
    system prompt and user prompt and return the raw text response.

    The provider contract is intentionally narrow:

    - It knows nothing about MLPlan, DatasetContext, or merging.
    - It accepts text prompts and returns text.
    - Errors are wrapped in :class:`AIProviderError`.
    """

    @abc.abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a text completion from the given prompts.

        Args:
            system_prompt: System-level instructions for the AI model.
            user_prompt: User-level content / context for the AI model.

        Returns:
            Raw text output from the AI model.

        Raises:
            AIProviderError: If the provider encounters any error.
        """
        ...
