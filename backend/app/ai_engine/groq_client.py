"""Reusable client for the Groq LLM API.

This module wraps the official ``groq`` SDK and provides a single
:meth:`GroqClient.generate_recommendations` entry-point that sends a
prompt to **llama-3.3-70b-versatile** and returns the raw completion text.

The API key is resolved from :pyattr:`backend.app.core.config.settings`
unless explicitly overridden via the *api_key* constructor parameter.
"""

from __future__ import annotations

import logging
from typing import Optional

from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
)

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE: float = 0.1


class GroqClientError(Exception):
    """Base exception for all :class:`GroqClient` failures."""


class GroqAuthenticationError(GroqClientError):
    """Raised when the Groq API rejects the provided credentials."""


class GroqNetworkError(GroqClientError):
    """Raised when a network / connection error prevents the API call."""


class GroqAPIError(GroqClientError):
    """Raised when the Groq API returns a non-authentication error status."""


class GroqClient:
    """Thin, production-ready wrapper around the Groq chat-completions API.

    Parameters
    ----------
    api_key:
        Optional override for the API key.  When *None* (default) the key
        is read from ``settings.GROQ_API_KEY``.
    client:
        Optional pre-configured :class:`groq.Groq` instance for
        dependency-injection / testing.  When supplied, *api_key* is
        ignored.

    Raises
    ------
    ValueError
        If no API key can be resolved and no *client* is injected.

    Usage::

        client = GroqClient()
        text   = client.generate_recommendations(prompt)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Groq] = None,
    ) -> None:
        if client is not None:
            self._client = client
            logger.info("GroqClient initialised with injected client.")
            return

        resolved_key = api_key or settings.GROQ_API_KEY
        if not resolved_key:
            raise ValueError(
                "Groq API key is missing. Configure 'GROQ_API_KEY' in "
                "your environment / .env file or pass 'api_key' explicitly."
            )

        self._client = Groq(api_key=resolved_key)
        logger.info("GroqClient initialised (model=%s).", DEFAULT_MODEL)

    # ── public API ──────────────────────────────────────────────────────

    def generate_recommendations(self, prompt: str) -> str:
        """Send *prompt* to the Groq LLM and return the raw response text.

        Parameters
        ----------
        prompt:
            The fully-assembled prompt (typically built by
            :class:`~backend.app.ai_engine.prompt_builder.PromptBuilder`).

        Returns
        -------
        str
            The raw content string from the model's first choice.

        Raises
        ------
        GroqAuthenticationError
            Invalid or expired API key.
        GroqNetworkError
            DNS / timeout / connection failure.
        GroqAPIError
            Any other HTTP-level error from the Groq API.
        GroqClientError
            Catch-all for unexpected failures.
        """
        logger.info(
            "Sending recommendation request to Groq (model=%s, temp=%s).",
            DEFAULT_MODEL,
            DEFAULT_TEMPERATURE,
        )

        try:
            response = self._client.chat.completions.create(
                model=DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            content: str = response.choices[0].message.content or ""
            logger.info(
                "Groq response received (status=ok, length=%d).", len(content)
            )
            return content

        except AuthenticationError as exc:
            logger.error("Groq authentication failed: %s", exc)
            raise GroqAuthenticationError(
                "Authentication with Groq failed. Verify your API key."
            ) from exc

        except APIConnectionError as exc:
            logger.error("Network error communicating with Groq: %s", exc)
            raise GroqNetworkError(
                "Could not connect to the Groq API. Check your network."
            ) from exc

        except APIStatusError as exc:
            logger.error("Groq API error (status=%s): %s", exc.status_code, exc)
            raise GroqAPIError(
                f"Groq API returned an error (HTTP {exc.status_code})."
            ) from exc

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during Groq request.")
            raise GroqClientError(
                f"Unexpected error: {exc}"
            ) from exc
