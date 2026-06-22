"""
aaizaql.llm.claude_provider
───────────────────────────
Anthropic Claude adapter via the official ``anthropic`` SDK.

Supported models (as of 2025):
  - claude-sonnet-4-20250514  ← recommended default
  - claude-opus-4-8           ← most capable
  - claude-haiku-4-5-20251001 ← fastest, most cost-effective

Get your API key at: https://console.anthropic.com

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import anthropic
import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider via the official SDK.

    Args:
        settings: Library-wide settings. Must have ``anthropic_api_key`` set.

    Raises:
        LLMError: If ``anthropic_api_key`` is missing.

    Example::

        engine = QueryEngine(
            llm="claude",
            database="sqlite",
            dsn="sqlite:///my.db",
            anthropic_api_key="sk-ant-...",
            claude_model="claude-sonnet-4-20250514",
        )
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise LLMError(
                "claude",
                "AAIZAQL_ANTHROPIC_API_KEY is not set. "
                "Export it or pass anthropic_api_key= to QueryEngine.",
            )
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        self._model = settings.claude_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("claude.ready", model=self._model)

    @property
    def name(self) -> str:
        """Return the provider/model identifier used in logs.

        Returns:
            String in the form ``"claude/<model>"``.
        """
        return f"claude/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send a prompt to Claude and return the raw text response.

        Args:
            prompt: User-facing content assembled by the SQL generator.
            system: System instruction override. Falls back to the library
                default when empty.
            timeout: Seconds before the call is cancelled. Uses the value from
                settings when ``0``.

        Returns:
            Raw text response from the model.

        Raises:
            LLMTimeoutError: When the Anthropic API does not respond in time.
            LLMError: On any other API failure.
        """
        effective_timeout = timeout or self._timeout
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system or SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                timeout=effective_timeout,
            )
            text_blocks = [b for b in message.content if hasattr(b, "text") and b.text is not None]
            text = text_blocks[0].text if text_blocks else ""
            logger.debug("claude.complete", model=self._model, tokens=message.usage.output_tokens)
            return text
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError("claude", effective_timeout) from exc
        except anthropic.APIError as exc:
            raise LLMError("claude", str(exc)) from exc
