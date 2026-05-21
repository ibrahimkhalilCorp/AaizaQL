"""
aaizaql.llm.claude_provider
──────────────────────────
Anthropic Claude adapter.
"""

from __future__ import annotations

import anthropic
import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude via the official SDK."""

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

    @property
    def name(self) -> str:
        return f"claude/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        effective_timeout = timeout or self._timeout
        try:
            with anthropic.Anthropic(
                api_key=self._client.api_key,
                timeout=effective_timeout,
            ) as client:
                message = client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system or SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            response = str(message.content[0].text)
            logger.debug("llm.complete", provider=self.name, tokens=message.usage.output_tokens)
            return response
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError("claude", effective_timeout) from exc
        except anthropic.APIError as exc:
            raise LLMError("claude", str(exc)) from exc