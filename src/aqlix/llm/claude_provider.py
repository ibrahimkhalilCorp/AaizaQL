"""
aqlix.llm.claude_provider
──────────────────────────
Anthropic Claude adapter.
"""

from __future__ import annotations

import anthropic

from aqlix.core.config import Settings
from aqlix.core.exceptions import LLMError
from aqlix.llm.base import LLMProvider
from aqlix.nlp.prompts import SYSTEM_PROMPT
import structlog

logger = structlog.get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude via the official SDK."""

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise LLMError(
                "claude",
                "AQLIX_ANTHROPIC_API_KEY is not set. "
                "Export it or pass anthropic_api_key= to QueryEngine.",
            )
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        self._model = settings.claude_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    @property
    def name(self) -> str:
        return f"claude/{self._model}"

    def complete(self, prompt: str, system: str = "") -> str:
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system or SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            response = str(message.content[0].text)
            logger.debug("llm.complete", provider=self.name, tokens=message.usage.output_tokens)
            return response
        except anthropic.APIError as exc:
            raise LLMError("claude", str(exc)) from exc
