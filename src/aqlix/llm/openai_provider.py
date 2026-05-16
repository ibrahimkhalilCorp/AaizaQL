"""
aqlix.llm.openai_provider
──────────────────────────
OpenAI / GPT adapter.
"""

from __future__ import annotations

import openai
import structlog
from aqlix.core.config import Settings
from aqlix.core.exceptions import LLMError
from aqlix.llm.base import LLMProvider
from aqlix.nlp.prompts import SYSTEM_PROMPT


logger = structlog.get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT via the official SDK."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise LLMError(
                "openai",
                "AQLIX_OPENAI_API_KEY is not set. "
                "Export it or pass openai_api_key= to QueryEngine.",
            )
        self._client = openai.OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.openai_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def complete(self, prompt: str, system: str = "") -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system or SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            logger.debug(
                "llm.complete", provider=self.name, tokens=response.usage.completion_tokens
            )
            return text
        except openai.OpenAIError as exc:
            raise LLMError("openai", str(exc)) from exc
