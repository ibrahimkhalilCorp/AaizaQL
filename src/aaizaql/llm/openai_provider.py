"""
aaizaql.llm.openai_provider
──────────────────────────
OpenAI / GPT adapter.
"""

from __future__ import annotations

import openai
import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT via the official SDK."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise LLMError(
                "openai",
                "AAIZAQL_OPENAI_API_KEY is not set. "
                "Export it or pass openai_api_key= to QueryEngine.",
            )
        self._client = openai.OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.openai_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 30) -> str:
        effective_timeout = timeout or self._timeout
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                timeout=effective_timeout,
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
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("openai", effective_timeout) from exc
        except openai.OpenAIError as exc:
            raise LLMError("openai", str(exc)) from exc