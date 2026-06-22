"""
aaizaql.llm.openai_provider
───────────────────────────
OpenAI GPT adapter via the official ``openai`` SDK.

Supported models (as of 2025):
  - gpt-4o          ← recommended default, best quality
  - gpt-4o-mini     ← faster, more cost-effective
  - gpt-4-turbo     ← large context window

Get your API key at: https://platform.openai.com/api-keys

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import openai
import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT LLM provider via the official SDK.

    Args:
        settings: Library-wide settings. Must have ``openai_api_key`` set.

    Raises:
        LLMError: If ``openai_api_key`` is missing.

    Example::

        engine = QueryEngine(
            llm="openai",
            database="sqlite",
            dsn="sqlite:///my.db",
            openai_api_key="sk-...",
            openai_model="gpt-4o",
        )
    """

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

        logger.info("openai.ready", model=self._model)

    @property
    def name(self) -> str:
        """Return the provider/model identifier used in logs.

        Returns:
            String in the form ``"openai/<model>"``.
        """
        return f"openai/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send a prompt to OpenAI and return the raw text response.

        Args:
            prompt: User-facing content assembled by the SQL generator.
            system: System instruction override. Falls back to the library
                default when empty.
            timeout: Seconds before the call is cancelled. Uses the value from
                settings when ``0``.

        Returns:
            Raw text response from the model.

        Raises:
            LLMTimeoutError: When the OpenAI API does not respond in time.
            LLMError: On any other API failure.
        """
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
                "openai.complete",
                model=self._model,
                tokens=response.usage.completion_tokens if response.usage else None,
            )
            return text
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("openai", effective_timeout) from exc
        except openai.OpenAIError as exc:
            raise LLMError("openai", str(exc)) from exc
