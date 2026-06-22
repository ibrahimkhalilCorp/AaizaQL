"""
aaizaql.llm.deepseek_provider
─────────────────────────────
DeepSeek LLM adapter via the OpenAI-compatible API.

Supported models (as of 2025):
  - deepseek-chat      ← recommended default, best quality/cost tradeoff
  - deepseek-reasoner  ← chain-of-thought reasoning for complex queries

Get your API key at: https://platform.deepseek.com

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment,misc]


class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM provider via the OpenAI-compatible REST API.

    Args:
        settings: Library-wide settings. Must have ``deepseek_api_key`` set.

    Raises:
        LLMError: If ``deepseek_api_key`` is missing or the ``openai`` package
            is not installed.

    Example::

        engine = QueryEngine(
            llm="deepseek",
            database="sqlite",
            dsn="sqlite:///my.db",
            deepseek_api_key="sk-...",
            deepseek_model="deepseek-chat",
        )
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.deepseek_api_key:
            raise LLMError(
                "deepseek",
                "AAIZAQL_DEEPSEEK_API_KEY is not set.\n"
                "Get your key at https://platform.deepseek.com\n"
                "Then set it:  $env:AAIZAQL_DEEPSEEK_API_KEY='sk-...'",
            )

        if OpenAI is None:
            raise LLMError(
                "deepseek",
                "openai package is not installed. Run:  pip install openai",
            )

        api_key = (
            settings.deepseek_api_key.get_secret_value()
            if hasattr(settings.deepseek_api_key, "get_secret_value")
            else str(settings.deepseek_api_key)
        )
        self._client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self._model = settings.deepseek_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("deepseek.ready", model=self._model)

    @property
    def name(self) -> str:
        """Return the provider/model identifier used in logs.

        Returns:
            String in the form ``"deepseek/<model>"``.
        """
        return f"deepseek/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send a prompt to DeepSeek and return the raw text response.

        Args:
            prompt: User-facing content assembled by the SQL generator.
            system: System instruction override. Falls back to the library
                default when empty.
            timeout: Seconds before the call is cancelled. Uses the value from
                settings when ``0``.

        Returns:
            Raw text response from the model.

        Raises:
            LLMTimeoutError: When the DeepSeek API does not respond in time.
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
                "deepseek.complete",
                model=self._model,
                input_tokens=response.usage.prompt_tokens if response.usage else None,
                output_tokens=response.usage.completion_tokens if response.usage else None,
            )
            return text
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("deepseek", effective_timeout) from exc
        except Exception as exc:
            raise LLMError("deepseek", str(exc)) from exc
