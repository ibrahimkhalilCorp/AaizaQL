"""
aaizaql.llm.deepseek_provider
─────────────────────────────
DeepSeek adapter — high-quality reasoning models at very low cost.
DeepSeek uses an OpenAI-compatible API, so integration is straightforward.

Supported models (as of 2025):
  - deepseek-chat       ← DeepSeek V3, best for SQL generation (recommended)
  - deepseek-reasoner   ← DeepSeek R1, slower but stronger on complex queries

Get your API key at: https://platform.deepseek.com
"""

from __future__ import annotations

import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)

DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment,misc]


class DeepSeekProvider(LLMProvider):
    """
    DeepSeek LLM provider.

    DeepSeek offers powerful reasoning models at a fraction of the cost
    of GPT-4. deepseek-chat (V3) is excellent for SQL generation.

    Usage:
        engine = QueryEngine(
            llm="deepseek",
            database="sqlite",
            dsn="sqlite:///my.db",
            deepseek_api_key="sk-...",
            deepseek_model="deepseek-chat",   # optional
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

        self._client = OpenAI(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=DEEPSEEK_BASE_URL,
        )
        self._model = settings.deepseek_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("deepseek.ready", model=self._model)

    @property
    def name(self) -> str:
        return f"deepseek/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send prompt to DeepSeek and return the SQL response."""
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
