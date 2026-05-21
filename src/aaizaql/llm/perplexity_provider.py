"""
aaizaql.llm.perplexity_provider
────────────────────────────────
Perplexity AI adapter — online models with real-time web search capability.
Uses an OpenAI-compatible API.

Supported models (as of 2025):
  - sonar                ← fast, lightweight, recommended for SQL generation
  - sonar-pro            ← more powerful, better reasoning
  - sonar-reasoning      ← chain-of-thought reasoning model
  - sonar-reasoning-pro  ← most powerful reasoning

Get your API key at: https://www.perplexity.ai/settings/api
"""

from __future__ import annotations

import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)

DEFAULT_PERPLEXITY_MODEL = "sonar"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment,misc]


class PerplexityProvider(LLMProvider):
    """
    Perplexity AI LLM provider.

    Perplexity's Sonar models are fast and cost-effective. The Pro variants
    offer stronger reasoning for complex multi-join SQL queries.

    Usage:
        engine = QueryEngine(
            llm="perplexity",
            database="sqlite",
            dsn="sqlite:///my.db",
            perplexity_api_key="pplx-...",
            perplexity_model="sonar",   # optional
        )
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.perplexity_api_key:
            raise LLMError(
                "perplexity",
                "AAIZAQL_PERPLEXITY_API_KEY is not set.\n"
                "Get your key at https://www.perplexity.ai/settings/api\n"
                "Then set it:  $env:AAIZAQL_PERPLEXITY_API_KEY='pplx-...'",
            )

        if OpenAI is None:
            raise LLMError(
                "perplexity",
                "openai package is not installed. Run:  pip install openai",
            )

        self._client = OpenAI(
            api_key=settings.perplexity_api_key.get_secret_value(),
            base_url=PERPLEXITY_BASE_URL,
        )
        self._model = settings.perplexity_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("perplexity.ready", model=self._model)

    @property
    def name(self) -> str:
        return f"perplexity/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send prompt to Perplexity and return the SQL response."""
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
                "perplexity.complete",
                model=self._model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return text

        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("perplexity", effective_timeout) from exc
        except Exception as exc:
            raise LLMError("perplexity", str(exc)) from exc
