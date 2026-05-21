"""
aaizaql.llm.groq_provider
────────────────────────
Groq adapter — ultra-fast inference via Groq Cloud.
Groq uses an OpenAI-compatible API, so the implementation is straightforward.

Supported models (as of 2025):
  - llama-3.3-70b-versatile  ← best accuracy, recommended
  - llama-3.1-8b-instant     ← faster, lighter
  - mixtral-8x7b-32768       ← large context window
  - gemma2-9b-it             ← Google Gemma via Groq

Get your free API key at: https://console.groq.com
"""

from __future__ import annotations

import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


try:
    from groq import APITimeoutError as _GroqAPITimeoutError
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment,misc]
    _GroqAPITimeoutError = None  # type: ignore[assignment,misc]


class GroqProvider(LLMProvider):
    """
    Groq Cloud LLM provider.

    Groq runs open-source models (LLaMA3, Mixtral, Gemma) at extremely
    high speed using their custom LPU hardware. Free tier is generous.

    Usage:
        engine = QueryEngine(
            llm="groq",
            database="sqlite",
            dsn="sqlite:///my.db",
            groq_api_key="gsk_...",
            groq_model="llama-3.3-70b-versatile",   # optional
        )
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise LLMError(
                "groq",
                "AAIZAQL_GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com\n"
                "Then set it:  $env:AAIZAQL_GROQ_API_KEY='gsk_api_key'",
            )

        # Check at call-time so tests can simulate absence via
        # patch.dict("sys.modules", {"groq": None}).
        import sys

        if sys.modules.get("groq") is None or Groq is None:
            raise LLMError(
                "groq",
                "groq package is not installed. Run:  pip install groq",
            )

        api_key = (
            settings.groq_api_key.get_secret_value()
            if hasattr(settings.groq_api_key, "get_secret_value")
            else settings.groq_api_key
        )
        self._client = Groq(api_key=api_key)
        self._model = settings.groq_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("groq.ready", model=self._model)

    @property
    def name(self) -> str:
        return f"groq/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send prompt to Groq and return the SQL response."""
        from groq import APITimeoutError

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
                "groq.complete",
                model=self._model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return text

        except APITimeoutError as exc:
            raise LLMTimeoutError("groq", effective_timeout) from exc
        except Exception as exc:
            raise LLMError("groq", str(exc)) from exc
