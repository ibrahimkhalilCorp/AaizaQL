"""
aaizaql.llm.groq_provider
─────────────────────────
Groq Cloud LLM adapter — ultra-fast inference via custom LPU hardware.

Groq uses an OpenAI-compatible API, so the implementation is straightforward.

Supported models (as of 2025):
  - llama-3.3-70b-versatile  ← best accuracy, recommended
  - llama-3.1-8b-instant     ← faster, lighter
  - mixtral-8x7b-32768       ← large context window
  - gemma2-9b-it             ← Google Gemma via Groq

Get your free API key at: https://console.groq.com

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

try:
    from groq import APITimeoutError as _GroqAPITimeoutError
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment,misc]
    _GroqAPITimeoutError = None  # type: ignore[assignment,misc]


class GroqProvider(LLMProvider):
    """Groq Cloud LLM provider.

    Runs open-source models (LLaMA 3, Mixtral, Gemma) at extremely high speed
    using Groq's custom LPU hardware. Free tier is generous.

    Args:
        settings: Library-wide settings. Must have ``groq_api_key`` set.

    Raises:
        LLMError: If ``groq_api_key`` is missing or the ``groq`` package is not
            installed.

    Example::

        engine = QueryEngine(
            llm="groq",
            database="sqlite",
            dsn="sqlite:///my.db",
            groq_api_key="gsk_...",
            groq_model="llama-3.3-70b-versatile",
        )
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise LLMError(
                "groq",
                "AAIZAQL_GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com\n"
                "Then set it:  $env:AAIZAQL_GROQ_API_KEY='gsk_...'",
            )

        import sys

        if sys.modules.get("groq") is None or Groq is None:
            raise LLMError(
                "groq",
                "groq package is not installed. Run:  pip install groq",
            )

        api_key = (
            settings.groq_api_key.get_secret_value()
            if hasattr(settings.groq_api_key, "get_secret_value")
            else str(settings.groq_api_key)
        )
        self._client = Groq(api_key=api_key)
        self._model = settings.groq_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("groq.ready", model=self._model)

    @property
    def name(self) -> str:
        """Return the provider/model identifier used in logs.

        Returns:
            String in the form ``"groq/<model>"``.
        """
        return f"groq/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send a prompt to Groq and return the raw text response.

        Args:
            prompt: User-facing content assembled by the SQL generator.
            system: System instruction override. Falls back to the library
                default when empty.
            timeout: Seconds before the call is cancelled. Uses the value from
                settings when ``0``.

        Returns:
            Raw text response from the model.

        Raises:
            LLMTimeoutError: When the Groq API does not respond in time.
            LLMError: On any other API failure.
        """
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
                input_tokens=response.usage.prompt_tokens if response.usage else None,
                output_tokens=response.usage.completion_tokens if response.usage else None,
            )
            return text
        except APITimeoutError as exc:
            raise LLMTimeoutError("groq", effective_timeout) from exc
        except Exception as exc:
            raise LLMError("groq", str(exc)) from exc
