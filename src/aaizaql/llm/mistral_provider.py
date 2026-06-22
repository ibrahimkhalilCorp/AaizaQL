"""
aaizaql.llm.mistral_provider
────────────────────────────
Mistral AI adapter via the official ``mistralai`` SDK.

Supported models (as of 2025):
  - mistral-large-latest  ← most capable, recommended for complex SQL
  - mistral-small-latest  ← fast and cost-effective
  - codestral-latest      ← code-specialized, excellent for SQL generation
  - open-mistral-nemo     ← open-weight, good balance

Get your API key at: https://console.mistral.ai

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
    from mistralai import Mistral
except ImportError:
    Mistral = None  # type: ignore[assignment,misc]


class MistralProvider(LLMProvider):
    """Mistral AI LLM provider via the official SDK.

    Codestral is particularly well-suited for SQL generation as it is
    specialized for code tasks. ``mistral-large`` is better for complex
    multi-step reasoning.

    Args:
        settings: Library-wide settings. Must have ``mistral_api_key`` set.

    Raises:
        LLMError: If ``mistral_api_key`` is missing or the ``mistralai``
            package is not installed.

    Example::

        engine = QueryEngine(
            llm="mistral",
            database="sqlite",
            dsn="sqlite:///my.db",
            mistral_api_key="...",
            mistral_model="codestral-latest",
        )
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.mistral_api_key:
            raise LLMError(
                "mistral",
                "AAIZAQL_MISTRAL_API_KEY is not set.\n"
                "Get your key at https://console.mistral.ai\n"
                "Then set it:  $env:AAIZAQL_MISTRAL_API_KEY='...'",
            )

        if Mistral is None:
            raise LLMError(
                "mistral",
                "mistralai package is not installed. Run:  pip install mistralai",
            )

        # timeout_ms=None defers timeout management to each complete() call.
        self._client = Mistral(
            api_key=settings.mistral_api_key.get_secret_value(),
            timeout_ms=None,
        )
        self._model = settings.mistral_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("mistral.ready", model=self._model)

    @property
    def name(self) -> str:
        """Return the provider/model identifier used in logs.

        Returns:
            String in the form ``"mistral/<model>"``.
        """
        return f"mistral/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send a prompt to Mistral and return the raw text response.

        Uses the native ``timeout_ms`` parameter so no thread leak occurs on
        slow responses.

        Args:
            prompt: User-facing content assembled by the SQL generator.
            system: System instruction override. Falls back to the library
                default when empty.
            timeout: Seconds before the call is cancelled. Uses the value from
                settings when ``0``.

        Returns:
            Raw text response from the model.

        Raises:
            LLMTimeoutError: When the Mistral API does not respond in time.
            LLMError: On any other API failure.
        """
        effective_timeout = timeout or self._timeout
        try:
            response = self._client.chat.complete(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                timeout_ms=int(effective_timeout * 1000),
                messages=[
                    {"role": "system", "content": system or SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            logger.debug(
                "mistral.complete",
                model=self._model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return text
        except Exception as exc:
            if "timeout" in str(exc).lower() or isinstance(exc, TimeoutError):
                raise LLMTimeoutError("mistral", effective_timeout) from exc
            raise LLMError("mistral", str(exc)) from exc
