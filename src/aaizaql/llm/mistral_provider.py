"""
aaizaql.llm.mistral_provider
─────────────────────────────
Mistral AI adapter via the official mistralai SDK.

Supported models (as of 2025):
  - mistral-large-latest    ← most capable, best for complex SQL (recommended)
  - mistral-small-latest    ← fast and cost-effective
  - codestral-latest        ← code-specialized, excellent for SQL generation
  - open-mistral-nemo       ← open-weight, good balance

Get your API key at: https://console.mistral.ai
"""

from __future__ import annotations

import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)

DEFAULT_MISTRAL_MODEL = "mistral-large-latest"

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None  # type: ignore[assignment,misc]


class MistralProvider(LLMProvider):
    """
    Mistral AI LLM provider.

    Codestral is particularly well-suited for SQL generation as it is
    specialized for code tasks. mistral-large is better for complex reasoning.

    Usage:
        engine = QueryEngine(
            llm="mistral",
            database="sqlite",
            dsn="sqlite:///my.db",
            mistral_api_key="...",
            mistral_model="codestral-latest",   # optional
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

        self._client = Mistral(
            api_key=settings.mistral_api_key.get_secret_value(),
            timeout_ms=None,  # we manage timeout per-call
        )
        self._model = settings.mistral_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("mistral.ready", model=self._model)

    @property
    def name(self) -> str:
        return f"mistral/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send prompt to Mistral and return the SQL response."""
        import concurrent.futures

        effective_timeout = timeout or self._timeout

        def _call() -> str:
            response = self._client.chat.complete(
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
                "mistral.complete",
                model=self._model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return text

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                try:
                    return future.result(timeout=effective_timeout)
                except concurrent.futures.TimeoutError as exc:
                    raise LLMTimeoutError("mistral", effective_timeout) from exc
        except LLMTimeoutError:
            raise
        except Exception as exc:
            raise LLMError("mistral", str(exc)) from exc
