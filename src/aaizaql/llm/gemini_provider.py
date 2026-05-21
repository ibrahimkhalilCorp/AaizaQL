"""
aaizaql.llm.gemini_provider
────────────────────────────
Google Gemini adapter via the official google-genai SDK.

Supported models (as of 2025):
  - gemini-2.5-flash   ← fast, cost-effective, recommended (default)
  - gemini-2.5-pro     ← most capable, best for complex queries
  - gemini-2.0-flash   ← previous generation fast model

Get your API key at: https://aistudio.google.com/app/apikey
"""

from __future__ import annotations

import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider.

    Gemini 2.5 Flash offers an excellent speed/quality tradeoff for SQL
    generation with a very large context window (1M tokens).

    Usage:
        engine = QueryEngine(
            llm="gemini",
            database="sqlite",
            dsn="sqlite:///my.db",
            gemini_api_key="AIza...",
            gemini_model="gemini-2.5-flash",   # optional
        )
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise LLMError(
                "gemini",
                "AAIZAQL_GEMINI_API_KEY is not set.\n"
                "Get your key at https://aistudio.google.com/app/apikey\n"
                "Then set it:  $env:AAIZAQL_GEMINI_API_KEY='AIza...'",
            )

        if genai is None:
            raise LLMError(
                "gemini",
                "google-genai package is not installed. Run:  pip install google-genai",
            )

        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self._model = settings.gemini_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_seconds

        logger.info("gemini.ready", model=self._model)

    @property
    def name(self) -> str:
        return f"gemini/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 30) -> str:
        """Send prompt to Gemini and return the SQL response."""
        import concurrent.futures

        effective_timeout = timeout or self._timeout

        def _call() -> str:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system or SYSTEM_PROMPT,
                    max_output_tokens=self._max_tokens,
                    temperature=self._temperature,
                ),
            )
            text = response.text or ""
            logger.debug(
                "gemini.complete",
                model=self._model,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            )
            return text

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                try:
                    return future.result(timeout=effective_timeout)
                except concurrent.futures.TimeoutError as exc:
                    raise LLMTimeoutError("gemini", effective_timeout) from exc
        except LLMTimeoutError:
            raise
        except Exception as exc:
            raise LLMError("gemini", str(exc)) from exc