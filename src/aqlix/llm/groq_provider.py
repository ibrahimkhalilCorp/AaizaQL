"""
aqlix.llm.groq_provider
────────────────────────
Groq adapter — ultra-fast inference via Groq Cloud.
Groq uses an OpenAI-compatible API, so the implementation is straightforward.

Supported models (as of 2025):
  - llama3-70b-8192       ← best accuracy, recommended
  - llama3-8b-8192        ← faster, lighter
  - mixtral-8x7b-32768    ← large context window
  - gemma2-9b-it          ← Google Gemma via Groq

Get your free API key at: https://console.groq.com
"""

from __future__ import annotations

from aqlix.core.config import Settings
from aqlix.core.exceptions import LLMError
from aqlix.llm.base import LLMProvider
from aqlix.nlp.prompts import SYSTEM_PROMPT
import structlog

logger = structlog.get_logger(__name__)

# Default model — best balance of speed + accuracy for SQL generation
DEFAULT_GROQ_MODEL = "llama3-70b-8192"


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
            groq_model="llama3-70b-8192",   # optional
        )
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise LLMError(
                "groq",
                "AQLIX_GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com\n"
                "Then set it:  $env:AQLIX_GROQ_API_KEY='gsk_api_key'",
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMError(
                "groq",
                "groq package is not installed. Run:  pip install groq",
            ) from exc

        self._client = Groq(api_key=settings.groq_api_key.get_secret_value())
        self._model = settings.groq_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

        logger.info("groq.ready", model=self._model)

    @property
    def name(self) -> str:
        return f"groq/{self._model}"

    def complete(self, prompt: str, system: str = "") -> str:
        """Send prompt to Groq and return the SQL response."""
        try:
            response = self._client.chat.completions.create(
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
                "groq.complete",
                model=self._model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return text

        except Exception as exc:
            # Groq raises groq.APIError, groq.RateLimitError, etc.
            raise LLMError("groq", str(exc)) from exc
