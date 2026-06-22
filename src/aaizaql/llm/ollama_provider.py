"""
aaizaql.llm.ollama_provider
───────────────────────────
Ollama local model adapter — free, private, no API key required.

Ollama runs open-source models (LLaMA 3, Mistral, Phi, Gemma, etc.) entirely
on your own hardware. Start the server with ``ollama serve`` before use.

Get Ollama at: https://ollama.com

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import requests  # type: ignore[import-untyped]
import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Local Ollama server adapter.

    No API key is needed. The Ollama server must be running at
    ``ollama_base_url`` (default: ``http://localhost:11434``) before
    :meth:`complete` is called.

    Args:
        settings: Library-wide settings. Reads ``ollama_base_url``,
            ``ollama_model``, ``llm_max_tokens``, and
            ``llm_timeout_seconds``.

    Example::

        engine = QueryEngine(
            llm="ollama",
            database="sqlite",
            dsn="sqlite:///my.db",
            ollama_model="llama3",
            ollama_base_url="http://localhost:11434",
        )
    """

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._max_tokens = settings.llm_max_tokens
        self._timeout = settings.llm_timeout_seconds

        logger.info("ollama.ready", model=self._model, base_url=self._base_url)

    @property
    def name(self) -> str:
        """Return the provider/model identifier used in logs.

        Returns:
            String in the form ``"ollama/<model>"``.
        """
        return f"ollama/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send a prompt to the local Ollama server and return the raw response.

        The system prompt and user prompt are concatenated into a single string
        because Ollama's ``/api/generate`` endpoint does not support separate
        ``system`` and ``user`` message roles.

        Args:
            prompt: User-facing content assembled by the SQL generator.
            system: System instruction override. Falls back to the library
                default when empty.
            timeout: Seconds before the HTTP request is cancelled. Uses the
                value from settings when ``0``.

        Returns:
            Raw text response from the model.

        Raises:
            LLMTimeoutError: When the Ollama server does not respond in time.
            LLMError: When the server is unreachable or returns an error.
        """
        effective_timeout = timeout or self._timeout
        full_prompt = f"{system or SYSTEM_PROMPT}\n\n{prompt}"
        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"num_predict": self._max_tokens},
                },
                timeout=effective_timeout,
            )
            response.raise_for_status()
            text = str(response.json().get("response", ""))
            logger.debug("ollama.complete", model=self._model)
            return text
        except requests.Timeout as exc:
            raise LLMTimeoutError("ollama", effective_timeout) from exc
        except requests.RequestException as exc:
            raise LLMError(
                "ollama",
                f"Cannot reach Ollama at {self._base_url}. "
                f"Is it running? (ollama serve). Detail: {exc}",
            ) from exc
