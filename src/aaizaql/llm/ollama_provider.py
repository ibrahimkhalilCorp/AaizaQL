"""
aaizaql.llm.ollama_provider
──────────────────────────
Ollama local model adapter — free, private, no API key needed.
"""

from __future__ import annotations

import requests
import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Local Ollama server adapter."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._max_tokens = settings.llm_max_tokens
        self._timeout = settings.llm_timeout_seconds

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def complete(self, prompt: str, system: str = "", timeout: int = 30) -> str:
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
            logger.debug("llm.complete", provider=self.name)
            return text
        except requests.Timeout as exc:
            raise LLMTimeoutError("ollama", effective_timeout) from exc
        except requests.RequestException as exc:
            raise LLMError(
                "ollama",
                f"Cannot reach Ollama at {self._base_url}. "
                f"Is it running? (ollama serve). Detail: {exc}",
            ) from exc
