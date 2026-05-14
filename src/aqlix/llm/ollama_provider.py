"""
aqlix.llm.ollama_provider
──────────────────────────
Ollama local model adapter — free, private, no API key needed.
"""

from __future__ import annotations

import requests

from aqlix.core.config import Settings
from aqlix.core.exceptions import LLMError
from aqlix.llm.base import LLMProvider
from aqlix.nlp.prompts import SYSTEM_PROMPT
import structlog

logger = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Local Ollama server adapter."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._max_tokens = settings.llm_max_tokens

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def complete(self, prompt: str, system: str = "") -> str:
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
                timeout=120,
            )
            response.raise_for_status()
            text = str(response.json().get("response", ""))
            logger.debug("llm.complete", provider=self.name)
            return text
        except requests.RequestException as exc:
            raise LLMError(
                "ollama",
                f"Cannot reach Ollama at {self._base_url}. "
                f"Is it running? (ollama serve). Detail: {exc}",
            ) from exc
