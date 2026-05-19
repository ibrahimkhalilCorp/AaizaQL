"""
AAIZAQL.llm
─────────
LLM provider factory.
"""

from __future__ import annotations

from AAIZAQL.core.config import Settings
from AAIZAQL.core.exceptions import LLMProviderNotFound
from AAIZAQL.llm.base import LLMProvider


def build_llm_provider(name: str, settings: Settings) -> LLMProvider:
    """Return the correct LLMProvider instance for the given name."""
    name = name.lower()
    if name == "claude":
        from AAIZAQL.llm.claude_provider import ClaudeProvider

        return ClaudeProvider(settings)
    if name == "openai":
        from AAIZAQL.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    if name == "ollama":
        from AAIZAQL.llm.ollama_provider import OllamaProvider

        return OllamaProvider(settings)
    if name == "groq":
        from AAIZAQL.llm.groq_provider import GroqProvider

        return GroqProvider(settings)
    raise LLMProviderNotFound(name)


__all__ = ["LLMProvider", "build_llm_provider"]
