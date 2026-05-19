"""
aaizaql.llm
─────────
LLM provider factory.
"""

from __future__ import annotations

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMProviderNotFound
from aaizaql.llm.base import LLMProvider


def build_llm_provider(name: str, settings: Settings) -> LLMProvider:
    """Return the correct LLMProvider instance for the given name."""
    name = name.lower()
    if name == "claude":
        from aaizaql.llm.claude_provider import ClaudeProvider

        return ClaudeProvider(settings)
    if name == "openai":
        from aaizaql.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    if name == "ollama":
        from aaizaql.llm.ollama_provider import OllamaProvider

        return OllamaProvider(settings)
    if name == "groq":
        from aaizaql.llm.groq_provider import GroqProvider

        return GroqProvider(settings)
    raise LLMProviderNotFound(name)


__all__ = ["LLMProvider", "build_llm_provider"]
