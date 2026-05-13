"""
aqlix.llm
─────────
LLM provider factory.
"""

from __future__ import annotations

from aqlix.core.config import Settings
from aqlix.core.exceptions import LLMProviderNotFound
from aqlix.llm.base import LLMProvider


def build_llm_provider(name: str, settings: Settings) -> LLMProvider:
    """Return the correct LLMProvider instance for the given name."""
    name = name.lower()
    if name == "claude":
        from aqlix.llm.claude_provider import ClaudeProvider
        return ClaudeProvider(settings)
    if name == "openai":
        from aqlix.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(settings)
    if name == "ollama":
        from aqlix.llm.ollama_provider import OllamaProvider
        return OllamaProvider(settings)
    if name == "groq":
        from aqlix.llm.groq_provider import GroqProvider
        return GroqProvider(settings)
    raise LLMProviderNotFound(name)


__all__ = ["LLMProvider", "build_llm_provider"]
