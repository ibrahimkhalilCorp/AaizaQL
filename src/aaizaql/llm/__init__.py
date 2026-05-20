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
    if name == "deepseek":
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(settings)
    if name == "perplexity":
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        return PerplexityProvider(settings)
    if name == "gemini":
        from aaizaql.llm.gemini_provider import GeminiProvider

        return GeminiProvider(settings)
    if name == "mistral":
        from aaizaql.llm.mistral_provider import MistralProvider

        return MistralProvider(settings)
    raise LLMProviderNotFound(name)


__all__ = ["LLMProvider", "build_llm_provider"]