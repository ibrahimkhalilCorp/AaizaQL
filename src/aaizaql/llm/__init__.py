"""
aaizaql.llm
─────────
LLM provider factory.
"""

from __future__ import annotations

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMProviderNotFound
from aaizaql.llm.base import LLMProvider

# Central registry of all supported LLM provider names.
# Add new providers here — LLMProviderNotFound will always stay current.
_PROVIDER_REGISTRY: dict[str, str] = {
    "claude": "aaizaql.llm.claude_provider.ClaudeProvider",
    "openai": "aaizaql.llm.openai_provider.OpenAIProvider",
    "ollama": "aaizaql.llm.ollama_provider.OllamaProvider",
    "groq": "aaizaql.llm.groq_provider.GroqProvider",
    "deepseek": "aaizaql.llm.deepseek_provider.DeepSeekProvider",
    "perplexity": "aaizaql.llm.perplexity_provider.PerplexityProvider",
    "gemini": "aaizaql.llm.gemini_provider.GeminiProvider",
    "mistral": "aaizaql.llm.mistral_provider.MistralProvider",
}

# Public read-only view consumed by LLMProviderNotFound
REGISTRY: frozenset[str] = frozenset(_PROVIDER_REGISTRY)

def build_llm_provider(name: str, settings: Settings) -> LLMProvider:
    """Return the correct LLMProvider instance for the given name."""
    name = name.lower()
    if name not in _PROVIDER_REGISTRY:
        raise LLMProviderNotFound(name, available=sorted(_PROVIDER_REGISTRY.keys()))

    # Lazy import — keeps optional SDK deps out of the import chain
    module_path, class_name = _PROVIDER_REGISTRY[name].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(settings)

__all__ = ["LLMProvider", "REGISTRY", "build_llm_provider"]
