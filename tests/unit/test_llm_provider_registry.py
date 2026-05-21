"""
tests/unit/test_llm_provider_registry.py
─────────────────────────────────────────
Unit tests for task 0.3 — dynamic provider list in LLMProviderNotFound.

Verifies:
- aaizaql.llm.REGISTRY contains all 8 current providers
- LLMProviderNotFound error message lists them all (not the old hardcoded 3)
- build_llm_provider raises LLMProviderNotFound for unknown names
- REGISTRY stays in sync with build_llm_provider (no silent drift)
No API keys or network calls required.
"""

from __future__ import annotations

import pytest

from aaizaql.core.exceptions import LLMProviderNotFound
from aaizaql.llm import REGISTRY, build_llm_provider

EXPECTED_PROVIDERS = frozenset(
    {"claude", "openai", "ollama", "groq", "deepseek", "perplexity", "gemini", "mistral"}
)


class TestRegistry:
    def test_registry_contains_all_eight_providers(self) -> None:
        assert REGISTRY == EXPECTED_PROVIDERS

    def test_registry_is_frozenset(self) -> None:
        assert isinstance(REGISTRY, frozenset)

    def test_registry_names_are_lowercase(self) -> None:
        for name in REGISTRY:
            assert name == name.lower(), f"Provider name '{name}' is not lowercase"


class TestLLMProviderNotFound:
    def test_unknown_name_raises(self) -> None:
        with pytest.raises(LLMProviderNotFound):
            build_llm_provider("nonexistent", None)  # type: ignore[arg-type]

    def test_error_message_lists_all_providers(self) -> None:
        try:
            build_llm_provider("bad_provider", None)  # type: ignore[arg-type]
        except LLMProviderNotFound as exc:
            msg = str(exc)
            for provider in EXPECTED_PROVIDERS:
                assert (
                    provider in msg
                ), f"Provider '{provider}' missing from LLMProviderNotFound message: {msg}"

    def test_error_message_does_not_contain_stale_hardcoded_list(self) -> None:
        """Regression: old message hardcoded only 'claude', 'openai', 'ollama'."""
        try:
            build_llm_provider("bad_provider", None)  # type: ignore[arg-type]
        except LLMProviderNotFound as exc:
            msg = str(exc)
            # The stale message was exactly this substring
            assert "'claude', 'openai', 'ollama'" not in msg

    def test_error_message_includes_unknown_name(self) -> None:
        try:
            build_llm_provider("gpt5", None)  # type: ignore[arg-type]
        except LLMProviderNotFound as exc:
            assert "gpt5" in str(exc)

    def test_case_insensitive_lookup_still_works(self) -> None:
        """build_llm_provider lowercases before lookup; bad names still raise."""
        with pytest.raises(LLMProviderNotFound):
            build_llm_provider("UNKNOWN", None)  # type: ignore[arg-type]

    def test_direct_exception_construction_uses_registry(self) -> None:
        """LLMProviderNotFound can be constructed standalone and still reflects registry."""
        exc = LLMProviderNotFound("whatever")
        msg = str(exc)
        for provider in EXPECTED_PROVIDERS:
            assert provider in msg

    def test_registry_and_factory_are_in_sync(self) -> None:
        """Every name in REGISTRY must be reachable by build_llm_provider
        (i.e. not silently fall through to LLMProviderNotFound)."""
        from unittest.mock import MagicMock, patch

        for name in REGISTRY:
            mock_cls = MagicMock(return_value=MagicMock())
            # Patch importlib.import_module so no real SDK is needed
            with patch("importlib.import_module") as mock_import:
                mock_module = MagicMock()
                mock_import.return_value = mock_module
                mock_module.__dict__[mock_module.__class__.__name__] = mock_cls
                # Just confirm it doesn't raise LLMProviderNotFound
                try:
                    build_llm_provider(name, MagicMock())
                except LLMProviderNotFound:
                    pytest.fail(
                        f"build_llm_provider raised LLMProviderNotFound for '{name}', "
                        "which is in REGISTRY — registry and factory are out of sync."
                    )
                except Exception:
                    # Any other exception (import error, bad mock, etc.) is fine —
                    # the provider was looked up, just not instantiated cleanly
                    pass
