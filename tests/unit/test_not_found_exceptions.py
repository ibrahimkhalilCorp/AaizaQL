"""
tests/unit/test_not_found_exceptions.py
────────────────────────────────────────
Unit tests for task 0.5 — ConnectorNotFound and LLMProviderNotFound
no longer import registries inside __init__; available keys are passed
explicitly at the raise site.

Verifies:
- Exceptions can be constructed with and without the `available` kwarg
- available keys appear in the message
- No import of aaizaql.connectors or aaizaql.llm occurs during construction
- Call sites (get_connector, build_llm_provider) pass correct available lists
- .name and .available attributes are set on both exceptions
No API keys or network calls required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from aaizaql.core.exceptions import ConnectorNotFound, LLMProviderNotFound


# ── ConnectorNotFound ─────────────────────────────────────────────────────────


class TestConnectorNotFound:
    def test_basic_construction(self) -> None:
        exc = ConnectorNotFound("redis", available=["sqlite", "postgres"])
        assert "redis" in str(exc)
        assert "sqlite" in str(exc)
        assert "postgres" in str(exc)

    def test_name_attribute(self) -> None:
        exc = ConnectorNotFound("redis", available=["sqlite"])
        assert exc.name == "redis"

    def test_available_attribute(self) -> None:
        exc = ConnectorNotFound("redis", available=["sqlite", "postgres"])
        assert sorted(exc.available) == ["postgres", "sqlite"]

    def test_no_available_kwarg_does_not_crash(self) -> None:
        # Should construct cleanly with an empty list, not raise ImportError
        exc = ConnectorNotFound("redis")
        assert "redis" in str(exc)
        assert exc.available == []

    def test_no_registry_import_during_construction(self) -> None:
        """Construction must NOT trigger an import of aaizaql.connectors."""
        # Remove the module from sys.modules so any import would be detectable
        connectors_key = "aaizaql.connectors"
        original = sys.modules.pop(connectors_key, None)
        try:
            # This must not re-import aaizaql.connectors
            exc = ConnectorNotFound("bad", available=["sqlite"])
            assert "bad" in str(exc)
            assert connectors_key not in sys.modules
        finally:
            if original is not None:
                sys.modules[connectors_key] = original

    def test_is_aaizaql_error(self) -> None:
        from aaizaql.core.exceptions import AAIZAQLError

        assert isinstance(ConnectorNotFound("x", available=[]), AAIZAQLError)


# ── LLMProviderNotFound ───────────────────────────────────────────────────────


class TestLLMProviderNotFound:
    def test_basic_construction(self) -> None:
        exc = LLMProviderNotFound("gpt5", available=["claude", "openai", "groq"])
        assert "gpt5" in str(exc)
        assert "claude" in str(exc)
        assert "openai" in str(exc)
        assert "groq" in str(exc)

    def test_name_attribute(self) -> None:
        exc = LLMProviderNotFound("gpt5", available=["claude"])
        assert exc.name == "gpt5"

    def test_available_attribute(self) -> None:
        exc = LLMProviderNotFound("gpt5", available=["claude", "openai"])
        assert sorted(exc.available) == ["claude", "openai"]

    def test_no_available_kwarg_does_not_crash(self) -> None:
        exc = LLMProviderNotFound("gpt5")
        assert "gpt5" in str(exc)
        assert exc.available == []

    def test_no_registry_import_during_construction(self) -> None:
        """Construction must NOT trigger an import of aaizaql.llm."""
        llm_key = "aaizaql.llm"
        original = sys.modules.pop(llm_key, None)
        try:
            exc = LLMProviderNotFound("bad", available=["claude"])
            assert "bad" in str(exc)
            assert llm_key not in sys.modules
        finally:
            if original is not None:
                sys.modules[llm_key] = original

    def test_is_aaizaql_error(self) -> None:
        from aaizaql.core.exceptions import AAIZAQLError

        assert isinstance(LLMProviderNotFound("x", available=[]), AAIZAQLError)


# ── Call-site integration: get_connector ─────────────────────────────────────


class TestGetConnectorCallSite:
    def test_raises_with_correct_available_list(self) -> None:
        from aaizaql.connectors import REGISTRY, get_connector

        with pytest.raises(ConnectorNotFound) as exc_info:
            get_connector("nonexistent_db")

        exc = exc_info.value
        assert exc.name == "nonexistent_db"
        # Every key currently in REGISTRY must appear in exc.available
        for key in REGISTRY:
            assert key in exc.available, f"'{key}' missing from exc.available"

    def test_available_matches_registry_at_raise_time(self) -> None:
        from aaizaql.connectors import REGISTRY, get_connector

        with pytest.raises(ConnectorNotFound) as exc_info:
            get_connector("__no_such_connector__")

        assert set(exc_info.value.available) == set(REGISTRY.keys())


# ── Call-site integration: build_llm_provider ─────────────────────────────────


class TestBuildLLMProviderCallSite:
    def test_raises_with_correct_available_list(self) -> None:
        from aaizaql.llm import REGISTRY, build_llm_provider

        with pytest.raises(LLMProviderNotFound) as exc_info:
            build_llm_provider("nonexistent_llm", MagicMock())

        exc = exc_info.value
        assert exc.name == "nonexistent_llm"
        for key in REGISTRY:
            assert key in exc.available, f"'{key}' missing from exc.available"

    def test_available_matches_registry_at_raise_time(self) -> None:
        from aaizaql.llm import REGISTRY, build_llm_provider

        with pytest.raises(LLMProviderNotFound) as exc_info:
            build_llm_provider("__no_such_llm__", MagicMock())

        assert set(exc_info.value.available) == set(REGISTRY)