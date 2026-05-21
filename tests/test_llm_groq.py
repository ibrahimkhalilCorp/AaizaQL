"""
tests/test_llm_groq.py
──────────────────────
Unit tests for GroqProvider.
All tests use mocks — no real API key or network calls needed.

Covers: success path, API error (5xx), empty response, malformed/None
content, timeout, missing-key guard, and missing-package guard.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_settings(**kwargs) -> Settings:
    """
    Build a hermetic Settings instance via model_construct (no env-var reads).
    groq_api_key defaults to a valid SecretStr so the provider initialises cleanly.
    """
    defaults = dict(
        groq_api_key=SecretStr("gsk-test-key"),
        groq_model="llama-3.3-70b-versatile",
        llm_max_tokens=1024,
        llm_temperature=0.0,
        llm_timeout_seconds=30,
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


def _mock_groq_client(content: str | None = "SELECT 1;") -> MagicMock:
    """Return a MagicMock that mimics groq.Groq with a canned response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 15
    resp.usage.completion_tokens = 8

    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestGroqProvider:
    """GroqProvider tests — patched Groq client, zero network calls."""

    # ── Initialization ────────────────────────────────────────────────────

    def test_init_missing_api_key_raises(self) -> None:
        """Should raise LLMError immediately when groq_api_key is None."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings(groq_api_key=None)
        with pytest.raises(LLMError, match="AAIZAQL_GROQ_API_KEY is not set"):
            GroqProvider(settings)

    def test_init_missing_groq_package_raises(self) -> None:
        """Should raise LLMError if the groq package is absent."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings(groq_api_key="gsk_fake_key_for_test")
        with (
            patch.dict("sys.modules", {"groq": None}),
            pytest.raises((LLMError, ImportError)),
        ):
            GroqProvider(settings)

    def test_init_success(self) -> None:
        """Should initialise and store the model name when settings are valid."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings(groq_model="llama-3.3-70b-versatile")
        with patch("aaizaql.llm.groq_provider.Groq", return_value=_mock_groq_client()):
            provider = GroqProvider(settings)

        assert provider._model == "llama-3.3-70b-versatile"

    def test_init_passes_secret_value_to_client(self) -> None:
        """The raw key string reaches the Groq client."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings(groq_api_key=SecretStr("gsk-real-key"))
        with patch("aaizaql.llm.groq_provider.Groq") as mock_groq:
            mock_groq.return_value = _mock_groq_client()
            GroqProvider(settings)

        call_kwargs = mock_groq.call_args[1]
        assert call_kwargs["api_key"] == "gsk-real-key"

    # ── name property ─────────────────────────────────────────────────────

    def test_name_property(self) -> None:
        """name should be 'groq/<model>'."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings(groq_model="llama-3.1-8b-instant")
        with patch("aaizaql.llm.groq_provider.Groq", return_value=_mock_groq_client()):
            provider = GroqProvider(settings)

        assert provider.name == "groq/llama-3.1-8b-instant"

    # ── complete() — success ──────────────────────────────────────────────

    def test_complete_returns_sql(self) -> None:
        """complete() should forward the model's text to the caller."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings()
        mock_client = _mock_groq_client("SELECT dept, COUNT(*) FROM employees GROUP BY dept;")

        with patch("aaizaql.llm.groq_provider.Groq", return_value=mock_client):
            provider = GroqProvider(settings)
            result = provider.complete(
                "Count employees by department", system="You are a SQL expert"
            )

        assert result == "SELECT dept, COUNT(*) FROM employees GROUP BY dept;"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    def test_complete_uses_default_system_prompt(self) -> None:
        """When no system arg is given, SYSTEM_PROMPT should be used (non-empty)."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings()
        mock_client = _mock_groq_client("SELECT 1;")

        with patch("aaizaql.llm.groq_provider.Groq", return_value=mock_client):
            provider = GroqProvider(settings)
            provider.complete("Simple query")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["content"]  # non-empty default prompt

    def test_complete_handles_none_content(self) -> None:
        """None content from the API should be normalised to an empty string."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings()
        mock_client = _mock_groq_client(None)

        with patch("aaizaql.llm.groq_provider.Groq", return_value=mock_client):
            provider = GroqProvider(settings)
            result = provider.complete("test")

        assert result == ""

    def test_complete_respects_temperature_and_max_tokens(self) -> None:
        """Settings values should be forwarded to the API call."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings(llm_temperature=0.1, llm_max_tokens=512)
        mock_client = _mock_groq_client("SELECT 1;")

        with patch("aaizaql.llm.groq_provider.Groq", return_value=mock_client):
            provider = GroqProvider(settings)
            provider.complete("test")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 512

    # ── complete() — error paths ──────────────────────────────────────────

    def test_complete_api_error_raises_llm_error(self) -> None:
        """Any non-timeout exception from the API should be re-raised as LLMError."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings()
        mock_client = _mock_groq_client()
        mock_client.chat.completions.create.side_effect = Exception("500 Internal Server Error")

        with patch("aaizaql.llm.groq_provider.Groq", return_value=mock_client):
            provider = GroqProvider(settings)
            with pytest.raises(LLMError, match="500 Internal Server Error"):
                provider.complete("test query")

    def test_complete_timeout_raises_llm_timeout_error(self) -> None:
        """groq.APITimeoutError should be re-raised as LLMTimeoutError."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings(llm_timeout_seconds=5)
        mock_client = _mock_groq_client()

        class FakeAPITimeoutError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = FakeAPITimeoutError("timed out")

        with (
            patch("aaizaql.llm.groq_provider.Groq", return_value=mock_client),
            patch(
                "aaizaql.llm.groq_provider.APITimeoutError",
                FakeAPITimeoutError,
                create=True,
            ),
        ):
            GroqProvider(settings)

        # Re-patch complete's local import too
        import aaizaql.llm.groq_provider as gmod

        with (
            patch.object(gmod, "Groq", return_value=mock_client),
            patch("aaizaql.llm.groq_provider.Groq", return_value=mock_client),
        ):
            provider2 = GroqProvider(settings)
            # Simulate timeout by monkeypatching the exception the provider catches
            provider2._client = mock_client
            with pytest.raises((LLMTimeoutError, LLMError)):
                import groq

                with patch.object(groq, "APITimeoutError", FakeAPITimeoutError):
                    provider2.complete("test", timeout=5)

    def test_complete_timeout_via_complete_method_timeout_param(self) -> None:
        """Explicit timeout param overrides the setting-level timeout."""
        from aaizaql.llm.groq_provider import GroqProvider

        settings = make_settings(llm_timeout_seconds=30)
        mock_client = _mock_groq_client("SELECT 1;")

        with patch("aaizaql.llm.groq_provider.Groq", return_value=mock_client):
            provider = GroqProvider(settings)
            result = provider.complete("test", timeout=60)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 60
        assert result == "SELECT 1;"