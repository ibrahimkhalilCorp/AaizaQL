"""
tests/test_llm_openai.py
────────────────────────
Unit tests for OpenAIProvider.
All tests use mocks — no real API key or network calls needed.

Covers: success path, API error (5xx), empty response, malformed/None
content, timeout, and missing-key guard.
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
    openai_api_key defaults to a valid SecretStr so the provider initialises cleanly.
    """
    defaults = dict(
        openai_api_key=SecretStr("sk-test-openai-key"),
        openai_model="gpt-4o",
        llm_max_tokens=1024,
        llm_temperature=0.0,
        llm_timeout_seconds=30,
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


def _mock_openai_client(content: str | None = "SELECT 1;") -> MagicMock:
    """Return a MagicMock that mimics openai.OpenAI with a canned response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.completion_tokens = 10

    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestOpenAIProvider:
    """OpenAIProvider tests — patched OpenAI client, zero network calls."""

    # ── Initialization ────────────────────────────────────────────────────

    def test_init_missing_api_key_raises(self) -> None:
        """Should raise LLMError immediately when openai_api_key is None."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings(openai_api_key=None)
        with pytest.raises(LLMError, match="AAIZAQL_OPENAI_API_KEY is not set"):
            OpenAIProvider(settings)

    def test_init_success(self) -> None:
        """Should initialise and store the model name when settings are valid."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings(openai_model="gpt-4o")
        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = _mock_openai_client()
            provider = OpenAIProvider(settings)

        assert provider._model == "gpt-4o"

    def test_init_passes_secret_value_to_client(self) -> None:
        """The raw key string reaches the OpenAI client."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings(openai_api_key=SecretStr("sk-real-openai-key"))
        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = _mock_openai_client()
            OpenAIProvider(settings)

        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert call_kwargs["api_key"] == "sk-real-openai-key"

    # ── name property ─────────────────────────────────────────────────────

    def test_name_property(self) -> None:
        """name should be 'openai/<model>'."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings(openai_model="gpt-4o-mini")
        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = _mock_openai_client()
            provider = OpenAIProvider(settings)

        assert provider.name == "openai/gpt-4o-mini"

    # ── complete() — success ──────────────────────────────────────────────

    def test_complete_returns_sql(self) -> None:
        """complete() should forward the model's text to the caller."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings()
        mock_client = _mock_openai_client("SELECT id, name FROM users LIMIT 10;")

        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.APITimeoutError = TimeoutError
            mock_openai.OpenAIError = Exception
            provider = OpenAIProvider(settings)
            result = provider.complete("Get first 10 users", system="You are a SQL expert")

        assert result == "SELECT id, name FROM users LIMIT 10;"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    def test_complete_uses_default_system_prompt(self) -> None:
        """When no system arg is given, SYSTEM_PROMPT should be used (non-empty)."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings()
        mock_client = _mock_openai_client("SELECT 1;")

        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.APITimeoutError = TimeoutError
            mock_openai.OpenAIError = Exception
            provider = OpenAIProvider(settings)
            provider.complete("Simple query")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["content"]  # non-empty default prompt

    def test_complete_handles_none_content(self) -> None:
        """None content from the API should be normalised to an empty string."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings()
        mock_client = _mock_openai_client(None)

        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.APITimeoutError = TimeoutError
            mock_openai.OpenAIError = Exception
            provider = OpenAIProvider(settings)
            result = provider.complete("test")

        assert result == ""

    def test_complete_respects_temperature_and_max_tokens(self) -> None:
        """Settings values should be forwarded to the API call."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings(llm_temperature=0.3, llm_max_tokens=256)
        mock_client = _mock_openai_client("SELECT 1;")

        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.APITimeoutError = TimeoutError
            mock_openai.OpenAIError = Exception
            provider = OpenAIProvider(settings)
            provider.complete("test")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 256

    # ── complete() — error paths ──────────────────────────────────────────

    def test_complete_api_error_raises_llm_error(self) -> None:
        """openai.OpenAIError should be re-raised as LLMError."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings()
        mock_client = _mock_openai_client()

        class FakeOpenAIError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = FakeOpenAIError("503 Service Unavailable")

        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.APITimeoutError = TimeoutError
            mock_openai.OpenAIError = FakeOpenAIError
            provider = OpenAIProvider(settings)
            with pytest.raises(LLMError, match="503 Service Unavailable"):
                provider.complete("test query")

    def test_complete_timeout_raises_llm_timeout_error(self) -> None:
        """openai.APITimeoutError should be re-raised as LLMTimeoutError."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings(llm_timeout_seconds=10)
        mock_client = _mock_openai_client()

        class FakeAPITimeoutError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = FakeAPITimeoutError("timed out")

        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.APITimeoutError = FakeAPITimeoutError
            mock_openai.OpenAIError = Exception
            provider = OpenAIProvider(settings)
            with pytest.raises(LLMTimeoutError):
                provider.complete("test query")

    def test_complete_timeout_message_includes_seconds(self) -> None:
        """LLMTimeoutError message should reference the timeout value."""
        from aaizaql.llm.openai_provider import OpenAIProvider

        settings = make_settings(llm_timeout_seconds=20)
        mock_client = _mock_openai_client()

        class FakeAPITimeoutError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = FakeAPITimeoutError()

        with patch("aaizaql.llm.openai_provider.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.APITimeoutError = FakeAPITimeoutError
            mock_openai.OpenAIError = Exception
            provider = OpenAIProvider(settings)
            with pytest.raises(LLMTimeoutError, match="20"):
                provider.complete("test", timeout=20)
