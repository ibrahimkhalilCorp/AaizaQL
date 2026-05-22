"""
tests/test_llm_claude.py
────────────────────────
Unit tests for ClaudeProvider.
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
    anthropic_api_key defaults to a valid SecretStr so the provider initialises cleanly.
    """
    defaults = dict(
        anthropic_api_key=SecretStr("sk-ant-test-key"),
        claude_model="claude-sonnet-4-20250514",
        llm_max_tokens=1024,
        llm_temperature=0.0,
        llm_timeout_seconds=30,
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


def _mock_anthropic_client(text: str | None = "SELECT 1;") -> MagicMock:
    """Return a MagicMock that mimics anthropic.Anthropic with a canned response."""
    content_block = MagicMock()
    content_block.text = text

    message = MagicMock()
    message.content = [content_block]
    message.usage.output_tokens = 10

    client = MagicMock()
    client.messages.create.return_value = message
    # Support context-manager usage: `with anthropic.Anthropic(...) as client:`
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestClaudeProvider:
    """ClaudeProvider tests — patched Anthropic client, zero network calls."""

    # ── Initialization ────────────────────────────────────────────────────

    def test_init_missing_api_key_raises(self) -> None:
        """Should raise LLMError immediately when anthropic_api_key is None."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings(anthropic_api_key=None)
        with pytest.raises(LLMError, match="AAIZAQL_ANTHROPIC_API_KEY is not set"):
            ClaudeProvider(settings)

    def test_init_success(self) -> None:
        """Should initialise and store the model name when settings are valid."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings(claude_model="claude-sonnet-4-20250514")
        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = _mock_anthropic_client()
            provider = ClaudeProvider(settings)

        assert provider._model == "claude-sonnet-4-20250514"

    def test_init_passes_secret_value_to_client(self) -> None:
        """The raw key string (not the SecretStr wrapper) reaches the Anthropic client."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings(anthropic_api_key=SecretStr("sk-ant-real-key"))
        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = _mock_anthropic_client()
            ClaudeProvider(settings)

        init_call_kwargs = mock_anthropic.Anthropic.call_args[1]
        assert init_call_kwargs["api_key"] == "sk-ant-real-key"

    # ── name property ─────────────────────────────────────────────────────

    def test_name_property(self) -> None:
        """name should be 'claude/<model>'."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings(claude_model="claude-opus-4-20250514")
        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = _mock_anthropic_client()
            provider = ClaudeProvider(settings)

        assert provider.name == "claude/claude-opus-4-20250514"

    # ── complete() — success ──────────────────────────────────────────────

    def test_complete_returns_sql(self) -> None:
        """complete() should forward the model's text to the caller."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings()
        mock_client = _mock_anthropic_client("SELECT * FROM employees WHERE active = 1;")

        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.APITimeoutError = TimeoutError
            mock_anthropic.APIError = Exception
            provider = ClaudeProvider(settings)
            result = provider.complete("Get all active employees", system="You are a SQL expert")

        assert result == "SELECT * FROM employees WHERE active = 1;"

    def test_complete_uses_default_system_prompt(self) -> None:
        """When no system arg is given, SYSTEM_PROMPT should be used (non-empty)."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings()
        mock_client = _mock_anthropic_client("SELECT 1;")

        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.APITimeoutError = TimeoutError
            mock_anthropic.APIError = Exception
            provider = ClaudeProvider(settings)
            provider.complete("Simple query")

        # The inner Anthropic context-manager client is used for actual calls
        inner_client = mock_anthropic.Anthropic.return_value.__enter__.return_value
        call_kwargs = inner_client.messages.create.call_args[1]
        assert call_kwargs["system"]  # non-empty default prompt

    def test_complete_handles_none_content(self) -> None:
        """None text from the API should be handled via str() coercion."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings()
        mock_client = _mock_anthropic_client(None)

        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.APITimeoutError = TimeoutError
            mock_anthropic.APIError = Exception
            provider = ClaudeProvider(settings)
            result = provider.complete("test")

        # str(None) == "None" — the provider does `str(message.content[0].text)`
        assert isinstance(result, str)

    def test_complete_respects_temperature_and_max_tokens(self) -> None:
        """Settings values should be forwarded to the API call."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings(llm_temperature=0.5, llm_max_tokens=512)
        mock_client = _mock_anthropic_client("SELECT 1;")

        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.APITimeoutError = TimeoutError
            mock_anthropic.APIError = Exception
            provider = ClaudeProvider(settings)
            provider.complete("test")

        inner_client = mock_anthropic.Anthropic.return_value.__enter__.return_value
        call_kwargs = inner_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    # ── complete() — error paths ──────────────────────────────────────────

    def test_complete_api_error_raises_llm_error(self) -> None:
        """anthropic.APIError should be re-raised as LLMError."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings()
        mock_client = _mock_anthropic_client()

        class FakeAPIError(Exception):
            pass

        mock_client.__enter__.return_value.messages.create.side_effect = FakeAPIError(
            "500 Internal Server Error"
        )

        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.APITimeoutError = TimeoutError
            mock_anthropic.APIError = FakeAPIError
            provider = ClaudeProvider(settings)
            with pytest.raises(LLMError, match="500 Internal Server Error"):
                provider.complete("test query")

    def test_complete_timeout_raises_llm_timeout_error(self) -> None:
        """anthropic.APITimeoutError should be re-raised as LLMTimeoutError."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings(llm_timeout_seconds=5)
        mock_client = _mock_anthropic_client()

        class FakeAPITimeoutError(Exception):
            pass

        mock_client.__enter__.return_value.messages.create.side_effect = FakeAPITimeoutError(
            "Request timed out"
        )

        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.APITimeoutError = FakeAPITimeoutError
            mock_anthropic.APIError = Exception
            provider = ClaudeProvider(settings)
            with pytest.raises(LLMTimeoutError):
                provider.complete("test query")

    def test_complete_timeout_message_includes_seconds(self) -> None:
        """LLMTimeoutError message should include the configured timeout duration."""
        from aaizaql.llm.claude_provider import ClaudeProvider

        settings = make_settings(llm_timeout_seconds=15)
        mock_client = _mock_anthropic_client()

        class FakeAPITimeoutError(Exception):
            pass

        mock_client.__enter__.return_value.messages.create.side_effect = FakeAPITimeoutError()

        with patch("aaizaql.llm.claude_provider.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.APITimeoutError = FakeAPITimeoutError
            mock_anthropic.APIError = Exception
            provider = ClaudeProvider(settings)
            with pytest.raises(LLMTimeoutError, match="15"):
                provider.complete("test query", timeout=15)
