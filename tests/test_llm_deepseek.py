"""
tests/test_llm_deepseek.py
──────────────────────────
Unit tests for DeepSeekProvider.
All tests use mocks — no real API key or network calls needed.

Fix: api_key fields are typed SecretStr, so plain string assignment does NOT
coerce — `settings.deepseek_api_key = "sk-..."` leaves a bare str that
crashes on `.get_secret_value()`.  Every test now wraps the key in SecretStr.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_settings(**kwargs) -> Settings:
    """
    Build a Settings instance with all env-var sources disabled so tests are
    hermetic, then layer on caller-supplied overrides.

    Using model_construct skips validation entirely — no env file, no env vars
    are read, and SecretStr fields receive proper SecretStr values.
    """
    defaults = dict(
        deepseek_api_key=SecretStr("sk-test-key"),
        deepseek_model="deepseek-chat",
        llm_max_tokens=1024,
        llm_temperature=0.0,
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDeepSeekProvider:
    """DeepSeekProvider tests — patched OpenAI client, zero network calls."""

    # ── Initialization ────────────────────────────────────────────────────

    def test_init_missing_api_key_raises(self) -> None:
        """Should raise LLMError immediately when deepseek_api_key is None."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings(deepseek_api_key=None)
        with pytest.raises(LLMError, match="AAIZAQL_DEEPSEEK_API_KEY is not set"):
            DeepSeekProvider(settings)

    def test_init_missing_openai_package_raises(self) -> None:
        """Should raise LLMError if the openai package is absent."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings()
        with patch.dict("sys.modules", {"openai": None}), pytest.raises(LLMError, match="openai package is not installed"):
                DeepSeekProvider(settings)

    def test_init_success(self) -> None:
        """Should initialise and store the model name when settings are valid."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings(deepseek_model="deepseek-chat")
        with patch("aaizaql.llm.deepseek_provider.OpenAI") as mock_openai:
            provider = DeepSeekProvider(settings)

        assert provider._model == "deepseek-chat"
        mock_openai.assert_called_once()

    def test_init_passes_secret_value_to_client(self) -> None:
        """The raw key string (not the SecretStr wrapper) reaches the OpenAI client."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings(deepseek_api_key=SecretStr("sk-real-key"))
        with patch("aaizaql.llm.deepseek_provider.OpenAI") as mock_openai:
            DeepSeekProvider(settings)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "sk-real-key"

    # ── name property ─────────────────────────────────────────────────────

    def test_name_property(self) -> None:
        """name should be 'deepseek/<model>'."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings(deepseek_model="deepseek-reasoner")
        with patch("aaizaql.llm.deepseek_provider.OpenAI"):
            provider = DeepSeekProvider(settings)

        assert provider.name == "deepseek/deepseek-reasoner"

    # ── complete() ────────────────────────────────────────────────────────

    def _mock_response(self, content: str | None) -> MagicMock:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.usage.prompt_tokens = 50
        resp.usage.completion_tokens = 20
        return resp

    def test_complete_returns_sql(self) -> None:
        """complete() should forward the model's text to the caller."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings(deepseek_model="deepseek-chat")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            "SELECT * FROM users WHERE age > 18;"
        )

        with patch("aaizaql.llm.deepseek_provider.OpenAI", return_value=mock_client):
            provider = DeepSeekProvider(settings)
            result = provider.complete("Get all adult users", system="You are a SQL expert")

        assert result == "SELECT * FROM users WHERE age > 18;"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    def test_complete_uses_default_system_prompt(self) -> None:
        """When no system arg is given, SYSTEM_PROMPT should be used (non-empty)."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("SELECT 1;")

        with patch("aaizaql.llm.deepseek_provider.OpenAI", return_value=mock_client):
            provider = DeepSeekProvider(settings)
            provider.complete("Simple query")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["content"]  # non-empty default prompt

    def test_complete_handles_none_content(self) -> None:
        """None content from the API should be normalised to an empty string."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(None)

        with patch("aaizaql.llm.deepseek_provider.OpenAI", return_value=mock_client):
            provider = DeepSeekProvider(settings)
            result = provider.complete("test")

        assert result == ""

    def test_complete_api_error_raises_llm_error(self) -> None:
        """Any exception from the API should be re-raised as LLMError."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API rate limit exceeded")

        with patch("aaizaql.llm.deepseek_provider.OpenAI", return_value=mock_client):
            provider = DeepSeekProvider(settings)
            with pytest.raises(LLMError, match="API rate limit exceeded"):
                provider.complete("test query")

    def test_complete_respects_temperature_and_max_tokens(self) -> None:
        """Settings values should be forwarded to the API call."""
        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        settings = make_settings(llm_temperature=0.7, llm_max_tokens=500)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("SELECT 1;")

        with patch("aaizaql.llm.deepseek_provider.OpenAI", return_value=mock_client):
            provider = DeepSeekProvider(settings)
            provider.complete("test")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 500

    def test_complete_uses_correct_base_url(self) -> None:
        """OpenAI client must be pointed at DeepSeek's base URL."""
        from aaizaql.llm.deepseek_provider import DEEPSEEK_BASE_URL, DeepSeekProvider

        settings = make_settings()
        with patch("aaizaql.llm.deepseek_provider.OpenAI") as mock_openai:
            DeepSeekProvider(settings)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["base_url"] == DEEPSEEK_BASE_URL
        assert call_kwargs["base_url"] == "https://api.deepseek.com"