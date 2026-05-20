"""
tests/test_llm_mistral.py
─────────────────────────
Unit tests for MistralProvider.
All tests use mocks — no real API key or network calls needed.

Fix: mistral_api_key is SecretStr — plain string assignment won't coerce and
.get_secret_value() would crash.  Every test now uses make_settings() which
passes SecretStr("...") via model_construct, bypassing env-var loading.
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
    Build a hermetic Settings instance via model_construct (no env-var reads).
    mistral_api_key defaults to a valid SecretStr so providers initialise cleanly.
    """
    defaults = dict(
        mistral_api_key=SecretStr("test-mistral-key"),
        mistral_model="mistral-large-latest",
        llm_max_tokens=1024,
        llm_temperature=0.0,
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


def _mock_mistral_client(content: str | None = "SELECT 1;") -> MagicMock:
    """Return a MagicMock that mimics the Mistral client with a canned response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5

    client = MagicMock()
    client.chat.complete.return_value = resp
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMistralProvider:
    """MistralProvider tests — patched Mistral client, zero network calls."""

    # ── Initialization ────────────────────────────────────────────────────

    def test_init_missing_api_key_raises(self) -> None:
        """Should raise LLMError immediately when mistral_api_key is None."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings(mistral_api_key=None)
        with pytest.raises(LLMError, match="AAIZAQL_MISTRAL_API_KEY is not set"):
            MistralProvider(settings)

    def test_init_missing_mistralai_package_raises(self) -> None:
        """Should raise LLMError if the mistralai package is absent."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings()
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(LLMError, match="mistralai package is not installed"):
                MistralProvider(settings)

    def test_init_success(self) -> None:
        """Should initialise and store the model name when settings are valid."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings(mistral_model="codestral-latest")
        with patch("aaizaql.llm.mistral_provider.Mistral") as mock_mistral:
            mock_mistral.return_value = MagicMock()
            provider = MistralProvider(settings)

        assert provider._model == "codestral-latest"
        mock_mistral.assert_called_once()

    def test_init_passes_secret_value_to_client(self) -> None:
        """The raw API key string (not SecretStr) should reach the Mistral client."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings(mistral_api_key=SecretStr("my-real-key"))
        with patch("aaizaql.llm.mistral_provider.Mistral") as mock_mistral:
            mock_mistral.return_value = MagicMock()
            MistralProvider(settings)

        call_kwargs = mock_mistral.call_args[1]
        assert call_kwargs["api_key"] == "my-real-key"

    # ── name property ─────────────────────────────────────────────────────

    def test_name_property(self) -> None:
        """name should be 'mistral/<model>'."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings(mistral_model="mistral-large-latest")
        with patch("aaizaql.llm.mistral_provider.Mistral"):
            provider = MistralProvider(settings)

        assert provider.name == "mistral/mistral-large-latest"

    # ── complete() ────────────────────────────────────────────────────────

    def test_complete_returns_sql(self) -> None:
        """complete() should return the model's text."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings(mistral_model="codestral-latest")
        mock_client = _mock_mistral_client(
            "SELECT product_name, SUM(quantity) FROM orders GROUP BY product_name;"
        )

        with patch("aaizaql.llm.mistral_provider.Mistral", return_value=mock_client):
            provider = MistralProvider(settings)
            result = provider.complete("Total quantity per product", system="You are a SQL generator")

        assert result == "SELECT product_name, SUM(quantity) FROM orders GROUP BY product_name;"
        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs["model"] == "codestral-latest"
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    def test_complete_uses_default_system_prompt(self) -> None:
        """When no system arg is given, SYSTEM_PROMPT should be used (non-empty)."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings()
        mock_client = _mock_mistral_client()

        with patch("aaizaql.llm.mistral_provider.Mistral", return_value=mock_client):
            provider = MistralProvider(settings)
            provider.complete("Simple query")

        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs["messages"][0]["content"]  # non-empty default prompt

    def test_complete_handles_none_content(self) -> None:
        """None content from the API should be normalised to an empty string."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings()
        mock_client = _mock_mistral_client(None)

        with patch("aaizaql.llm.mistral_provider.Mistral", return_value=mock_client):
            provider = MistralProvider(settings)
            result = provider.complete("test")

        assert result == ""

    def test_complete_api_error_raises_llm_error(self) -> None:
        """Any exception from the API should be re-raised as LLMError."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings()
        mock_client = MagicMock()
        mock_client.chat.complete.side_effect = Exception("Server error 500")

        with patch("aaizaql.llm.mistral_provider.Mistral", return_value=mock_client):
            provider = MistralProvider(settings)
            with pytest.raises(LLMError, match="Server error 500"):
                provider.complete("test query")

    def test_complete_respects_temperature_and_max_tokens(self) -> None:
        """Settings values should be forwarded to the API call."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings(llm_temperature=0.5, llm_max_tokens=1500)
        mock_client = _mock_mistral_client()

        with patch("aaizaql.llm.mistral_provider.Mistral", return_value=mock_client):
            provider = MistralProvider(settings)
            provider.complete("test")

        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1500

    def test_complete_with_mistral_small_model(self) -> None:
        """Should work correctly when mistral-small-latest is selected."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings(mistral_model="mistral-small-latest")
        mock_client = _mock_mistral_client("SELECT * FROM users LIMIT 10;")

        with patch("aaizaql.llm.mistral_provider.Mistral", return_value=mock_client):
            provider = MistralProvider(settings)
            result = provider.complete("First 10 users")

        assert "LIMIT 10" in result
        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs["model"] == "mistral-small-latest"

    def test_complete_with_custom_system_message(self) -> None:
        """A custom system string should appear in the messages sent to the API."""
        from aaizaql.llm.mistral_provider import MistralProvider

        settings = make_settings()
        mock_client = _mock_mistral_client()

        with patch("aaizaql.llm.mistral_provider.Mistral", return_value=mock_client):
            provider = MistralProvider(settings)
            provider.complete("test", system="Custom SQL expert prompt")

        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs["messages"][0]["content"] == "Custom SQL expert prompt"