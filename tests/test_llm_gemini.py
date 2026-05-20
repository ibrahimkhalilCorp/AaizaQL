"""
tests/test_llm_gemini.py
────────────────────────
Unit tests for GeminiProvider.
All tests use mocks — no real API key or network calls needed.

Fix: gemini_api_key is SecretStr — plain string assignment won't coerce and
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
    gemini_api_key defaults to a valid SecretStr so providers initialise cleanly.
    """
    defaults = dict(
        gemini_api_key=SecretStr("AIza-test-key"),
        gemini_model="gemini-2.5-flash",
        llm_max_tokens=1024,
        llm_temperature=0.0,
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


def _mock_genai_client(content: str | None = "SELECT 1;") -> MagicMock:
    """Return a MagicMock that mimics genai.Client with a canned response."""
    resp = MagicMock()
    resp.text = content
    resp.usage_metadata.prompt_token_count = 10
    resp.usage_metadata.candidates_token_count = 5

    client = MagicMock()
    client.models.generate_content.return_value = resp
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGeminiProvider:
    """GeminiProvider tests — patched google.genai client, zero network calls."""

    # ── Initialization ────────────────────────────────────────────────────

    def test_init_missing_api_key_raises(self) -> None:
        """Should raise LLMError immediately when gemini_api_key is None."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings(gemini_api_key=None)
        with pytest.raises(LLMError, match="AAIZAQL_GEMINI_API_KEY is not set"):
            GeminiProvider(settings)

    def test_init_missing_google_genai_package_raises(self) -> None:
        """Should raise LLMError if google-genai is not installed."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings()
        with patch("aaizaql.llm.gemini_provider.genai", None), pytest.raises(LLMError, match="google-genai package is not installed"):
            GeminiProvider(settings)

    def test_init_success(self) -> None:
        """Should initialise and store the model name when settings are valid."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings(gemini_model="gemini-2.5-pro")
        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            provider = GeminiProvider(settings)

        assert provider._model == "gemini-2.5-pro"
        mock_genai.Client.assert_called_once()

    def test_init_passes_secret_value_to_client(self) -> None:
        """The raw API key string (not SecretStr) should reach genai.Client."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings(gemini_api_key=SecretStr("AIza-real-key"))
        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            GeminiProvider(settings)

        call_kwargs = mock_genai.Client.call_args[1]
        assert call_kwargs["api_key"] == "AIza-real-key"

    # ── name property ─────────────────────────────────────────────────────

    def test_name_property(self) -> None:
        """name should be 'gemini/<model>'."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings(gemini_model="gemini-2.5-flash")
        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            provider = GeminiProvider(settings)

        assert provider.name == "gemini/gemini-2.5-flash"

    # ── complete() ────────────────────────────────────────────────────────

    def test_complete_returns_sql(self) -> None:
        """complete() should return the model's text."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings(gemini_model="gemini-2.5-flash")
        mock_client = _mock_genai_client(
            "SELECT id, name, salary FROM employees WHERE salary > 50000;"
        )

        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(settings)
            result = provider.complete("Get high-earning employees", system="You are a SQL expert")

        assert result == "SELECT id, name, salary FROM employees WHERE salary > 50000;"
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["contents"] == "Get high-earning employees"

    def test_complete_uses_default_system_prompt(self) -> None:
        """When no system arg is supplied, SYSTEM_PROMPT should be used (non-empty)."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings()
        mock_client = _mock_genai_client()

        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(settings)
            provider.complete("Simple query")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["config"].system_instruction  # non-empty

    def test_complete_handles_none_text(self) -> None:
        """None text from the API should be normalised to an empty string."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings()
        mock_client = _mock_genai_client(None)

        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(settings)
            result = provider.complete("test")

        assert result == ""

    def test_complete_api_error_raises_llm_error(self) -> None:
        """Any exception from the API should be re-raised as LLMError."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API quota exceeded")

        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(settings)
            with pytest.raises(LLMError, match="API quota exceeded"):
                provider.complete("test query")

    def test_complete_respects_temperature_and_max_tokens(self) -> None:
        """Settings values should be forwarded inside GenerateContentConfig."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings(llm_temperature=0.9, llm_max_tokens=2000)
        mock_client = _mock_genai_client()

        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(settings)
            provider.complete("test")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["config"].temperature == 0.9
        assert call_kwargs["config"].max_output_tokens == 2000

    def test_complete_passes_custom_system_instruction(self) -> None:
        """A custom system string should reach the GenerateContentConfig."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings()
        mock_client = _mock_genai_client()

        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(settings)
            provider.complete("test", system="Custom system message")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["config"].system_instruction == "Custom system message"

    def test_complete_with_gemini_pro_model(self) -> None:
        """Should work correctly when gemini-2.5-pro is selected."""
        from aaizaql.llm.gemini_provider import GeminiProvider

        settings = make_settings(gemini_model="gemini-2.5-pro")
        mock_client = _mock_genai_client(
            "SELECT AVG(price) FROM products WHERE category = 'Electronics';"
        )

        with patch("aaizaql.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(settings)
            result = provider.complete("Average electronics price")

        assert "AVG(price)" in result
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"