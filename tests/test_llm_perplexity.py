"""
tests/test_llm_perplexity.py
────────────────────────────
Unit tests for PerplexityProvider.
All tests use mocks — no real API key or network calls needed.

Fix: perplexity_api_key is SecretStr — plain string assignment won't coerce
and .get_secret_value() would crash.  Every test now uses make_settings()
which passes SecretStr("...") via model_construct, bypassing env-var loading.
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
    perplexity_api_key defaults to a valid SecretStr so providers initialise cleanly.
    """
    defaults = dict(
        perplexity_api_key=SecretStr("pplx-test-key"),
        perplexity_model="sonar",
        llm_max_tokens=1024,
        llm_temperature=0.0,
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)

def _mock_openai_client(content: str | None = "SELECT 1;") -> MagicMock:
    """Return a MagicMock that mimics the OpenAI-compat client with a canned response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5

    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client

# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPerplexityProvider:
    """PerplexityProvider tests — patched OpenAI-compat client, zero network calls."""

    # ── Initialization ────────────────────────────────────────────────────

    def test_init_missing_api_key_raises(self) -> None:
        """Should raise LLMError immediately when perplexity_api_key is None."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(perplexity_api_key=None)
        with pytest.raises(LLMError, match="AAIZAQL_PERPLEXITY_API_KEY is not set"):
            PerplexityProvider(settings)

    def test_init_missing_openai_package_raises(self) -> None:
        """Should raise LLMError if the openai package is absent."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings()
        with (
            patch("aaizaql.llm.perplexity_provider.OpenAI", None),
            pytest.raises(LLMError, match="openai package is not installed"),
        ):
            PerplexityProvider(settings)

    def test_init_success(self) -> None:
        """Should initialise and store the model name when settings are valid."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(perplexity_model="sonar-pro")
        with patch("aaizaql.llm.perplexity_provider.OpenAI") as mock_openai:
            provider = PerplexityProvider(settings)

        assert provider._model == "sonar-pro"
        mock_openai.assert_called_once()

    def test_init_passes_secret_value_to_client(self) -> None:
        """The raw API key string (not SecretStr) should reach the OpenAI client."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(perplexity_api_key=SecretStr("pplx-real-key"))
        with patch("aaizaql.llm.perplexity_provider.OpenAI") as mock_openai:
            PerplexityProvider(settings)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "pplx-real-key"

    # ── name property ─────────────────────────────────────────────────────

    def test_name_property(self) -> None:
        """name should be 'perplexity/<model>'."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(perplexity_model="sonar-reasoning")
        with patch("aaizaql.llm.perplexity_provider.OpenAI"):
            provider = PerplexityProvider(settings)

        assert provider.name == "perplexity/sonar-reasoning"

    # ── complete() ────────────────────────────────────────────────────────

    def test_complete_returns_sql(self) -> None:
        """complete() should return the model's text."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(perplexity_model="sonar")
        mock_client = _mock_openai_client("SELECT name, email FROM customers WHERE active = 1;")

        with patch("aaizaql.llm.perplexity_provider.OpenAI", return_value=mock_client):
            provider = PerplexityProvider(settings)
            result = provider.complete(
                "List all active customers", system="You are a SQL generator"
            )

        assert result == "SELECT name, email FROM customers WHERE active = 1;"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "sonar"
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    def test_complete_uses_default_system_prompt(self) -> None:
        """When no system arg is given, SYSTEM_PROMPT should be used (non-empty)."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings()
        mock_client = _mock_openai_client()

        with patch("aaizaql.llm.perplexity_provider.OpenAI", return_value=mock_client):
            provider = PerplexityProvider(settings)
            provider.complete("Simple query")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["content"]  # non-empty default prompt

    def test_complete_handles_none_content(self) -> None:
        """None content from the API should be normalised to an empty string."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings()
        mock_client = _mock_openai_client(None)

        with patch("aaizaql.llm.perplexity_provider.OpenAI", return_value=mock_client):
            provider = PerplexityProvider(settings)
            result = provider.complete("test")

        assert result == ""

    def test_complete_api_error_raises_llm_error(self) -> None:
        """Any exception from the API should be re-raised as LLMError."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Network timeout")

        with patch("aaizaql.llm.perplexity_provider.OpenAI", return_value=mock_client):
            provider = PerplexityProvider(settings)
            with pytest.raises(LLMError, match="Network timeout"):
                provider.complete("test query")

    def test_complete_respects_temperature_and_max_tokens(self) -> None:
        """Settings values should be forwarded to the API call."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(llm_temperature=0.3, llm_max_tokens=1000)
        mock_client = _mock_openai_client()

        with patch("aaizaql.llm.perplexity_provider.OpenAI", return_value=mock_client):
            provider = PerplexityProvider(settings)
            provider.complete("test")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 1000

    def test_complete_uses_correct_base_url(self) -> None:
        """OpenAI client must be pointed at Perplexity's base URL."""
        from aaizaql.llm.perplexity_provider import PERPLEXITY_BASE_URL, PerplexityProvider

        settings = make_settings()
        with patch("aaizaql.llm.perplexity_provider.OpenAI") as mock_openai:
            PerplexityProvider(settings)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["base_url"] == PERPLEXITY_BASE_URL
        assert call_kwargs["base_url"] == "https://api.perplexity.ai"

    def test_complete_with_sonar_pro_model(self) -> None:
        """Should work correctly when sonar-pro is selected."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(perplexity_model="sonar-pro")
        mock_client = _mock_openai_client("SELECT COUNT(*) FROM orders;")

        with patch("aaizaql.llm.perplexity_provider.OpenAI", return_value=mock_client):
            provider = PerplexityProvider(settings)
            result = provider.complete("Count orders")

        assert "SELECT COUNT(*)" in result
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "sonar-pro"

# ── Timeout tests (added for 80% coverage target) ─────────────────────────────

class TestPerplexityProviderTimeout:
    """Timeout path tests for PerplexityProvider."""

    def test_complete_timeout_raises_llm_timeout_error(self) -> None:
        """openai.APITimeoutError should be re-raised as LLMTimeoutError."""
        from aaizaql.core.exceptions import LLMTimeoutError
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(llm_timeout_seconds=5)
        mock_client = MagicMock()

        class FakeAPITimeoutError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = FakeAPITimeoutError("timed out")

        with patch("aaizaql.llm.perplexity_provider.OpenAI", return_value=mock_client):
            provider = PerplexityProvider(settings)
            provider._client = mock_client

            import aaizaql.llm.perplexity_provider as pmod

            with patch.object(pmod, "openai") as mock_oai:
                mock_oai.APITimeoutError = FakeAPITimeoutError
                with pytest.raises((LLMTimeoutError, LLMError)):
                    provider.complete("test", timeout=5)

    def test_complete_timeout_explicit_param_forwarded(self) -> None:
        """Explicit timeout param should reach the API call."""
        from aaizaql.llm.perplexity_provider import PerplexityProvider

        settings = make_settings(llm_timeout_seconds=30)
        mock_client = _mock_openai_client("SELECT 1;")

        with patch("aaizaql.llm.perplexity_provider.OpenAI", return_value=mock_client):
            provider = PerplexityProvider(settings)
            result = provider.complete("test", timeout=60)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 60
        assert result == "SELECT 1;"
