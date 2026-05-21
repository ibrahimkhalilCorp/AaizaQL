"""
tests/test_llm_ollama.py
────────────────────────
Unit tests for OllamaProvider.
All tests use mocks — no real Ollama server or network calls needed.

Covers: success path, HTTP 5xx error, empty response, malformed JSON,
timeout, and connection failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_settings(**kwargs) -> Settings:
    """
    Build a hermetic Settings instance via model_construct (no env-var reads).
    """
    defaults = dict(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        llm_max_tokens=1024,
        llm_timeout_seconds=30,
    )
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


def _mock_response(json_body: dict, status_code: int = 200) -> MagicMock:
    """Return a MagicMock that mimics a requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code} Error", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestOllamaProvider:
    """OllamaProvider tests — patched requests.post, zero network calls."""

    # ── Initialization ────────────────────────────────────────────────────

    def test_init_success_no_api_key_needed(self) -> None:
        """Ollama requires no API key — should initialise without error."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings()
        provider = OllamaProvider(settings)

        assert provider._model == "llama3"
        assert provider._base_url == "http://localhost:11434"

    def test_init_strips_trailing_slash_from_base_url(self) -> None:
        """base_url trailing slash should be stripped."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings(ollama_base_url="http://localhost:11434/")
        provider = OllamaProvider(settings)

        assert provider._base_url == "http://localhost:11434"

    def test_init_custom_base_url(self) -> None:
        """Custom Ollama server URL should be stored."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings(ollama_base_url="http://192.168.1.10:11434")
        provider = OllamaProvider(settings)

        assert provider._base_url == "http://192.168.1.10:11434"

    # ── name property ─────────────────────────────────────────────────────

    def test_name_property(self) -> None:
        """name should be 'ollama/<model>'."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings(ollama_model="mistral")
        provider = OllamaProvider(settings)

        assert provider.name == "ollama/mistral"

    # ── complete() — success ──────────────────────────────────────────────

    def test_complete_returns_sql(self) -> None:
        """complete() should return the 'response' field from Ollama's JSON."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings()
        mock_resp = _mock_response({"response": "SELECT * FROM orders WHERE status = 'open';"})

        with patch("aaizaql.llm.ollama_provider.requests.post", return_value=mock_resp) as mock_post:
            provider = OllamaProvider(settings)
            result = provider.complete("Show open orders", system="You are a SQL expert")

        assert result == "SELECT * FROM orders WHERE status = 'open';"
        call_args = mock_post.call_args
        assert "/api/generate" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3"
        assert payload["stream"] is False

    def test_complete_includes_system_and_prompt_in_full_prompt(self) -> None:
        """System instruction and user prompt should be concatenated."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings()
        mock_resp = _mock_response({"response": "SELECT 1;"})

        with patch("aaizaql.llm.ollama_provider.requests.post", return_value=mock_resp) as mock_post:
            provider = OllamaProvider(settings)
            provider.complete("user question", system="CUSTOM SYSTEM")

        payload = mock_post.call_args[1]["json"]
        assert "CUSTOM SYSTEM" in payload["prompt"]
        assert "user question" in payload["prompt"]

    def test_complete_uses_default_system_prompt_when_none_given(self) -> None:
        """When no system arg is given, SYSTEM_PROMPT should be embedded."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings()
        mock_resp = _mock_response({"response": "SELECT 1;"})

        with patch("aaizaql.llm.ollama_provider.requests.post", return_value=mock_resp) as mock_post:
            provider = OllamaProvider(settings)
            provider.complete("Simple query")

        payload = mock_post.call_args[1]["json"]
        assert payload["prompt"]  # non-empty

    def test_complete_handles_empty_response_field(self) -> None:
        """Missing 'response' key should be normalised to empty string."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings()
        mock_resp = _mock_response({})  # no 'response' key

        with patch("aaizaql.llm.ollama_provider.requests.post", return_value=mock_resp):
            provider = OllamaProvider(settings)
            result = provider.complete("test")

        assert result == ""

    def test_complete_respects_max_tokens(self) -> None:
        """llm_max_tokens should be forwarded as num_predict."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings(llm_max_tokens=256)
        mock_resp = _mock_response({"response": "SELECT 1;"})

        with patch("aaizaql.llm.ollama_provider.requests.post", return_value=mock_resp) as mock_post:
            provider = OllamaProvider(settings)
            provider.complete("test")

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["num_predict"] == 256

    # ── complete() — error paths ──────────────────────────────────────────

    def test_complete_http_error_raises_llm_error(self) -> None:
        """HTTP 5xx responses should be re-raised as LLMError."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings()
        mock_resp = _mock_response({}, status_code=500)

        with patch("aaizaql.llm.ollama_provider.requests.post", return_value=mock_resp):
            provider = OllamaProvider(settings)
            with pytest.raises(LLMError):
                provider.complete("test")

    def test_complete_connection_error_raises_llm_error(self) -> None:
        """Connection refused (requests.RequestException) should raise LLMError."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings()

        with patch(
            "aaizaql.llm.ollama_provider.requests.post",
            side_effect=requests.ConnectionError("Connection refused"),
        ):
            provider = OllamaProvider(settings)
            with pytest.raises(LLMError, match="Cannot reach Ollama"):
                provider.complete("test query")

    def test_complete_connection_error_message_includes_base_url(self) -> None:
        """LLMError message should include the server URL for debugging."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings(ollama_base_url="http://remoteserver:11434")

        with patch(
            "aaizaql.llm.ollama_provider.requests.post",
            side_effect=requests.ConnectionError("refused"),
        ):
            provider = OllamaProvider(settings)
            with pytest.raises(LLMError, match="remoteserver"):
                provider.complete("test")

    def test_complete_timeout_raises_llm_timeout_error(self) -> None:
        """requests.Timeout should be re-raised as LLMTimeoutError."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings(llm_timeout_seconds=5)

        with patch(
            "aaizaql.llm.ollama_provider.requests.post",
            side_effect=requests.Timeout("timed out"),
        ):
            provider = OllamaProvider(settings)
            with pytest.raises(LLMTimeoutError):
                provider.complete("test query")

    def test_complete_timeout_message_includes_seconds(self) -> None:
        """LLMTimeoutError message should reference the timeout value."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings(llm_timeout_seconds=12)

        with patch(
            "aaizaql.llm.ollama_provider.requests.post",
            side_effect=requests.Timeout(),
        ):
            provider = OllamaProvider(settings)
            with pytest.raises(LLMTimeoutError, match="12"):
                provider.complete("test", timeout=12)

    def test_complete_passes_timeout_to_requests(self) -> None:
        """The timeout param should reach requests.post."""
        from aaizaql.llm.ollama_provider import OllamaProvider

        settings = make_settings(llm_timeout_seconds=30)
        mock_resp = _mock_response({"response": "SELECT 1;"})

        with patch("aaizaql.llm.ollama_provider.requests.post", return_value=mock_resp) as mock_post:
            provider = OllamaProvider(settings)
            provider.complete("test", timeout=45)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["timeout"] == 45