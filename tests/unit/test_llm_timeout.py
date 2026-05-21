"""
tests/unit/test_llm_timeout.py
───────────────────────────────
Unit tests for LLM call timeout (task 0.2).

Verifies:
- llm_timeout_seconds exposed in Settings (default 30, configurable)
- LLMTimeoutError is a subclass of LLMError
- LLMTimeoutError carries correct provider/timeout attributes
- Each provider's complete() raises LLMTimeoutError on timeout
- timeout param on complete() overrides the instance default
- generator.py and corrector.py forward the timeout
No real API keys or network calls are made.
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest
import requests

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import LLMError, LLMTimeoutError


# ── Settings ──────────────────────────────────────────────────────────────────


class TestSettings:
    def test_default_timeout(self) -> None:
        s = Settings()
        assert s.llm_timeout_seconds == 30

    def test_custom_timeout_via_kwarg(self) -> None:
        s = Settings(llm_timeout_seconds=60)
        assert s.llm_timeout_seconds == 60

    def test_timeout_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AAIZAQL_LLM_TIMEOUT_SECONDS", "45")
        s = Settings()
        assert s.llm_timeout_seconds == 45

    def test_timeout_minimum(self) -> None:
        with pytest.raises(Exception):
            Settings(llm_timeout_seconds=0)

    def test_timeout_maximum(self) -> None:
        with pytest.raises(Exception):
            Settings(llm_timeout_seconds=301)


# ── LLMTimeoutError ───────────────────────────────────────────────────────────


class TestLLMTimeoutError:
    def test_is_llm_error_subclass(self) -> None:
        err = LLMTimeoutError("groq", 30)
        assert isinstance(err, LLMError)

    def test_attributes(self) -> None:
        err = LLMTimeoutError("openai", 15)
        assert err.provider == "openai"
        assert err.timeout == 15

    def test_message_contains_seconds(self) -> None:
        err = LLMTimeoutError("claude", 20)
        assert "20" in str(err)
        assert "claude" in str(err)


# ── OllamaProvider timeout ────────────────────────────────────────────────────


class TestOllamaTimeout:
    def _make_provider(self, timeout: int = 5) -> "OllamaProvider":  # noqa: F821
        from aaizaql.llm.ollama_provider import OllamaProvider

        s = Settings(llm_timeout_seconds=timeout)
        return OllamaProvider(s)

    def test_timeout_raises_llm_timeout_error(self) -> None:
        provider = self._make_provider(timeout=5)
        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            with pytest.raises(LLMTimeoutError) as exc_info:
                provider.complete("SELECT 1")
        assert exc_info.value.provider == "ollama"
        assert exc_info.value.timeout == 5

    def test_timeout_param_overrides_settings(self) -> None:
        provider = self._make_provider(timeout=30)
        captured: list[int] = []

        def fake_post(url: str, json: dict, timeout: int) -> None:  # type: ignore[override]
            captured.append(timeout)
            raise requests.Timeout("forced")

        with patch("requests.post", side_effect=fake_post):
            with pytest.raises(LLMTimeoutError):
                provider.complete("SELECT 1", timeout=7)

        assert captured == [7]

    def test_requests_exception_not_timeout_raises_llm_error(self) -> None:
        provider = self._make_provider()
        with patch(
            "requests.post",
            side_effect=requests.ConnectionError("refused"),
        ):
            with pytest.raises(LLMError) as exc_info:
                provider.complete("SELECT 1")
        assert not isinstance(exc_info.value, LLMTimeoutError)


# ── OpenAI-compatible providers (openai, deepseek, perplexity) ────────────────
#   All three use the openai SDK; same mock pattern applies.


def _make_openai_timeout_exc() -> "openai.APITimeoutError":  # noqa: F821
    import openai

    return openai.APITimeoutError(request=MagicMock())


class TestOpenAITimeout:
    def _make_provider(self, timeout: int = 5) -> "OpenAIProvider":  # noqa: F821
        import openai

        from aaizaql.llm.openai_provider import OpenAIProvider

        s = Settings(openai_api_key="sk-fake", llm_timeout_seconds=timeout)
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = MagicMock(spec=openai.OpenAI)
        provider._model = "gpt-4o"
        provider._max_tokens = 1024
        provider._temperature = 0.0
        provider._timeout = timeout
        return provider

    def test_timeout_raises_llm_timeout_error(self) -> None:
        import openai

        provider = self._make_provider(timeout=5)
        provider._client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )
        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.complete("SELECT 1")
        assert exc_info.value.provider == "openai"
        assert exc_info.value.timeout == 5

    def test_other_openai_error_raises_llm_error(self) -> None:
        import openai

        provider = self._make_provider()
        provider._client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="bad key",
            response=MagicMock(),
            body=None,
        )
        with pytest.raises(LLMError) as exc_info:
            provider.complete("SELECT 1")
        assert not isinstance(exc_info.value, LLMTimeoutError)

    def test_timeout_kwarg_forwarded_to_sdk(self) -> None:
        provider = self._make_provider(timeout=30)
        provider._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))],
            usage=MagicMock(completion_tokens=5),
        )
        provider.complete("SELECT 1", timeout=12)
        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 12


class TestDeepSeekTimeout:
    def _make_provider(self, timeout: int = 5) -> "DeepSeekProvider":  # noqa: F821
        import openai

        from aaizaql.llm.deepseek_provider import DeepSeekProvider

        provider = DeepSeekProvider.__new__(DeepSeekProvider)
        provider._client = MagicMock(spec=openai.OpenAI)
        provider._model = "deepseek-chat"
        provider._max_tokens = 1024
        provider._temperature = 0.0
        provider._timeout = timeout
        return provider

    def test_timeout_raises_llm_timeout_error(self) -> None:
        import openai

        provider = self._make_provider(timeout=5)
        provider._client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )
        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.complete("SELECT 1")
        assert exc_info.value.provider == "deepseek"


class TestPerplexityTimeout:
    def _make_provider(self, timeout: int = 5) -> "PerplexityProvider":  # noqa: F821
        import openai

        from aaizaql.llm.perplexity_provider import PerplexityProvider

        provider = PerplexityProvider.__new__(PerplexityProvider)
        provider._client = MagicMock(spec=openai.OpenAI)
        provider._model = "sonar"
        provider._max_tokens = 1024
        provider._temperature = 0.0
        provider._timeout = timeout
        return provider

    def test_timeout_raises_llm_timeout_error(self) -> None:
        import openai

        provider = self._make_provider(timeout=5)
        provider._client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )
        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.complete("SELECT 1")
        assert exc_info.value.provider == "perplexity"


# ── Groq timeout ──────────────────────────────────────────────────────────────


class TestGroqTimeout:
    def _make_provider(self, timeout: int = 5) -> "GroqProvider":  # noqa: F821
        from aaizaql.llm.groq_provider import GroqProvider

        provider = GroqProvider.__new__(GroqProvider)
        provider._client = MagicMock()
        provider._model = "llama-3.3-70b-versatile"
        provider._max_tokens = 1024
        provider._temperature = 0.0
        provider._timeout = timeout
        return provider

    def test_timeout_raises_llm_timeout_error(self) -> None:
        from groq import APITimeoutError

        provider = self._make_provider(timeout=5)
        provider._client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )
        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.complete("SELECT 1")
        assert exc_info.value.provider == "groq"
        assert exc_info.value.timeout == 5

    def test_timeout_kwarg_forwarded_to_sdk(self) -> None:
        provider = self._make_provider(timeout=30)
        provider._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        provider.complete("SELECT 1", timeout=8)
        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 8


# ── Gemini timeout (ThreadPoolExecutor path) ──────────────────────────────────


class TestGeminiTimeout:
    def _make_provider(self, timeout: int = 1) -> "GeminiProvider":  # noqa: F821
        from aaizaql.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider._client = MagicMock()
        provider._model = "gemini-2.5-flash"
        provider._max_tokens = 1024
        provider._temperature = 0.0
        provider._timeout = timeout
        return provider

    def test_timeout_raises_llm_timeout_error(self) -> None:
        import time

        provider = self._make_provider(timeout=1)

        def slow_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            time.sleep(5)
            return MagicMock(text="SELECT 1", usage_metadata=MagicMock())

        provider._client.models.generate_content.side_effect = slow_call

        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.complete("SELECT 1")
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.timeout == 1

    def test_api_error_raises_llm_error(self) -> None:
        provider = self._make_provider(timeout=5)
        provider._client.models.generate_content.side_effect = RuntimeError("api error")
        with pytest.raises(LLMError) as exc_info:
            provider.complete("SELECT 1")
        assert not isinstance(exc_info.value, LLMTimeoutError)


# ── Mistral timeout (ThreadPoolExecutor path) ─────────────────────────────────


class TestMistralTimeout:
    def _make_provider(self, timeout: int = 1) -> "MistralProvider":  # noqa: F821
        from aaizaql.llm.mistral_provider import MistralProvider

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._model = "mistral-large-latest"
        provider._max_tokens = 1024
        provider._temperature = 0.0
        provider._timeout = timeout
        return provider

    def test_timeout_raises_llm_timeout_error(self) -> None:
        import time

        provider = self._make_provider(timeout=1)

        def slow_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            time.sleep(5)

        provider._client.chat.complete.side_effect = slow_call

        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.complete("SELECT 1")
        assert exc_info.value.provider == "mistral"
        assert exc_info.value.timeout == 1


# ── Claude timeout ────────────────────────────────────────────────────────────


class TestClaudeTimeout:
    def _make_provider(self, timeout: int = 5) -> "ClaudeProvider":  # noqa: F821
        import anthropic

        from aaizaql.llm.claude_provider import ClaudeProvider

        provider = ClaudeProvider.__new__(ClaudeProvider)
        provider._client = MagicMock(spec=anthropic.Anthropic)
        provider._client.api_key = "sk-fake"
        provider._model = "claude-sonnet-4-20250514"
        provider._max_tokens = 1024
        provider._temperature = 0.0
        provider._timeout = timeout
        return provider

    def test_timeout_raises_llm_timeout_error(self) -> None:
        import anthropic

        provider = self._make_provider(timeout=5)

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = anthropic.APITimeoutError(
            request=MagicMock()
        )
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        with patch("anthropic.Anthropic", return_value=mock_client_instance):
            with pytest.raises(LLMTimeoutError) as exc_info:
                provider.complete("SELECT 1")
        assert exc_info.value.provider == "claude"
        assert exc_info.value.timeout == 5


# ── Generator forwards timeout ────────────────────────────────────────────────


class TestGeneratorForwardsTimeout:
    def test_generate_passes_timeout_to_complete(self) -> None:
        from aaizaql.nlp.generator import SQLGenerator

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "SELECT 1"
        mock_llm.name = "mock/test"

        mock_vs = MagicMock()
        mock_vs.search.return_value = []

        mock_semantic = MagicMock()
        mock_semantic.has_enums.return_value = False
        mock_semantic.search_documentation.return_value = ""

        settings = Settings(llm_timeout_seconds=42)
        gen = SQLGenerator(mock_llm, mock_vs, settings, mock_semantic)
        gen.generate("show all users", history=[])

        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs.get("timeout") == 42


# ── Corrector forwards timeout ────────────────────────────────────────────────


class TestCorrectorForwardsTimeout:
    def test_correction_passes_timeout_to_complete(self) -> None:
        import pandas as pd

        from aaizaql.core.exceptions import DatabaseError
        from aaizaql.nlp.corrector import SelfCorrector

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "SELECT 1"

        mock_executor = MagicMock()
        mock_executor.execute.side_effect = [
            DatabaseError("syntax error", sql="BAD SQL"),
            pd.DataFrame({"id": [1]}),
        ]

        settings = Settings(llm_timeout_seconds=55, max_self_correction_retries=2)
        corrector = SelfCorrector(mock_llm, settings)
        corrector.execute_with_correction("BAD SQL", mock_executor, "get all users")

        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs.get("timeout") == 55