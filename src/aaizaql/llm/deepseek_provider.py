"""
DeepSeek LLM provider for AaizaQL.
"""

from __future__ import annotations

try:
    import openai
    from openai import OpenAI
except ImportError as _exc:
    raise ImportError(
        "DeepSeek requires the openai package, which is not installed.\n"
        'Fix: pip install "aaizaql[deepseek]"\n'
        "Or combined with a database driver: "
        'pip install "aaizaql[postgres,deepseek]"'
    ) from _exc

from aaizaql.core.exceptions import LLMError, LLMTimeoutError

# Public constants (tests import these directly)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-chat"


class DeepSeekProvider:
    """LLM provider that calls DeepSeek via the OpenAI-compatible API."""

    def __init__(
        self,
        settings,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if model is None:
            model = getattr(settings, "deepseek_model", None) or _DEFAULT_MODEL

        raw_key = getattr(settings, "deepseek_api_key", None)
        if raw_key is None:
            raise LLMError("AAIZAQL_DEEPSEEK_API_KEY is not set")
        api_key = (
            raw_key.get_secret_value() if hasattr(raw_key, "get_secret_value") else str(raw_key)
        )
        if not api_key:
            raise LLMError("AAIZAQL_DEEPSEEK_API_KEY is not set")

        self._model = model
        self._timeout = timeout
        if not callable(OpenAI):
            raise LLMError("openai package is not installed")
        self._client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"deepseek/{self._model}"

    def complete(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        system: str | None = None,
        timeout: float | None = None,
    ) -> str:
        effective_system = system_prompt or system
        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": user_prompt})

        # Always include temperature and max_tokens so callers can assert on them
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("deepseek", int(self._timeout)) from exc
        except Exception as exc:
            raise LLMError(f"DeepSeek API error: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(user_prompt=user_prompt, system_prompt=system_prompt)

    def __repr__(self) -> str:
        return f"DeepSeekProvider(model={self._model!r})"
