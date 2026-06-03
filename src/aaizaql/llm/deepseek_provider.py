"""
DeepSeek LLM provider for AaizaQL.

DeepSeek exposes an OpenAI-compatible REST API, so this provider is a thin
wrapper around the ``openai`` SDK pointed at DeepSeek's base URL.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Keep the ``openai`` module reference at module level so tests can patch
# both ``aaizaql.llm.deepseek_provider.OpenAI`` (class) and
# ``aaizaql.llm.deepseek_provider.openai`` (module).
# ---------------------------------------------------------------------------
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

from aaizaql.core.exceptions import LLMError

# ---------------------------------------------------------------------------
# Public constants (tests import these directly)
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"   # no /v1 suffix — tests assert exact value
_DEFAULT_MODEL = "deepseek-chat"


class DeepSeekProvider:
    """LLM provider that calls DeepSeek via the OpenAI-compatible API.

    Parameters
    ----------
    settings:
        AaizaQL Settings object. ``settings.deepseek_api_key`` is read and
        its secret value extracted.
    model:
        Model name — ``"deepseek-chat"`` (default) or ``"deepseek-reasoner"``.
    timeout:
        Request timeout in seconds (default: 60).
    """

    def __init__(
        self,
        settings,               # Settings object — tests pass a Settings instance
        model: str = _DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        # Extract the raw API key from the Settings object.
        raw_key = getattr(settings, "deepseek_api_key", None)
        if raw_key is None:
            raise LLMError("AAIZAQL_DEEPSEEK_API_KEY is not set")
        # Support both plain strings and Pydantic SecretStr.
        if hasattr(raw_key, "get_secret_value"):
            api_key = raw_key.get_secret_value()
        else:
            api_key = str(raw_key)
        if not api_key:
            raise LLMError("AAIZAQL_DEEPSEEK_API_KEY is not set")

        self._model = model
        self._timeout = timeout
        self._client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        """Return ``deepseek/<model>`` — e.g. ``deepseek/deepseek-chat``."""
        return f"deepseek/{self._model}"

    # ------------------------------------------------------------------
    # Core interface expected by QueryEngine / SQLGenerator
    # ------------------------------------------------------------------

    def complete(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        system: str | None = None,       # alias accepted by some callers
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Return the model's text completion.

        Parameters
        ----------
        user_prompt:
            The natural-language question from the user.
        system_prompt / system:
            Optional system instructions (either kwarg name is accepted).
        temperature:
            Sampling temperature forwarded to the API.
        max_tokens:
            Maximum tokens in the completion.
        timeout:
            Per-request timeout override in seconds.
        """
        effective_system = system_prompt or system
        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict = {"model": self._model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            from aaizaql.core.exceptions import LLMTimeoutError
            raise LLMTimeoutError(f"DeepSeek request timed out: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"DeepSeek API error: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            return ""   # tests expect empty string, not an exception
        return content

    # Convenience alias used by some internal callers.
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(user_prompt=user_prompt, system_prompt=system_prompt)

    def __repr__(self) -> str:
        return f"DeepSeekProvider(model={self._model!r})"
