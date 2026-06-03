"""
DeepSeek LLM provider for AaizaQL.

DeepSeek exposes an OpenAI-compatible REST API, so this provider is a thin
wrapper around the ``openai`` SDK pointed at DeepSeek's base URL.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Eager import guard — expose OpenAI at module level so tests can patch it
# ---------------------------------------------------------------------------
try:
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
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_DEFAULT_MODEL = "deepseek-chat"


class DeepSeekProvider:
    """LLM provider that calls DeepSeek via the OpenAI-compatible API.

    Parameters
    ----------
    api_key:
        Your DeepSeek API key (``AAIZAQL_DEEPSEEK_API_KEY`` env var).
    model:
        Model name — ``"deepseek-chat"`` (default) or
        ``"deepseek-reasoner"``.
    timeout:
        Request timeout in seconds (default: 60).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMError("AAIZAQL_DEEPSEEK_API_KEY is not set")
        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "deepseek"

    # ------------------------------------------------------------------
    # Core interface expected by QueryEngine / SQLGenerator
    # ------------------------------------------------------------------

    def complete(self, user_prompt: str, system_prompt: str | None = None) -> str:
        """Return the model's text completion for the given prompts.

        Parameters
        ----------
        user_prompt:
            The natural-language question from the user.
        system_prompt:
            Optional instructions / schema context injected as the ``system`` role.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        except Exception as exc:
            raise LLMError(f"DeepSeek API error: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            raise LLMError(f"DeepSeek returned an empty response for model '{self._model}'.")
        return content

    # Convenience alias used by some internal callers.
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(user_prompt=user_prompt, system_prompt=system_prompt)

    def __repr__(self) -> str:
        return f"DeepSeekProvider(model={self._model!r})"
