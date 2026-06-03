"""
DeepSeek LLM provider for AaizaQL.

DeepSeek exposes an OpenAI-compatible REST API, so this provider is a thin
wrapper around the ``openai`` SDK pointed at DeepSeek's base URL.

Fix (Issue #2): An explicit import check at module load time raises an
``ImportError`` that points users to the correct install command::

    pip install "aaizaql[deepseek]"

instead of the unhelpful ``pip install openai`` message that appeared before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Eager import guard — Issue #2
# ---------------------------------------------------------------------------
# The openai package is NOT in the base dependencies and NOT pulled in by
# aaizaql[postgres].  Check for it immediately so users get a clear, actionable
# error at QueryEngine.__init__ time rather than a confusing AttributeError
# deep inside the call stack.
# ---------------------------------------------------------------------------
try:
    import openai as _openai
except ImportError as _exc:
    raise ImportError(
        "DeepSeek requires the openai package, which is not installed.\n"
        'Fix: pip install "aaizaql[deepseek]"\n'
        "Or combined with a database driver: "
        'pip install "aaizaql[postgres,deepseek]"'
    ) from _exc

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
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
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._client = _openai.OpenAI(
            api_key=api_key,
            base_url=_DEEPSEEK_BASE_URL,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Core interface expected by QueryEngine / SQLGenerator
    # ------------------------------------------------------------------

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text completion for the given prompts.

        Parameters
        ----------
        system_prompt:
            Instructions / schema context injected as the ``system`` role.
        user_prompt:
            The natural-language question from the user.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"DeepSeek returned an empty response for model '{self._model}'.")
        return content

    # Convenience alias used by some internal callers.
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete(system_prompt, user_prompt)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DeepSeekProvider(model={self._model!r})"
