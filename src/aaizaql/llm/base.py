"""
aaizaql.llm.base
────────────────
Abstract base class for all LLM provider adapters.

Every provider in ``aaizaql.llm`` must subclass :class:`LLMProvider` and
implement :meth:`complete`.  The engine calls only this interface — new
providers require zero changes to ``core/engine.py``.

Plugin contract::

    class MyProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "myprovider"

        def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
            ...  # call your LLM API and return the raw text response

    # Register in llm/__init__.py so build_llm_provider() can find it.

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for all LLM provider adapters.

    Subclass this and implement :meth:`complete` and :attr:`name` to add a new
    LLM provider.  Register the subclass in ``llm/__init__.py``; no other file
    needs to change.
    """

    @abstractmethod
    def complete(self, prompt: str, system: str = "", timeout: int = 0) -> str:
        """Send a prompt to the LLM and return the raw text response.

        Args:
            prompt: User-facing content — typically the natural language question
                plus schema context assembled by the generator.
            system: Optional system instruction override.  When empty, providers
                use their own default system prompt.
            timeout: Seconds before the API call is cancelled.  Pass ``0`` to
                use the value from :attr:`~aaizaql.core.config.Settings.llm_timeout_seconds`.

        Returns:
            Raw text response from the model, with no post-processing applied.

        Raises:
            LLMError: On any API failure (network error, auth error, rate limit,
                etc.).  Providers must wrap provider-specific exceptions in
                :exc:`~aaizaql.core.exceptions.LLMError`.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier used in logs and error messages.

        Returns:
            Lowercase string, e.g. ``"groq"``, ``"claude"``, ``"openai"``.
        """
        ...
