"""
AAIZAQL.llm.base
──────────────
Abstract base class for all LLM providers.
All providers expose a single method: complete(prompt, system) → str.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# from AAIZAQL.core.exceptions import LLMProviderNotFound


class LLMProvider(ABC):
    """Abstract base — every LLM adapter must implement complete()."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """
        Send a prompt to the LLM and return the text response.

        Parameters
        ----------
        prompt : str   The user-facing content (question + context).
        system : str   Optional system instruction override.

        Returns
        -------
        str  Raw text response from the model.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for logging."""
        ...
