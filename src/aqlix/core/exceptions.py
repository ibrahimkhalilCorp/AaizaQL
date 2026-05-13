"""
aqlix.core.exceptions
─────────────────────
All custom exceptions used throughout the library.
Every layer raises a specific exception — never a bare Exception.
"""

from __future__ import annotations


class AQLIXError(Exception):
    """Base exception for all aqlix errors."""


# ── Security ────────────────────────────────────────────────────────────────


class SecurityException(AQLIXError):
    """Raised when SQL fails security validation before execution."""

    def __init__(self, reason: str, sql: str = "") -> None:
        self.reason = reason
        self.sql = sql
        super().__init__(f"Security violation: {reason}" + (f"\nSQL: {sql}" if sql else ""))


class PromptInjectionDetected(SecurityException):
    """Raised when prompt injection is detected in the user's question."""


# ── SQL Generation ───────────────────────────────────────────────────────────


class SQLGenerationError(AQLIXError):
    """Raised when the LLM fails to produce valid SQL."""

    def __init__(self, question: str, detail: str = "") -> None:
        self.question = question
        super().__init__(f"Failed to generate SQL for: '{question}'. {detail}".strip())


class UnsupportedQueryError(AQLIXError):
    """Raised when the question cannot be answered with a SELECT statement."""

    def __init__(self, question: str) -> None:
        self.question = question
        super().__init__(
            f"Query cannot be answered with SELECT: '{question}'. "
            "Try rephrasing as a data retrieval question."
        )


class MaxRetriesExceeded(AQLIXError):
    """Raised when the self-correction loop exhausts all retries."""

    def __init__(self, sql: str, last_error: str, attempts: int = 3) -> None:
        self.sql = sql
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(
            f"SQL failed after {attempts} self-correction attempts.\n"
            f"Last error: {last_error}\n"
            f"Last SQL: {sql}"
        )


# ── Database / Connectors ────────────────────────────────────────────────────


class DatabaseError(AQLIXError):
    """Raised when query execution fails at the database level."""

    def __init__(self, message: str, sql: str = "", connector: str = "") -> None:
        self.sql = sql
        self.connector = connector
        parts = [message]
        if connector:
            parts.append(f"Connector: {connector}")
        if sql:
            parts.append(f"SQL: {sql}")
        super().__init__("\n".join(parts))


class ConnectionError(AQLIXError):
    """Raised when a database connection cannot be established."""

    def __init__(self, connector: str, dsn_hint: str = "", detail: str = "") -> None:
        self.connector = connector
        super().__init__(
            f"Cannot connect to '{connector}'"
            + (f" ({dsn_hint})" if dsn_hint else "")
            + (f": {detail}" if detail else "")
        )


class ConnectorNotFound(AQLIXError):
    """Raised when an unknown connector name is requested."""

    def __init__(self, name: str) -> None:
        from aqlix.connectors import REGISTRY  # lazy to avoid circular import

        super().__init__(
            f"No connector registered for '{name}'. " f"Available: {sorted(REGISTRY.keys())}"
        )


# ── LLM Providers ────────────────────────────────────────────────────────────


class LLMError(AQLIXError):
    """Raised when an LLM API call fails."""

    def __init__(self, provider: str, detail: str = "") -> None:
        self.provider = provider
        super().__init__(f"LLM provider '{provider}' error" + (f": {detail}" if detail else ""))


class LLMProviderNotFound(AQLIXError):
    """Raised when an unknown LLM provider name is requested."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"No LLM provider registered for '{name}'. " "Available: 'claude', 'openai', 'ollama'"
        )


# ── Vector Store / RAG ───────────────────────────────────────────────────────


class VectorStoreError(AQLIXError):
    """Raised when a vector store operation fails."""


class SchemaIngestionError(AQLIXError):
    """Raised when schema introspection or ingestion fails."""


# ── Federation ───────────────────────────────────────────────────────────────


class FederationError(AQLIXError):
    """Raised when a federated query fails."""

    def __init__(self, detail: str, sources: list[str] | None = None) -> None:
        self.sources = sources or []
        suffix = f" (sources: {self.sources})" if self.sources else ""
        super().__init__(f"Federation failed{suffix}: {detail}")


class CredentialError(AQLIXError):
    """Raised when credential encryption/decryption fails."""
