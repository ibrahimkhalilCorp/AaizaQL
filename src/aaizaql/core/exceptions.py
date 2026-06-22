"""
aaizaql.core.exceptions
───────────────────────
All custom exceptions used throughout the library.
Every layer raises a specific subclass — never a bare ``Exception``.

Exception hierarchy
───────────────────
::

    AAIZAQLError                        # base for every library error
    ├── SecurityException               # SQL failed security gate
    │   └── PromptInjectionDetected     # user question contained injection attempt
    ├── SQLGenerationError              # LLM could not produce valid SQL
    ├── RateLimitError                  # tenant query quota exceeded
    ├── UnsupportedQueryError           # question requires non-SELECT statement
    ├── MaxRetriesExceeded              # self-correction loop ran out of attempts
    ├── DatabaseError                   # query execution failed at DB level
    ├── ConnectionError                 # could not establish a DB connection
    ├── ConnectorNotFound               # unknown connector name requested
    ├── LLMError                        # LLM API call failed
    │   └── LLMTimeoutError             # LLM API call exceeded timeout
    ├── LLMProviderNotFound             # unknown LLM provider name requested
    ├── VectorStoreError                # vector store operation failed
    ├── SchemaIngestionError            # schema introspection or ingestion failed
    ├── FederationError                 # federated query failed
    └── CredentialError                 # credential encryption/decryption failed

Catching exceptions
───────────────────
Catch the most specific subclass you care about; fall back to ``AAIZAQLError``
to handle any library error uniformly::

    from aaizaql.core.exceptions import AAIZAQLError, RateLimitError

    try:
        result = engine.query("How many users signed up last week?")
    except RateLimitError as e:
        time.sleep(e.retry_after_seconds)
    except AAIZAQLError as e:
        logger.error("Query failed: %s", e)

.. note::
    ``ConnectionError`` defined here shadows the Python built-in of the same
    name within this module.  Always import it explicitly from this module
    rather than relying on the built-in.
"""

from __future__ import annotations

__all__ = [
    "AAIZAQLError",
    "SecurityException",
    "PromptInjectionDetected",
    "SQLGenerationError",
    "RateLimitError",
    "UnsupportedQueryError",
    "MaxRetriesExceeded",
    "DatabaseError",
    "ConnectionError",
    "ConnectorNotFound",
    "LLMError",
    "LLMTimeoutError",
    "LLMProviderNotFound",
    "VectorStoreError",
    "SchemaIngestionError",
    "FederationError",
    "CredentialError",
]


class AAIZAQLError(Exception):
    """Base exception for all AAIZAQL errors.

    Catch this class to handle any library error with a single ``except``
    clause, or catch a specific subclass for finer-grained error handling.
    """


# ── Security ─────────────────────────────────────────────────────────────────


class SecurityException(AAIZAQLError):
    """Raised when SQL fails security validation before execution.

    Attributes:
        reason: Human-readable explanation of the violation.
        sql: The offending SQL string (empty string if not applicable).
    """

    def __init__(self, reason: str, sql: str = "") -> None:
        self.reason = reason
        self.sql = sql
        super().__init__(f"Security violation: {reason}" + (f"\nSQL: {sql}" if sql else ""))


class PromptInjectionDetected(SecurityException):
    """Raised when a prompt injection attempt is detected in the user's question.

    This is a specialisation of :class:`SecurityException` — the ``reason``
    attribute describes the specific pattern that triggered detection.
    """


# ── SQL Generation ────────────────────────────────────────────────────────────


class SQLGenerationError(AAIZAQLError):
    """Raised when the LLM fails to produce valid SQL for a given question.

    Attributes:
        question: The natural-language question that could not be translated.
    """

    def __init__(self, question: str, detail: str = "") -> None:
        self.question = question
        super().__init__(f"Failed to generate SQL for: '{question}'. {detail}".strip())


class RateLimitError(AAIZAQLError):
    """Raised when a tenant exceeds their configured query rate limit.

    Attributes:
        tenant_id: Identifier of the tenant that hit the limit.
        retry_after_seconds: Suggested wait time before retrying.
    """

    def __init__(self, tenant_id: str, retry_after_seconds: int = 60) -> None:
        self.tenant_id = tenant_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Rate limit exceeded for tenant '{tenant_id}'. " f"Retry after {retry_after_seconds}s."
        )


class UnsupportedQueryError(AAIZAQLError):
    """Raised when the question cannot be answered with a SELECT statement.

    AAIZAQL only executes read-only queries.  Questions that imply INSERT,
    UPDATE, DELETE, or DDL operations raise this exception.

    Attributes:
        question: The original natural-language question.
    """

    def __init__(self, question: str) -> None:
        self.question = question
        super().__init__(
            f"Query cannot be answered with SELECT: '{question}'. "
            "Try rephrasing as a data retrieval question."
        )


class MaxRetriesExceeded(AAIZAQLError):
    """Raised when the self-correction loop exhausts all retry attempts.

    The engine retries failed SQL by feeding the error back to the LLM.
    This exception is raised when every attempt produces invalid SQL.

    Attributes:
        sql: The last SQL string that was attempted.
        last_error: The database error message from the final attempt.
        attempts: Total number of attempts made.
    """

    def __init__(self, sql: str, last_error: str, attempts: int = 3) -> None:
        self.sql = sql
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(
            f"SQL failed after {attempts} self-correction attempts.\n"
            f"Last error: {last_error}\n"
            f"Last SQL: {sql}"
        )


# ── Database / Connectors ─────────────────────────────────────────────────────


class DatabaseError(AAIZAQLError):
    """Raised when query execution fails at the database level.

    Attributes:
        sql: The SQL string that caused the failure (may be empty).
        connector: Name of the connector that raised the error (may be empty).
    """

    def __init__(self, message: str, sql: str = "", connector: str = "") -> None:
        self.sql = sql
        self.connector = connector
        parts = [message]
        if connector:
            parts.append(f"Connector: {connector}")
        if sql:
            parts.append(f"SQL: {sql}")
        super().__init__("\n".join(parts))


class ConnectionError(AAIZAQLError):
    """Raised when a database connection cannot be established.

    .. note::
        This class shadows the Python built-in ``ConnectionError``.  Import it
        explicitly: ``from aaizaql.core.exceptions import ConnectionError``.

    Attributes:
        connector: Name of the connector that failed to connect.
    """

    def __init__(self, connector: str, dsn_hint: str = "", detail: str = "") -> None:
        self.connector = connector
        super().__init__(
            f"Cannot connect to '{connector}'"
            + (f" ({dsn_hint})" if dsn_hint else "")
            + (f": {detail}" if detail else "")
        )


class ConnectorNotFound(AAIZAQLError):
    """Raised when an unknown connector name is requested.

    Attributes:
        name: The connector name that was not found.
        available: List of registered connector names at the time of the error.
    """

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available or []
        super().__init__(
            f"No connector registered for '{name}'. " f"Available: {sorted(self.available)}"
        )


# ── LLM Providers ─────────────────────────────────────────────────────────────


class LLMError(AAIZAQLError):
    """Raised when an LLM API call fails.

    Attributes:
        provider: Name of the LLM provider (e.g. ``"openai"``, ``"deepseek"``).
    """

    def __init__(self, provider: str, detail: str = "") -> None:
        self.provider = provider
        super().__init__(f"LLM provider '{provider}' error" + (f": {detail}" if detail else ""))


class LLMTimeoutError(LLMError):
    """Raised when an LLM API call exceeds the configured ``llm_timeout_seconds``.

    Attributes:
        provider: Name of the LLM provider that timed out.
        timeout: The timeout value (in seconds) that was exceeded.
    """

    def __init__(self, provider: str, timeout: int) -> None:
        self.timeout = timeout
        super().__init__(provider, f"call timed out after {timeout}s")


class LLMProviderNotFound(AAIZAQLError):
    """Raised when an unknown LLM provider name is requested.

    Attributes:
        name: The provider name that was not found.
        available: List of registered provider names at the time of the error.
    """

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available if available is not None else []
        super().__init__(
            f"No LLM provider registered for '{name}'. " f"Available: {sorted(self.available)}"
        )


# ── Vector Store / RAG ────────────────────────────────────────────────────────


class VectorStoreError(AAIZAQLError):
    """Raised when a vector store operation fails.

    Covers embedding insertion, similarity search, and index management
    failures across all supported vector backends.
    """


class SchemaIngestionError(AAIZAQLError):
    """Raised when schema introspection or ingestion fails.

    Typically indicates a permissions problem reading ``information_schema``,
    or a serialisation failure when persisting schema metadata.
    """


# ── Federation ────────────────────────────────────────────────────────────────


class FederationError(AAIZAQLError):
    """Raised when a federated query spanning multiple data sources fails.

    Attributes:
        sources: List of data source names involved in the failed query.
    """

    def __init__(self, detail: str, sources: list[str] | None = None) -> None:
        self.sources = sources or []
        suffix = f" (sources: {self.sources})" if self.sources else ""
        super().__init__(f"Federation failed{suffix}: {detail}")


# ── Credentials ───────────────────────────────────────────────────────────────


class CredentialError(AAIZAQLError):
    """Raised when credential encryption or decryption fails.

    Signals a configuration problem (e.g. wrong encryption key) rather than
    an authentication failure at the database level.
    """
