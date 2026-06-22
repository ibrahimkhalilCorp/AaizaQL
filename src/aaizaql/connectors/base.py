"""
aaizaql.connectors.base
───────────────────────
Abstract base class for all database connector adapters.

Every adapter in ``aaizaql.connectors`` must subclass :class:`DatabaseConnector`
and implement :meth:`connect`, :meth:`execute`, and :meth:`get_schema`.
The engine calls only this interface — new connectors require zero changes
to ``core/engine.py``.

Plugin contract::

    class MyConnector(DatabaseConnector):
        name = "mydb"
        requires_sql_validation = True

        def connect(self, dsn: str) -> None: ...
        def execute(self, sql: str) -> pd.DataFrame: ...
        def get_schema(self) -> str: ...

    # Register in connectors/__init__.py so the engine's REGISTRY finds it.

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

from abc import ABC, abstractmethod

import pandas as pd


class DatabaseConnector(ABC):
    """Abstract base class for all database adapters.

    Subclass this and implement :meth:`connect`, :meth:`execute`, and
    :meth:`get_schema` to add a new database backend.  Register the subclass
    in ``connectors/__init__.py``; no other file needs to change.

    Class Attributes:
        name: Human-readable connector identifier used in logs and the SQL
            dialect selector, e.g. ``"sqlite"``, ``"postgresql"``.
        requires_sql_validation: Set to ``False`` for connectors that do not
            speak SQL (e.g. MongoDB) so the security validator is skipped.
    """

    name: str = ""
    requires_sql_validation: bool = True

    @abstractmethod
    def connect(self, dsn: str) -> None:
        """Establish a connection to the database.

        Args:
            dsn: SQLAlchemy-compatible connection string, e.g.
                ``"postgresql+psycopg2://user:pass@localhost/mydb"``.

        Raises:
            ConnectionError: If the connection cannot be established.
        """
        ...

    @abstractmethod
    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL statement and return results as a DataFrame.

        Args:
            sql: A validated SQL statement (typically a SELECT or WITH query).

        Returns:
            Query results as a DataFrame. Returns an empty DataFrame when the
            query produces no rows.

        Raises:
            DatabaseError: On any execution failure (syntax error, constraint
                violation, connection loss, etc.).
        """
        ...

    @abstractmethod
    def get_schema(self) -> str:
        """Return the full DDL schema of all tables in the connected database.

        Returns:
            DDL string containing CREATE TABLE statements for all user tables,
            suitable for embedding in the LLM prompt as schema context.

        Raises:
            DatabaseError: If the schema cannot be retrieved.
        """
        ...

    def close(self) -> None:  # noqa: B027 — intentionally non-abstract empty default
        """Close the database connection and release resources.

        The default implementation is a no-op. Override when the connector
        holds a connection pool or other resources that must be explicitly
        released (e.g. SQLAlchemy engines, connection pools).
        """

    def test_connection(self) -> bool:
        """Return ``True`` if the database connection is alive.

        Runs a trivial ``SELECT 1`` query. Exceptions are caught and converted
        to ``False`` so callers receive a simple boolean rather than having to
        handle connector-specific error types.

        Returns:
            ``True`` if the connection is healthy, ``False`` otherwise.
        """
        try:
            self.execute("SELECT 1")
            return True
        except Exception:
            return False
