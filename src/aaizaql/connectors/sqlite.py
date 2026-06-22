"""
aaizaql.connectors.sqlite
─────────────────────────
SQLite connector — zero-dependency, great for local development and testing.

DSN format::

    sqlite:///path/to/file.db
    sqlite:////abs/path/file.db
    sqlite:///:memory:

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import re
import sqlite3

import pandas as pd
import structlog

from aaizaql.connectors._limit import inject_limit
from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


class SQLiteConnector(DatabaseConnector):
    """SQLite database adapter using the stdlib ``sqlite3`` module.

    No extra packages required. Supports both file-based and in-memory databases.
    Auto-reconnects when the underlying connection is dropped.

    Args (set at construction, no direct params):
        Call :meth:`connect` with a DSN string after instantiation.
    """

    name = "sqlite"

    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None
        self._path: str = ""

    def connect(self, dsn: str) -> None:
        """Establish a connection to a SQLite database file or in-memory DB.

        Args:
            dsn: SQLite connection string. The ``sqlite:///`` prefix is stripped
                automatically. Use ``:memory:`` for an ephemeral in-memory DB.

        Raises:
            ConnectionError: If SQLite cannot open the file at the given path.
        """
        path = re.sub(r"^sqlite:///", "", dsn)
        self._path = path or ":memory:"
        try:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            logger.info("sqlite.connected", path=self._path)
        except sqlite3.Error as exc:
            raise ConnectionError("sqlite", dsn[:40], str(exc)) from exc

    def _ensure_connection(self, sql: str) -> None:
        """Raise or reconnect if the SQLite connection is not alive.

        SQLite connections can silently drop in long-running processes.
        This method probes with ``SELECT 1`` and reconnects to the same path
        when the probe fails, rather than surfacing an opaque error.

        Args:
            sql: The SQL about to be executed — used for error context only.

        Raises:
            DatabaseError: If :meth:`connect` was never called.
        """
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="sqlite")
        try:
            self._conn.execute("SELECT 1")
        except Exception:
            logger.warning("sqlite.reconnecting", path=self._path)
            self._conn = sqlite3.connect(self._path, check_same_thread=False)

    def execute(self, sql: str, _max_rows: int = 10000) -> pd.DataFrame:
        """Execute a SQL statement and return results as a DataFrame.

        Injects a LIMIT clause when none is present to cap memory usage.

        Args:
            sql: A validated SQL statement to execute.
            _max_rows: Maximum rows to return. Passed through to
                :func:`~aaizaql.connectors._limit.inject_limit`.

        Returns:
            Query results as a DataFrame. Returns an empty DataFrame for DDL
            or DML statements that produce no rows.

        Raises:
            DatabaseError: On any SQLite execution error.
        """
        self._ensure_connection(sql)
        sql, _ = inject_limit(sql, _max_rows, dialect="sqlite")
        try:
            return pd.read_sql_query(sql, self._conn)
        except TypeError:
            # DDL/DML statements return no cursor.description;
            # pd.read_sql_query raises TypeError in that case.
            return pd.DataFrame()
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="sqlite") from exc

    def get_schema(self) -> str:
        """Return CREATE TABLE DDL for all user tables in the database.

        Returns:
            DDL string with one CREATE TABLE block per table, separated by
            blank lines. Returns an empty string if not connected.
        """
        if self._conn is None:
            return ""
        cursor = self._conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        rows = cursor.fetchall()
        return "\n\n".join(ddl for _, ddl in rows if ddl)

    def close(self) -> None:
        """Close the SQLite connection and release the file handle."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("sqlite.closed", path=self._path)
