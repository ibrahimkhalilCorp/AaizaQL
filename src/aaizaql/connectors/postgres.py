"""
aaizaql.connectors.postgres
───────────────────────────
PostgreSQL connector using psycopg2.

DSN format::

    postgresql://user:password@host:5432/dbname
    postgresql://user:password@host/dbname

Install::

    pip install "aaizaql[postgres]"   # or: pip install psycopg2-binary

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


def _encode_dsn_password(dsn: str) -> str:
    """Percent-encode special characters in the DSN password component.

    psycopg2 requires URL-encoded passwords when the password contains
    characters like ``@``, ``/``, or ``?``.

    Args:
        dsn: Raw connection string that may contain an unencoded password.

    Returns:
        DSN with the password component percent-encoded, or the original
        string unchanged if no password is present.
    """
    parsed = urlparse(dsn)
    if not parsed.password:
        return dsn
    encoded_password = quote(parsed.password, safe="")
    netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


class PostgresConnector(DatabaseConnector):
    """PostgreSQL adapter via psycopg2.

    Supports context-manager usage::

        with PostgresConnector() as conn:
            conn.connect("postgresql://user:pass@localhost/mydb")
            df = conn.execute("SELECT * FROM employees")

    Args (set at construction, no direct params):
        Call :meth:`connect` with a DSN string after instantiation, or pass
        ``dsn`` to the constructor for convenience.
    """

    name = "postgresql"

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn: str = _encode_dsn_password(dsn) if dsn else ""
        self._conn: Any = None

    def connect(self, dsn: str | None = None) -> None:
        """Open the PostgreSQL connection.

        Args:
            dsn: Optional DSN override. When provided, replaces the DSN given
                at construction. Special characters in the password are
                percent-encoded automatically.

        Raises:
            ConnectionError: If psycopg2 is not installed, or the server
                cannot be reached with the given credentials.
        """
        try:
            import psycopg2
        except ImportError as exc:
            raise ConnectionError(
                "postgresql",
                (dsn or self._dsn)[:40],
                'psycopg2 is not installed. Run: pip install "aaizaql[postgres]"',
            ) from exc

        if dsn:
            self._dsn = _encode_dsn_password(dsn)
        try:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = True
            logger.info("postgresql.connected", dsn_hint=self._dsn[:40])
        except Exception as exc:
            raise ConnectionError("postgresql", self._dsn[:40], str(exc)) from exc

    def disconnect(self) -> None:
        """Close the connection and release the socket.

        Safe to call when not connected — does nothing in that case.
        """
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def close(self) -> None:
        """Alias for :meth:`disconnect` — satisfies the base-class contract."""
        self.disconnect()

    def __enter__(self) -> "PostgresConnector":
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute *sql* and return results as a DataFrame.

        Reconnects automatically when the connection is found closed before
        execution.

        Args:
            sql: A validated SQL statement to execute.

        Returns:
            Query results as a DataFrame. Returns an empty DataFrame for
            statements that produce no rows (e.g. DDL).

        Raises:
            DatabaseError: On any psycopg2 execution error.
        """
        if self._conn is None or self._conn.closed:
            self.connect()
        if self._conn is None:
            raise DatabaseError(
                "Connection could not be established.", sql=sql, connector="postgresql"
            )
        try:
            import psycopg2.extras

            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                if cur.description is None:
                    return pd.DataFrame()
                rows = [dict(row) for row in cur.fetchall()]
                return pd.DataFrame(rows)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="postgresql") from exc

    def get_schema(self) -> str:
        """Return a human-readable schema string for all public tables.

        Queries ``information_schema.columns`` to reconstruct a compact
        ``table(col type[?], ...)`` representation suitable for LLM prompts.

        Returns:
            One line per table, e.g. ``employees(id int4, name varchar?)``.
            Returns an empty string if not connected.
        """
        schema_sql = """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """
        if self._conn is None or self._conn.closed:
            self.connect()
        if self._conn is None:
            return ""

        try:
            import psycopg2.extras

            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(schema_sql)
                rows = [dict(row) for row in cur.fetchall()]
        except Exception:
            return ""

        table_map: dict[str, list[dict]] = {}
        for row in rows:
            table_map.setdefault(row["table_name"], []).append(
                {
                    "column": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                }
            )

        lines = []
        for table, cols in table_map.items():
            col_strs = ", ".join(
                f"{c['column']} {c['type']}{'?' if c['nullable'] else ''}" for c in cols
            )
            lines.append(f"{table}({col_strs})")
        return "\n".join(lines)

    def test_connection(self) -> bool:
        """Return ``True`` if a connection can be established.

        Leaves the connection open on success for subsequent calls.

        Returns:
            ``True`` if the server is reachable, ``False`` otherwise.
        """
        try:
            self.connect()
            if self._conn is None:
                return False
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False


# Alias expected by connectors/__init__.py and the test suite.
PostgreSQLConnector = PostgresConnector
