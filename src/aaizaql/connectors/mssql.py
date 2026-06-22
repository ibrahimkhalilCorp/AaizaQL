"""
aaizaql.connectors.mssql
────────────────────────
Microsoft SQL Server connector using pyodbc.

Requires the Microsoft ODBC Driver for SQL Server to be installed on the OS.
Download at: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

DSN format::

    mssql://user:password@host:1433/dbname
    mssql://user:password@host/dbname

Install::

    pip install "aaizaql[mssql]"   # or: pip install pyodbc

Author: Ibrahim
Date: 2026-06-15
Version: 1.0.0
"""

import re
from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)

_DSN_PATTERN = re.compile(r"mssql(?:\+pyodbc)?://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/(.+)")


class MSSQLConnector(DatabaseConnector):
    """Microsoft SQL Server adapter via pyodbc.

    Converts the URL-style DSN to a native pyodbc connection string, then
    connects using ``ODBC Driver 18 for SQL Server`` with
    ``TrustServerCertificate=yes`` for self-signed certs in dev/test.

    Args (set at construction, no direct params):
        Call :meth:`connect` with a DSN string after instantiation.

    Example::

        engine = QueryEngine(
            llm="groq",
            database="mssql",
            dsn="mssql://sa:password@localhost:1433/mydb",
        )
    """

    name = "mssql"

    def __init__(self) -> None:
        self._conn: Any = None
        self._dsn: str = ""

    def connect(self, dsn: str) -> None:
        """Establish a connection to SQL Server via pyodbc.

        Accepts a URL-style DSN and converts it to a pyodbc connection string.
        Passthrough raw pyodbc strings (starting with ``Driver=`` or ``SERVER=``)
        are forwarded unchanged.

        Args:
            dsn: SQL Server connection string.

        Raises:
            ConnectionError: If pyodbc is not installed, the ODBC driver is
                missing, or the server cannot be reached.
        """
        try:
            import pyodbc
        except ImportError as exc:
            raise ConnectionError(
                "mssql",
                dsn[:40],
                "pyodbc is not installed. Run: pip install pyodbc",
            ) from exc

        self._dsn = dsn
        conn_str = self._build_conn_str(dsn)
        try:
            self._conn = pyodbc.connect(conn_str, autocommit=True)
            logger.info("mssql.connected", dsn_hint=dsn[:40])
        except Exception as exc:
            raise ConnectionError("mssql", dsn[:40], str(exc)) from exc

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL statement and return results as a DataFrame.

        Args:
            sql: A validated SQL statement to execute.

        Returns:
            Query results as a DataFrame. Returns an empty DataFrame for
            statements that produce no rows.

        Raises:
            DatabaseError: On any SQL Server execution error.
        """
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="mssql")
        try:
            return pd.read_sql_query(sql, self._conn)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="mssql") from exc

    def get_schema(self) -> str:
        """Return CREATE TABLE DDL for all user tables in the database.

        Reconstructs DDL from ``INFORMATION_SCHEMA`` using ``STRING_AGG`` to
        build column definitions, filtered to exclude system schemas.

        Returns:
            DDL string with one CREATE TABLE block per table. Returns an
            empty string if not connected or on any error.
        """
        if self._conn is None:
            return ""
        query = """
            SELECT
                'CREATE TABLE [' + t.TABLE_SCHEMA + '].[' + t.TABLE_NAME + '] (' +
                STRING_AGG(
                    '[' + c.COLUMN_NAME + '] ' + c.DATA_TYPE +
                    CASE
                        WHEN c.CHARACTER_MAXIMUM_LENGTH IS NOT NULL
                        THEN '(' + CAST(c.CHARACTER_MAXIMUM_LENGTH AS VARCHAR) + ')'
                        WHEN c.DATA_TYPE IN ('decimal', 'numeric')
                        THEN '(' + CAST(c.NUMERIC_PRECISION AS VARCHAR) + ',' +
                             CAST(c.NUMERIC_SCALE AS VARCHAR) + ')'
                        ELSE ''
                    END +
                    CASE WHEN c.IS_NULLABLE = 'NO' THEN ' NOT NULL' ELSE '' END,
                    ', '
                ) WITHIN GROUP (ORDER BY c.ORDINAL_POSITION) + ');' AS ddl
            FROM INFORMATION_SCHEMA.TABLES t
            JOIN INFORMATION_SCHEMA.COLUMNS c
                ON t.TABLE_NAME = c.TABLE_NAME AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
            WHERE t.TABLE_TYPE = 'BASE TABLE'
              AND t.TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')
            GROUP BY t.TABLE_SCHEMA, t.TABLE_NAME
            ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME;
        """
        try:
            df = pd.read_sql_query(query, self._conn)
            return "\n\n".join(df["ddl"].tolist())
        except Exception:
            return ""

    def close(self) -> None:
        """Close the pyodbc connection and release the ODBC handle."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("mssql.closed")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_conn_str(dsn: str) -> str:
        """Convert a URL-style DSN to a pyodbc connection string.

        Args:
            dsn: Either a URL DSN (``mssql://...``) or a raw pyodbc string
                (starting with ``Driver=`` or ``SERVER=``).

        Returns:
            pyodbc-compatible connection string.

        Raises:
            ValueError: If the URL DSN does not match the expected format.
        """
        # Raw pyodbc connection string — pass through unchanged.
        if dsn.startswith("Driver=") or dsn.startswith("SERVER="):
            return dsn

        match = _DSN_PATTERN.match(dsn)
        if not match:
            raise ValueError(
                f"Cannot parse MSSQL DSN: {dsn!r}\n"
                "Expected format: mssql://user:password@host:1433/dbname"
            )
        user, password, host, port, dbname = match.groups()
        port = port or "1433"
        return (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={host},{port};"
            f"DATABASE={dbname};"
            f"UID={user};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )
