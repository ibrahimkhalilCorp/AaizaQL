"""
aaizaql.connectors.mssql
─────────────────────────
Microsoft SQL Server connector using pyodbc.

DSN format:
  mssql://user:password@host:1433/dbname
  mssql+pyodbc://user:password@host/dbname?driver=ODBC+Driver+18+for+SQL+Server

Install:
  pip install aaizaql[mssql]   # or: pip install pyodbc
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)

class MSSQLConnector(DatabaseConnector):
    """
    Microsoft SQL Server adapter via pyodbc.

    Usage:
        engine = QueryEngine(
            llm="groq",
            database="mssql",
            dsn="mssql://user:password@localhost:1433/mydb",
        )

    Requires the Microsoft ODBC Driver for SQL Server to be installed on the OS.
    Download at: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
    """

    name = "mssql"

    def __init__(self) -> None:
        self._conn: Any = None
        self._dsn: str = ""

    def connect(self, dsn: str) -> None:
        """
        Accepts a SQLAlchemy-style DSN and converts it to a pyodbc connection string.

        DSN examples:
          mssql://sa:password@localhost:1433/mydb
          mssql://sa:password@localhost/mydb
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

    def _build_conn_str(self, dsn: str) -> str:
        """Convert a URL-style DSN to a pyodbc connection string."""
        import re

        # Already a raw pyodbc connection string
        if dsn.startswith("Driver=") or dsn.startswith("SERVER="):
            return dsn

        # Parse: mssql://user:password@host:port/dbname
        pattern = r"mssql(?:\+pyodbc)?://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/(.+)"
        m = re.match(pattern, dsn)
        if not m:
            raise ValueError(
                f"Cannot parse MSSQL DSN: {dsn!r}\n"
                "Expected format: mssql://user:password@host:1433/dbname"
            )
        user, password, host, port, dbname = m.groups()
        port = port or "1433"

        return (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={host},{port};"
            f"DATABASE={dbname};"
            f"UID={user};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="mssql")
        try:
            return pd.read_sql_query(sql, self._conn)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="mssql") from exc

    def get_schema(self) -> str:
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
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("mssql.closed")
