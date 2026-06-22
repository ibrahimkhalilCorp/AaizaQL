"""
aaizaql.connectors.oracle
─────────────────────────
Oracle Database connector using python-oracledb in thin mode.

Thin mode requires NO Oracle Instant Client installation — pure Python.

DSN format::

    oracle://user:password@host:1521/service_name
    oracle://user:password@host:1521/?sid=ORCL

Install::

    pip install "aaizaql[oracle]"   # or: pip install oracledb

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

_DSN_PATTERN = re.compile(r"oracle://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/(.+)")


class OracleConnector(DatabaseConnector):
    """Oracle Database adapter via python-oracledb (thin mode).

    Thin mode connects directly to Oracle without the Oracle Instant Client,
    making it easy to install in any Python environment.

    Args (set at construction, no direct params):
        Call :meth:`connect` with a DSN string after instantiation.

    Example::

        engine = QueryEngine(
            llm="groq",
            database="oracle",
            dsn="oracle://hr:password@localhost:1521/XEPDB1",
        )
    """

    name = "oracle"

    def __init__(self) -> None:
        self._conn: Any = None
        self._dsn: str = ""

    def connect(self, dsn: str) -> None:
        """Establish a connection to an Oracle Database instance.

        Args:
            dsn: Oracle connection string, e.g.
                ``"oracle://user:password@host:1521/service_name"``.

        Raises:
            ConnectionError: If oracledb is not installed, or the server
                cannot be reached with the given credentials.
        """
        try:
            import oracledb
        except ImportError as exc:
            raise ConnectionError(
                "oracle",
                dsn[:40],
                "oracledb is not installed. Run: pip install oracledb",
            ) from exc

        self._dsn = dsn
        user, password, host, port, service = self._parse_dsn(dsn)
        try:
            self._conn = oracledb.connect(
                user=user,
                password=password,
                dsn=f"{host}:{port}/{service}",
            )
            logger.info("oracle.connected", dsn_hint=dsn[:40])
        except Exception as exc:
            raise ConnectionError("oracle", dsn[:40], str(exc)) from exc

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL statement and return results as a DataFrame.

        Args:
            sql: A validated SQL statement to execute.

        Returns:
            Query results as a DataFrame. Returns an empty DataFrame for
            statements that produce no rows.

        Raises:
            DatabaseError: On any Oracle execution error.
        """
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="oracle")
        try:
            return pd.read_sql_query(sql, self._conn)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="oracle") from exc

    def get_schema(self) -> str:
        """Return CREATE TABLE DDL for all user tables in the Oracle schema.

        Queries ``USER_TAB_COLUMNS`` and uses ``LISTAGG`` to reconstruct
        column definitions in a single query.

        Returns:
            DDL string with one CREATE TABLE block per table. Returns an
            empty string if not connected or on any error.
        """
        if self._conn is None:
            return ""
        query = """
            SELECT
                'CREATE TABLE "' || c.TABLE_NAME || '" (' ||
                LISTAGG(
                    '"' || c.COLUMN_NAME || '" ' || c.DATA_TYPE ||
                    CASE
                        WHEN c.DATA_TYPE IN ('VARCHAR2', 'CHAR', 'NVARCHAR2', 'NCHAR')
                        THEN '(' || c.DATA_LENGTH || ')'
                        WHEN c.DATA_TYPE = 'NUMBER' AND c.DATA_PRECISION IS NOT NULL
                        THEN '(' || c.DATA_PRECISION || ',' || NVL(c.DATA_SCALE, 0) || ')'
                        ELSE ''
                    END ||
                    CASE WHEN c.NULLABLE = 'N' THEN ' NOT NULL' ELSE '' END,
                    ', '
                ) WITHIN GROUP (ORDER BY c.COLUMN_ID) || ');' AS ddl
            FROM USER_TAB_COLUMNS c
            GROUP BY c.TABLE_NAME
            ORDER BY c.TABLE_NAME
        """
        try:
            df = pd.read_sql_query(query, self._conn)
            return "\n\n".join(df["DDL"].tolist())
        except Exception:
            return ""

    def close(self) -> None:
        """Close the Oracle connection and release resources."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("oracle.closed")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dsn(dsn: str) -> tuple[str, str, str, str, str]:
        """Parse an Oracle DSN URL into connection components.

        Args:
            dsn: Oracle connection string in the form
                ``"oracle://user:password@host:port/service_name"``.

        Returns:
            Tuple of ``(user, password, host, port, service)``.

        Raises:
            ValueError: If the DSN does not match the expected format.
        """
        match = _DSN_PATTERN.match(dsn)
        if not match:
            raise ValueError(
                f"Cannot parse Oracle DSN: {dsn!r}\n"
                "Expected format: oracle://user:password@host:1521/service_name"
            )
        user, password, host, port, service = match.groups()
        return user, password, host, port or "1521", service
