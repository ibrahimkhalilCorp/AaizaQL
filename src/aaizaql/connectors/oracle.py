"""
aaizaql.connectors.oracle
──────────────────────────
Oracle Database connector using python-oracledb (thin mode — no Oracle Client needed).

DSN format:
  oracle://user:password@host:1521/service_name
  oracle://user:password@host:1521/?sid=ORCL

Install:
  pip install aaizaql[oracle]   # or: pip install oracledb
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


class OracleConnector(DatabaseConnector):
    """
    Oracle Database adapter via python-oracledb (thin mode).

    Thin mode requires NO Oracle Instant Client installation — pure Python.

    Usage:
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
        """
        DSN examples:
          oracle://user:password@host:1521/service_name
          oracle://user:password@host:1521/?sid=ORCL
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

    def _parse_dsn(self, dsn: str) -> tuple[str, str, str, str, str]:
        """Parse oracle://user:password@host:port/service into components."""
        import re

        pattern = r"oracle://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/(.+)"
        m = re.match(pattern, dsn)
        if not m:
            raise ValueError(
                f"Cannot parse Oracle DSN: {dsn!r}\n"
                "Expected format: oracle://user:password@host:1521/service_name"
            )
        user, password, host, port, service = m.groups()
        return user, password, host, port or "1521", service

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="oracle")
        try:
            return pd.read_sql_query(sql, self._conn)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="oracle") from exc

    def get_schema(self) -> str:
        """Return DDL-style schema for all user tables."""
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
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("oracle.closed")