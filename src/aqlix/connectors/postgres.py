"""
aqlix.connectors.postgres
──────────────────────────
PostgreSQL connector using psycopg2.
DSN format: postgresql://user:password@host:5432/dbname
"""

from __future__ import annotations
from typing import Any
import pandas as pd
import structlog

from aqlix.connectors.base import DatabaseConnector
from aqlix.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


class PostgreSQLConnector(DatabaseConnector):
    name = "postgresql"

    def __init__(self) -> None:
        self._conn: Any = None

    def connect(self, dsn: str) -> None:
        try:
            import psycopg2

            self._conn = psycopg2.connect(dsn)
            self._conn.autocommit = True
            logger.info("postgres.connected")
        except Exception as exc:
            raise ConnectionError("postgresql", dsn[:40], str(exc)) from exc

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            raise DatabaseError("Not connected.", sql=sql, connector="postgresql")
        try:
            return pd.read_sql_query(sql, self._conn)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="postgresql") from exc

    def get_schema(self) -> str:
        if self._conn is None:
            return ""
        query = """
            SELECT
                'CREATE TABLE ' || table_schema || '.' || table_name || ' (' ||
                string_agg(
                    column_name || ' ' || data_type ||
                    CASE WHEN character_maximum_length IS NOT NULL
                         THEN '(' || character_maximum_length || ')'
                         ELSE '' END,
                    ', '
                    ORDER BY ordinal_position
                ) || ');' AS ddl
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            GROUP BY table_schema, table_name
            ORDER BY table_schema, table_name;
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
            logger.info("postgres.closed")
