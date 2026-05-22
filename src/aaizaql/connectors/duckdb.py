"""
aaizaql.connectors.duckdb
────────────────────────
DuckDB connector — supports both file-based and in-memory databases.
Also serves as the ephemeral workspace engine for Phase 3 federation.

DSN format:
  duckdb:///path/to/file.db    — persistent file database
  duckdb:///:memory:           — ephemeral in-memory database
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)

class DuckDBConnector(DatabaseConnector):
    name = "duckdb"

    def __init__(self) -> None:
        self._conn: Any = None
        self._path: str = ""

    def connect(self, dsn: str) -> None:
        """
        dsn examples:
          duckdb:///./analytics.db
          duckdb:////abs/path/analytics.db
          duckdb:///:memory:
        """
        path = re.sub(r"^duckdb:///", "", dsn)
        self._path = path or ":memory:"
        try:
            import duckdb

            self._conn = duckdb.connect(self._path)
            logger.info("duckdb.connected", path=self._path)
        except Exception as exc:
            raise ConnectionError("duckdb", dsn[:40], str(exc)) from exc

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="duckdb")
        try:
            return self._conn.execute(sql).df()
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="duckdb") from exc

    def get_schema(self) -> str:
        """
        Return CREATE TABLE DDL for all user tables in the DuckDB database.
        Uses DuckDB's built-in SHOW TABLES and duckdb_columns() view.
        """
        if self._conn is None:
            return ""
        try:
            tables_df = self._conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).df()
            if tables_df.empty:
                return ""

            ddl_parts: list[str] = []
            for table_name in tables_df["table_name"].tolist():
                cols_df = self._conn.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'main' AND table_name = ? "
                    "ORDER BY ordinal_position",
                    [table_name],
                ).df()

                col_defs = []
                for _, row in cols_df.iterrows():
                    nullable = "" if row["is_nullable"] == "YES" else " NOT NULL"
                    col_defs.append(f"  {row['column_name']} {row['data_type']}{nullable}")

                ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"
                ddl_parts.append(ddl)

            return "\n\n".join(ddl_parts)
        except Exception:
            return ""

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("duckdb.closed", path=self._path)

    # ── Federation helpers ────────────────────────────────────────────────────

    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """
        Register a DataFrame as a virtual table in this DuckDB connection.
        Used by the FederationCoordinator to load sub-query results for joining.
        """
        if self._conn is None:
            raise DatabaseError("Not connected.", connector="duckdb")
        try:
            self._conn.register(name, df)
            logger.debug("duckdb.registered", table=name, rows=len(df))
        except Exception as exc:
            raise DatabaseError(str(exc), connector="duckdb") from exc

    def test_connection(self) -> bool:
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False
