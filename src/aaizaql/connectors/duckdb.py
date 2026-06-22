"""
aaizaql.connectors.duckdb
─────────────────────────
DuckDB connector — supports both file-based and in-memory databases.

Also serves as the ephemeral workspace engine for Phase 4 federated queries,
where sub-query results from multiple databases are joined via DuckDB's
in-memory columnar engine.

DSN format::

    duckdb:///path/to/file.db    — persistent file database
    duckdb:///:memory:           — ephemeral in-memory database

Install::

    pip install "aaizaql[duckdb]"   # or: pip install duckdb

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


class DuckDBConnector(DatabaseConnector):
    """DuckDB database adapter.

    Supports in-memory and file-backed databases. The in-memory mode is used
    by the Phase 4 federation coordinator to join sub-query results from
    multiple heterogeneous databases without writing to disk.

    Args (set at construction, no direct params):
        Call :meth:`connect` with a DSN string after instantiation.
    """

    name = "duckdb"

    def __init__(self) -> None:
        self._conn: Any = None
        self._path: str = ""

    def connect(self, dsn: str) -> None:
        """Establish a connection to a DuckDB database.

        Args:
            dsn: DuckDB connection string. The ``duckdb:///`` prefix is
                stripped automatically. Use ``:memory:`` for an ephemeral DB.

        Raises:
            ConnectionError: If the duckdb package is not installed or the
                file path is not accessible.
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
        """Execute a SQL statement and return results as a DataFrame.

        Args:
            sql: A validated SQL statement to execute.

        Returns:
            Query results as a DataFrame. Returns an empty DataFrame for
            statements that produce no rows.

        Raises:
            DatabaseError: On any DuckDB execution error.
        """
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="duckdb")
        try:
            return self._conn.execute(sql).df()
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="duckdb") from exc

    def get_schema(self) -> str:
        """Return CREATE TABLE DDL for all user tables in the DuckDB database.

        Queries ``information_schema`` to list tables, then fetches column
        metadata per table to reconstruct DDL statements.

        Returns:
            DDL string with one CREATE TABLE block per table. Returns an
            empty string if not connected, no tables exist, or on any error.
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
        """Close the DuckDB connection and release resources."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("duckdb.closed", path=self._path)

    def test_connection(self) -> bool:
        """Return ``True`` if the DuckDB connection is alive.

        Returns:
            ``True`` if a trivial query succeeds, ``False`` otherwise.
        """
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ── Federation helpers ────────────────────────────────────────────────────

    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """Register a DataFrame as a virtual table in this DuckDB connection.

        Used by the Phase 4 FederationCoordinator to load sub-query results
        from other connectors so they can be joined in DuckDB's engine.

        Args:
            name: Virtual table name to register under.
            df: DataFrame to expose as a queryable table.

        Raises:
            DatabaseError: If not connected or the registration fails.
        """
        if self._conn is None:
            raise DatabaseError("Not connected.", connector="duckdb")
        try:
            self._conn.register(name, df)
            logger.debug("duckdb.registered", table=name, rows=len(df))
        except Exception as exc:
            raise DatabaseError(str(exc), connector="duckdb") from exc
