"""
aaizaql.connectors.postgres
──────────────────────────
PostgreSQL connector using psycopg2.
DSN format: postgresql://user:password@host:5432/dbname
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors._limit import inject_limit
from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


class PostgreSQLConnector(DatabaseConnector):
    name = "postgresql"

    def __init__(self) -> None:
        self._pool: Any = None
        self._dsn: str = ""

    def connect(self, dsn: str, pool_size: int = 5) -> None:
        try:
            from psycopg2 import pool as pg_pool

            self._dsn = dsn
            self._pool = pg_pool.ThreadedConnectionPool(
                minconn=1, maxconn=pool_size, dsn=dsn
            )
            logger.info("postgres.connected", pool_size=pool_size)
        except Exception as exc:
            raise ConnectionError("postgresql", dsn[:40], str(exc)) from exc

    def execute(self, sql: str, _max_rows: int = 10000) -> pd.DataFrame:
        if self._pool is None:
            raise DatabaseError("Not connected.", sql=sql, connector="postgresql")
        sql, _ = inject_limit(sql, _max_rows, dialect="postgres")  # T1.4
        conn = self._pool.getconn()
        try:
            return pd.read_sql_query(sql, conn)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="postgresql") from exc
        finally:
            self._pool.putconn(conn)

    def get_schema(self) -> str:
        # T3.3 — enriched DDL with PK, FK, and indexes
        if self._pool is None:
            return ""
        col_query = """
            SELECT
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                c.character_maximum_length,
                c.ordinal_position,
                c.is_nullable
            FROM information_schema.columns c
            WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY c.table_schema, c.table_name, c.ordinal_position;
        """
        pk_query = """
            SELECT kcu.table_schema, kcu.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY';
        """
        fk_query = """
            SELECT
                kcu.table_schema, kcu.table_name, kcu.column_name,
                ccu.table_name AS foreign_table, ccu.column_name AS foreign_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY';
        """
        conn = self._pool.getconn()
        try:
            col_df = pd.read_sql_query(col_query, conn)
            pk_df = pd.read_sql_query(pk_query, conn)
            fk_df = pd.read_sql_query(fk_query, conn)

            pk_set = set(zip(pk_df["table_schema"], pk_df["table_name"], pk_df["column_name"]))
            fk_map: dict = {}
            for _, row in fk_df.iterrows():
                key = (row["table_schema"], row["table_name"], row["column_name"])
                fk_map[key] = (row["foreign_table"], row["foreign_column"])

            parts: list[str] = []
            for (schema, table), grp in col_df.groupby(["table_schema", "table_name"]):
                col_defs = []
                pk_cols = []
                for _, row in grp.iterrows():
                    dtype = row["data_type"]
                    if pd.notna(row["character_maximum_length"]):
                        dtype = f"{dtype}({int(row['character_maximum_length'])})"
                    not_null = " NOT NULL" if row["is_nullable"] == "NO" else ""
                    col_defs.append(f"  {row['column_name']} {dtype}{not_null}")
                    if (schema, table, row["column_name"]) in pk_set:
                        pk_cols.append(row["column_name"])

                if pk_cols:
                    col_defs.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")

                for _, row in fk_df[
                    (fk_df["table_schema"] == schema) & (fk_df["table_name"] == table)
                ].iterrows():
                    col_defs.append(
                        f"  FOREIGN KEY ({row['column_name']}) "
                        f"REFERENCES {row['foreign_table']}({row['foreign_column']})"
                    )

                ddl = f"CREATE TABLE {schema}.{table} (\n" + ",\n".join(col_defs) + "\n);"
                parts.append(ddl)

            return "\n\n".join(parts)
        except Exception:
            return ""
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("postgres.closed")
