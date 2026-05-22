"""
aaizaql.connectors.snowflake
───────────────────────────
Snowflake connector using snowflake-connector-python.

DSN format:
  snowflake://user:password@account/database/schema?warehouse=WH&role=ROLE

All query parameters are optional but warehouse is strongly recommended.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)

class SnowflakeConnector(DatabaseConnector):
    name = "snowflake"

    def __init__(self) -> None:
        self._conn: Any = None
        self._account: str = ""
        self._database: str = ""
        self._schema: str = "PUBLIC"

    def connect(self, dsn: str) -> None:
        """
        dsn examples:
          snowflake://myuser:mypass@myaccount/mydb/myschema?warehouse=COMPUTE_WH
          snowflake://myuser:mypass@myaccount.us-east-1/mydb?warehouse=WH&role=ANALYST
        """
        try:
            import snowflake.connector

            params = self._parse_dsn(dsn)
            self._account = params["account"]
            self._database = params.get("database", "")
            self._schema = params.get("schema", "PUBLIC")

            connect_kwargs: dict[str, Any] = {
                "user": params["user"],
                "password": params["password"],
                "account": params["account"],
            }
            if params.get("database"):
                connect_kwargs["database"] = params["database"]
            if params.get("schema"):
                connect_kwargs["schema"] = params["schema"]
            if params.get("warehouse"):
                connect_kwargs["warehouse"] = params["warehouse"]
            if params.get("role"):
                connect_kwargs["role"] = params["role"]

            self._conn = snowflake.connector.connect(**connect_kwargs)
            logger.info(
                "snowflake.connected",
                account=self._account,
                database=self._database,
            )
        except Exception as exc:
            raise ConnectionError("snowflake", dsn[:40], str(exc)) from exc

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            raise DatabaseError(
                "Not connected. Call connect() first.", sql=sql, connector="snowflake"
            )
        try:
            cur = self._conn.cursor()
            cur.execute(sql)
            if cur.description is None:
                return pd.DataFrame()
            columns = [col[0] for col in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="snowflake") from exc

    def get_schema(self) -> str:
        """
        Return a DDL-style schema string for all tables in the current database/schema.
        Snowflake doesn't expose GET_DDL easily via INFORMATION_SCHEMA, so we
        reconstruct CREATE TABLE statements from column metadata.
        """
        if self._conn is None:
            return ""

        db_filter = f"AND TABLE_CATALOG = '{self._database}'" if self._database else ""
        schema_filter = f"AND TABLE_SCHEMA = '{self._schema}'" if self._schema else ""

        query = f"""
            SELECT
                TABLE_NAME,
                COLUMN_NAME,
                DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH,
                IS_NULLABLE,
                ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA')
            {db_filter}
            {schema_filter}
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        try:
            df = self.execute(query)
            if df.empty:
                return ""

            ddl_parts: list[str] = []
            for table_name, group in df.groupby("TABLE_NAME"):
                col_defs: list[str] = []
                for _, row in group.iterrows():
                    dtype = row["DATA_TYPE"]
                    if row["CHARACTER_MAXIMUM_LENGTH"] and pd.notna(
                        row["CHARACTER_MAXIMUM_LENGTH"]
                    ):
                        dtype = f"{dtype}({int(row['CHARACTER_MAXIMUM_LENGTH'])})"
                    # T3.7 — removed dead ternary; only the correct assignment remains
                    not_null = " NOT NULL" if row["IS_NULLABLE"] == "NO" else ""
                    col_defs.append(f"  {row['COLUMN_NAME']} {dtype}{not_null}")

                qual_name = (
                    f"{self._database}.{self._schema}.{table_name}"
                    if self._database
                    else table_name
                )
                ddl = f"CREATE TABLE {qual_name} (\n" + ",\n".join(col_defs) + "\n);"
                ddl_parts.append(ddl)

            return "\n\n".join(ddl_parts)
        except Exception:
            return ""

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("snowflake.closed", account=self._account)

    def test_connection(self) -> bool:
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT CURRENT_VERSION()")
            return True
        except Exception:
            return False

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dsn(dsn: str) -> dict[str, Any]:
        """
        Parse snowflake://user:pass@account/database/schema?warehouse=WH&role=ROLE
        into a flat dict of connection parameters.
        """
        parsed = urlparse(dsn)
        if parsed.scheme != "snowflake":
            raise ValueError(
                f"Invalid scheme {parsed.scheme!r}. DSN must start with 'snowflake://'."
            )

        # Path is /database/schema — both optional
        path_parts = [p for p in parsed.path.split("/") if p]
        database = path_parts[0] if len(path_parts) > 0 else ""
        schema = path_parts[1] if len(path_parts) > 1 else "PUBLIC"

        # Query string params
        qs = parse_qs(parsed.query)

        return {
            "user": parsed.username or "",
            "password": parsed.password or "",
            "account": parsed.hostname or "",
            "database": database,
            "schema": schema,
            "warehouse": qs.get("warehouse", [None])[0],
            "role": qs.get("role", [None])[0],
        }
