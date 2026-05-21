"""
aaizaql.connectors.mysql
───────────────────────
MySQL connector using pymysql.
DSN format: mysql://user:password@host:3306/dbname
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


class MySQLConnector(DatabaseConnector):
    name = "mysql"

    def __init__(self) -> None:
        self._conn: Any = None
        self._dsn: str = ""

    def connect(self, dsn: str) -> None:
        """
        dsn examples:
          mysql://user:password@localhost:3306/mydb
          mysql://user:password@host/mydb
        """
        self._dsn = dsn
        try:
            import pymysql
            import pymysql.cursors

            parsed = self._parse_dsn(dsn)
            self._conn = pymysql.connect(
                host=parsed["host"],
                port=parsed["port"],
                user=parsed["user"],
                password=parsed["password"],
                database=parsed["database"],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
                charset="utf8mb4",
            )
            logger.info("mysql.connected", host=parsed["host"], db=parsed["database"])
        except Exception as exc:
            raise ConnectionError("mysql", dsn[:40], str(exc)) from exc

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="mysql")
        try:
            import pymysql

            # Use a plain (non-Dict) cursor so pandas receives raw tuples
            # and constructs column names from the cursor description itself.
            with self._conn.cursor(pymysql.cursors.Cursor) as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                columns = [d[0] for d in cur.description] if cur.description else []
            return pd.DataFrame(rows, columns=columns)
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="mysql") from exc

    def get_schema(self) -> str:
        if self._conn is None:
            return ""
        query = """
            SELECT
                CONCAT(
                    'CREATE TABLE `', TABLE_NAME, '` (',
                    GROUP_CONCAT(
                        CONCAT(
                            '`', COLUMN_NAME, '` ', COLUMN_TYPE,
                            IF(IS_NULLABLE = 'NO', ' NOT NULL', ''),
                            IF(COLUMN_DEFAULT IS NOT NULL,
                               CONCAT(' DEFAULT ', QUOTE(COLUMN_DEFAULT)), '')
                        )
                        ORDER BY ORDINAL_POSITION
                        SEPARATOR ', '
                    ),
                    ');'
                ) AS ddl
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            GROUP BY TABLE_NAME
            ORDER BY TABLE_NAME;
        """
        try:
            import pymysql

            with self._conn.cursor(pymysql.cursors.Cursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
            return "\n\n".join(row[0] for row in rows if row[0])
        except Exception:
            return ""

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("mysql.closed")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dsn(dsn: str) -> dict[str, Any]:
        """
        Parse mysql://user:password@host:port/database into components.
        Falls back to localhost:3306 if port is omitted.
        """
        import re

        pattern = re.compile(
            r"mysql://(?P<user>[^:@]+)(?::(?P<password>[^@]*))?@"
            r"(?P<host>[^:/]+)(?::(?P<port>\d+))?/(?P<database>.+)"
        )
        m = pattern.match(dsn)
        if not m:
            raise ValueError(
                f"Cannot parse MySQL DSN: {dsn!r}. "
                "Expected format: mysql://user:password@host:3306/dbname"
            )
        return {
            "host": m.group("host"),
            "port": int(m.group("port") or 3306),
            "user": m.group("user"),
            "password": m.group("password") or "",
            "database": m.group("database"),
        }
