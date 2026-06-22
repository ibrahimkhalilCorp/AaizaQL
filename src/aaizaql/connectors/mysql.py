"""
aaizaql.connectors.mysql
────────────────────────
MySQL connector using pymysql.

DSN format::

    mysql://user:password@host:3306/dbname
    mysql://user:password@host/dbname

Install::

    pip install "aaizaql[mysql]"   # or: pip install pymysql

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

_DSN_PATTERN = re.compile(
    r"mysql://(?P<user>[^:@]+)(?::(?P<password>[^@]*))?@"
    r"(?P<host>[^:/]+)(?::(?P<port>\d+))?/(?P<database>.+)"
)


class MySQLConnector(DatabaseConnector):
    """MySQL database adapter via pymysql.

    Args (set at construction, no direct params):
        Call :meth:`connect` with a DSN string after instantiation.
    """

    name = "mysql"

    def __init__(self) -> None:
        self._conn: Any = None
        self._dsn: str = ""

    def connect(self, dsn: str) -> None:
        """Establish a connection to a MySQL server.

        Args:
            dsn: MySQL connection string, e.g.
                ``"mysql://user:password@localhost:3306/mydb"``.

        Raises:
            ConnectionError: If pymysql is not installed, the host is
                unreachable, or authentication fails.
        """
        self._dsn = dsn
        try:
            import pymysql  # type: ignore[import-untyped]
            import pymysql.cursors  # type: ignore[import-untyped]

            parsed = self._parse_dsn(dsn)
            self._conn = pymysql.connect(
                host=parsed["host"],
                port=parsed["port"],
                user=parsed["user"],
                password=parsed["password"],
                database=parsed["database"],
                cursorclass=pymysql.cursors.Cursor,
                autocommit=True,
                charset="utf8mb4",
            )
            logger.info("mysql.connected", host=parsed["host"], db=parsed["database"])
        except Exception as exc:
            raise ConnectionError("mysql", dsn[:40], str(exc)) from exc

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL statement and return results as a DataFrame.

        Args:
            sql: A validated SQL statement to execute.

        Returns:
            Query results as a DataFrame. Returns an empty DataFrame for
            statements that produce no rows.

        Raises:
            DatabaseError: On any MySQL execution error.
        """
        if self._conn is None:
            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="mysql")
        try:
            return pd.read_sql_query(sql, self._conn)
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(str(exc), sql=sql, connector="mysql") from exc

    def get_schema(self) -> str:
        """Return CREATE TABLE DDL for all tables in the connected database.

        Reconstructs DDL from ``information_schema.COLUMNS`` using MySQL's
        ``GROUP_CONCAT`` to build column definitions in a single query.

        Returns:
            DDL string with one CREATE TABLE block per table. Returns an
            empty string if not connected or on any error.
        """
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
            df = pd.read_sql_query(query, self._conn)
            return "\n\n".join(df["ddl"].dropna())
        except Exception:
            return ""

    def close(self) -> None:
        """Close the MySQL connection and release resources."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("mysql.closed")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dsn(dsn: str) -> dict[str, Any]:
        """Parse a MySQL DSN URL into a dict of connection parameters.

        Args:
            dsn: MySQL connection string in the form
                ``"mysql://user:password@host:port/database"``.

        Returns:
            Dict with keys ``host``, ``port`` (int), ``user``, ``password``,
            and ``database``.

        Raises:
            ValueError: If the DSN does not match the expected format.
        """
        match = _DSN_PATTERN.match(dsn)
        if not match:
            raise ValueError(
                f"Cannot parse MySQL DSN: {dsn!r}. "
                "Expected format: mysql://user:password@host:3306/dbname"
            )
        return {
            "host": match.group("host"),
            "port": int(match.group("port") or 3306),
            "user": match.group("user"),
            "password": match.group("password") or "",
            "database": match.group("database"),
        }
