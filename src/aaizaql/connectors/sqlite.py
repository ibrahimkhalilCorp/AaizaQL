"""

aaizaql.connectors.sqlite

────────────────────────

SQLite connector — great for local development and testing.

DSN format: sqlite:///path/to/file.db  or  sqlite:///:memory:

"""

from __future__ import annotations

import re
import sqlite3

import pandas as pd
import structlog

from aaizaql.connectors._limit import inject_limit
from aaizaql.connectors.base import DatabaseConnector
from aaizaql.core.exceptions import ConnectionError, DatabaseError

logger = structlog.get_logger(__name__)


class SQLiteConnector(DatabaseConnector):

    name = "sqlite"

    def __init__(self) -> None:

        self._conn: sqlite3.Connection | None = None

        self._path: str = ""

    def connect(self, dsn: str) -> None:
        """

        dsn examples:

          sqlite:///./my.db

          sqlite:////abs/path/my.db

          sqlite:///:memory:

        """

        # Strip the sqlite:/// prefix

        path = re.sub(r"^sqlite:///", "", dsn)

        self._path = path or ":memory:"

        try:

            self._conn = sqlite3.connect(self._path, check_same_thread=False)

            logger.info("sqlite.connected", path=self._path)

        except sqlite3.Error as exc:

            raise ConnectionError("sqlite", dsn[:40], str(exc)) from exc

    def _ensure_connection(self, sql: str) -> None:
        """T2.1 — Reconnect if the SQLite connection was dropped."""

        if self._conn is None:

            raise DatabaseError("Not connected. Call connect() first.", sql=sql, connector="sqlite")

        try:

            self._conn.execute("SELECT 1")

        except Exception:

            logger.warning("sqlite.reconnecting", path=self._path)

            self._conn = sqlite3.connect(self._path, check_same_thread=False)

    def execute(self, sql: str, _max_rows: int = 10000) -> pd.DataFrame:

        self._ensure_connection(sql)

        sql, _ = inject_limit(sql, _max_rows, dialect="sqlite")  # T1.4

        try:

            return pd.read_sql_query(sql, self._conn)

        except TypeError:

            # DDL / DML statements (CREATE, INSERT…) return no cursor.description.

            # pd.read_sql_query raises TypeError in this case — return empty DataFrame.

            return pd.DataFrame()

        except Exception as exc:

            raise DatabaseError(str(exc), sql=sql, connector="sqlite") from exc

    def get_schema(self) -> str:

        if self._conn is None:

            return ""

        cursor = self._conn.cursor()

        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")

        rows = cursor.fetchall()

        return "\n\n".join(sql for _, sql in rows if sql)

    def close(self) -> None:

        if self._conn:

            self._conn.close()

            self._conn = None

            logger.info("sqlite.closed", path=self._path)
