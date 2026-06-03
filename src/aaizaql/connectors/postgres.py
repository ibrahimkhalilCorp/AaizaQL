"""
PostgreSQL connector for AaizaQL.
"""

from __future__ import annotations

from urllib.parse import quote, urlparse, urlunparse

try:
    import psycopg2
    import psycopg2.extras
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'PostgreSQL support requires psycopg2. Run: pip install "aaizaql[postgres]"'
    ) from exc


def _encode_dsn_password(dsn: str) -> str:
    parsed = urlparse(dsn)
    if not parsed.password:
        return dsn
    encoded_password = quote(parsed.password, safe="")
    netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


class PostgresConnector:
    """Thin wrapper around psycopg2."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn: str = _encode_dsn_password(dsn) if dsn else ""
        self._conn: psycopg2.extensions.connection | None = None

    def connect(self, dsn: str | None = None) -> None:
        """Open the database connection. Accepts an optional DSN override."""
        if dsn:
            self._dsn = _encode_dsn_password(dsn)
        try:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = True
        except Exception as exc:
            raise ConnectionError(f"Cannot connect to 'postgresql' ({self._dsn}): {exc}") from exc

    def disconnect(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> PostgresConnector:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    def execute(self, sql: str) -> list[dict]:
        if self._conn is None or self._conn.closed:
            self.connect()
        assert self._conn is not None
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            if cur.description is None:
                return []
            return [dict(row) for row in cur.fetchall()]

    def get_schema(self) -> dict[str, list[dict]]:
        sql = """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """
        rows = self.execute(sql)
        schema: dict[str, list[dict]] = {}
        for row in rows:
            tbl = row["table_name"]
            schema.setdefault(tbl, []).append({
                "column": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
            })
        return schema

    def test_connection(self) -> bool:
        try:
            self.connect()
            self.execute("SELECT 1")
            return True
        except Exception:
            return False
        finally:
            self.disconnect()


# Alias expected by connectors/__init__.py and the test suite
PostgreSQLConnector = PostgresConnector
