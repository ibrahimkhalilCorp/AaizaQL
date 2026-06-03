"""
PostgreSQL connector for AaizaQL.

Fix (Issue #1): DSN passwords containing special characters (e.g. ``@``) are
URL-encoded before the DSN is handed to psycopg2, preventing the URL parser
from misidentifying the host.
"""

from __future__ import annotations

from urllib.parse import quote, urlparse, urlunparse

try:
    import psycopg2
    import psycopg2.extras
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgreSQL support requires psycopg2. " 'Run: pip install "aaizaql[postgres]"'
    ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode_dsn_password(dsn: str) -> str:
    """Return *dsn* with the password component percent-encoded.

    psycopg2 / libpq parse the DSN as a URL, so any ``@`` (or other
    reserved character) inside the password must be encoded as ``%40``
    before the string is passed to the driver.  We re-assemble the URL
    from parsed components so that the *existing* ``@`` host separator
    is never double-encoded.

    Examples
    --------
    >>> _encode_dsn_password("postgresql://user:pass@word@host:5432/db")
    'postgresql://user:pass%40word@host:5432/db'
    >>> _encode_dsn_password("postgresql://user:simple@host/db")
    'postgresql://user:simple@host/db'
    """
    parsed = urlparse(dsn)

    # Nothing to do if there is no password or it is already safe.
    if not parsed.password:
        return dsn

    encoded_password = quote(parsed.password, safe="")

    # Rebuild netloc: user:encoded_pw@host[:port]
    netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"

    return urlunparse(parsed._replace(netloc=netloc))


# ---------------------------------------------------------------------------
# Public connector class
# ---------------------------------------------------------------------------


class PostgresConnector:
    """Thin wrapper around psycopg2 that exposes the interface expected by
    AaizaQL's ``QueryEngine``.

    Parameters
    ----------
    dsn:
        A ``postgresql://user:password@host:port/dbname`` connection string.
        Passwords containing URL-reserved characters (``@ : / ? # [ ] !``)
        are automatically percent-encoded before the DSN is forwarded to the
        driver (fixes Issue #1).
    """

    def __init__(self, dsn: str) -> None:
        self._dsn: str = _encode_dsn_password(dsn)
        self._conn: psycopg2.extensions.connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open (or re-open) the database connection."""
        try:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = True
        except Exception as exc:
            raise ConnectionError(f"Cannot connect to 'postgresql' ({self._dsn}): {exc}") from exc

    def disconnect(self) -> None:
        """Close the connection if open."""
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

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute(self, sql: str) -> list[dict]:
        """Execute *sql* and return rows as a list of dicts."""
        if self._conn is None or self._conn.closed:
            self.connect()

        assert self._conn is not None  # satisfy mypy

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            if cur.description is None:
                return []
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schema(self) -> dict[str, list[dict]]:
        """Return ``{table_name: [{column, type, nullable}, ...]}`` for the
        current database's public schema."""
        sql = """
            SELECT
                table_name,
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """
        rows = self.execute(sql)

        schema: dict[str, list[dict]] = {}
        for row in rows:
            tbl = row["table_name"]
            schema.setdefault(tbl, []).append(
                {
                    "column": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                }
            )
        return schema
