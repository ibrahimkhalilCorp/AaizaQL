"""
tests/test_connectors_new.py
─────────────────────────────
Unit tests for MySQL, DuckDB, and Snowflake connectors.
All tests use mocks or in-process DuckDB — no real servers needed,
no API key required.

Append the contents of this file into tests/test_connectors.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aaizaql.core.exceptions import ConnectionError, DatabaseError

# ══════════════════════════════════════════════════════════════════════════════
# MySQL
# ══════════════════════════════════════════════════════════════════════════════


class TestMySQLConnector:
    """
    MySQLConnector tests use a patched pymysql — no real MySQL server needed.
    """

    # ── DSN parser ────────────────────────────────────────────────────────────

    def test_parse_dsn_full(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        result = MySQLConnector._parse_dsn("mysql://alice:secret@db.example.com:3307/sales")
        assert result["host"] == "db.example.com"
        assert result["port"] == 3307
        assert result["user"] == "alice"
        assert result["password"] == "secret"
        assert result["database"] == "sales"

    def test_parse_dsn_default_port(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        result = MySQLConnector._parse_dsn("mysql://user:pass@localhost/mydb")
        assert result["port"] == 3306

    def test_parse_dsn_no_password(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        result = MySQLConnector._parse_dsn("mysql://user@localhost/mydb")
        assert result["password"] == ""

    def test_parse_dsn_invalid_raises(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        with pytest.raises(ValueError, match="Cannot parse MySQL DSN"):
            MySQLConnector._parse_dsn("not-a-valid-dsn")

    # ── connect ───────────────────────────────────────────────────────────────

    def test_connect_success(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        mock_conn = MagicMock()
        with patch("pymysql.connect", return_value=mock_conn):
            connector = MySQLConnector()
            connector.connect("mysql://user:pass@localhost/mydb")
            assert connector._conn is mock_conn

    def test_connect_failure_raises_connection_error(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        with patch("pymysql.connect", side_effect=Exception("refused")):
            connector = MySQLConnector()
            with pytest.raises(ConnectionError, match="mysql"):
                connector.connect("mysql://user:pass@localhost/mydb")

    # ── execute ───────────────────────────────────────────────────────────────

    def test_execute_returns_dataframe(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        mock_conn = MagicMock()
        expected_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        with patch("pandas.read_sql_query", return_value=expected_df):
            connector = MySQLConnector()
            connector._conn = mock_conn
            result = connector.execute("SELECT id, name FROM users")
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["id", "name"]
            assert len(result) == 2

    def test_execute_not_connected_raises(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        connector = MySQLConnector()
        with pytest.raises(DatabaseError, match="Not connected"):
            connector.execute("SELECT 1")

    def test_execute_db_error_raises_database_error(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        mock_conn = MagicMock()
        with patch("pandas.read_sql_query", side_effect=Exception("table not found")):
            connector = MySQLConnector()
            connector._conn = mock_conn
            with pytest.raises(DatabaseError, match="table not found"):
                connector.execute("SELECT * FROM nonexistent")

    # ── get_schema ────────────────────────────────────────────────────────────

    def test_get_schema_returns_ddl_string(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        mock_conn = MagicMock()
        schema_df = pd.DataFrame(
            {
                "ddl": [
                    "CREATE TABLE `users` (`id` int NOT NULL, `name` varchar(255));",
                    "CREATE TABLE `orders` (`id` int NOT NULL, `user_id` int);",
                ]
            }
        )
        with patch("pandas.read_sql_query", return_value=schema_df):
            connector = MySQLConnector()
            connector._conn = mock_conn
            schema = connector.get_schema()
            assert "users" in schema
            assert "orders" in schema
            assert "CREATE TABLE" in schema

    def test_get_schema_not_connected_returns_empty(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        connector = MySQLConnector()
        assert connector.get_schema() == ""

    # ── close ─────────────────────────────────────────────────────────────────

    def test_close_calls_connection_close(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        mock_conn = MagicMock()
        connector = MySQLConnector()
        connector._conn = mock_conn
        connector.close()
        mock_conn.close.assert_called_once()
        assert connector._conn is None

    def test_close_is_idempotent(self) -> None:
        from aaizaql.connectors.mysql import MySQLConnector

        connector = MySQLConnector()
        connector.close()  # no connection — should not raise
        connector.close()


# ══════════════════════════════════════════════════════════════════════════════
# DuckDB
# ══════════════════════════════════════════════════════════════════════════════


class TestDuckDBConnector:
    """
    DuckDBConnector tests use real in-process DuckDB (:memory:).
    No mocking needed — DuckDB is a zero-dependency embedded engine.
    """

    @pytest.fixture()
    def connector(self):
        from aaizaql.connectors.duckdb import DuckDBConnector

        c = DuckDBConnector()
        c.connect("duckdb:///:memory:")
        c.execute("""
            CREATE TABLE products (
                id    INTEGER,
                name  VARCHAR,
                price DOUBLE
            )
        """)
        c.execute("INSERT INTO products VALUES (1, 'Widget', 9.99), (2, 'Gadget', 24.99)")
        yield c
        c.close()

    # ── connect ───────────────────────────────────────────────────────────────

    def test_connect_memory(self) -> None:
        from aaizaql.connectors.duckdb import DuckDBConnector

        c = DuckDBConnector()
        c.connect("duckdb:///:memory:")
        assert c._conn is not None
        c.close()

    def test_connect_invalid_raises_connection_error(self) -> None:
        from aaizaql.connectors.duckdb import DuckDBConnector

        with patch("duckdb.connect", side_effect=Exception("bad path")):
            c = DuckDBConnector()
            with pytest.raises(ConnectionError, match="duckdb"):
                c.connect("duckdb:///bad/path/that/fails")

    # ── execute ───────────────────────────────────────────────────────────────

    def test_execute_simple_query(self, connector) -> None:
        df = connector.execute("SELECT * FROM products ORDER BY id")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "price"]

    def test_execute_aggregate(self, connector) -> None:
        df = connector.execute("SELECT COUNT(*) as cnt, AVG(price) as avg_price FROM products")
        assert df["cnt"].iloc[0] == 2
        assert round(float(df["avg_price"].iloc[0]), 2) == 17.49

    def test_execute_filtered(self, connector) -> None:
        df = connector.execute("SELECT name FROM products WHERE price > 10")
        assert len(df) == 1
        assert df["name"].iloc[0] == "Gadget"

    def test_execute_not_connected_raises(self) -> None:
        from aaizaql.connectors.duckdb import DuckDBConnector

        c = DuckDBConnector()
        with pytest.raises(DatabaseError, match="Not connected"):
            c.execute("SELECT 1")

    def test_execute_bad_sql_raises_database_error(self, connector) -> None:
        with pytest.raises(DatabaseError):
            connector.execute("SELECT * FROM nonexistent_table_xyz")

    # ── get_schema ────────────────────────────────────────────────────────────

    def test_get_schema_contains_table_name(self, connector) -> None:
        schema = connector.get_schema()
        assert "products" in schema.lower()

    def test_get_schema_contains_column_names(self, connector) -> None:
        schema = connector.get_schema()
        assert "name" in schema.lower()
        assert "price" in schema.lower()

    def test_get_schema_not_connected_returns_empty(self) -> None:
        from aaizaql.connectors.duckdb import DuckDBConnector

        c = DuckDBConnector()
        assert c.get_schema() == ""

    def test_get_schema_empty_db_returns_empty(self) -> None:
        from aaizaql.connectors.duckdb import DuckDBConnector

        c = DuckDBConnector()
        c.connect("duckdb:///:memory:")
        assert c.get_schema() == ""
        c.close()

    # ── register_dataframe (federation hook) ──────────────────────────────────

    def test_register_dataframe_queryable(self, connector) -> None:
        df = pd.DataFrame({"region": ["US", "EU"], "total": [1000, 850]})
        connector.register_dataframe("sales", df)
        result = connector.execute("SELECT region, total FROM sales ORDER BY total DESC")
        assert len(result) == 2
        assert result["region"].iloc[0] == "US"

    def test_register_dataframe_not_connected_raises(self) -> None:
        from aaizaql.connectors.duckdb import DuckDBConnector

        c = DuckDBConnector()
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(DatabaseError):
            c.register_dataframe("t", df)

    # ── close ─────────────────────────────────────────────────────────────────

    def test_close_is_idempotent(self, connector) -> None:
        connector.close()
        connector.close()  # should not raise

    def test_test_connection(self, connector) -> None:
        assert connector.test_connection() is True


# ══════════════════════════════════════════════════════════════════════════════
# Snowflake
# ══════════════════════════════════════════════════════════════════════════════

snowflake_connector = pytest.importorskip(
    "snowflake.connector",
    reason="snowflake-connector-python is not installed; skipping Snowflake tests",
)


class TestSnowflakeConnector:
    """
    SnowflakeConnector tests use a patched snowflake.connector —
    no real Snowflake account needed.
    """

    # ── DSN parser ────────────────────────────────────────────────────────────

    def test_parse_dsn_full(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        result = SnowflakeConnector._parse_dsn(
            "snowflake://alice:secret@myaccount/mydb/myschema" "?warehouse=COMPUTE_WH&role=ANALYST"
        )
        assert result["user"] == "alice"
        assert result["password"] == "secret"
        assert result["account"] == "myaccount"
        assert result["database"] == "mydb"
        assert result["schema"] == "myschema"
        assert result["warehouse"] == "COMPUTE_WH"
        assert result["role"] == "ANALYST"

    def test_parse_dsn_minimal(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        result = SnowflakeConnector._parse_dsn("snowflake://user:pass@account/db")
        assert result["database"] == "db"
        assert result["schema"] == "PUBLIC"  # default
        assert result["warehouse"] is None
        assert result["role"] is None

    def test_parse_dsn_no_database(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        result = SnowflakeConnector._parse_dsn("snowflake://user:pass@account/")
        assert result["database"] == ""

    def test_parse_dsn_wrong_scheme_raises(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        with pytest.raises(ValueError, match="Invalid scheme"):
            SnowflakeConnector._parse_dsn("mysql://user:pass@host/db")

    # ── connect ───────────────────────────────────────────────────────────────

    def test_connect_success(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        mock_conn = MagicMock()
        with patch("snowflake.connector.connect", return_value=mock_conn):
            c = SnowflakeConnector()
            c.connect("snowflake://user:pass@account/db/PUBLIC?warehouse=WH")
            assert c._conn is mock_conn
            assert c._account == "account"
            assert c._database == "db"

    def test_connect_failure_raises_connection_error(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        with patch("snowflake.connector.connect", side_effect=Exception("auth failed")):
            c = SnowflakeConnector()
            with pytest.raises(ConnectionError, match="snowflake"):
                c.connect("snowflake://user:pass@account/db")

    # ── execute ───────────────────────────────────────────────────────────────

    def test_execute_returns_dataframe(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        mock_cursor = MagicMock()
        mock_cursor.description = [("ID",), ("NAME",)]
        mock_cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        c = SnowflakeConnector()
        c._conn = mock_conn
        df = c.execute("SELECT ID, NAME FROM users")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["ID", "NAME"]
        assert len(df) == 2

    def test_execute_no_description_returns_empty(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        mock_cursor = MagicMock()
        mock_cursor.description = None

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        c = SnowflakeConnector()
        c._conn = mock_conn
        df = c.execute("CALL some_procedure()")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_execute_not_connected_raises(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        c = SnowflakeConnector()
        with pytest.raises(DatabaseError, match="Not connected"):
            c.execute("SELECT 1")

    def test_execute_error_raises_database_error(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("query failed")

        c = SnowflakeConnector()
        c._conn = mock_conn
        with pytest.raises(DatabaseError, match="query failed"):
            c.execute("SELECT * FROM bad_table")

    # ── get_schema ────────────────────────────────────────────────────────────

    def test_get_schema_not_connected_returns_empty(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        c = SnowflakeConnector()
        assert c.get_schema() == ""

    def test_get_schema_returns_ddl(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        cols_df = pd.DataFrame(
            {
                "TABLE_NAME": ["USERS", "USERS", "ORDERS"],
                "COLUMN_NAME": ["ID", "NAME", "ID"],
                "DATA_TYPE": ["NUMBER", "VARCHAR", "NUMBER"],
                "CHARACTER_MAXIMUM_LENGTH": [None, 255.0, None],
                "IS_NULLABLE": ["NO", "YES", "NO"],
            }
        )

        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("TABLE_NAME",),
            ("COLUMN_NAME",),
            ("DATA_TYPE",),
            ("CHARACTER_MAXIMUM_LENGTH",),
            ("IS_NULLABLE",),
        ]
        mock_cursor.fetchall.return_value = [tuple(row) for row in cols_df.values]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        c = SnowflakeConnector()
        c._conn = mock_conn
        c._database = "MYDB"
        c._schema = "PUBLIC"
        schema = c.get_schema()

        assert "USERS" in schema
        assert "ORDERS" in schema
        assert "CREATE TABLE" in schema

    # ── close ─────────────────────────────────────────────────────────────────

    def test_close_calls_connection_close(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        mock_conn = MagicMock()
        c = SnowflakeConnector()
        c._conn = mock_conn
        c.close()
        mock_conn.close.assert_called_once()
        assert c._conn is None

    def test_close_is_idempotent(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        c = SnowflakeConnector()
        c.close()
        c.close()  # no connection — should not raise

    # ── test_connection ───────────────────────────────────────────────────────

    def test_test_connection_success(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        c = SnowflakeConnector()
        c._conn = mock_conn
        assert c.test_connection() is True

    def test_test_connection_failure_returns_false(self) -> None:
        from aaizaql.connectors.snowflake import SnowflakeConnector

        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("network error")

        c = SnowflakeConnector()
        c._conn = mock_conn
        assert c.test_connection() is False
