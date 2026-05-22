"""
tests/integration/test_sqlite_pipeline.py
──────────────────────────────────────────
Integration tests: full query pipeline against an in-memory SQLite DB.
These tests use a MockLLMProvider so no API key is needed.
Mark: not requires_api_key
"""

import sqlite3

import pandas as pd
import pytest

from aaizaql.connectors.sqlite import SQLiteConnector
from aaizaql.core.config import Settings
from aaizaql.core.exceptions import SecurityException
from aaizaql.memory.context import ContextManager
from aaizaql.security.validator import SQLValidator

# ── Helpers ────────────────────────────────────────────────────────────────


@pytest.fixture
def db_conn():
    """Create a fresh in-memory SQLite DB with sample data."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE customers (
            id       INTEGER PRIMARY KEY,
            name     TEXT NOT NULL,
            country  TEXT NOT NULL,
            revenue  REAL DEFAULT 0
        );
        CREATE TABLE orders (
            id          INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            amount      REAL NOT NULL,
            placed_at   TEXT NOT NULL
        );
        INSERT INTO customers VALUES (1, 'Alice', 'US', 5000);
        INSERT INTO customers VALUES (2, 'Bob',   'DE', 3200);
        INSERT INTO customers VALUES (3, 'Carol', 'US', 8100);
        INSERT INTO orders VALUES (1, 1, 500,  '2025-06-01');
        INSERT INTO orders VALUES (2, 1, 300,  '2025-06-15');
        INSERT INTO orders VALUES (3, 2, 1200, '2025-06-20');
        INSERT INTO orders VALUES (4, 3, 950,  '2025-07-01');
    """)
    yield conn
    conn.close()


@pytest.fixture
def connector(db_conn):
    c = SQLiteConnector()
    c._conn = db_conn  # inject pre-built connection
    return c


# ── Connector tests ────────────────────────────────────────────────────────


class TestSQLiteConnector:
    def test_execute_returns_dataframe(self, connector):
        df = connector.execute("SELECT * FROM customers")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "name" in df.columns

    def test_get_schema_returns_ddl(self, connector):
        schema = connector.get_schema()
        assert "customers" in schema
        assert "orders" in schema
        assert "CREATE TABLE" in schema.upper()

    def test_test_connection_true(self, connector):
        assert connector.test_connection() is True

    def test_bad_sql_raises_database_error(self, connector):
        from aaizaql.core.exceptions import DatabaseError

        with pytest.raises(DatabaseError):
            connector.execute("SELECT * FROM nonexistent_table_xyz")

    def test_join_query(self, connector):
        df = connector.execute("""
            SELECT c.name, SUM(o.amount) as total
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
            GROUP BY c.name
            ORDER BY total DESC
        """)
        assert len(df) == 3
        assert df.iloc[0]["name"] == "Bob"  # 1200

    def test_aggregation(self, connector):
        df = connector.execute("SELECT COUNT(*) as cnt FROM orders")
        assert df["cnt"].iloc[0] == 4


# ── Validator pipeline tests ───────────────────────────────────────────────


class TestValidatorWithRealSQL:
    @pytest.fixture
    def validator(self):
        return SQLValidator(Settings())

    def test_valid_query_passes(self, validator, connector):
        sql = "SELECT name, revenue FROM customers ORDER BY revenue DESC LIMIT 3"
        validator.validate(sql)
        df = connector.execute(sql)
        assert len(df) == 3

    def test_blocked_query_never_executes(self, validator, connector):
        sql = "DROP TABLE customers"
        with pytest.raises(SecurityException):
            validator.validate(sql)
        # customers table still intact
        df = connector.execute("SELECT COUNT(*) as cnt FROM customers")
        assert df["cnt"].iloc[0] == 3


# ── Context memory integration ─────────────────────────────────────────────


class TestContextMemory:
    def test_multi_turn_history_accumulated(self):
        ctx = ContextManager(limit=10)
        sid = "test-session"

        ctx.add_turn(sid, "How many customers?", "SELECT COUNT(*) FROM customers", 1)
        ctx.add_turn(sid, "Show US customers", "SELECT * FROM customers WHERE country='US'", 2)

        history = ctx.get_history(sid)
        assert len(history) == 2
        assert history[0]["question"] == "How many customers?"
        assert history[1]["sql"] == "SELECT * FROM customers WHERE country='US'"
