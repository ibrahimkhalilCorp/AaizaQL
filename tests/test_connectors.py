"""
tests/test_connectors.py
─────────────────────────
Integration tests for database connectors.
Uses the real SQLite connector against an in-process temp database — no mocks.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aqlix.connectors.sqlite import SQLiteConnector
from aqlix.core.exceptions import DatabaseError


class TestSQLiteConnector:
    def test_connect_and_simple_query(self, sqlite_connector: SQLiteConnector) -> None:
        df = sqlite_connector.execute("SELECT * FROM employees")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "name" in df.columns

    def test_filtered_query(self, sqlite_connector: SQLiteConnector) -> None:
        df = sqlite_connector.execute(
            "SELECT name FROM employees WHERE dept = 'Engineering'"
        )
        assert len(df) == 2
        names = set(df["name"].tolist())
        assert names == {"Alice", "Charlie"}

    def test_aggregate_query(self, sqlite_connector: SQLiteConnector) -> None:
        df = sqlite_connector.execute(
            "SELECT dept, COUNT(*) as cnt, AVG(salary) as avg_sal "
            "FROM employees GROUP BY dept ORDER BY dept"
        )
        assert len(df) == 3
        assert "cnt" in df.columns

    def test_join_query(self, sqlite_connector: SQLiteConnector) -> None:
        df = sqlite_connector.execute(
            "SELECT e.name, d.name as department "
            "FROM employees e "
            "JOIN departments d ON e.dept = d.name"
        )
        assert len(df) == 4
        assert "department" in df.columns

    def test_empty_result(self, sqlite_connector: SQLiteConnector) -> None:
        df = sqlite_connector.execute(
            "SELECT * FROM employees WHERE salary > 999999"
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_get_schema_returns_ddl(self, sqlite_connector: SQLiteConnector) -> None:
        schema = sqlite_connector.get_schema()
        assert "employees" in schema.lower()
        assert "departments" in schema.lower()
        assert "CREATE TABLE" in schema

    def test_invalid_sql_raises_database_error(
        self, sqlite_connector: SQLiteConnector
    ) -> None:
        with pytest.raises(DatabaseError):
            sqlite_connector.execute("SELECT * FROM nonexistent_table_xyz")

    def test_not_connected_raises(self) -> None:
        conn = SQLiteConnector()
        with pytest.raises(DatabaseError):
            conn.execute("SELECT 1")

    def test_memory_database(self) -> None:
        conn = SQLiteConnector()
        conn.connect("sqlite:///:memory:")
        conn.execute("CREATE TABLE t (x INTEGER)")
        # CREATE is executed but returns empty result — no error
        conn.close()

    def test_test_connection(self, sqlite_connector: SQLiteConnector) -> None:
        assert sqlite_connector.test_connection() is True

    def test_close_is_idempotent(self, sqlite_connector: SQLiteConnector) -> None:
        sqlite_connector.close()
        sqlite_connector.close()  # second close should not raise
