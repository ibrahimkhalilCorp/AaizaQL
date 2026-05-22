"""
tests/integration/test_connector_matrix.py
──────────────────────────────────────────
Integration test matrix for all 9 database connectors.

Containerised (spun up via docker-compose.test.yml in CI):
  - PostgreSQL  →  POSTGRES_TEST_DSN  (default: postgresql://aaizaql:aaizaql@localhost:5432/aaizaql_test)
  - MySQL       →  MYSQL_TEST_DSN     (default: mysql://aaizaql:aaizaql@localhost:3306/aaizaql_test)
  - MSSQL       →  MSSQL_TEST_DSN     (default: mssql://sa:AaizaQL_2024!@localhost:1433/master)
  - MongoDB     →  MONGODB_TEST_DSN   (default: mongodb://aaizaql:aaizaql@localhost:27017/aaizaql_test?authSource=admin)

No container needed:
  - SQLite      →  always runs (in-memory)
  - DuckDB      →  always runs (in-memory)

Skipped unless env var gate is set:
  - Snowflake   →  SNOWFLAKE_DSN      (skipped when absent)
  - BigQuery    →  BIGQUERY_DSN       (skipped when absent)
  - Oracle      →  ORACLE_DSN         (skipped when absent)
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

# ── DSN helpers ───────────────────────────────────────────────────────────────

POSTGRES_DSN = os.getenv(
    "POSTGRES_TEST_DSN",
    "postgresql://aaizaql:aaizaql@localhost:5432/aaizaql_test",
)
MYSQL_DSN = os.getenv(
    "MYSQL_TEST_DSN",
    "mysql://aaizaql:aaizaql@localhost:3306/aaizaql_test",
)
MSSQL_DSN = os.getenv(
    "MSSQL_TEST_DSN",
    "mssql://sa:AaizaQL_2024!@localhost:1433/master",
)
MONGODB_DSN = os.getenv(
    "MONGODB_TEST_DSN",
    "mongodb://aaizaql:aaizaql@localhost:27017/aaizaql_test?authSource=admin",
)

# Cloud / licensed — skipped unless the caller provides the DSN
SNOWFLAKE_DSN = os.getenv("SNOWFLAKE_DSN", "")
BIGQUERY_DSN = os.getenv("BIGQUERY_DSN", "")
ORACLE_DSN = os.getenv("ORACLE_DSN", "")

# ── Shared DDL / seed helpers ─────────────────────────────────────────────────

_CREATE_EMPLOYEES = """
CREATE TABLE IF NOT EXISTS employees (
    id         INT PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    department VARCHAR(50)  NOT NULL,
    salary     DECIMAL(10,2) NOT NULL
)
"""

_CREATE_EMPLOYEES_MSSQL = """
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'employees') AND type = 'U')
BEGIN
    CREATE TABLE employees (
        id         INT PRIMARY KEY,
        name       NVARCHAR(100) NOT NULL,
        department NVARCHAR(50)  NOT NULL,
        salary     DECIMAL(10,2) NOT NULL
    )
END
"""

_SEED_EMPLOYEES = [
    (1, "Alice", "Engineering", 95000.00),
    (2, "Bob", "Marketing", 72000.00),
    (3, "Carol", "Engineering", 105000.00),
    (4, "Dave", "HR", 68000.00),
]

# ── Shared assertion helpers ──────────────────────────────────────────────────

def _assert_select_all(df: pd.DataFrame) -> None:
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4

def _assert_filter(df: pd.DataFrame) -> None:
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Alice and Carol

def _assert_aggregate(df: pd.DataFrame) -> None:
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
    # Use positional access — column may be named 'COUNT(*)' or 'n' depending on dialect
    assert int(df.iloc[0, 0]) == 4

def _assert_schema_nonempty(schema: str) -> None:
    assert isinstance(schema, str)
    assert len(schema) > 0

# ══════════════════════════════════════════════════════════════════════════════
# 1. SQLite  (no container — always runs)
# ══════════════════════════════════════════════════════════════════════════════

class TestSQLiteConnector:
    """Full connector test suite for SQLite (in-memory)."""

    @pytest.fixture(autouse=True)
    def connector(self):
        from aaizaql.connectors.sqlite import SQLiteConnector

        c = SQLiteConnector()
        c.connect("sqlite:///:memory:")
        c.execute(
            "CREATE TABLE employees ("
            "id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "department TEXT NOT NULL, salary REAL NOT NULL)"
        )
        for row in _SEED_EMPLOYEES:
            c.execute(f"INSERT INTO employees VALUES ({row[0]}, '{row[1]}', '{row[2]}', {row[3]})")
        yield c
        c.close()

    def test_connect_and_select_all(self, connector):
        _assert_select_all(connector.execute("SELECT * FROM employees"))

    def test_filter_by_department(self, connector):
        _assert_filter(connector.execute("SELECT * FROM employees WHERE department='Engineering'"))

    def test_aggregate_count(self, connector):
        _assert_aggregate(connector.execute("SELECT COUNT(*) FROM employees"))

    def test_get_schema_nonempty(self, connector):
        _assert_schema_nonempty(connector.get_schema())

    def test_schema_contains_table_name(self, connector):
        assert "employees" in connector.get_schema()

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_bad_sql_raises_database_error(self, connector):
        from aaizaql.core.exceptions import DatabaseError

        with pytest.raises(DatabaseError):
            connector.execute("SELECT * FROM nonexistent_xyz")

    def test_close_idempotent(self, connector):
        connector.close()
        connector.close()  # second close must not raise

# ══════════════════════════════════════════════════════════════════════════════
# 2. DuckDB  (no container — always runs)
# ══════════════════════════════════════════════════════════════════════════════

class TestDuckDBConnector:
    """Full connector test suite for DuckDB (in-memory)."""

    @pytest.fixture(autouse=True)
    def connector(self):
        pytest.importorskip("duckdb", reason="duckdb not installed")
        from aaizaql.connectors.duckdb import DuckDBConnector

        c = DuckDBConnector()
        c.connect("duckdb:///:memory:")
        c.execute(
            "CREATE TABLE employees ("
            "id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, "
            "department VARCHAR NOT NULL, salary DECIMAL(10,2) NOT NULL)"
        )
        for row in _SEED_EMPLOYEES:
            c.execute(f"INSERT INTO employees VALUES ({row[0]}, '{row[1]}', '{row[2]}', {row[3]})")
        yield c
        c.close()

    def test_connect_and_select_all(self, connector):
        _assert_select_all(connector.execute("SELECT * FROM employees"))

    def test_filter_by_department(self, connector):
        _assert_filter(connector.execute("SELECT * FROM employees WHERE department='Engineering'"))

    def test_aggregate_count(self, connector):
        _assert_aggregate(connector.execute("SELECT COUNT(*) FROM employees"))

    def test_get_schema_nonempty(self, connector):
        _assert_schema_nonempty(connector.get_schema())

    def test_schema_contains_table_name(self, connector):
        assert "employees" in connector.get_schema()

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_register_dataframe_and_query(self, connector):
        import pandas as pd

        df_input = pd.DataFrame({"x": [10, 20, 30]})
        connector.register_dataframe("temp_view", df_input)
        result = connector.execute("SELECT SUM(x) AS total FROM temp_view")
        assert int(result["total"].iloc[0]) == 60

    def test_bad_sql_raises_database_error(self, connector):
        from aaizaql.core.exceptions import DatabaseError

        with pytest.raises(DatabaseError):
            connector.execute("SELECT * FROM nonexistent_xyz")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PostgreSQL  (Docker container in CI)
# ══════════════════════════════════════════════════════════════════════════════

pytestmark_pg = pytest.mark.integration_pg

@pytest.mark.integration_pg
class TestPostgreSQLConnector:
    """Integration tests for PostgreSQL connector against a real Postgres instance."""

    @pytest.fixture(autouse=True)
    def connector(self):
        pytest.importorskip("psycopg2", reason="psycopg2 not installed")
        from aaizaql.connectors.postgres import PostgreSQLConnector
        from aaizaql.core.exceptions import ConnectionError as AaizaConnectionError

        c = PostgreSQLConnector()
        try:
            c.connect(POSTGRES_DSN)
        except AaizaConnectionError as exc:
            pytest.skip(f"Postgres not reachable: {exc}")

        # Use raw cursor for DDL — pd.read_sql_query requires a result set
        with c._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS employees")
            cur.execute(
                "CREATE TABLE employees ("
                "id INT PRIMARY KEY, name VARCHAR(100) NOT NULL, "
                "department VARCHAR(50) NOT NULL, salary DECIMAL(10,2) NOT NULL)"
            )
            for row in _SEED_EMPLOYEES:
                cur.execute("INSERT INTO employees VALUES (%s, %s, %s, %s)", row)
        c._conn.commit()
        yield c
        with c._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS employees")
        c._conn.commit()
        c.close()

    def test_connect_and_select_all(self, connector):
        _assert_select_all(connector.execute("SELECT * FROM employees"))

    def test_filter_by_department(self, connector):
        _assert_filter(connector.execute("SELECT * FROM employees WHERE department='Engineering'"))

    def test_aggregate_count(self, connector):
        _assert_aggregate(connector.execute("SELECT COUNT(*) FROM employees"))

    def test_get_schema_nonempty(self, connector):
        _assert_schema_nonempty(connector.get_schema())

    def test_schema_contains_table_name(self, connector):
        assert "employees" in connector.get_schema()

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_bad_sql_raises_database_error(self, connector):
        from aaizaql.core.exceptions import DatabaseError

        with pytest.raises(DatabaseError):
            connector.execute("SELECT * FROM nonexistent_xyz")

    def test_order_by_salary_desc(self, connector):
        df = connector.execute("SELECT name, salary FROM employees ORDER BY salary DESC")
        assert df["name"].iloc[0] == "Carol"

# ══════════════════════════════════════════════════════════════════════════════
# 4. MySQL  (Docker container in CI)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration_mysql
class TestMySQLConnector:
    """Integration tests for MySQL connector against a real MySQL instance."""

    @pytest.fixture(autouse=True)
    def connector(self):
        pytest.importorskip("pymysql", reason="pymysql not installed")
        from aaizaql.connectors.mysql import MySQLConnector
        from aaizaql.core.exceptions import ConnectionError as AaizaConnectionError

        c = MySQLConnector()
        try:
            c.connect(MYSQL_DSN)
        except AaizaConnectionError as exc:
            pytest.skip(f"MySQL not reachable: {exc}")

        # Use raw cursor for DDL
        with c._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS employees")
            cur.execute(
                "CREATE TABLE employees ("
                "id INT PRIMARY KEY, name VARCHAR(100) NOT NULL, "
                "department VARCHAR(50) NOT NULL, salary DECIMAL(10,2) NOT NULL)"
            )
            for row in _SEED_EMPLOYEES:
                cur.execute("INSERT INTO employees VALUES (%s, %s, %s, %s)", row)
        c._conn.commit()
        yield c
        with c._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS employees")
        c._conn.commit()
        c.close()

    def test_connect_and_select_all(self, connector):
        _assert_select_all(connector.execute("SELECT * FROM employees"))

    def test_filter_by_department(self, connector):
        _assert_filter(connector.execute("SELECT * FROM employees WHERE department='Engineering'"))

    def test_aggregate_count(self, connector):
        _assert_aggregate(connector.execute("SELECT COUNT(*) FROM employees"))

    def test_get_schema_nonempty(self, connector):
        _assert_schema_nonempty(connector.get_schema())

    def test_schema_contains_table_name(self, connector):
        assert "employees" in connector.get_schema()

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_bad_sql_raises_database_error(self, connector):
        from aaizaql.core.exceptions import DatabaseError

        with pytest.raises(DatabaseError):
            connector.execute("SELECT * FROM nonexistent_xyz")

    def test_order_by_salary_desc(self, connector):
        df = connector.execute("SELECT name, salary FROM employees ORDER BY salary DESC")
        assert df["name"].iloc[0] == "Carol"

# ══════════════════════════════════════════════════════════════════════════════
# 5. MSSQL  (Docker container in CI — SQL Server 2022 Developer Edition)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration_mssql
class TestMSSQLConnector:
    """Integration tests for MSSQL connector against SQL Server 2022."""

    @pytest.fixture(autouse=True)
    def connector(self):
        pytest.importorskip("pyodbc", reason="pyodbc not installed")
        from aaizaql.connectors.mssql import MSSQLConnector
        from aaizaql.core.exceptions import ConnectionError as AaizaConnectionError

        c = MSSQLConnector()
        try:
            c.connect(MSSQL_DSN)
        except AaizaConnectionError as exc:
            pytest.skip(f"MSSQL not reachable: {exc}")

        # Use raw cursor for DDL — pd.read_sql_query requires a result set
        cur = c._conn.cursor()
        try:
            cur.execute("DROP TABLE IF EXISTS employees")
            c._conn.commit()
            cur.execute(
                "CREATE TABLE employees ("
                "id INT PRIMARY KEY, name NVARCHAR(100) NOT NULL, "
                "department NVARCHAR(50) NOT NULL, salary DECIMAL(10,2) NOT NULL)"
            )
            for row in _SEED_EMPLOYEES:
                cur.execute("INSERT INTO employees VALUES (?, ?, ?, ?)", row)
            c._conn.commit()
        finally:
            cur.close()
        yield c
        cur = c._conn.cursor()
        try:
            cur.execute("DROP TABLE IF EXISTS employees")
            c._conn.commit()
        finally:
            cur.close()
        c.close()

    def test_connect_and_select_all(self, connector):
        _assert_select_all(connector.execute("SELECT * FROM employees"))

    def test_filter_by_department(self, connector):
        _assert_filter(connector.execute("SELECT * FROM employees WHERE department='Engineering'"))

    def test_aggregate_count(self, connector):
        _assert_aggregate(connector.execute("SELECT COUNT(*) FROM employees"))

    def test_get_schema_nonempty(self, connector):
        _assert_schema_nonempty(connector.get_schema())

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_bad_sql_raises_database_error(self, connector):
        from aaizaql.core.exceptions import DatabaseError

        with pytest.raises(DatabaseError):
            connector.execute("SELECT * FROM nonexistent_xyz")

    def test_order_by_salary_desc(self, connector):
        df = connector.execute("SELECT name, salary FROM employees ORDER BY salary DESC")
        assert df["name"].iloc[0] == "Carol"

# ══════════════════════════════════════════════════════════════════════════════
# 6. MongoDB  (Docker container in CI)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration_mongodb
class TestMongoDBConnector:
    """Integration tests for MongoDB connector against a real MongoDB 7 instance."""

    @pytest.fixture(autouse=True)
    def connector(self):
        pytest.importorskip("pymongo", reason="pymongo not installed")
        from aaizaql.connectors.mongodb import MongoDBConnector
        from aaizaql.core.exceptions import ConnectionError as AaizaConnectionError

        c = MongoDBConnector()
        try:
            c.connect(MONGODB_DSN)
        except AaizaConnectionError as exc:
            pytest.skip(f"MongoDB not reachable: {exc}")

        # Seed collection
        db = c._client["aaizaql_test"]
        db["employees"].drop()
        db["employees"].insert_many(
            [
                {"_id": row[0], "name": row[1], "department": row[2], "salary": row[3]}
                for row in _SEED_EMPLOYEES
            ]
        )
        yield c
        db["employees"].drop()
        c.close()

    def test_connect_returns_schema(self, connector):
        schema = connector.get_schema()
        assert isinstance(schema, str)

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_execute_find_all(self, connector):
        """MongoDB connector execute() runs an aggregation pipeline via JSON SQL shim."""
        df = connector.execute('{"collection": "employees", "filter": {}}')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_execute_filter(self, connector):
        df = connector.execute(
            '{"collection": "employees", "filter": {"department": "Engineering"}}'
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_bad_collection_raises_database_error(self, connector):
        from aaizaql.core.exceptions import DatabaseError

        with pytest.raises(DatabaseError):
            connector.execute(
                '{"collection": "__nonexistent_xyz__", "filter": {}, "raise_if_empty": true}'
            )

# ══════════════════════════════════════════════════════════════════════════════
# 7. Snowflake  (skipped unless SNOWFLAKE_DSN env var is set)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not SNOWFLAKE_DSN,
    reason="SNOWFLAKE_DSN env var not set — skipping Snowflake integration tests",
)
class TestSnowflakeConnector:
    """Integration tests for Snowflake — requires SNOWFLAKE_DSN env var."""

    @pytest.fixture(autouse=True)
    def connector(self):
        pytest.importorskip(
            "snowflake.connector", reason="snowflake-connector-python not installed"
        )
        from aaizaql.connectors.snowflake import SnowflakeConnector

        c = SnowflakeConnector()
        c.connect(SNOWFLAKE_DSN)
        yield c
        c.close()

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_get_schema_nonempty(self, connector):
        _assert_schema_nonempty(connector.get_schema())

    def test_execute_simple_select(self, connector):
        df = connector.execute("SELECT 1 AS n")
        assert int(df["N"].iloc[0]) == 1

# ══════════════════════════════════════════════════════════════════════════════
# 8. BigQuery  (skipped unless BIGQUERY_DSN env var is set)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not BIGQUERY_DSN,
    reason="BIGQUERY_DSN env var not set — skipping BigQuery integration tests",
)
class TestBigQueryConnector:
    """Integration tests for BigQuery — requires BIGQUERY_DSN env var."""

    @pytest.fixture(autouse=True)
    def connector(self):
        pytest.importorskip("google.cloud.bigquery", reason="google-cloud-bigquery not installed")
        from aaizaql.connectors.bigquery import BigQueryConnector

        c = BigQueryConnector()
        c.connect(BIGQUERY_DSN)
        yield c
        c.close()

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_get_schema_nonempty(self, connector):
        _assert_schema_nonempty(connector.get_schema())

    def test_execute_simple_select(self, connector):
        df = connector.execute("SELECT 1 AS n")
        assert int(df["n"].iloc[0]) == 1

# ══════════════════════════════════════════════════════════════════════════════
# 9. Oracle  (skipped unless ORACLE_DSN env var is set)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not ORACLE_DSN,
    reason="ORACLE_DSN env var not set — skipping Oracle integration tests",
)
class TestOracleConnector:
    """Integration tests for Oracle — requires ORACLE_DSN env var."""

    @pytest.fixture(autouse=True)
    def connector(self):
        pytest.importorskip("oracledb", reason="oracledb not installed")
        from aaizaql.connectors.oracle import OracleConnector

        c = OracleConnector()
        c.connect(ORACLE_DSN)
        yield c
        c.close()

    def test_test_connection_returns_true(self, connector):
        assert connector.test_connection() is True

    def test_get_schema_nonempty(self, connector):
        _assert_schema_nonempty(connector.get_schema())

    def test_execute_simple_select(self, connector):
        df = connector.execute("SELECT 1 AS n FROM DUAL")
        assert int(df["N"].iloc[0]) == 1
