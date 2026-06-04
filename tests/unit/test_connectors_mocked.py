"""
tests/unit/test_connectors_mocked.py
──────────────────────────────────────
Unit tests for the heavy connectors (Snowflake, BigQuery, MSSQL, Oracle, MySQL).
All third-party drivers are mocked — no real databases or credentials needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from aaizaql.core.exceptions import ConnectionError, DatabaseError

# ═══════════════════════════════════════════════════════════════════════════════
# SNOWFLAKE
# ═══════════════════════════════════════════════════════════════════════════════


class TestSnowflakeConnector:
    """SnowflakeConnector with mocked snowflake.connector."""

    @pytest.fixture(autouse=True)
    def _mock_snowflake(self):
        """Inject a fake snowflake.connector into sys.modules before each test."""
        fake_sf = MagicMock()
        fake_conn = MagicMock()
        fake_sf.connector.connect.return_value = fake_conn
        with patch.dict(
            "sys.modules",
            {
                "snowflake": fake_sf,
                "snowflake.connector": fake_sf.connector,
            },
        ):
            yield fake_sf, fake_conn

    def _make_connector(self):
        from aaizaql.connectors.snowflake import SnowflakeConnector

        return SnowflakeConnector()

    # ── _parse_dsn ────────────────────────────────────────────────────────────

    def test_parse_dsn_full(self):
        from aaizaql.connectors.snowflake import SnowflakeConnector

        result = SnowflakeConnector._parse_dsn(
            "snowflake://myuser:mypass@myaccount/mydb/myschema?warehouse=WH&role=ANALYST"
        )
        assert result["user"] == "myuser"
        assert result["password"] == "mypass"
        assert result["account"] == "myaccount"
        assert result["database"] == "mydb"
        assert result["schema"] == "myschema"
        assert result["warehouse"] == "WH"
        assert result["role"] == "ANALYST"

    def test_parse_dsn_minimal(self):
        from aaizaql.connectors.snowflake import SnowflakeConnector

        result = SnowflakeConnector._parse_dsn("snowflake://u:p@acct/db")
        assert result["schema"] == "PUBLIC"
        assert result["warehouse"] is None

    def test_parse_dsn_wrong_scheme_raises(self):
        from aaizaql.connectors.snowflake import SnowflakeConnector

        with pytest.raises(ValueError, match="snowflake://"):
            SnowflakeConnector._parse_dsn("postgres://u:p@host/db")

    # ── connect ───────────────────────────────────────────────────────────────

    def test_connect_calls_driver(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        conn = self._make_connector()
        conn.connect("snowflake://user:pass@account/db/PUBLIC?warehouse=WH")
        fake_sf.connector.connect.assert_called_once()

    def test_connect_driver_error_raises_connection_error(self, _mock_snowflake):
        fake_sf, _ = _mock_snowflake
        fake_sf.connector.connect.side_effect = Exception("auth failed")
        conn = self._make_connector()
        with pytest.raises(ConnectionError):
            conn.connect("snowflake://u:p@acct/db")

    def test_connect_missing_driver_raises_connection_error(self):
        """If snowflake package is missing, connect() raises ConnectionError."""
        from aaizaql.connectors.snowflake import SnowflakeConnector

        conn = SnowflakeConnector()
        with (
            patch.dict("sys.modules", {"snowflake": None, "snowflake.connector": None}),
            pytest.raises((ConnectionError, Exception)),
        ):
            conn.connect("snowflake://u:p@acct/db")

    # ── execute ───────────────────────────────────────────────────────────────

    def test_execute_not_connected_raises(self):
        conn = self._make_connector()
        with pytest.raises(DatabaseError):
            conn.execute("SELECT 1")

    def test_execute_returns_dataframe(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        cur = MagicMock()
        cur.description = [("id",), ("name",)]
        cur.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
        fake_conn.cursor.return_value = cur

        conn = self._make_connector()
        conn.connect("snowflake://u:p@acct/db")
        conn._conn = fake_conn

        df = conn.execute("SELECT id, name FROM users")
        assert list(df.columns) == ["id", "name"]
        assert len(df) == 2

    def test_execute_no_description_returns_empty(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        cur = MagicMock()
        cur.description = None
        fake_conn.cursor.return_value = cur

        conn = self._make_connector()
        conn._conn = fake_conn
        df = conn.execute("INSERT INTO t VALUES (1)")
        assert df.empty

    def test_execute_db_error_raises(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        cur = MagicMock()
        cur.execute.side_effect = Exception("syntax error")
        fake_conn.cursor.return_value = cur

        conn = self._make_connector()
        conn._conn = fake_conn
        with pytest.raises(DatabaseError):
            conn.execute("BAD SQL")

    # ── get_schema ────────────────────────────────────────────────────────────

    def test_get_schema_not_connected_returns_empty(self):
        conn = self._make_connector()
        assert conn.get_schema() == ""

    def test_get_schema_returns_ddl(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        df = pd.DataFrame(
            {
                "TABLE_NAME": ["users", "users"],
                "COLUMN_NAME": ["id", "name"],
                "DATA_TYPE": ["NUMBER", "VARCHAR"],
                "CHARACTER_MAXIMUM_LENGTH": [None, 255],
                "IS_NULLABLE": ["NO", "YES"],
                "ORDINAL_POSITION": [1, 2],
            }
        )
        cur = MagicMock()
        cur.description = [(c,) for c in df.columns]
        cur.fetchall.return_value = list(df.itertuples(index=False, name=None))
        fake_conn.cursor.return_value = cur

        conn = self._make_connector()
        conn._conn = fake_conn
        conn._database = "mydb"
        conn._schema = "PUBLIC"

        with patch.object(conn, "execute", return_value=df):
            schema = conn.get_schema()
        assert "CREATE TABLE" in schema
        assert "users" in schema

    def test_get_schema_execute_error_returns_empty(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        conn = self._make_connector()
        conn._conn = fake_conn
        with patch.object(conn, "execute", side_effect=Exception("boom")):
            assert conn.get_schema() == ""

    def test_get_schema_empty_result_returns_empty(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        conn = self._make_connector()
        conn._conn = fake_conn
        with patch.object(conn, "execute", return_value=pd.DataFrame()):
            assert conn.get_schema() == ""

    # ── close / test_connection ───────────────────────────────────────────────

    def test_close_disconnects(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        conn = self._make_connector()
        conn._conn = fake_conn
        conn.close()
        fake_conn.close.assert_called_once()
        assert conn._conn is None

    def test_close_noop_when_not_connected(self):
        conn = self._make_connector()
        conn.close()  # should not raise

    def test_test_connection_success(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        cur = MagicMock()
        fake_conn.cursor.return_value = cur
        conn = self._make_connector()
        conn._conn = fake_conn
        assert conn.test_connection() is True

    def test_test_connection_failure(self, _mock_snowflake):
        fake_sf, fake_conn = _mock_snowflake
        fake_conn.cursor.side_effect = Exception("down")
        conn = self._make_connector()
        conn._conn = fake_conn
        assert conn.test_connection() is False


# ═══════════════════════════════════════════════════════════════════════════════
# BIGQUERY
# ═══════════════════════════════════════════════════════════════════════════════


class TestBigQueryConnector:
    """BigQueryConnector with mocked google-cloud-bigquery."""

    @pytest.fixture(autouse=True)
    def _mock_bq(self):
        fake_bq_mod = MagicMock()
        fake_client = MagicMock()
        fake_bq_mod.Client.return_value = fake_client

        fake_google = MagicMock()
        fake_google.cloud.bigquery = fake_bq_mod

        with patch.dict(
            "sys.modules",
            {
                "google": fake_google,
                "google.cloud": fake_google.cloud,
                "google.cloud.bigquery": fake_bq_mod,
                "google.oauth2": MagicMock(),
                "google.oauth2.service_account": MagicMock(),
            },
        ):
            yield fake_bq_mod, fake_client

    def _make_connector(self):
        from aaizaql.connectors.bigquery import BigQueryConnector

        return BigQueryConnector()

    # ── _parse_dsn ────────────────────────────────────────────────────────────

    def test_parse_dsn_basic(self):
        from aaizaql.connectors.bigquery import BigQueryConnector

        conn = BigQueryConnector()
        project, dataset, creds = conn._parse_dsn("bigquery://my-project/my_dataset")
        assert project == "my-project"
        assert dataset == "my_dataset"
        assert creds == ""

    def test_parse_dsn_with_credentials(self):
        from aaizaql.connectors.bigquery import BigQueryConnector

        conn = BigQueryConnector()
        project, dataset, creds = conn._parse_dsn(
            "bigquery://my-project/my_dataset?credentials_path=/tmp/key.json"
        )
        assert creds == "/tmp/key.json"

    def test_parse_dsn_invalid_raises(self):
        from aaizaql.connectors.bigquery import BigQueryConnector

        conn = BigQueryConnector()
        with pytest.raises(ValueError, match="Cannot parse"):
            conn._parse_dsn("not-a-bq-dsn")

    # ── connect ───────────────────────────────────────────────────────────────

    def test_connect_creates_client(self, _mock_bq):
        fake_bq_mod, fake_client = _mock_bq
        conn = self._make_connector()
        conn.connect("bigquery://my-project/my_dataset")
        fake_bq_mod.Client.assert_called_once_with(project="my-project")
        assert conn._project == "my-project"
        assert conn._dataset == "my_dataset"

    def test_connect_client_error_raises_connection_error(self, _mock_bq):
        fake_bq_mod, _ = _mock_bq
        fake_bq_mod.Client.side_effect = Exception("auth error")
        conn = self._make_connector()
        with pytest.raises(ConnectionError):
            conn.connect("bigquery://proj/ds")

    def test_connect_missing_package_raises_connection_error(self):
        from aaizaql.connectors.bigquery import BigQueryConnector

        conn = BigQueryConnector()
        with (
            patch.dict(
                "sys.modules",
                {"google": None, "google.cloud": None, "google.cloud.bigquery": None},
            ),
            pytest.raises((ConnectionError, Exception)),
        ):
            conn.connect("bigquery://proj/ds")

    # ── execute ───────────────────────────────────────────────────────────────

    def test_execute_not_connected_raises(self):
        conn = self._make_connector()
        with pytest.raises(DatabaseError):
            conn.execute("SELECT 1")

    def test_execute_returns_dataframe(self, _mock_bq):
        fake_bq_mod, fake_client = _mock_bq
        expected_df = pd.DataFrame({"col": [1, 2]})
        job = MagicMock()
        job.to_dataframe.return_value = expected_df
        fake_client.query.return_value = job

        conn = self._make_connector()
        conn._client = fake_client
        df = conn.execute("SELECT col FROM t")
        assert list(df.columns) == ["col"]

    def test_execute_error_raises_database_error(self, _mock_bq):
        fake_bq_mod, fake_client = _mock_bq
        fake_client.query.side_effect = Exception("quota exceeded")
        conn = self._make_connector()
        conn._client = fake_client
        with pytest.raises(DatabaseError):
            conn.execute("SELECT 1")

    # ── get_schema ────────────────────────────────────────────────────────────

    def test_get_schema_not_connected_returns_empty(self):
        conn = self._make_connector()
        assert conn.get_schema() == ""

    def test_get_schema_returns_ddl(self, _mock_bq):
        fake_bq_mod, fake_client = _mock_bq

        field = MagicMock()
        field.name = "id"
        field.field_type = "INTEGER"
        field.mode = "REQUIRED"

        table = MagicMock()
        table.schema = [field]
        table.table_id = "users"

        table_ref = MagicMock()
        fake_client.list_tables.return_value = [table_ref]
        fake_client.get_table.return_value = table

        conn = self._make_connector()
        conn._client = fake_client
        conn._project = "proj"
        conn._dataset = "ds"

        schema = conn.get_schema()
        assert "CREATE TABLE" in schema
        assert "users" in schema
        assert "INTEGER NOT NULL" in schema

    def test_get_schema_error_returns_empty(self, _mock_bq):
        fake_bq_mod, fake_client = _mock_bq
        fake_client.list_tables.side_effect = Exception("permission denied")
        conn = self._make_connector()
        conn._client = fake_client
        conn._project = "p"
        conn._dataset = "d"
        assert conn.get_schema() == ""

    # ── close ─────────────────────────────────────────────────────────────────

    def test_close_disconnects(self, _mock_bq):
        fake_bq_mod, fake_client = _mock_bq
        conn = self._make_connector()
        conn._client = fake_client
        conn.close()
        fake_client.close.assert_called_once()
        assert conn._client is None

    def test_close_noop_when_not_connected(self):
        conn = self._make_connector()
        conn.close()  # no raise


# ═══════════════════════════════════════════════════════════════════════════════
# MSSQL
# ═══════════════════════════════════════════════════════════════════════════════


class TestMSSQLConnector:
    """MSSQLConnector with mocked pyodbc."""

    @pytest.fixture(autouse=True)
    def _mock_pyodbc(self):
        fake_pyodbc = MagicMock()
        fake_conn = MagicMock()
        fake_pyodbc.connect.return_value = fake_conn
        with patch.dict("sys.modules", {"pyodbc": fake_pyodbc}):
            yield fake_pyodbc, fake_conn

    def _make_connector(self):
        from aaizaql.connectors.mssql import MSSQLConnector

        return MSSQLConnector()

    # ── _build_conn_str ───────────────────────────────────────────────────────

    def test_build_conn_str_from_url(self):
        from aaizaql.connectors.mssql import MSSQLConnector

        conn = MSSQLConnector()
        cs = conn._build_conn_str("mssql://sa:secret@localhost:1433/mydb")
        assert "SERVER=localhost,1433" in cs
        assert "DATABASE=mydb" in cs
        assert "UID=sa" in cs

    def test_build_conn_str_default_port(self):
        from aaizaql.connectors.mssql import MSSQLConnector

        conn = MSSQLConnector()
        cs = conn._build_conn_str("mssql://sa:secret@localhost/mydb")
        assert "1433" in cs

    def test_build_conn_str_passthrough(self):
        from aaizaql.connectors.mssql import MSSQLConnector

        conn = MSSQLConnector()
        raw = "Driver={ODBC Driver 18 for SQL Server};SERVER=host;DATABASE=db;UID=u;PWD=p;"
        assert conn._build_conn_str(raw) == raw

    def test_build_conn_str_invalid_raises(self):
        from aaizaql.connectors.mssql import MSSQLConnector

        conn = MSSQLConnector()
        with pytest.raises(ValueError, match="Cannot parse"):
            conn._build_conn_str("badformat")

    # ── connect ───────────────────────────────────────────────────────────────

    def test_connect_calls_pyodbc(self, _mock_pyodbc):
        fake_pyodbc, fake_conn = _mock_pyodbc
        conn = self._make_connector()
        conn.connect("mssql://sa:pass@localhost:1433/mydb")
        fake_pyodbc.connect.assert_called_once()

    def test_connect_error_raises_connection_error(self, _mock_pyodbc):
        fake_pyodbc, _ = _mock_pyodbc
        fake_pyodbc.connect.side_effect = Exception("login failed")
        conn = self._make_connector()
        with pytest.raises(ConnectionError):
            conn.connect("mssql://sa:pass@localhost/mydb")

    def test_connect_missing_pyodbc_raises(self):
        from aaizaql.connectors.mssql import MSSQLConnector

        conn = MSSQLConnector()
        with (
            patch.dict("sys.modules", {"pyodbc": None}),
            pytest.raises((ConnectionError, Exception)),
        ):
            conn.connect("mssql://sa:pass@localhost/mydb")

    # ── execute ───────────────────────────────────────────────────────────────

    def test_execute_not_connected_raises(self):
        conn = self._make_connector()
        with pytest.raises(DatabaseError):
            conn.execute("SELECT 1")

    def test_execute_returns_dataframe(self, _mock_pyodbc):
        fake_pyodbc, fake_conn = _mock_pyodbc
        expected = pd.DataFrame({"id": [1, 2]})
        with patch("pandas.read_sql_query", return_value=expected):
            conn = self._make_connector()
            conn._conn = fake_conn
            df = conn.execute("SELECT id FROM t")
        assert list(df.columns) == ["id"]

    def test_execute_error_raises_database_error(self, _mock_pyodbc):
        fake_pyodbc, fake_conn = _mock_pyodbc
        with patch("pandas.read_sql_query", side_effect=Exception("timeout")):
            conn = self._make_connector()
            conn._conn = fake_conn
            with pytest.raises(DatabaseError):
                conn.execute("SELECT 1")

    # ── get_schema ────────────────────────────────────────────────────────────

    def test_get_schema_not_connected_returns_empty(self):
        conn = self._make_connector()
        assert conn.get_schema() == ""

    def test_get_schema_returns_ddl(self, _mock_pyodbc):
        fake_pyodbc, fake_conn = _mock_pyodbc
        df = pd.DataFrame({"ddl": ["CREATE TABLE [dbo].[users] (id INT NOT NULL);"]})
        with patch("pandas.read_sql_query", return_value=df):
            conn = self._make_connector()
            conn._conn = fake_conn
            schema = conn.get_schema()
        assert "CREATE TABLE" in schema

    def test_get_schema_error_returns_empty(self, _mock_pyodbc):
        fake_pyodbc, fake_conn = _mock_pyodbc
        with patch("pandas.read_sql_query", side_effect=Exception("boom")):
            conn = self._make_connector()
            conn._conn = fake_conn
            assert conn.get_schema() == ""

    # ── close ─────────────────────────────────────────────────────────────────

    def test_close_disconnects(self, _mock_pyodbc):
        fake_pyodbc, fake_conn = _mock_pyodbc
        conn = self._make_connector()
        conn._conn = fake_conn
        conn.close()
        fake_conn.close.assert_called_once()
        assert conn._conn is None

    def test_close_noop_when_not_connected(self):
        conn = self._make_connector()
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ORACLE
# ═══════════════════════════════════════════════════════════════════════════════


class TestOracleConnector:
    """OracleConnector with mocked oracledb."""

    @pytest.fixture(autouse=True)
    def _mock_oracledb(self):
        fake_oracledb = MagicMock()
        fake_conn = MagicMock()
        fake_oracledb.connect.return_value = fake_conn
        with patch.dict("sys.modules", {"oracledb": fake_oracledb}):
            yield fake_oracledb, fake_conn

    def _make_connector(self):
        from aaizaql.connectors.oracle import OracleConnector

        return OracleConnector()

    # ── _parse_dsn ────────────────────────────────────────────────────────────

    def test_parse_dsn_with_port(self):
        from aaizaql.connectors.oracle import OracleConnector

        conn = OracleConnector()
        user, pw, host, port, svc = conn._parse_dsn("oracle://hr:secret@db.host:1521/XEPDB1")
        assert user == "hr"
        assert pw == "secret"
        assert host == "db.host"
        assert port == "1521"
        assert svc == "XEPDB1"

    def test_parse_dsn_default_port(self):
        from aaizaql.connectors.oracle import OracleConnector

        conn = OracleConnector()
        _, _, _, port, _ = conn._parse_dsn("oracle://u:p@host/svc")
        assert port == "1521"

    def test_parse_dsn_invalid_raises(self):
        from aaizaql.connectors.oracle import OracleConnector

        conn = OracleConnector()
        with pytest.raises(ValueError, match="Cannot parse"):
            conn._parse_dsn("baddsn")

    # ── connect ───────────────────────────────────────────────────────────────

    def test_connect_calls_driver(self, _mock_oracledb):
        fake_oracledb, fake_conn = _mock_oracledb
        conn = self._make_connector()
        conn.connect("oracle://hr:pass@localhost:1521/XEPDB1")
        fake_oracledb.connect.assert_called_once_with(
            user="hr", password="pass", dsn="localhost:1521/XEPDB1"
        )

    def test_connect_error_raises_connection_error(self, _mock_oracledb):
        fake_oracledb, _ = _mock_oracledb
        fake_oracledb.connect.side_effect = Exception("ORA-01017")
        conn = self._make_connector()
        with pytest.raises(ConnectionError):
            conn.connect("oracle://u:p@host:1521/svc")

    def test_connect_missing_package_raises(self):
        from aaizaql.connectors.oracle import OracleConnector

        conn = OracleConnector()
        with (
            patch.dict("sys.modules", {"oracledb": None}),
            pytest.raises((ConnectionError, Exception)),
        ):
            conn.connect("oracle://u:p@host/svc")

    # ── execute ───────────────────────────────────────────────────────────────

    def test_execute_not_connected_raises(self):
        conn = self._make_connector()
        with pytest.raises(DatabaseError):
            conn.execute("SELECT 1 FROM DUAL")

    def test_execute_returns_dataframe(self, _mock_oracledb):
        fake_oracledb, fake_conn = _mock_oracledb
        expected = pd.DataFrame({"id": [10, 20]})
        with patch("pandas.read_sql_query", return_value=expected):
            conn = self._make_connector()
            conn._conn = fake_conn
            df = conn.execute("SELECT id FROM emp")
        assert list(df.columns) == ["id"]

    def test_execute_error_raises_database_error(self, _mock_oracledb):
        fake_oracledb, fake_conn = _mock_oracledb
        with patch("pandas.read_sql_query", side_effect=Exception("table not found")):
            conn = self._make_connector()
            conn._conn = fake_conn
            with pytest.raises(DatabaseError):
                conn.execute("SELECT * FROM nope")

    # ── get_schema ────────────────────────────────────────────────────────────

    def test_get_schema_not_connected_returns_empty(self):
        conn = self._make_connector()
        assert conn.get_schema() == ""

    def test_get_schema_returns_ddl(self, _mock_oracledb):
        fake_oracledb, fake_conn = _mock_oracledb
        df = pd.DataFrame({"DDL": ['CREATE TABLE "EMP" (id NUMBER NOT NULL);']})
        with patch("pandas.read_sql_query", return_value=df):
            conn = self._make_connector()
            conn._conn = fake_conn
            schema = conn.get_schema()
        assert "CREATE TABLE" in schema

    def test_get_schema_error_returns_empty(self, _mock_oracledb):
        fake_oracledb, fake_conn = _mock_oracledb
        with patch("pandas.read_sql_query", side_effect=Exception("ORA-00942")):
            conn = self._make_connector()
            conn._conn = fake_conn
            assert conn.get_schema() == ""

    # ── close ─────────────────────────────────────────────────────────────────

    def test_close_disconnects(self, _mock_oracledb):
        fake_oracledb, fake_conn = _mock_oracledb
        conn = self._make_connector()
        conn._conn = fake_conn
        conn.close()
        fake_conn.close.assert_called_once()
        assert conn._conn is None

    def test_close_noop_when_not_connected(self):
        conn = self._make_connector()
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MYSQL
# ═══════════════════════════════════════════════════════════════════════════════


class TestMySQLConnector:
    """MySQLConnector with mocked pymysql."""

    @pytest.fixture(autouse=True)
    def _mock_pymysql(self):
        fake_pymysql = MagicMock()
        fake_conn = MagicMock()
        fake_pymysql.connect.return_value = fake_conn
        with patch.dict(
            "sys.modules",
            {
                "pymysql": fake_pymysql,
                "pymysql.cursors": fake_pymysql.cursors,
            },
        ):
            yield fake_pymysql, fake_conn

    def _make_connector(self):
        from aaizaql.connectors.mysql import MySQLConnector

        return MySQLConnector()

    # ── _parse_dsn ────────────────────────────────────────────────────────────

    def test_parse_dsn_full(self):
        from aaizaql.connectors.mysql import MySQLConnector

        result = MySQLConnector._parse_dsn("mysql://root:secret@localhost:3306/mydb")
        assert result["user"] == "root"
        assert result["password"] == "secret"
        assert result["host"] == "localhost"
        assert result["port"] == 3306
        assert result["database"] == "mydb"

    def test_parse_dsn_default_port(self):
        from aaizaql.connectors.mysql import MySQLConnector

        result = MySQLConnector._parse_dsn("mysql://root:pass@localhost/mydb")
        assert result["port"] == 3306

    def test_parse_dsn_no_password(self):
        from aaizaql.connectors.mysql import MySQLConnector

        result = MySQLConnector._parse_dsn("mysql://root:@localhost/mydb")
        assert result["password"] == ""

    def test_parse_dsn_invalid_raises(self):
        from aaizaql.connectors.mysql import MySQLConnector

        with pytest.raises(ValueError, match="Cannot parse"):
            MySQLConnector._parse_dsn("notmysql://x")

    # ── connect ───────────────────────────────────────────────────────────────

    def test_connect_calls_pymysql(self, _mock_pymysql):
        fake_pymysql, fake_conn = _mock_pymysql
        conn = self._make_connector()
        conn.connect("mysql://root:pass@localhost:3306/mydb")
        fake_pymysql.connect.assert_called_once()

    def test_connect_error_raises_connection_error(self, _mock_pymysql):
        fake_pymysql, _ = _mock_pymysql
        fake_pymysql.connect.side_effect = Exception("access denied")
        conn = self._make_connector()
        with pytest.raises(ConnectionError):
            conn.connect("mysql://root:pass@localhost/mydb")

    # ── execute ───────────────────────────────────────────────────────────────

    def test_execute_not_connected_raises(self):
        conn = self._make_connector()
        with pytest.raises(DatabaseError):
            conn.execute("SELECT 1")

    def test_execute_returns_dataframe(self, _mock_pymysql):
        fake_pymysql, fake_conn = _mock_pymysql
        expected = pd.DataFrame({"name": ["Alice"]})
        with patch("pandas.read_sql_query", return_value=expected):
            conn = self._make_connector()
            conn._conn = fake_conn
            df = conn.execute("SELECT name FROM users")
        assert list(df.columns) == ["name"]

    def test_execute_error_raises_database_error(self, _mock_pymysql):
        fake_pymysql, fake_conn = _mock_pymysql
        with patch("pandas.read_sql_query", side_effect=Exception("unknown table")):
            conn = self._make_connector()
            conn._conn = fake_conn
            with pytest.raises(DatabaseError):
                conn.execute("SELECT * FROM nope")

    def test_execute_re_raises_database_error_unchanged(self, _mock_pymysql):
        """DatabaseError raised inside execute() should propagate as-is."""
        fake_pymysql, fake_conn = _mock_pymysql
        with patch(
            "pandas.read_sql_query",
            side_effect=DatabaseError("orig", sql="SELECT 1", connector="mysql"),
        ):
            conn = self._make_connector()
            conn._conn = fake_conn
            with pytest.raises(DatabaseError, match="orig"):
                conn.execute("SELECT 1")

    # ── get_schema ────────────────────────────────────────────────────────────

    def test_get_schema_not_connected_returns_empty(self):
        conn = self._make_connector()
        assert conn.get_schema() == ""

    def test_get_schema_returns_ddl(self, _mock_pymysql):
        fake_pymysql, fake_conn = _mock_pymysql
        df = pd.DataFrame({"ddl": ["CREATE TABLE `users` (id INT NOT NULL);"]})
        with patch("pandas.read_sql_query", return_value=df):
            conn = self._make_connector()
            conn._conn = fake_conn
            schema = conn.get_schema()
        assert "CREATE TABLE" in schema

    def test_get_schema_error_returns_empty(self, _mock_pymysql):
        fake_pymysql, fake_conn = _mock_pymysql
        with patch("pandas.read_sql_query", side_effect=Exception("error")):
            conn = self._make_connector()
            conn._conn = fake_conn
            assert conn.get_schema() == ""

    # ── close ─────────────────────────────────────────────────────────────────

    def test_close_disconnects(self, _mock_pymysql):
        fake_pymysql, fake_conn = _mock_pymysql
        conn = self._make_connector()
        conn._conn = fake_conn
        conn.close()
        fake_conn.close.assert_called_once()
        assert conn._conn is None

    def test_close_noop_when_not_connected(self):
        conn = self._make_connector()
        conn.close()
