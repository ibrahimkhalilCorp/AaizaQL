"""
aaizaql.connectors
─────────────────
Connector registry — maps name strings to connector classes.
Add a new database by subclassing DatabaseConnector and registering it here.
"""

from __future__ import annotations

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.connectors.postgres import PostgreSQLConnector
from aaizaql.connectors.sqlite import SQLiteConnector
from aaizaql.core.exceptions import ConnectorNotFound as _ConnectorNotFound

# Central registry — name → class (not instance)
REGISTRY: dict[str, type[DatabaseConnector]] = {
    "sqlite": SQLiteConnector,
    "postgresql": PostgreSQLConnector,
    "postgres": PostgreSQLConnector,  # alias
}

# Lazy-register heavier connectors only when available
try:
    from aaizaql.connectors.mysql import MySQLConnector

    REGISTRY["mysql"] = MySQLConnector
except ImportError:
    pass

try:
    from aaizaql.connectors.snowflake import SnowflakeConnector

    REGISTRY["snowflake"] = SnowflakeConnector
except ImportError:
    pass

try:
    from aaizaql.connectors.duckdb import DuckDBConnector

    REGISTRY["duckdb"] = DuckDBConnector
except ImportError:
    pass

try:
    from aaizaql.connectors.mssql import MSSQLConnector

    REGISTRY["mssql"] = MSSQLConnector
    REGISTRY["sqlserver"] = MSSQLConnector  # alias
except ImportError:
    pass

try:
    from aaizaql.connectors.oracle import OracleConnector

    REGISTRY["oracle"] = OracleConnector
except ImportError:
    pass

try:
    from aaizaql.connectors.mongodb import MongoDBConnector

    REGISTRY["mongodb"] = MongoDBConnector
    REGISTRY["mongo"] = MongoDBConnector  # alias
except ImportError:
    pass

try:
    from aaizaql.connectors.bigquery import BigQueryConnector

    REGISTRY["bigquery"] = BigQueryConnector
except ImportError:
    pass


def get_connector(name: str) -> DatabaseConnector:
    """Instantiate and return a connector by name."""
    name = name.lower()
    if name not in REGISTRY:
        raise _ConnectorNotFound(name, available=sorted(REGISTRY.keys()))
    return REGISTRY[name]()


__all__ = ["DatabaseConnector", "REGISTRY", "get_connector"]
