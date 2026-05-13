"""
aqlix.connectors
─────────────────
Connector registry — maps name strings to connector classes.
Add a new database by subclassing DatabaseConnector and registering it here.
"""

from __future__ import annotations

from aqlix.connectors.base import DatabaseConnector
from aqlix.connectors.sqlite import SQLiteConnector
from aqlix.connectors.postgres import PostgreSQLConnector
from aqlix.core.exceptions import ConnectorNotFound as _ConnectorNotFound

# Central registry — name → class (not instance)
REGISTRY: dict[str, type[DatabaseConnector]] = {
    "sqlite": SQLiteConnector,
    "postgresql": PostgreSQLConnector,
    "postgres": PostgreSQLConnector,   # alias
}

# Lazy-register heavier connectors only when available
try:
    from aqlix.connectors.mysql import MySQLConnector
    REGISTRY["mysql"] = MySQLConnector
except ImportError:
    pass

try:
    from aqlix.connectors.snowflake import SnowflakeConnector
    REGISTRY["snowflake"] = SnowflakeConnector
except ImportError:
    pass

try:
    from aqlix.connectors.duckdb import DuckDBConnector
    REGISTRY["duckdb"] = DuckDBConnector
except ImportError:
    pass


def get_connector(name: str) -> DatabaseConnector:
    """Instantiate and return a connector by name."""
    name = name.lower()
    if name not in REGISTRY:
        raise _ConnectorNotFound(name)
    return REGISTRY[name]()


__all__ = ["DatabaseConnector", "REGISTRY", "get_connector"]
