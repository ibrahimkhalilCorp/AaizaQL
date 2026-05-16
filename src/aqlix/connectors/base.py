"""
aqlix.connectors.base
──────────────────────
Abstract base class for all database connectors.
Every adapter must implement connect(), execute(), get_schema(), and close().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DatabaseConnector(ABC):
    """Base class for all database adapters."""

    # Subclasses set this to their connector name string
    name: str = ""

    @abstractmethod
    def connect(self, dsn: str) -> None:
        """Establish a connection to the database."""
        ...

    @abstractmethod
    def execute(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL SELECT statement and return results as a DataFrame.

        Parameters
        ----------
        sql : str  A validated SQL SELECT statement.

        Returns
        -------
        pd.DataFrame  Query results. Empty DataFrame if no rows returned.

        Raises
        ------
        aqlix.core.exceptions.DatabaseError  On execution failure.
        """
        ...

    @abstractmethod
    def get_schema(self) -> str:
        """
        Return the full DDL schema of all tables in the connected database.

        Returns
        -------
        str  DDL string (CREATE TABLE statements).
        """
        ...

    def close(self) -> None:
        """Close the database connection. Override if cleanup is needed."""

    def test_connection(self) -> bool:
        """
        Return True if the connection is alive. Used in the UI connection wizard.
        Default implementation runs a trivial query.
        """
        try:
            self.execute("SELECT 1")
            return True
        except Exception:
            return False
