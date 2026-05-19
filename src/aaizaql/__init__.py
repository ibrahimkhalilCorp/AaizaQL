"""
AAIZAQL — Natural Language to SQL library.

Quick start::

    from aaizaql import QueryEngine

    engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///my.db")
    engine.ingest_schema()
    result = engine.query("Show all users who signed up last month")
    print(result.sql)
    print(result.data)
"""

from __future__ import annotations

from aaizaql.core.engine import QueryEngine, QueryResult
from aaizaql.core.exceptions import (
    AAIZAQLError,
    ConnectorNotFound,
    DatabaseError,
    LLMError,
    MaxRetriesExceeded,
    SecurityException,
    SQLGenerationError,
    UnsupportedQueryError,
)

__version__ = "0.1.1"

__all__ = [
    "QueryEngine",
    "QueryResult",
    "AAIZAQLError",
    "SecurityException",
    "SQLGenerationError",
    "UnsupportedQueryError",
    "MaxRetriesExceeded",
    "DatabaseError",
    "ConnectorNotFound",
    "LLMError",
    "__version__",
]
