"""
aqlix — Natural Language to SQL library.

Quick start::

    from aqlix import QueryEngine

    engine = QueryEngine(llm="groq", database="sqlite", dsn="sqlite:///my.db")
    engine.ingest_schema()
    result = engine.query("Show all users who signed up last month")
    print(result.sql)
    print(result.data)
"""

from __future__ import annotations

from aqlix.core.engine import QueryEngine, QueryResult
from aqlix.core.exceptions import (
    AQLIXError,
    SecurityException,
    SQLGenerationError,
    UnsupportedQueryError,
    MaxRetriesExceeded,
    DatabaseError,
    ConnectorNotFound,
    LLMError,
)

__version__ = "0.1.0"

__all__ = [
    "QueryEngine",
    "QueryResult",
    "AQLIXError",
    "SecurityException",
    "SQLGenerationError",
    "UnsupportedQueryError",
    "MaxRetriesExceeded",
    "DatabaseError",
    "ConnectorNotFound",
    "LLMError",
    "__version__",
]
