"""
aqlix — Natural Language to SQL
────────────────────────────────
Query any database in plain English.

Quick start:
    from aqlix import QueryEngine

    engine = QueryEngine(llm="claude", database="sqlite", dsn="sqlite:///my.db")
    engine.ingest_schema()
    result = engine.query("How many orders were placed last month?")
    print(result.sql)
    print(result.data)
    result.chart.show()
    print(result.summary)
"""

from aqlix.core.engine import QueryEngine, QueryResult
from aqlix.core.exceptions import (
    AQLIXError,
    SecurityException,
    SQLGenerationError,
    UnsupportedQueryError,
    MaxRetriesExceeded,
    DatabaseError,
    ConnectionError,
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
    "ConnectionError",
]
