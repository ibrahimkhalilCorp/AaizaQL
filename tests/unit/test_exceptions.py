"""
tests/unit/test_exceptions.py
──────────────────────────────
Verify all custom exceptions carry the right attributes and messages.
"""

# import pytest
from aaizaql.core.exceptions import (
    ConnectionError,
    DatabaseError,
    FederationError,
    LLMError,
    MaxRetriesExceeded,
    PromptInjectionDetected,
    SecurityException,
    SQLGenerationError,
    UnsupportedQueryError,
)


def test_security_exception_carries_sql():
    exc = SecurityException(reason="DROP not allowed", sql="DROP TABLE x")
    assert "DROP not allowed" in str(exc)
    assert exc.sql == "DROP TABLE x"
    assert exc.reason == "DROP not allowed"

def test_prompt_injection_is_security_exception():
    exc = PromptInjectionDetected(reason="pattern matched")
    assert isinstance(exc, SecurityException)

def test_sql_generation_error():
    exc = SQLGenerationError(question="How many orders?", detail="LLM timed out")
    assert "How many orders?" in str(exc)
    assert "LLM timed out" in str(exc)

def test_unsupported_query_error():
    exc = UnsupportedQueryError(question="Delete all users")
    assert "Delete all users" in str(exc)
    assert exc.question == "Delete all users"

def test_max_retries_exceeded():
    exc = MaxRetriesExceeded(sql="SELECT bad", last_error="column not found", attempts=3)
    assert "3" in str(exc)
    assert exc.sql == "SELECT bad"
    assert exc.attempts == 3

def test_database_error_with_connector():
    exc = DatabaseError("syntax error", sql="SELECT ?", connector="sqlite")
    assert "sqlite" in str(exc)
    assert "SELECT ?" in str(exc)

def test_connection_error():
    exc = ConnectionError("postgresql", dsn_hint="postgresql://...", detail="refused")
    assert "postgresql" in str(exc)
    assert "refused" in str(exc)

def test_llm_error():
    exc = LLMError("claude", detail="rate limited")
    assert "claude" in str(exc)
    assert "rate limited" in str(exc)

def test_federation_error_with_sources():
    exc = FederationError("join failed", sources=["snowflake", "postgres"])
    assert "snowflake" in str(exc)
    assert "postgres" in str(exc)
