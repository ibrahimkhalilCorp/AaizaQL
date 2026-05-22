"""
tests/conftest.py
─────────────────
Shared pytest fixtures.  All tests that need a real SQLite database,
a mock LLM, or a vector store can import these via dependency injection.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aaizaql.connectors.sqlite import SQLiteConnector
from aaizaql.core.config import Settings
from aaizaql.memory.vector_store import VectorStoreAdapter

# ── Settings ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings wired to a temporary ChromaDB directory."""
    return Settings(
        llm_provider="groq",
        vector_store="chroma",
        chroma_persist_dir=str(tmp_path / "chroma"),
        vector_store_namespace="test",
        max_self_correction_retries=1,
        enable_injection_detection=True,
    )


# ── SQLite database ───────────────────────────────────────────────────────────


@pytest.fixture()
def sqlite_db(tmp_path: Path) -> Generator[str, None, None]:
    """
    Create a small in-memory-equivalent SQLite database for testing.
    Yields the DSN string.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE employees (
            id      INTEGER PRIMARY KEY,
            name    TEXT    NOT NULL,
            dept    TEXT    NOT NULL,
            salary  REAL    NOT NULL,
            status  INTEGER NOT NULL DEFAULT 1
        );
        INSERT INTO employees VALUES
            (1, 'Alice',   'Engineering', 95000, 1),
            (2, 'Bob',     'Marketing',   72000, 2),
            (3, 'Charlie', 'Engineering', 88000, 1),
            (4, 'Diana',   'HR',          65000, 3);

        CREATE TABLE departments (
            id   INTEGER PRIMARY KEY,
            name TEXT    NOT NULL,
            head TEXT
        );
        INSERT INTO departments VALUES
            (1, 'Engineering', 'Alice'),
            (2, 'Marketing',   'Bob'),
            (3, 'HR',          'Diana');
    """)
    conn.commit()
    conn.close()
    yield f"sqlite:///{db_path}"


@pytest.fixture()
def sqlite_connector(sqlite_db: str) -> Generator[SQLiteConnector, None, None]:
    """A connected SQLiteConnector pointing at the test database."""
    conn = SQLiteConnector()
    conn.connect(sqlite_db)
    yield conn
    conn.close()


# ── Mock LLM ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_llm() -> MagicMock:
    """
    A mock LLMProvider that returns valid SQL by default.
    Override mock_llm.complete.return_value in individual tests.
    """
    llm = MagicMock()
    llm.name = "mock/test"
    llm.complete.return_value = "SELECT * FROM employees"
    return llm


# ── Vector store ──────────────────────────────────────────────────────────────


@pytest.fixture()
def vector_store(settings: Settings) -> VectorStoreAdapter:
    """A real ChromaDB-backed vector store in a temp directory."""
    try:
        return VectorStoreAdapter(settings)
    except Exception:
        pytest.skip("ChromaDB not available in this environment")


def pytest_collection_modifyitems(config, items):
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return  # keys present — run everything
    skip = pytest.mark.skip(reason="No API key set — skipping LLM tests")
    for item in items:
        if item.get_closest_marker("requires_api_key"):
            item.add_marker(skip)
