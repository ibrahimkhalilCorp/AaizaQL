"""
tests/test_engine.py
─────────────────────
Integration tests for QueryEngine using a mock LLM and a real SQLite DB.
These tests verify the full pipeline: NL → SQL → execute → result.
No API keys required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aaizaql import QueryEngine, QueryResult
from aaizaql.core.exceptions import SecurityException, UnsupportedQueryError


def _make_engine(sqlite_db: str, mock_llm: MagicMock, tmp_path: Path) -> QueryEngine:
    """Helper: build a QueryEngine with a mock LLM and temp vector store."""
    with (
        patch("aaizaql.core.engine.build_llm_provider", return_value=mock_llm),
        patch(
            "aaizaql.core.engine.VectorStoreAdapter",
            side_effect=lambda s: _make_mock_vs(),
        ),
    ):
        engine = QueryEngine(
            llm="groq",
            database="sqlite",
            dsn=sqlite_db,
        )
    return engine

def _make_mock_vs() -> MagicMock:
    """Mock vector store that returns empty search results."""
    vs = MagicMock()
    vs.search.return_value = []
    vs.upsert.return_value = None
    vs.count.return_value = 0
    return vs

class TestQueryEngine:
    def test_simple_query_returns_dataframe(
        self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path
    ) -> None:
        mock_llm.complete.return_value = "SELECT * FROM employees"
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)

        result = engine.query("Show me all employees")

        assert isinstance(result, QueryResult)
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 4
        assert result.sql == "SELECT * FROM employees"

    def test_query_result_fields(self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path) -> None:
        mock_llm.complete.return_value = "SELECT COUNT(*) as total FROM employees"
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)

        result = engine.query("How many employees?")

        assert result.question == "How many employees?"
        assert isinstance(result.execution_time_ms, int)
        assert result.execution_time_ms >= 0
        assert result.session_id != ""

    def test_session_memory_across_turns(
        self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path
    ) -> None:
        mock_llm.complete.return_value = "SELECT * FROM employees"
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)

        engine.query("Show all employees", session_id="test-session")
        engine.query("Filter by Engineering", session_id="test-session")

        # Second call should have included history in the prompt
        second_call_prompt = mock_llm.complete.call_args_list[1][0][0]
        assert "Show all employees" in second_call_prompt

    def test_unsupported_query_raises(
        self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path
    ) -> None:
        mock_llm.complete.return_value = "UNSUPPORTED"
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)

        with pytest.raises(UnsupportedQueryError):
            engine.query("Delete all employees")

    def test_security_blocks_dangerous_sql(
        self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path
    ) -> None:
        mock_llm.complete.return_value = "DROP TABLE employees"
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)

        with pytest.raises(SecurityException):
            engine.query("Destroy the database")

    def test_train_documentation(self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path) -> None:
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)
        # Should not raise
        engine.train(documentation="employees.status: 1=Active, 2=Resigned")

    def test_define_enum(self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path) -> None:
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)
        engine.define_enum("employees", "status", {1: "Active", 2: "Resigned"})
        info = engine.training_info()
        assert info["enum_count"] == 1
        assert "employees.status" in info["enums"]

    def test_train_sql_pair(self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path) -> None:
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)
        engine.train(
            question="How many employees?",
            sql="SELECT COUNT(*) FROM employees",
        )

    def test_context_manager(self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path) -> None:
        mock_llm.complete.return_value = "SELECT 1"
        with _make_engine(sqlite_db, mock_llm, tmp_path) as engine:
            assert engine is not None

    def test_reset_session(self, sqlite_db: str, mock_llm: MagicMock, tmp_path: Path) -> None:
        mock_llm.complete.return_value = "SELECT * FROM employees"
        engine = _make_engine(sqlite_db, mock_llm, tmp_path)
        engine.query("Q1", session_id="s1")
        engine.reset_session("s1")
        # After reset, history is empty — second call should not include Q1
        engine.query("Q2", session_id="s1")
        second_prompt = mock_llm.complete.call_args_list[-1][0][0]
        assert "Q1" not in second_prompt
