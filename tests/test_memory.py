"""
tests/test_memory.py
─────────────────────
Unit tests for ContextManager (short-term session memory).
"""

from __future__ import annotations

from aaizaql.memory.context import ContextManager


class TestContextManager:
    def test_empty_session_returns_empty_list(self) -> None:
        ctx = ContextManager()
        assert ctx.get_history("session-1") == []

    def test_add_and_retrieve_turn(self) -> None:
        ctx = ContextManager()
        ctx.add_turn(
            "s1", question="How many users?", sql="SELECT COUNT(*) FROM users", row_count=1
        )
        history = ctx.get_history("s1")
        assert len(history) == 1
        assert history[0]["question"] == "How many users?"
        assert history[0]["sql"] == "SELECT COUNT(*) FROM users"
        assert history[0]["row_count"] == 1

    def test_multiple_turns(self) -> None:
        ctx = ContextManager()
        for i in range(5):
            ctx.add_turn("s1", question=f"Q{i}", sql=f"SELECT {i}", row_count=i)
        history = ctx.get_history("s1")
        assert len(history) == 5
        assert history[0]["question"] == "Q0"
        assert history[4]["question"] == "Q4"

    def test_history_limit_enforced(self) -> None:
        ctx = ContextManager(limit=3)
        for i in range(10):
            ctx.add_turn("s1", question=f"Q{i}", sql=f"SELECT {i}", row_count=0)
        history = ctx.get_history("s1")
        assert len(history) == 3
        # Oldest items evicted — only last 3 remain
        assert history[0]["question"] == "Q7"
        assert history[2]["question"] == "Q9"

    def test_sessions_are_independent(self) -> None:
        ctx = ContextManager()
        ctx.add_turn("s1", question="Q from s1", sql="SELECT 1", row_count=0)
        ctx.add_turn("s2", question="Q from s2", sql="SELECT 2", row_count=0)
        assert len(ctx.get_history("s1")) == 1
        assert len(ctx.get_history("s2")) == 1
        assert ctx.get_history("s1")[0]["question"] == "Q from s1"

    def test_clear_session(self) -> None:
        ctx = ContextManager()
        ctx.add_turn("s1", question="Q", sql="SELECT 1", row_count=0)
        ctx.clear("s1")
        assert ctx.get_history("s1") == []

    def test_clear_nonexistent_session_no_error(self) -> None:
        ctx = ContextManager()
        ctx.clear("nonexistent")  # should not raise

    def test_session_count(self) -> None:
        ctx = ContextManager()
        ctx.add_turn("s1", question="Q", sql="SELECT 1", row_count=0)
        ctx.add_turn("s2", question="Q", sql="SELECT 2", row_count=0)
        assert ctx.session_count() == 2
