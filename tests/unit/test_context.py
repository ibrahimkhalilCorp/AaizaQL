"""
tests/unit/test_context.py
───────────────────────────
Unit tests for the session context manager.
"""

import pytest

from aqlix.memory.context import ContextManager


@pytest.fixture
def ctx() -> ContextManager:
    return ContextManager(limit=3)


def test_empty_session_returns_empty_list(ctx):
    assert ctx.get_history("sess_1") == []


def test_add_and_retrieve_turn(ctx):
    ctx.add_turn("s1", question="How many users?", sql="SELECT COUNT(*) FROM users", row_count=1)
    history = ctx.get_history("s1")
    assert len(history) == 1
    assert history[0]["question"] == "How many users?"
    assert history[0]["sql"] == "SELECT COUNT(*) FROM users"


def test_limit_is_respected(ctx):
    for i in range(5):
        ctx.add_turn("s1", question=f"Q{i}", sql=f"SELECT {i}", row_count=i)
    history = ctx.get_history("s1")
    assert len(history) == 3  # limit=3
    assert history[0]["question"] == "Q2"  # oldest in window


def test_sessions_are_isolated(ctx):
    ctx.add_turn("sess_a", question="A", sql="SELECT 'a'")
    ctx.add_turn("sess_b", question="B", sql="SELECT 'b'")
    assert len(ctx.get_history("sess_a")) == 1
    assert len(ctx.get_history("sess_b")) == 1
    assert ctx.get_history("sess_a")[0]["question"] == "A"


def test_clear_removes_session(ctx):
    ctx.add_turn("s1", question="Q", sql="SELECT 1")
    ctx.clear("s1")
    assert ctx.get_history("s1") == []


def test_session_count(ctx):
    ctx.add_turn("s1", question="Q1", sql="SELECT 1")
    ctx.add_turn("s2", question="Q2", sql="SELECT 2")
    assert ctx.session_count() == 2
