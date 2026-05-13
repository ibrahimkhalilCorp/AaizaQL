"""
aqlix.memory.context
─────────────────────
ContextManager: short-term session memory for multi-turn conversations.
Stores the last N Q&A pairs per session_id in memory.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TypedDict


class Turn(TypedDict):
    question: str
    sql: str
    row_count: int


class ContextManager:
    """
    In-process session memory.

    Each session_id has an independent conversation history.
    History is capped at `limit` turns to control prompt length.
    """

    def __init__(self, limit: int = 10) -> None:
        self._limit = limit
        self._sessions: dict[str, deque[Turn]] = defaultdict(lambda: deque(maxlen=self._limit))

    def add_turn(self, session_id: str, question: str, sql: str, row_count: int = 0) -> None:
        """Append a completed Q&A turn to the session history."""
        self._sessions[session_id].append(Turn(question=question, sql=sql, row_count=row_count))

    def get_history(self, session_id: str) -> list[Turn]:
        """Return the current history for a session (oldest first)."""
        return list(self._sessions[session_id])

    def clear(self, session_id: str) -> None:
        """Wipe all history for a session."""
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        """Number of active sessions currently tracked."""
        return len(self._sessions)
