"""
aaizaql.memory.context
──────────────────────
ContextManager: short-term session memory for multi-turn conversations.
Stores the last N Q&A pairs per session_id in memory.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

from collections import defaultdict, deque
from typing import TypedDict


class Turn(TypedDict):
    question: str
    sql: str
    row_count: int


class ContextManager:
    """In-process session memory.

    Each session_id has an independent conversation history.
    History is capped at ``limit`` turns to control prompt length.

    Args:
        limit: Maximum number of turns to keep per session.
    """

    def __init__(self, limit: int = 10) -> None:
        self._limit = limit
        self._sessions: dict[str, deque[Turn]] = defaultdict(lambda: deque(maxlen=self._limit))

    def add_turn(self, session_id: str, question: str, sql: str, row_count: int = 0) -> None:
        """Append a completed Q&A turn to the session history.

        Args:
            session_id: Caller-supplied session identifier.
            question: Natural language question asked by the user.
            sql: SQL that was generated and executed.
            row_count: Number of rows returned by the query.
        """
        self._sessions[session_id].append(Turn(question=question, sql=sql, row_count=row_count))

    def get_history(self, session_id: str) -> list[Turn]:
        """Return the current history for a session (oldest first).

        Args:
            session_id: Session identifier.

        Returns:
            List of :class:`Turn` dicts ordered from oldest to newest.
            Returns an empty list if the session has no history.
        """
        return list(self._sessions[session_id])

    def clear(self, session_id: str) -> None:
        """Wipe all history for a session.

        Args:
            session_id: Session identifier to clear. No-op if the session
                does not exist.
        """
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        """Return the number of active sessions currently tracked.

        Returns:
            Count of sessions that have at least one recorded turn.
        """
        return len(self._sessions)
