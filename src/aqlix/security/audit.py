"""
aqlix.security.audit
─────────────────────
AuditLogger: structured, append-only log of every query and security event.
Writes to a JSONL file (one JSON object per line) when audit_log_file is set.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class AuditLogger:
    """
    Append-only structured audit log.

    Each record is a JSON object written to the configured file.
    When no file is configured the events are logged at DEBUG level only.
    """

    def __init__(self, log_file: str | None = None) -> None:
        self._path: Path | None = Path(log_file) if log_file else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("audit.logger_ready", path=str(self._path))

    def log_query(
        self,
        user_id: str,
        session_id: str,
        question: str,
        sql: str,
        rows: int,
        execution_ms: int,
        was_corrected: bool = False,
    ) -> None:
        self._write(
            event="query",
            user_id=user_id,
            session_id=session_id,
            question=question,
            sql=sql,
            rows=rows,
            execution_ms=execution_ms,
            was_corrected=was_corrected,
        )

    def log_security_event(
        self,
        event_type: str,
        detail: str,
        user_id: str = "",
        sql: str = "",
    ) -> None:
        self._write(
            event=f"security.{event_type}",
            user_id=user_id,
            detail=detail,
            sql=sql,
        )

    def log_correction(
        self,
        original_sql: str,
        error: str,
        corrected_sql: str,
    ) -> None:
        self._write(
            event="self_correction",
            original_sql=original_sql,
            error=error,
            corrected_sql=corrected_sql,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _write(self, **fields: Any) -> None:
        record: dict[str, Any] = {"ts": time.time(), **fields}
        logger.debug("audit.event", **fields)
        if self._path:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
            except OSError as exc:
                logger.warning("audit.write_failed", detail=str(exc))
