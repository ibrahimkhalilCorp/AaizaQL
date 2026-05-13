"""
aqlix.security.validator
─────────────────────────
SQLValidator: runs every generated SQL through three safety checks before execution.

1. Whitelist check  — only SELECT / WITH allowed
2. Injection scan   — detect prompt injection patterns in the SQL text
3. Dialect parse    — use sqlglot to confirm syntactic validity
"""

from __future__ import annotations

import re

import sqlglot
import sqlglot.errors

from aqlix.core.config import Settings
from aqlix.core.exceptions import PromptInjectionDetected, SecurityException
import structlog

logger = structlog.get_logger(__name__)

# Dangerous statement types — any of these in the SQL → hard reject
_BLOCKED_STATEMENT_TYPES = {
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
    "ALTER", "CREATE", "REPLACE", "EXEC", "EXECUTE",
    "GRANT", "REVOKE", "CALL", "LOAD", "IMPORT",
}

# Prompt injection signatures commonly injected into user questions
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(your\s+)?(prior|previous|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"new\s+persona", re.IGNORECASE),
    re.compile(r"(drop|delete|truncate)\s+table", re.IGNORECASE),
    re.compile(r";\s*(drop|delete|truncate|insert|update)", re.IGNORECASE),
    re.compile(r"--\s*bypass", re.IGNORECASE),
    re.compile(r"\/\*.*?\*\/", re.DOTALL),           # block comments
    re.compile(r"xp_cmdshell", re.IGNORECASE),        # MSSQL command execution
    re.compile(r"INTO\s+OUTFILE", re.IGNORECASE),     # MySQL file export
]


class SQLValidator:
    """
    Validates generated SQL before execution.
    Raises SecurityException (or PromptInjectionDetected) on any violation.
    """

    def __init__(self, settings: Settings) -> None:
        self._allowed_ops = set(settings.allowed_sql_operations)
        self._injection_detection = settings.enable_injection_detection

    def validate(self, sql: str) -> None:
        """
        Run all validation checks on the given SQL string.

        Parameters
        ----------
        sql : str  The raw SQL string produced by the LLM.

        Raises
        ------
        SecurityException         On whitelist violation or parse failure.
        PromptInjectionDetected   On injection pattern detection.
        """
        self._check_injection(sql)
        self._check_whitelist(sql)
        self._check_parse(sql)
        logger.info("validator.passed", sql_preview=sql[:60])

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_injection(self, sql: str) -> None:
        if not self._injection_detection:
            return
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(sql):
                logger.warning("validator.injection_detected", pattern=pattern.pattern[:40])
                raise PromptInjectionDetected(
                    reason=f"Injection pattern matched: {pattern.pattern[:60]}",
                    sql=sql,
                )

    def _check_whitelist(self, sql: str) -> None:
        """
        Parse the SQL with sqlglot and check that the root statement type is
        in the allowed whitelist.
        """
        try:
            statements = sqlglot.parse(sql)
        except sqlglot.errors.ParseError:
            # Let _check_parse handle the error message
            return

        for stmt in statements:
            if stmt is None:
                continue
            stmt_type = type(stmt).__name__.upper()

            # Map sqlglot class names to our keyword set
            # e.g. Select → SELECT, Drop → DROP
            keyword = stmt_type.replace("STATEMENT", "").strip()

            if keyword not in self._allowed_ops:
                # Also check the string representation
                sql_upper = sql.strip().upper()
                for blocked in _BLOCKED_STATEMENT_TYPES:
                    if sql_upper.startswith(blocked):
                        raise SecurityException(
                            reason=f"Statement type '{blocked}' is not allowed. "
                                   f"Only {sorted(self._allowed_ops)} are permitted.",
                            sql=sql,
                        )

    def _check_parse(self, sql: str) -> None:
        """Attempt to parse the SQL with sqlglot — catch hard syntax errors."""
        try:
            result = sqlglot.parse(sql)
            if not result or all(s is None for s in result):
                raise SecurityException(
                    reason="SQL could not be parsed — it may be empty or malformed.",
                    sql=sql,
                )
        except sqlglot.errors.ParseError as exc:
            # Warn but don't block — the DB will surface the real error
            logger.warning("validator.parse_warning", detail=str(exc)[:120])
