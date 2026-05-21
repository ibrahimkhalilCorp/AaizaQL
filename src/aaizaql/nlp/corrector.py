"""
aaizaql.nlp.corrector
────────────────────
SelfCorrector: executes SQL and retries with LLM correction on failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import DatabaseError, MaxRetriesExceeded
from aaizaql.llm.base import LLMProvider
from aaizaql.nlp.prompts import SELF_CORRECTION_TEMPLATE

if TYPE_CHECKING:
    from aaizaql.security.validator import SQLValidator

logger = structlog.get_logger(__name__)


class SelfCorrector:
    """
    Executes SQL against a connector; on DatabaseError sends the error
    back to the LLM for correction and retries up to MAX_RETRIES times.

    The corrected SQL returned by the LLM is re-validated through
    ``SQLValidator`` before execution to prevent hallucinated dangerous
    statements (e.g. DROP TABLE) from slipping through on a correction pass.
    """

    def __init__(
        self,
        llm: LLMProvider,
        settings: Settings,
        validator: SQLValidator | None = None,
    ) -> None:
        self._llm = llm
        self._max_retries = settings.max_self_correction_retries
        self._validator = validator
        self._timeout = settings.llm_timeout_seconds
        self.last_sql: str = ""  # Updated to the final (possibly corrected) SQL

    def execute_with_correction(
        self,
        sql: str,
        executor: Any,  # DatabaseConnector — avoid circular import with Any
        question: str,
    ) -> tuple[pd.DataFrame, bool, int]:
        """
        Execute SQL with automatic self-correction on failure.

        Each LLM-corrected SQL string is validated through ``SQLValidator``
        before being executed, ensuring that a malicious or hallucinated
        correction (e.g. ``DROP TABLE``) is caught before it reaches the DB.

        Returns
        -------
        (DataFrame, was_corrected, attempts)
        """
        self.last_sql = sql
        was_corrected = False

        for attempt in range(self._max_retries + 1):
            try:
                data = executor.execute(self.last_sql)
                logger.info(
                    "corrector.success",
                    attempt=attempt,
                    rows=len(data),
                    was_corrected=was_corrected,
                )
                return data, was_corrected, attempt

            except DatabaseError as exc:
                if attempt >= self._max_retries:
                    raise MaxRetriesExceeded(
                        sql=self.last_sql,
                        last_error=str(exc),
                        attempts=attempt + 1,
                    ) from exc

                logger.warning(
                    "corrector.retrying",
                    attempt=attempt + 1,
                    error=str(exc)[:120],
                )

                # Ask LLM to fix the broken SQL
                correction_prompt = SELF_CORRECTION_TEMPLATE.format(
                    sql=self.last_sql,
                    error=str(exc),
                    schema_chunks="(see previously provided schema)",
                )
                corrected = self._llm.complete(correction_prompt, timeout=self._timeout)
                self.last_sql = corrected.strip().strip("`").strip()
                was_corrected = True

                # Re-validate the corrected SQL through the security layer
                # before executing it.  A hallucinated correction could contain
                # dangerous statements (DROP TABLE, DELETE, …) that must be
                # blocked regardless of context.
                if self._validator is not None:
                    self._validator.validate(self.last_sql)
                    logger.debug(
                        "corrector.revalidated",
                        attempt=attempt + 1,
                        sql_preview=self.last_sql[:60],
                    )

        # Should never reach here
        raise MaxRetriesExceeded(
            sql=self.last_sql, last_error="unknown", attempts=self._max_retries
        )
