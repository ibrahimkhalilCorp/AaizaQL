"""
AAIZAQL.nlp.corrector
────────────────────
SelfCorrector: executes SQL and retries with LLM correction on failure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from AAIZAQL.core.config import Settings
from AAIZAQL.core.exceptions import DatabaseError, MaxRetriesExceeded
from AAIZAQL.llm.base import LLMProvider
from AAIZAQL.nlp.prompts import SELF_CORRECTION_TEMPLATE

logger = structlog.get_logger(__name__)


class SelfCorrector:
    """
    Executes SQL against a connector; on DatabaseError sends the error
    back to the LLM for correction and retries up to MAX_RETRIES times.
    """

    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self._llm = llm
        self._max_retries = settings.max_self_correction_retries
        self.last_sql: str = ""  # Updated to the final (possibly corrected) SQL

    def execute_with_correction(
        self,
        sql: str,
        executor: Any,  # DatabaseConnector — avoid circular import with Any
        question: str,
    ) -> tuple[pd.DataFrame, bool, int]:
        """
        Execute SQL with automatic self-correction on failure.

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
                corrected = self._llm.complete(correction_prompt)
                self.last_sql = corrected.strip().strip("`").strip()
                was_corrected = True

        # Should never reach here
        raise MaxRetriesExceeded(
            sql=self.last_sql, last_error="unknown", attempts=self._max_retries
        )
