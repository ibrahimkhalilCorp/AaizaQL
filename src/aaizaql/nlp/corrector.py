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
from aaizaql.nlp.utils import parse_sql_response

if TYPE_CHECKING:

    from aaizaql.schema.semantic_store import SemanticStore
    from aaizaql.security.validator import SQLValidator

logger = structlog.get_logger(__name__)


class SelfCorrector:
    """

    Executes SQL against a connector; on DatabaseError sends the error

    back to the LLM for correction and retries up to MAX_RETRIES times.

    The corrected SQL returned by the LLM is re-validated through

    ``SQLValidator`` before execution to prevent hallucinated dangerous

    statements (e.g. DROP TABLE) from slipping through on a correction pass.

    Parameters
    ----------
    semantic_store:
        Optional SemanticStore reference.  When provided, enum mappings and
        relevant documentation are appended to every correction prompt — the
        same enrichment the initial generator receives — so the LLM has full
        business context when it rewrites the query.
    dialect:
        Database dialect label (e.g. ``"sqlite"``, ``"postgresql"``).  Passed
        straight into ``SELF_CORRECTION_TEMPLATE`` so the LLM knows which SQL
        flavour to target.  Defaults to the empty string (backward-compatible).
    """

    def __init__(
        self,
        llm: LLMProvider,
        settings: Settings,
        validator: "SQLValidator | None" = None,
        vector_store: Any | None = None,  # T2.7 — for schema context on retry
        semantic_store: "SemanticStore | None" = None,  # fix: enum + doc context on retry
        dialect: str = "",  # fix: dialect label for correction prompt
    ) -> None:

        self._llm = llm

        self._max_retries = settings.max_self_correction_retries

        self._validator = validator

        self._vector_store = vector_store  # T2.7

        self._semantic_store = semantic_store  # fix: for enum/doc blocks in retry prompt

        self._dialect = dialect  # fix: e.g. "sqlite", "postgresql"

        self._schema_top_k = settings.schema_top_k

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

                # T2.7 — retrieve real schema context for the correction prompt

                schema_chunks = "(schema context unavailable)"

                if self._vector_store is not None:

                    try:

                        hits = self._vector_store.search(
                            query=question,
                            filter_type="ddl",
                            top_k=self._schema_top_k,
                        )

                        if hits:

                            schema_chunks = "\n\n".join(h.text for h in hits)

                    except Exception:

                        pass

                # fix: inject enum mappings — same data the generator always sends,
                # so the corrected query uses the right integer codes.
                enum_block = ""
                if self._semantic_store is not None and self._semantic_store.has_enums():
                    enum_block = (
                        "\n--- COLUMN CODE MAPPINGS ---\n"
                        + self._semantic_store.get_enum_block()
                        + "\n"
                    )

                # fix: inject relevant documentation retrieved for this question.
                doc_block = ""
                if self._semantic_store is not None:
                    try:
                        docs = self._semantic_store.search_documentation(question, top_k=3)
                        if docs:
                            doc_block = "\n--- BUSINESS CONTEXT ---\n" + docs + "\n"
                    except Exception:
                        pass

                # Ask LLM to fix the broken SQL

                correction_prompt = SELF_CORRECTION_TEMPLATE.format(
                    sql=self.last_sql,
                    error=str(exc),
                    schema_chunks=schema_chunks,
                    dialect=self._dialect or "unknown",
                    enum_block=enum_block,
                    doc_block=doc_block,
                )

                corrected = self._llm.complete(correction_prompt, timeout=self._timeout)

                self.last_sql = parse_sql_response(corrected)  # T1.3

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
