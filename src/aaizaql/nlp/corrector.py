"""
aaizaql.nlp.corrector
─────────────────────
SelfCorrector: executes SQL and retries with LLM-guided correction on failure.

On each ``DatabaseError`` the corrector:

1. Retrieves fresh schema context from the vector store.
2. Appends enum mappings and business documentation (same enrichment the
   initial generator uses) so the LLM has full context when rewriting.
3. Sends the broken SQL + error message to the LLM via
   ``SELF_CORRECTION_TEMPLATE``.
4. Re-validates the corrected SQL through ``SQLValidator`` before executing —
   a hallucinated correction (e.g. ``DROP TABLE``) is blocked before it
   reaches the database.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

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
    """Executes SQL with automatic LLM-guided self-correction on failure.

    Args:
        llm: LLM provider used to generate corrected SQL.
        settings: Library-wide settings (controls ``max_self_correction_retries``,
            ``schema_top_k``, and ``llm_timeout_seconds``).
        validator: Optional ``SQLValidator`` instance. When provided, every
            LLM-corrected SQL is re-validated before execution to prevent
            hallucinated dangerous statements from slipping through.
        vector_store: Optional vector store adapter. When provided, fresh DDL
            schema chunks are retrieved for each correction prompt.
        semantic_store: Optional SemanticStore. When provided, enum mappings
            and relevant documentation are appended to each correction prompt —
            the same enrichment the initial generator receives.
        dialect: SQL dialect label (e.g. ``"sqlite"``, ``"postgresql"``) passed
            into the correction prompt template.
    """

    def __init__(
        self,
        llm: LLMProvider,
        settings: Settings,
        validator: "SQLValidator | None" = None,
        vector_store: Any | None = None,
        semantic_store: "SemanticStore | None" = None,
        dialect: str = "",
    ) -> None:
        self._llm = llm
        self._max_retries = settings.max_self_correction_retries
        self._validator = validator
        self._vector_store = vector_store
        self._semantic_store = semantic_store
        self._dialect = dialect
        self._schema_top_k = settings.schema_top_k
        self._timeout = settings.llm_timeout_seconds
        self.last_sql: str = ""

    def execute_with_correction(
        self,
        sql: str,
        executor: Any,
        question: str,
    ) -> tuple[pd.DataFrame, bool, int]:
        """Execute SQL with automatic self-correction on ``DatabaseError``.

        Args:
            sql: Initial SQL statement to execute.
            executor: DatabaseConnector instance with an ``execute(sql)`` method.
            question: Original user question — used to retrieve schema context
                for correction prompts.

        Returns:
            Tuple of ``(data, was_corrected, attempts)`` where ``data`` is the
            result DataFrame, ``was_corrected`` is ``True`` if at least one
            retry occurred, and ``attempts`` is the number of retries made.

        Raises:
            MaxRetriesExceeded: When all correction attempts are exhausted.
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

                schema_chunks = self._retrieve_schema_context(question)
                enum_block = self._build_enum_block()
                doc_block = self._build_doc_block(question)

                correction_prompt = SELF_CORRECTION_TEMPLATE.format(
                    sql=self.last_sql,
                    error=str(exc),
                    schema_chunks=schema_chunks,
                    dialect=self._dialect or "unknown",
                    enum_block=enum_block,
                    doc_block=doc_block,
                )

                corrected = self._llm.complete(correction_prompt, timeout=self._timeout)
                self.last_sql = parse_sql_response(corrected)
                was_corrected = True

                if self._validator is not None:
                    self._validator.validate(self.last_sql)
                    logger.debug(
                        "corrector.revalidated",
                        attempt=attempt + 1,
                        sql_preview=self.last_sql[:60],
                    )

        # Unreachable: the loop always either returns or raises MaxRetriesExceeded.
        raise MaxRetriesExceeded(
            sql=self.last_sql, last_error="unknown", attempts=self._max_retries
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _retrieve_schema_context(self, question: str) -> str:
        """Retrieve DDL schema chunks relevant to *question* from the vector store.

        Args:
            question: User's question used as the search query.

        Returns:
            Newline-separated DDL string, or a fallback message when the
            vector store is unavailable or returns no results.
        """
        if self._vector_store is None:
            return "(schema context unavailable)"
        try:
            hits = self._vector_store.search(
                query=question,
                filter_type="ddl",
                top_k=self._schema_top_k,
            )
            if hits:
                return "\n\n".join(h.text for h in hits)
        except Exception:
            pass
        return "(schema context unavailable)"

    def _build_enum_block(self) -> str:
        """Return formatted enum mappings for the correction prompt.

        Enum mappings are injected on every correction attempt — not retrieved
        via RAG — so integer codes are never lost between retries.

        Returns:
            Formatted enum block string, or an empty string if no enums exist.
        """
        if self._semantic_store is None or not self._semantic_store.has_enums():
            return ""
        return "\n--- COLUMN CODE MAPPINGS ---\n" + self._semantic_store.get_enum_block() + "\n"

    def _build_doc_block(self, question: str) -> str:
        """Retrieve relevant business documentation for the correction prompt.

        Args:
            question: User's question used as the semantic search query.

        Returns:
            Formatted documentation block string, or an empty string if no
            docs are found or the SemanticStore is not set.
        """
        if self._semantic_store is None:
            return ""
        try:
            docs = self._semantic_store.search_documentation(question, top_k=3)
            if docs:
                return "\n--- BUSINESS CONTEXT ---\n" + docs + "\n"
        except Exception:
            pass
        return ""
