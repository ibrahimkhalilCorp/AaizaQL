"""

aaizaql.nlp.generator

────────────────────

SQLGenerator: builds context from RAG + SemanticStore, assembles prompt, calls LLM.

"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import SQLGenerationError
from aaizaql.llm.base import LLMProvider
from aaizaql.memory.context import Turn
from aaizaql.memory.vector_store import VectorStoreAdapter
from aaizaql.nlp.prompts import (
    CONTEXT_TEMPLATE,
    COT_PROMPT_PREFIX,
    COT_PROMPT_SUFFIX,
    DOC_BLOCK_TEMPLATE,
    ENUM_BLOCK_TEMPLATE,
    SYSTEM_PROMPT,
)
from aaizaql.nlp.utils import parse_sql_response

if TYPE_CHECKING:

    from aaizaql.schema.semantic_store import SemanticStore

logger = structlog.get_logger(__name__)

_COT_KEYWORDS = {
    # T3.1 — removed "per" (false-positive: expenses, temperature, department…)
    "join",
    "joining",
    "combine",
    "merge",
    "group by",
    "grouped",
    "by month",
    "by year",
    "by week",
    "compare",
    "difference",
    "trend",
    "average",
    "sum",
    "count",
    "most",
    "least",
    "top",
    "bottom",
    "rank",
    "ranking",
    "between",
    "range",
    "having",
}


def _needs_cot(question: str) -> bool:

    # T3.1 — word-boundary match to prevent "expenses" → COT on "per"

    q = question.lower()

    return any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in _COT_KEYWORDS)


class SQLGenerator:
    """

    Generates SQL from a natural language question using RAG + SemanticStore + LLM.

    Pipeline:

    1. Retrieve schema chunks (DDL) from vector store

    2. Inject enum mappings — ALWAYS, no retrieval miss possible

    3. Retrieve relevant documentation from semantic store

    4. Retrieve similar Q→SQL pairs (few-shot examples)

    5. Assemble structured prompt

    6. Call LLM

    7. Parse SQL from response

    """

    def __init__(
        self,
        llm: LLMProvider,
        vector_store: VectorStoreAdapter,
        settings: Settings,
        semantic_store: SemanticStore | None = None,
        connector: object | None = None,
    ) -> None:

        self._llm = llm

        self._vs = vector_store

        self._settings = settings

        self._semantic = semantic_store

        self._connector = connector  # used for dialect label (T1.1)

        self._timeout = settings.llm_timeout_seconds

        # T1.1 — Validate at construction time so dialect errors surface
        # immediately, not silently at query time.
        if connector is None:
            raise ValueError(
                "SQLGenerator requires a connector instance. "
                "Pass connector=<DatabaseConnector> so the correct SQL dialect "
                "is used in prompts. Do not leave this as None."
            )
        if not getattr(connector, "name", ""):
            raise ValueError(
                f"Connector {type(connector).__name__!r} has an empty or missing "
                "'name' attribute. Every DatabaseConnector subclass must set "
                "name = '<dialect>' (e.g. 'sqlite', 'postgresql')."
            )

    def generate(self, question: str, history: list[Turn]) -> str:

        schema_chunks, example_pairs = self._build_rag_context(question)

        enum_block = self._build_enum_block()

        doc_block = self._build_doc_block(question)

        use_cot = _needs_cot(question)

        prompt = self._assemble_prompt(
            question=question,
            schema_chunks=schema_chunks,
            example_pairs=example_pairs,
            enum_block=enum_block,
            doc_block=doc_block,
            history=history,
            use_cot=use_cot,
        )

        logger.debug(
            "generator.calling_llm",
            provider=self._llm.name,
            use_cot=use_cot,
            has_enums=bool(enum_block),
            has_docs=bool(doc_block),
            question=question[:60],
        )

        raw = self._llm.complete(prompt, system=SYSTEM_PROMPT, timeout=self._timeout)

        sql = self._parse_response(raw, use_cot=use_cot)

        if not sql:

            raise SQLGenerationError(question, f"LLM raw response was: {raw[:200]}")

        logger.info("generator.sql_ready", sql_preview=sql[:80])

        return sql

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_rag_context(self, question: str) -> tuple[str, str]:

        schema_hits = self._vs.search(
            query=question,
            filter_type="ddl",
            top_k=self._settings.schema_top_k,
        )

        example_hits = self._vs.search(
            query=question,
            filter_type="qa_pair",
            top_k=self._settings.examples_top_k,
        )

        schema_block = "\n\n".join(h.text for h in schema_hits) or "(no schema ingested yet)"

        examples_block = "\n\n".join(h.text for h in example_hits) or "(no examples yet)"

        return schema_block, examples_block

    def _build_enum_block(self) -> str:
        """Always injected — never a retrieval miss."""

        if not self._semantic or not self._semantic.has_enums():

            return ""

        return ENUM_BLOCK_TEMPLATE.format(enums=self._semantic.get_enum_block())

    def _build_doc_block(self, question: str) -> str:
        """Retrieve relevant documentation via RAG."""

        if not self._semantic:

            return ""

        docs = self._semantic.search_documentation(question, top_k=3)

        if not docs:

            return ""

        return DOC_BLOCK_TEMPLATE.format(docs=docs)

    def _assemble_prompt(
        self,
        question: str,
        schema_chunks: str,
        example_pairs: str,
        enum_block: str,
        doc_block: str,
        history: list[Turn],
        use_cot: bool,
    ) -> str:

        # T3.4 — removed redundant slice; ContextManager.deque(maxlen=N) already caps history

        history_text = (
            "\n".join(f"User: {turn['question']}\nSQL: {turn['sql']}" for turn in history)
            or "(no prior conversation)"
        )

        context = CONTEXT_TEMPLATE.format(
            dialect=self._connector.name,  # type: ignore[union-attr]  # T1.1
            schema_chunks=schema_chunks,
            enum_block=enum_block,
            doc_block=doc_block,
            example_pairs=example_pairs,
            history=history_text,
            question=question,
        )

        if use_cot:

            return COT_PROMPT_PREFIX + context + COT_PROMPT_SUFFIX

        return context

    def _parse_response(self, raw: str, use_cot: bool) -> str:

        # T1.3 — delegate to shared utility

        return parse_sql_response(raw, use_cot=use_cot)
