"""
aaizaql.nlp.generator
─────────────────────
SQLGenerator: builds context from RAG + SemanticStore, assembles a prompt,
and calls the LLM to produce a SQL statement.

Pipeline per call to :meth:`SQLGenerator.generate`:

1. Retrieve schema DDL chunks from the vector store (RAG).
2. Inject enum mappings from the SemanticStore — always, no retrieval miss.
3. Retrieve relevant documentation via semantic search.
4. Retrieve similar Q→SQL pairs for few-shot examples.
5. Assemble a structured prompt (optionally with chain-of-thought prefix).
6. Call the LLM.
7. Parse raw SQL from the response.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

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

# Keywords that signal a question needs chain-of-thought reasoning.
# "per" was deliberately excluded — it produces false positives on words like
# "expenses", "temperature", and "department".
_COT_KEYWORDS = {
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
    """Return ``True`` when the question contains a CoT keyword.

    Uses word-boundary matching to prevent partial-word matches like
    "expenses" triggering on a hypothetical "per" keyword.

    Args:
        question: Natural language question from the user.

    Returns:
        ``True`` if chain-of-thought prompting should be used.
    """
    q = question.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in _COT_KEYWORDS)


class SQLGenerator:
    """Generates SQL from a natural language question via RAG + LLM.

    Requires a ``connector`` at construction time so the correct SQL dialect
    (e.g. ``"sqlite"``, ``"postgresql"``) is embedded in every prompt.

    Args:
        llm: LLM provider used for SQL generation.
        vector_store: Vector store adapter for schema and example retrieval.
        settings: Library-wide settings (controls top-k values, timeout).
        semantic_store: Optional SemanticStore for enum and doc injection.
        connector: DatabaseConnector instance. Must have a non-empty ``name``
            attribute identifying the SQL dialect.

    Raises:
        ValueError: If ``connector`` is ``None`` or has an empty ``name``.
    """

    def __init__(
        self,
        llm: LLMProvider,
        vector_store: VectorStoreAdapter,
        settings: Settings,
        semantic_store: "SemanticStore | None" = None,
        connector: object | None = None,
    ) -> None:
        # Validate connector at construction so dialect errors surface
        # immediately rather than silently at query time.
        if connector is None:
            raise ValueError(
                "SQLGenerator requires a connector instance. "
                "Pass connector=<DatabaseConnector> so the correct SQL dialect "
                "is used in prompts."
            )
        if not getattr(connector, "name", ""):
            raise ValueError(
                f"Connector {type(connector).__name__!r} has an empty or missing "
                "'name' attribute. Every DatabaseConnector subclass must set "
                "name = '<dialect>' (e.g. 'sqlite', 'postgresql')."
            )

        self._llm = llm
        self._vs = vector_store
        self._settings = settings
        self._semantic = semantic_store
        self._connector = connector
        self._timeout = settings.llm_timeout_seconds

    def generate(self, question: str, history: list[Turn]) -> str:
        """Convert a natural language question to SQL.

        Args:
            question: User's natural language question.
            history: Previous conversation turns for multi-turn context.

        Returns:
            A SQL statement string ready for validation and execution.

        Raises:
            SQLGenerationError: If the LLM returns an empty response after
                parsing.
        """
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
        """Retrieve schema DDL chunks and Q→SQL example pairs from the vector store.

        Args:
            question: User's question used as the search query.

        Returns:
            Tuple of ``(schema_block, examples_block)`` — both formatted as
            newline-separated strings ready for prompt injection.
        """
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
        """Return the formatted enum block for prompt injection.

        Enum mappings are always injected directly into the prompt rather than
        retrieved via RAG — this guarantees they are never missed regardless of
        vector-store retrieval quality.

        Returns:
            Formatted enum block string, or an empty string if no enums are
            registered.
        """
        if not self._semantic or not self._semantic.has_enums():
            return ""
        return ENUM_BLOCK_TEMPLATE.format(enums=self._semantic.get_enum_block())

    def _build_doc_block(self, question: str) -> str:
        """Retrieve relevant business documentation via semantic search.

        Args:
            question: User's question used as the search query.

        Returns:
            Formatted documentation block string, or an empty string if no
            relevant docs are found or the SemanticStore is not set.
        """
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
        """Assemble the final LLM prompt from all context components.

        Args:
            question: User's natural language question.
            schema_chunks: DDL schema context from RAG.
            example_pairs: Few-shot Q→SQL examples from RAG.
            enum_block: Formatted enum mappings block.
            doc_block: Formatted business documentation block.
            history: Previous conversation turns.
            use_cot: When ``True``, wraps the prompt with CoT prefix/suffix.

        Returns:
            Complete prompt string ready to send to the LLM.
        """
        # ContextManager's deque(maxlen=N) already caps history length;
        # no additional slicing is needed here.
        history_text = (
            "\n".join(f"User: {turn['question']}\nSQL: {turn['sql']}" for turn in history)
            or "(no prior conversation)"
        )

        context = CONTEXT_TEMPLATE.format(
            dialect=self._connector.name,  # type: ignore[attr-defined]
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
        """Strip markdown fences and extract SQL from the raw LLM response.

        Args:
            raw: Raw text response from the LLM.
            use_cot: When ``True``, splits on the ``[SQL]`` marker first.

        Returns:
            Clean SQL string with fences and whitespace removed.
        """
        return parse_sql_response(raw, use_cot=use_cot)
