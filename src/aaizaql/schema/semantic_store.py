"""
aaizaql.schema.semantic_store
─────────────────────────────
SemanticStore: holds three kinds of training data that improve SQL accuracy.

1. **Documentation** — free-text business context (retrieved via RAG).
2. **Enums** — integer-code → label mappings (always injected, no miss).
3. **Q→SQL pairs** — sample (question, correct SQL) pairs (retrieved via RAG).

This module is the single source of truth for all user-supplied knowledge.
Enum mappings are kept in RAM and injected into every prompt regardless of
vector-search results, guaranteeing they are never missed.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

import hashlib
from typing import Any

import structlog

from aaizaql.memory.vector_store import VectorStoreAdapter

logger = structlog.get_logger(__name__)


class EnumMapping:
    """A code → label mapping for a single table column.

    Used to tell the LLM what integer codes mean in business terms.
    Always injected into every prompt — no RAG retrieval miss is possible.

    Args:
        table: Table name the column belongs to.
        column: Column name that holds the numeric codes.
        mapping: Dict of ``{code: label}`` pairs. Keys and values are
            coerced to strings for consistent serialisation.

    Example::

        EnumMapping("employees", "status", {1: "Active", 2: "Resigned"})
        # → "employees.status: 1=Active, 2=Resigned"
    """

    def __init__(self, table: str, column: str, mapping: dict[Any, str]) -> None:
        self.table = table
        self.column = column
        self.mapping = {str(k): str(v) for k, v in mapping.items()}

    def to_prompt_text(self) -> str:
        """Format this mapping as a single line ready for prompt injection.

        Returns:
            String in the form ``"table.column: code=label, code=label, ..."``.
        """
        pairs = ", ".join(f"{k}={v}" for k, v in self.mapping.items())
        return f"{self.table}.{self.column}: {pairs}"


class SemanticStore:
    """Manages all user-supplied knowledge for a QueryEngine instance.

    The store is per-engine (not shared across engines) but persists across
    queries within the same process via the underlying vector store backend.

    Args:
        vector_store: Vector store adapter for RAG-based retrieval of
            documentation and Q→SQL pairs. May be ``None`` during deferred
            initialisation — enum mappings still work without it.
    """

    def __init__(self, vector_store: VectorStoreAdapter) -> None:
        self._vs = vector_store
        self._enums: dict[str, dict[Any, str]] = {}

    # ── Documentation ─────────────────────────────────────────────────────────

    def train_documentation(self, documentation: str) -> None:
        """Store free-text business documentation for RAG retrieval.

        Long texts are split on double newlines so each paragraph is
        retrievable independently. Duplicate content is deduplicated by
        SHA-256 fingerprint on the vector-store upsert.

        Args:
            documentation: Free-text string containing business rules, column
                explanations, or domain knowledge.
        """
        paragraphs = [p.strip() for p in documentation.split("\n\n") if p.strip()]
        if not paragraphs:
            return

        if self._vs is None:
            logger.info("semantic.documentation_trained", paragraphs=len(paragraphs))
            return

        for para in paragraphs:
            doc_id = f"doc_{self._fingerprint(para)}"
            embedding = self._embed(para)
            self._vs.upsert(
                doc_id=doc_id,
                text=para,
                embedding=embedding,
                metadata={"type": "documentation"},
            )
        logger.info("semantic.documentation_trained", paragraphs=len(paragraphs))

    def search_documentation(self, question: str, top_k: int = 3) -> str:
        """Retrieve the most relevant documentation chunks for a question.

        Args:
            question: User's natural language question used as the search query.
            top_k: Maximum number of documentation chunks to retrieve.

        Returns:
            Newline-separated documentation string ready for prompt injection,
            or an empty string when no relevant docs are found.
        """
        if self._vs is None:
            return ""
        hits = self._vs.search(
            query=question,
            filter_type="documentation",
            top_k=top_k,
        )
        if not hits:
            return ""
        return "\n\n".join(h.text for h in hits)

    # ── Q→SQL Pairs ───────────────────────────────────────────────────────────

    def train_sql_pair(self, question: str, sql: str) -> None:
        """Store a verified (question, SQL) pair for few-shot retrieval.

        The question is embedded and indexed; the combined ``Question: ...\nSQL: ...``
        string is stored as the document text so it renders cleanly in prompts.

        Args:
            question: Example natural language question.
            sql: Correct SQL statement for the question.
        """
        if self._vs is None:
            return
        text = f"Question: {question}\nSQL: {sql}"
        doc_id = f"pair_{self._fingerprint(question)}"
        embedding = self._embed(question)
        self._vs.upsert(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata={"type": "qa_pair"},
        )
        logger.debug("semantic.pair_trained", question=question[:60])

    # ── Enum Mappings ─────────────────────────────────────────────────────────

    def define_enum(self, table: str, column: str, mapping: dict[Any, str]) -> None:
        """Register a code → label mapping for a table column.

        Enum mappings are stored in RAM and injected into every prompt, so
        they are never missed regardless of RAG retrieval quality. They are
        also upserted to the vector store for reference when available.

        Args:
            table: Table name the column belongs to.
            column: Column name that holds the numeric codes.
            mapping: Dict of ``{code: label}`` pairs.
        """
        key = f"{table}.{column}"
        str_mapping = {str(k): str(v) for k, v in mapping.items()}
        self._enums[key] = str_mapping
        logger.info("semantic.enum_defined", key=key, values=len(mapping))

        if self._vs is None:
            return

        text = f"{key}: " + ", ".join(f"{k}={v}" for k, v in str_mapping.items())
        doc_id = f"enum_{key}"
        embedding = self._embed(text)
        self._vs.upsert(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata={"type": "enum", "table": table, "column": column},
        )

    def has_enums(self) -> bool:
        """Return ``True`` if any enum mappings have been registered.

        Returns:
            ``True`` when the enum registry is non-empty.
        """
        return bool(self._enums)

    def enum_count(self) -> int:
        """Return the number of registered enum column mappings.

        Returns:
            Count of unique ``table.column`` enum registrations.
        """
        return len(self._enums)

    def list_enums(self) -> list[dict]:
        """Return all registered enums as a list of dicts.

        Integer-stringified keys (e.g. ``"1"``) are restored to ``int`` where
        possible for usability.

        Returns:
            List of dicts with keys ``"table"``, ``"column"``, and
            ``"mapping"`` (the original code → label dict).
        """
        result = []
        for key, mapping in self._enums.items():
            table, column = key.split(".", 1)
            restored: dict = {}
            for k, v in mapping.items():
                try:
                    restored[int(k)] = v
                except (ValueError, TypeError):
                    restored[k] = v
            result.append({"table": table, "column": column, "mapping": restored})
        return result

    def get_enum_block(self) -> str:
        """Format all registered enums as a prompt-ready block.

        Returns:
            Newline-separated string with one line per registered column::

                employees.status: 1=Active, 2=On Leave, 3=Resigned
                employees.job_grade: 1=Junior, 2=Mid, 3=Senior
        """
        lines: list[str] = []
        for key, mapping in self._enums.items():
            pairs = ", ".join(f"{k}={v}" for k, v in mapping.items())
            lines.append(f"{key}: {pairs}")
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Delegate to the singleton EmbeddingService.

        Uses the shared singleton to avoid loading the 92 MB model more than
        once per process. Falls back to a zero vector on any failure so the
        store remains functional even if sentence-transformers is unavailable.

        Args:
            text: Input string to embed.

        Returns:
            List of ``384`` floats, or all-zeros on error.
        """
        try:
            from aaizaql.schema.embedder import EmbeddingService

            return EmbeddingService.get_instance().embed(text)
        except Exception:
            return [0.0] * 384

    @staticmethod
    def _fingerprint(text: str) -> str:
        """Return a stable 16-character hex fingerprint for deduplication.

        Args:
            text: Any string to fingerprint.

        Returns:
            First 16 hex characters of the SHA-256 digest.
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]
