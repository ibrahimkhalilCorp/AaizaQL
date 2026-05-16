"""
aqlix.schema.semantic_store
────────────────────────────
SemanticStore: holds three kinds of "training" data that improve SQL accuracy:

1. Documentation  — free-text business context (retrieved via RAG).
2. Enums          — integer-code → label mappings (ALWAYS injected, no miss).
3. Q→SQL pairs    — sample (question, correct SQL) pairs (retrieved via RAG).

This module is the single source of truth for all user-supplied knowledge.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog

from aqlix.memory.vector_store import VectorStoreAdapter

logger = structlog.get_logger(__name__)

class EnumMapping:
    """
    A code → label mapping for a single table column.

    Used to tell the LLM what integer codes mean in business terms.
    Always injected into every prompt — no RAG retrieval miss possible.

    Example::

        EnumMapping("employees", "status", {1: "Active", 2: "Resigned"})
        # → "employees.status: 1=Active, 2=Resigned"
    """

    def __init__(
        self,
        table: str,
        column: str,
        mapping: dict[Any, str],
    ) -> None:
        self.table = table
        self.column = column
        self.mapping = {str(k): str(v) for k, v in mapping.items()}

    def to_prompt_text(self) -> str:
        """Format as a single line ready for prompt injection."""
        pairs = ", ".join(f"{k}={v}" for k, v in self.mapping.items())
        return f"{self.table}.{self.column}: {pairs}"


class SemanticStore:
    """
    Manages all user-supplied knowledge for a QueryEngine instance.

    The store is per-engine (not shared across engines) but persists across
    queries within the same process via the underlying vector store.
    """

    def __init__(self, vector_store: VectorStoreAdapter) -> None:
        self._vs = vector_store
        # Enum registry — stored in RAM, always injected into every prompt.
        # Structure: { "table.column": { code: label, ... }, ... }
        self._enums: dict[str, dict[Any, str]] = {}

    # ── Documentation ─────────────────────────────────────────────────────────

    def train_documentation(self, documentation: str) -> None:
        """
        Store free-text business documentation for RAG retrieval.

        Long texts are split into paragraphs so each chunk is retrievable
        independently. Duplicate content is deduplicated by fingerprint.
        """
        paragraphs = [p.strip() for p in documentation.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [documentation.strip()]

        for para in paragraphs:
            doc_id = f"doc_{self._fingerprint(para)}"
            embedding = self._embed(para)
            self._vs.upsert(
                id=doc_id,
                text=para,
                embedding=embedding,
                metadata={"type": "documentation"},
            )
        logger.info("semantic.documentation_trained", paragraphs=len(paragraphs))

    def search_documentation(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve the most relevant documentation chunks for a question.
        Returns a joined string ready for prompt injection.
        """
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
        """Store a verified (question, SQL) pair for few-shot retrieval."""
        text = f"Question: {question}\nSQL: {sql}"
        doc_id = f"pair_{self._fingerprint(question)}"
        embedding = self._embed(question)
        self._vs.upsert(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata={"type": "qa_pair"},
        )
        logger.debug("semantic.pair_trained", question=question[:60])

    # ── Enum mappings ─────────────────────────────────────────────────────────

    def define_enum(
        self,
        table: str,
        column: str,
        mapping: dict[Any, str],
    ) -> None:
        """
        Register a code → label mapping for a table column.
        These are always injected into every prompt — no RAG retrieval required.
        """
        key = f"{table}.{column}"
        self._enums[key] = {str(k): str(v) for k, v in mapping.items()}
        logger.info("semantic.enum_defined", key=key, values=len(mapping))

    def has_enums(self) -> bool:
        return bool(self._enums)

    def enum_count(self) -> int:
        return len(self._enums)

    def list_enums(self) -> dict[str, dict[str, str]]:
        return dict(self._enums)

    def get_enum_block(self) -> str:
        """
        Format all registered enums as a prompt-ready block.

        Example output::

            employees.status: 1=Active, 2=On Leave, 3=Resigned, 4=Terminated
            employees.job_grade: 1=Junior, 2=Mid, 3=Senior
        """
        lines: list[str] = []
        for key, mapping in self._enums.items():
            pairs = ", ".join(f"{k}={v}" for k, v in mapping.items())
            lines.append(f"{key}: {pairs}")
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Delegate embedding to the vector store's internal embedder."""
        # We reuse the schema ingester's embedder approach via a local import
        # to keep SemanticStore free of heavy dependencies.
        try:
            from aqlix.schema.ingestion import _SentenceEmbedder
            if not hasattr(self, "_embedder"):
                self._embedder = _SentenceEmbedder()  # type: ignore[attr-defined]
            return self._embedder.embed(text)  # type: ignore[attr-defined]
        except Exception:
            return [0.0] * 384  # safe fallback

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:12]
