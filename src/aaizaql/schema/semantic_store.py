"""
aaizaql.schema.semantic_store
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

from aaizaql.memory.vector_store import VectorStoreAdapter

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
            doc_id=doc_id,
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
        Also upserted to the vector store for reference.
        """
        key = f"{table}.{column}"
        str_mapping = {str(k): str(v) for k, v in mapping.items()}
        self._enums[key] = str_mapping
        # Upsert to vector store so it is searchable/persistable
        text = f"{key}: " + ", ".join(f"{k}={v}" for k, v in str_mapping.items())
        doc_id = f"enum_{key}"
        embedding = self._embed(text)
        self._vs.upsert(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata={"type": "enum", "table": table, "column": column},
        )
        logger.info("semantic.enum_defined", key=key, values=len(mapping))

    def has_enums(self) -> bool:
        return bool(self._enums)

    def enum_count(self) -> int:
        return len(self._enums)

    def list_enums(self) -> list[dict]:
        """Return enums as a list of dicts with table, column, and mapping keys."""
        result = []
        for key, mapping in self._enums.items():
            table, column = key.split(".", 1)
            # Try to restore int keys for usability
            restored: dict = {}
            for k, v in mapping.items():
                try:
                    restored[int(k)] = v
                except (ValueError, TypeError):
                    restored[k] = v
            result.append({"table": table, "column": column, "mapping": restored})
        return result

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
        """T2.4 — Delegate to the singleton EmbeddingService (one model load)."""
        try:
            from aaizaql.schema.embedder import EmbeddingService

            return EmbeddingService.get_instance().embed(text)
        except Exception:
            return [0.0] * 384  # safe fallback

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]  # T3.6 SHA-256-16
