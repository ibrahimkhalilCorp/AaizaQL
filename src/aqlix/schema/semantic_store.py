"""
aqlix.schema.semantic_store
────────────────────────────
SemanticStore: manages three types of training data.

  1. Documentation  — free-text business context (Vanna-style)
  2. Enum mappings  — numeric code → label (always injected, never missed)
  3. Q→SQL pairs    — sample questions with correct SQL

This is the single source of truth for everything the engine "knows"
about the user's database beyond the raw DDL schema.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any

from aqlix.memory.vector_store import VectorStoreAdapter
from aqlix.schema.ingestion import _embed, _stable_id
import structlog

logger = structlog.get_logger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class EnumMapping:
    """A numeric code → label mapping for one column."""

    table: str
    column: str
    mapping: dict[int | str, str]  # {1: "Active", 2: "On Leave", ...}

    def to_prompt_text(self) -> str:
        """Format for injection into every LLM prompt."""
        pairs = ", ".join(f"{k}={v}" for k, v in self.mapping.items())
        return f"{self.table}.{self.column}: {pairs}"


@dataclass
class TrainingState:
    """
    In-memory registry of all enum mappings and documentation chunks.
    Enum mappings are always injected into prompts (never retrieved via RAG).
    Documentation is stored in the vector store for semantic retrieval.
    """

    enums: list[EnumMapping] = field(default_factory=list)


# ── SemanticStore ─────────────────────────────────────────────────────────────


class SemanticStore:
    """
    Manages all three training data types for Aqlix.

    Usage (mirrors Vanna API where possible):
        store.train(documentation="status 1=Active, 2=OnLeave...")
        store.train(question="Top 5 employees", sql="SELECT...")
        store.define_enum("employees", "status", {1:"Active", 2:"On Leave"})
    """

    def __init__(self, vector_store: VectorStoreAdapter) -> None:
        self._vs = vector_store
        self._state = TrainingState()

    # ── 1. Documentation ──────────────────────────────────────────────────────

    def train_documentation(self, documentation: str) -> None:
        """
        Store free-text business documentation for RAG retrieval.

        Use for:
          - Table/column descriptions in plain English
          - Business rules and calculations
          - Date format conventions for this database
          - Domain-specific terminology

        Example:
            engine.train(documentation='''
                employees.status: 1=Active, 2=On Leave, 3=Resigned, 4=Terminated
                sales_orders.order_status: 1=Pending, 2=Processing, 3=Delivered
                Use strftime() for SQLite date filtering.
            ''')
        """
        if not documentation.strip():
            logger.warning("semantic.empty_documentation")
            return

        # Split into paragraphs so each chunk is focused
        chunks = [c.strip() for c in documentation.split("\n\n") if c.strip()]
        if not chunks:
            chunks = [documentation.strip()]

        for chunk in chunks:
            self._vs.upsert(
                id=_stable_id("doc", chunk),
                text=chunk,
                embedding=_embed(chunk),
                metadata={"type": "documentation"},
            )

        logger.info("semantic.documentation_stored", chunks=len(chunks))

    # ── 2. Enum Mappings ──────────────────────────────────────────────────────

    def define_enum(
        self,
        table: str,
        column: str,
        mapping: dict[int | str, str],
    ) -> None:
        """
        Register a numeric code → label mapping for a column.

        Unlike documentation, enum mappings are ALWAYS injected into every
        prompt — they are never subject to RAG retrieval misses.

        Use for any column that stores numeric codes instead of text:
          - status columns (1=Active, 2=Inactive)
          - type/category enums (1=Manager, 2=Staff)
          - foreign-key lookups that are stored as integers

        Example:
            engine.define_enum("employees", "status", {
                1: "Active",
                2: "On Leave",
                3: "Resigned",
                4: "Terminated",
            })
        """
        # Remove existing mapping for same table.column (idempotent)
        self._state.enums = [
            e for e in self._state.enums if not (e.table == table and e.column == column)
        ]
        enum = EnumMapping(table=table, column=column, mapping=mapping)
        self._state.enums.append(enum)

        # Also store in vector store for documentation-style retrieval
        doc_text = f"Column {table}.{column} stores numeric codes: " + enum.to_prompt_text()
        self._vs.upsert(
            id=_stable_id("enum", table, column),
            text=doc_text,
            embedding=_embed(doc_text),
            metadata={"type": "enum", "table": table, "column": column},
        )

        logger.info(
            "semantic.enum_defined",
            table=table,
            column=column,
            values=len(mapping),
        )

    # ── 3. Q→SQL Pairs ───────────────────────────────────────────────────────

    def train_sql_pair(self, question: str, sql: str) -> None:
        """
        Store a verified (question → SQL) pair for few-shot retrieval.

        Example:
            engine.train(
                question="Top 5 employees by sales",
                sql="SELECT e.name, SUM(s.total_amount) FROM ..."
            )
        """
        text = f"Question: {question}\nSQL: {sql}"
        self._vs.upsert(
            id=_stable_id("qa", question),
            text=text,
            embedding=_embed(question),
            metadata={"type": "qa_pair"},
        )
        logger.info("semantic.sql_pair_stored", question=question[:60])

    # ── Prompt injection ──────────────────────────────────────────────────────

    def get_enum_block(self) -> str:
        """
        Return all enum mappings formatted for prompt injection.
        Returns empty string if no enums defined.
        """
        if not self._state.enums:
            return ""
        lines = [e.to_prompt_text() for e in self._state.enums]
        return "\n".join(lines)

    def search_documentation(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve relevant documentation chunks for a question.
        Returns formatted string for prompt injection.
        """
        hits = self._vs.search(
            query=question,
            filter_type="documentation",
            top_k=top_k,
        )
        if not hits:
            return ""
        return "\n\n".join(h.text for h in hits)

    def has_enums(self) -> bool:
        return len(self._state.enums) > 0

    def enum_count(self) -> int:
        return len(self._state.enums)

    def list_enums(self) -> list[dict[str, Any]]:
        """Return all registered enums as a list of dicts (for inspection)."""
        return [
            {"table": e.table, "column": e.column, "mapping": e.mapping} for e in self._state.enums
        ]
