"""
aaizaql.schema.ingestion
───────────────────────
SchemaIngester: reads a live database schema and loads it into the vector
store so the LLM always has accurate table/column context.

Two modes:
  1. ingest_from_database(connector) — auto-introspect a live DB
  2. ingest_ddl(ddl_string)          — ingest a raw DDL string directly

Each table becomes one vector-store document tagged with type="ddl".
The ingester is idempotent — re-ingesting the same schema just upserts
existing documents.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

import structlog

from aaizaql.core.exceptions import SchemaIngestionError
from aaizaql.memory.vector_store import VectorStoreAdapter
from aaizaql.schema.embedder import EmbeddingService

# Alias so tests can patch `ing_mod._SentenceEmbedder` (the embedder singleton class)
_SentenceEmbedder = EmbeddingService

if TYPE_CHECKING:
    from aaizaql.connectors.base import DatabaseConnector

logger = structlog.get_logger(__name__)


class SchemaIngester:
    """
    Reads database schema and stores it in the vector store.

    Each CREATE TABLE block is stored as a separate document so the retriever
    can surface only the relevant tables for a given question.
    """

    def __init__(self, vector_store: VectorStoreAdapter) -> None:
        self._vs = vector_store
        self._embedder = EmbeddingService.get_instance()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_from_database(self, connector: DatabaseConnector) -> int:
        """
        Auto-read schema from the connected database and index it.

        Returns
        -------
        int  Number of table chunks ingested.
        """
        try:
            ddl = connector.get_schema()
        except Exception as exc:
            raise SchemaIngestionError(f"Failed to read schema from database: {exc}") from exc

        if not ddl.strip():
            logger.warning("schema.empty", detail="Database returned no DDL.")
            return 0

        return self.ingest_ddl(ddl)

    def ingest_ddl(self, ddl: str) -> int:
        """
        Parse and ingest a raw DDL string.

        On each call the ingester reconciles the vector store against the
        current schema:
          - Tables present in the new DDL are upserted (added or updated).
          - Tables that existed in the vector store but are absent from the
            new DDL are deleted (stale vectors removed).
          - A ``schema_version`` sentinel document is upserted so callers can
            detect schema drift without re-reading the database.

        Safe to call multiple times — re-ingesting an identical schema is a
        no-op on the store (upsert is idempotent, zero stale deletions).

        Returns
        -------
        int  Number of table chunks ingested (added or updated).
        """
        chunks = self._chunk_by_table(ddl)
        if not chunks:
            logger.warning("schema.no_chunks", ddl_length=len(ddl))
            return 0

        # T2.3 — compute new IDs and delete stale ones before upserting
        new_ids = {f"ddl_{self._fingerprint(c)}" for c in chunks}
        old_ids = self._vs.list_ids(filter_type="ddl")
        for stale_id in old_ids - new_ids:
            self._vs.delete(stale_id)
            logger.debug("schema.stale_deleted", doc_id=stale_id)

        for chunk in chunks:
            table_name = self._extract_table_name(chunk)
            doc_id = f"ddl_{self._fingerprint(chunk)}"
            embedding = self._embedder.embed(chunk)
            self._vs.upsert(
                doc_id=doc_id,
                text=chunk,
                embedding=embedding,
                metadata={"type": "ddl", "table": table_name},
            )

        # T5.5 — store schema version hash for drift detection
        import hashlib

        version_hash = hashlib.sha256(ddl.encode()).hexdigest()[:16]
        self._vs.upsert(
            doc_id="schema_version",
            text=version_hash,
            embedding=[0.0] * 384,  # sentinel; not used for search
            metadata={"type": "schema_version", "hash": version_hash},
        )
        logger.info(
            "schema.ingested",
            tables=len(chunks),
            stale_removed=len(old_ids - new_ids),
            schema_version=version_hash,
        )
        return len(chunks)

    # T2.5 — ingest_sql_pair() removed (dead code; use SemanticStore.train_sql_pair() instead)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_by_table(ddl: str) -> list[str]:
        """
        Split a multi-table DDL string into one chunk per CREATE TABLE block.
        Handles both semicolon-terminated and newline-separated DDL.
        """
        # Normalise line endings
        ddl = ddl.replace("\r\n", "\n").strip()

        # Split on CREATE TABLE boundaries (case-insensitive)
        pattern = re.compile(r"(?=CREATE\s+TABLE\b)", re.IGNORECASE)
        raw_chunks = pattern.split(ddl)

        chunks = [c.strip() for c in raw_chunks if c.strip()]
        return chunks

    @staticmethod
    def _extract_table_name(ddl_chunk: str) -> str:
        """Pull the table name out of a CREATE TABLE statement."""
        match = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?",
            ddl_chunk,
            re.IGNORECASE,
        )
        return match.group(1) if match else "unknown"

    @staticmethod
    def _fingerprint(text: str) -> str:
        """Stable 12-char hex fingerprint of a string."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]  # T3.6 SHA-256-16
