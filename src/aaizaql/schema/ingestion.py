"""
aaizaql.schema.ingestion
────────────────────────
SchemaIngester: reads a live database schema and loads it into the vector
store so the LLM always has accurate table/column context.

Two ingestion modes:

1. :meth:`ingest_from_database` — auto-introspect a live database connector.
2. :meth:`ingest_ddl` — ingest a raw DDL string directly.

Each table becomes one vector-store document tagged ``type="ddl"``.
The ingester is idempotent — re-ingesting the same schema upserts existing
documents and deletes stale ones from the previous schema version.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

import hashlib
import re
from typing import TYPE_CHECKING

import structlog

from aaizaql.core.exceptions import SchemaIngestionError
from aaizaql.memory.vector_store import VectorStoreAdapter
from aaizaql.schema.embedder import EmbeddingService

if TYPE_CHECKING:
    from aaizaql.connectors.base import DatabaseConnector

logger = structlog.get_logger(__name__)


class SchemaIngester:
    """Reads database schema and indexes it in the vector store.

    Each CREATE TABLE block is stored as a separate document so the retriever
    can surface only the relevant tables for a given question rather than
    returning the entire schema on every query.

    Args:
        vector_store: Vector store adapter where DDL chunks are persisted.
    """

    def __init__(self, vector_store: VectorStoreAdapter) -> None:
        self._vs = vector_store
        self._embedder = EmbeddingService.get_instance()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_from_database(self, connector: "DatabaseConnector") -> int:
        """Auto-read schema from the connected database and index it.

        Args:
            connector: Connected database adapter whose ``get_schema()`` method
                returns CREATE TABLE DDL strings.

        Returns:
            Number of table chunks indexed.

        Raises:
            SchemaIngestionError: If the database schema cannot be read.
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
        """Parse and ingest a raw DDL string into the vector store.

        Reconciles the vector store against the current schema on each call:

        - Tables present in the new DDL are upserted (added or updated).
        - Tables absent from the new DDL but still in the store are deleted.
        - A ``schema_version`` sentinel document is upserted so callers can
          detect schema drift without re-reading the database.

        Safe to call multiple times — re-ingesting an identical schema is a
        no-op (upsert is idempotent, zero stale deletions).

        Args:
            ddl: Raw DDL string containing one or more CREATE TABLE statements.

        Returns:
            Number of table chunks ingested (added or updated).
        """
        chunks = self._chunk_by_table(ddl)
        if not chunks:
            logger.warning("schema.no_chunks", ddl_length=len(ddl))
            return 0

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

        version_hash = hashlib.sha256(ddl.encode()).hexdigest()[:16]
        self._vs.upsert(
            doc_id="schema_version",
            text=version_hash,
            embedding=[0.0] * 384,  # sentinel — zero vector, not used for similarity search
            metadata={"type": "schema_version", "hash": version_hash},
        )
        logger.info(
            "schema.ingested",
            tables=len(chunks),
            stale_removed=len(old_ids - new_ids),
            schema_version=version_hash,
        )
        return len(chunks)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_by_table(ddl: str) -> list[str]:
        """Split a multi-table DDL string into one chunk per CREATE TABLE block.

        Handles both semicolon-terminated and newline-separated DDL formats by
        splitting on ``CREATE TABLE`` boundaries rather than semicolons, which
        may be absent in some connectors' schema output.

        Args:
            ddl: Raw DDL string containing one or more CREATE TABLE statements.

        Returns:
            List of non-empty DDL strings, one per table.
        """
        ddl = ddl.replace("\r\n", "\n").strip()
        pattern = re.compile(r"(?=CREATE\s+TABLE\b)", re.IGNORECASE)
        raw_chunks = pattern.split(ddl)
        return [c.strip() for c in raw_chunks if c.strip()]

    @staticmethod
    def _extract_table_name(ddl_chunk: str) -> str:
        """Pull the table name out of a CREATE TABLE statement.

        Args:
            ddl_chunk: A single CREATE TABLE DDL string.

        Returns:
            Table name string, or ``"unknown"`` if no match is found.
        """
        match = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?",
            ddl_chunk,
            re.IGNORECASE,
        )
        return match.group(1) if match else "unknown"

    @staticmethod
    def _fingerprint(text: str) -> str:
        """Return a stable 16-character hex fingerprint for deduplication.

        Args:
            text: Any string to fingerprint.

        Returns:
            First 16 hex characters of the SHA-256 digest.
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]
