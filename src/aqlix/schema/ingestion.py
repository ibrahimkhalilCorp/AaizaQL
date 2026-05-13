"""
aqlix.schema.ingestion
───────────────────────
SchemaIngester: introspects databases and stores schema chunks + Q→SQL pairs
in the vector store for RAG retrieval.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from aqlix.memory.vector_store import VectorStoreAdapter
from aqlix.core.exceptions import SchemaIngestionError
import structlog

if TYPE_CHECKING:
    from aqlix.connectors.base import DatabaseConnector

logger = structlog.get_logger(__name__)

# ── Embedding helper (lazy-loaded to avoid slow import at startup) ────────────

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("embedder.loaded", model="all-MiniLM-L6-v2")
        except ImportError as exc:
            raise SchemaIngestionError(
                "sentence-transformers is not installed. " "Run: pip install sentence-transformers"
            ) from exc
    return _embedder


def _embed(text: str) -> list[float]:
    embedder = _get_embedder()
    return embedder.encode(text, normalize_embeddings=True).tolist()


def _stable_id(*parts: str) -> str:
    """Generate a stable, collision-resistant ID from string parts."""
    combined = "|".join(parts)
    return hashlib.sha1(combined.encode()).hexdigest()[:16]


class SchemaIngester:
    """
    Ingests database schema and Q→SQL pairs into the vector store.
    Must be called once before first query, and after schema changes.
    """

    def __init__(self, vector_store: VectorStoreAdapter) -> None:
        self._vs = vector_store

    def ingest_from_database(self, connector: "DatabaseConnector") -> int:
        """
        Auto-introspect the connected database and upsert all schema chunks.

        Returns
        -------
        int  Number of chunks upserted.
        """
        try:
            ddl = connector.get_schema()
        except Exception as exc:
            raise SchemaIngestionError(f"Schema introspection failed: {exc}") from exc

        if not ddl.strip():
            logger.warning("ingester.empty_schema")
            return 0

        chunks = self._chunk_by_table(ddl)
        for chunk in chunks:
            table_name = self._extract_table_name(chunk)
            self._vs.upsert(
                id=_stable_id("ddl", chunk),
                text=chunk,
                embedding=_embed(chunk),
                metadata={"type": "ddl", "table": table_name},
            )

        logger.info("ingester.schema_done", chunks=len(chunks))
        return len(chunks)

    def ingest_sql_pair(self, question: str, sql: str) -> None:
        """
        Store a verified (question, SQL) pair for future few-shot retrieval.
        Embed on the question so retrieval finds similar questions.
        """
        text = f"Question: {question}\nSQL: {sql}"
        self._vs.upsert(
            id=_stable_id("qa", question),
            text=text,
            embedding=_embed(question),
            metadata={"type": "qa_pair"},
        )
        logger.debug("ingester.pair_stored", question=question[:60])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _chunk_by_table(self, ddl: str) -> list[str]:
        """Split DDL into one chunk per CREATE TABLE statement."""
        # Split on CREATE TABLE (case-insensitive)
        pattern = re.compile(r"(?=CREATE\s+TABLE)", re.IGNORECASE)
        raw_chunks = pattern.split(ddl)
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        return chunks or [ddl.strip()]

    def _extract_table_name(self, chunk: str) -> str:
        """Extract the table name from a CREATE TABLE chunk."""
        match = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"']?(\w+)[`\"']?",
            chunk,
            re.IGNORECASE,
        )
        return match.group(1) if match else "unknown"
