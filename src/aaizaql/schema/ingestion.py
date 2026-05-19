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
        self._embedder = _SentenceEmbedder()

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

        Returns
        -------
        int  Number of table chunks ingested.
        """
        chunks = self._chunk_by_table(ddl)
        if not chunks:
            logger.warning("schema.no_chunks", ddl_length=len(ddl))
            return 0

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

        logger.info("schema.ingested", tables=len(chunks))
        return len(chunks)

    def ingest_sql_pair(self, question: str, sql: str) -> None:
        """
        Store a verified (question, SQL) pair for future few-shot retrieval.
        The embedding is computed on the *question* so retrieval is similarity-
        based on intent, not SQL syntax.
        """
        text = f"Question: {question}\nSQL: {sql}"
        doc_id = f"pair_{self._fingerprint(question)}"
        embedding = self._embedder.embed(question)
        self._vs.upsert(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata={"type": "qa_pair"},
        )
        logger.debug("schema.pair_ingested", question=question[:60])

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
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:12]


# ── Embedding helper ──────────────────────────────────────────────────────────


class _SentenceEmbedder:
    """
    Thin wrapper around sentence-transformers.
    Lazy-loads the model on first use to keep import time fast.
    Falls back to a deterministic zero-vector when the library is absent
    so unit tests can run without downloading a model.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"
    _DIM = 384

    def __init__(self) -> None:
        self._model: object | None = None

    def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for text."""
        if self._model is None:
            self._load()
        if self._model is None:
            # Fallback — deterministic pseudo-embedding for offline testing
            return self._fallback_embed(text)
        try:
            vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
            return vec.tolist()
        except Exception as exc:
            logger.warning("embedder.failed", detail=str(exc)[:80])
            return self._fallback_embed(text)

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer(self._MODEL_NAME)
            logger.info("embedder.loaded", model=self._MODEL_NAME)
        except ImportError:
            logger.warning(
                "embedder.missing",
                detail="sentence-transformers not installed. "
                "Falling back to hash-based embeddings (lower accuracy). "
                "Install with: pip install sentence-transformers",
            )

    @classmethod
    def _fallback_embed(cls, text: str) -> list[float]:
        """
        Hash-based pseudo-embedding.  Not semantically meaningful but allows
        the vector store to operate without the transformers library.
        One float per 4-char slice of the sha256 hex digest, normalised.
        """
        digest = hashlib.sha256(text.encode()).hexdigest()
        values = [
            int(digest[i : i + 2], 16) / 255.0 for i in range(0, min(len(digest), cls._DIM * 2), 2)
        ]
        # Pad or truncate to DIM
        values = (values + [0.0] * cls._DIM)[: cls._DIM]
        return values
