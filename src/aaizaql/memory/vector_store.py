"""
aaizaql.memory.vector_store
───────────────────────────
VectorStoreAdapter: thin abstraction over ChromaDB (dev) and Qdrant (prod).

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Any

import structlog

from aaizaql.core.config import Settings, VectorStoreBackend
from aaizaql.core.exceptions import VectorStoreError

logger = structlog.get_logger(__name__)


@dataclass
class SearchHit:
    """A single result from a vector store search."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStoreAdapter:
    """Unified interface over ChromaDB and Qdrant.

    Swap backends by changing the ``AAIZAQL_VECTOR_STORE`` env var.

    Args:
        settings: Application settings used to configure the backend,
            persistence path, and collection namespace.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._namespace = settings.vector_store_namespace
        self._backend = settings.vector_store
        self._client: Any = None
        self._collection: Any = None
        self._init_backend()

    def _init_backend(self) -> None:
        """Dispatch to the appropriate backend initialiser."""
        if self._backend == VectorStoreBackend.CHROMA:
            self._init_chroma()
        elif self._backend == VectorStoreBackend.QDRANT:
            self._init_qdrant()

    def _init_chroma(self) -> None:
        """Initialise a persistent ChromaDB client and collection.

        Raises:
            VectorStoreError: If ``chromadb`` is not installed.
        """
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._settings.chroma_persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=f"AAIZAQL_{self._namespace}",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("vector_store.chroma.ready", namespace=self._namespace)
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is not installed. "
                "Run: pip install 'aaizaql[rag]'  "
                "(or: pip install chromadb sentence-transformers)"
            ) from exc

    def _init_qdrant(self) -> None:
        """Raise immediately so the user gets a clear message at init time.

        Raises:
            VectorStoreError: Always — Qdrant is not yet implemented.
        """
        raise VectorStoreError(
            "Qdrant vector search is not yet fully implemented. "
            "Use the default ChromaDB backend: AAIZAQL_VECTOR_STORE=chroma"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def upsert(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a document in the vector store.

        Args:
            doc_id: Stable unique identifier for the document.
            text: Raw document text stored alongside the embedding.
            embedding: Pre-computed embedding vector.
            metadata: Optional key-value attributes stored with the document.
        """
        meta = metadata or {}
        if self._backend == VectorStoreBackend.CHROMA:
            self._collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[meta],
            )
        elif self._backend == VectorStoreBackend.QDRANT:
            from qdrant_client.models import PointStruct  # noqa: PLC0415

            self._client.upsert(
                collection_name=f"AAIZAQL_{self._namespace}",
                points=[
                    PointStruct(
                        id=abs(hash(id)) % (2**63), vector=embedding, payload={"text": text, **meta}
                    )
                ],
            )

    def search(
        self,
        query: str,
        filter_type: str | None = None,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[SearchHit]:
        """Semantic search over the vector store.

        Args:
            query: Natural language query string.
            filter_type: Filter by metadata ``type`` field (e.g. ``"ddl"``,
                ``"qa_pair"``). ``None`` returns all types.
            top_k: Maximum number of results to return.
            query_embedding: Pre-computed embedding. Computed internally if
                ``None``.

        Returns:
            List of :class:`SearchHit` objects ordered by descending score.

        Raises:
            NotImplementedError: If the active backend is not ChromaDB.
        """
        if self._backend == VectorStoreBackend.CHROMA:
            return self._search_chroma(query, filter_type, top_k)
        raise NotImplementedError(
            "Qdrant vector search is not yet implemented. "
            "Use the default ChromaDB backend (AAIZAQL_VECTOR_STORE=chroma) "
            "or follow https://github.com/ibrahimkhalilCorp/AaizaQL for Qdrant support."
        )

    def _search_chroma(
        self,
        query: str,
        filter_type: str | None,
        top_k: int,
    ) -> list[SearchHit]:
        """Execute a ChromaDB similarity query and return ranked hits.

        Args:
            query: Natural language query string passed to ChromaDB.
            filter_type: Optional metadata ``type`` filter.
            top_k: Maximum number of results to request.

        Returns:
            List of :class:`SearchHit` objects, or an empty list on error.
        """
        where = {"type": filter_type} if filter_type else None
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, max(1, self._collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("vector_store.search_failed", detail=str(exc)[:80])
            return []

        hits: list[SearchHit] = []
        for doc, meta, dist, id_ in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
            strict=True,
        ):
            hits.append(SearchHit(id=id_, text=doc, score=1 - dist, metadata=meta))
        return hits

    def count(self) -> int:
        """Return the total number of documents in the store.

        Returns:
            Document count as an integer.

        Raises:
            NotImplementedError: If the active backend is not ChromaDB.
        """
        if self._backend == VectorStoreBackend.CHROMA:
            return int(self._collection.count())
        raise NotImplementedError("Qdrant count() is not yet implemented. Use ChromaDB backend.")

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID.

        Args:
            doc_id: Document identifier to remove. Silently logs a warning if
                deletion fails (e.g. the document does not exist).

        Raises:
            NotImplementedError: If the active backend is not ChromaDB.
        """
        if self._backend == VectorStoreBackend.CHROMA:
            try:
                self._collection.delete(ids=[doc_id])
            except Exception as exc:
                logger.warning("vector_store.delete_failed", doc_id=doc_id, detail=str(exc)[:80])
        else:
            raise NotImplementedError("Qdrant delete() is not yet implemented.")

    def list_ids(self, filter_type: str | None = None, namespace: str | None = None) -> set[str]:
        """Return all document IDs, optionally filtered by metadata type.

        Args:
            filter_type: Optional metadata ``type`` filter (e.g. ``"enum"``).
            namespace: Reserved for future Qdrant namespace support; ignored
                by the ChromaDB backend.

        Returns:
            Set of document ID strings matching the filter.

        Raises:
            NotImplementedError: If the active backend is not ChromaDB.
        """
        if self._backend == VectorStoreBackend.CHROMA:
            where = {}
            if filter_type:
                where["type"] = filter_type
            try:
                results = self._collection.get(where=where or None, include=[])
                return set(results["ids"])
            except Exception:
                return set()
        raise NotImplementedError("Qdrant list_ids() is not yet implemented.")
