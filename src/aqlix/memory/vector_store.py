"""
aqlix.memory.vector_store
──────────────────────────
VectorStoreAdapter: thin abstraction over ChromaDB (dev) and Qdrant (prod).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aqlix.core.config import Settings, VectorStoreBackend
from aqlix.core.exceptions import VectorStoreError
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SearchHit:
    """A single result from a vector store search."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStoreAdapter:
    """
    Unified interface over ChromaDB and Qdrant.
    Swap backends by changing AQLIX_VECTOR_STORE env var.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._namespace = settings.vector_store_namespace
        self._backend = settings.vector_store
        self._client: Any = None
        self._collection: Any = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self._backend == VectorStoreBackend.CHROMA:
            self._init_chroma()
        elif self._backend == VectorStoreBackend.QDRANT:
            self._init_qdrant()

    def _init_chroma(self) -> None:
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._settings.chroma_persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=f"aqlix_{self._namespace}",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("vector_store.chroma.ready", namespace=self._namespace)
        except ImportError as exc:
            raise VectorStoreError("chromadb is not installed. Run: pip install chromadb") from exc

    def _init_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient

            api_key = (
                self._settings.qdrant_api_key.get_secret_value()
                if self._settings.qdrant_api_key
                else None
            )
            self._client = QdrantClient(
                url=self._settings.qdrant_url,
                api_key=api_key,
            )
            logger.info("vector_store.qdrant.ready", url=self._settings.qdrant_url)
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client is not installed. Run: pip install qdrant-client"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def upsert(
        self,
        id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a document in the vector store."""
        meta = metadata or {}
        if self._backend == VectorStoreBackend.CHROMA:
            self._collection.upsert(
                ids=[id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[meta],
            )
        elif self._backend == VectorStoreBackend.QDRANT:
            from qdrant_client.models import PointStruct

            self._client.upsert(
                collection_name=f"aqlix_{self._namespace}",
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
        """
        Semantic search over the vector store.

        Parameters
        ----------
        query          : str            Natural language query string.
        filter_type    : str | None     Filter by metadata 'type' field (e.g. "ddl", "qa_pair").
        top_k          : int            Maximum results to return.
        query_embedding: list | None    Pre-computed embedding (computed here if None).
        """
        if self._backend == VectorStoreBackend.CHROMA:
            return self._search_chroma(query, filter_type, top_k)
        return []  # Qdrant path — implement when Qdrant is configured

    def _search_chroma(
        self,
        query: str,
        filter_type: str | None,
        top_k: int,
    ) -> list[SearchHit]:
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
        ):
            hits.append(SearchHit(id=id_, text=doc, score=1 - dist, metadata=meta))
        return hits

    def count(self) -> int:
        """Return the total number of documents in the store."""
        if self._backend == VectorStoreBackend.CHROMA:
            return int(self._collection.count())
        return 0
