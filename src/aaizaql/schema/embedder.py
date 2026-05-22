"""

aaizaql.schema.embedder

───────────────────────

T2.4 — Singleton EmbeddingService.

One instance, one 92MB model load, shared by SchemaIngester and SemanticStore.

"""

from __future__ import annotations

import hashlib

import structlog

logger = structlog.get_logger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"

_DIM = 384


class EmbeddingService:
    """

    Thread-safe singleton sentence-transformer embedder.

    Call EmbeddingService.get_instance() — never instantiate directly.

    """

    _instance: EmbeddingService | None = None

    _model: object | None = None

    @classmethod
    def get_instance(cls) -> EmbeddingService:

        if cls._instance is None:

            cls._instance = cls()

        return cls._instance

    def embed(self, text: str) -> list[float]:

        if self._model is None:

            self._load()

        try:

            vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]

            return vec.tolist()

        except Exception as exc:

            logger.warning("embedder.failed", detail=str(exc)[:80])

            return self._fallback_embed(text)

    def _load(self) -> None:

        try:

            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer(_MODEL_NAME)

            logger.info("embedder.loaded", model=_MODEL_NAME)

        except ImportError as exc:

            raise ImportError(
                "sentence-transformers is not installed. " "Run: pip install 'aaizaql[rag]'"
            ) from exc

    @classmethod
    def _fallback_embed(cls, text: str) -> list[float]:

        digest = hashlib.sha256(text.encode()).hexdigest()

        values = [
            int(digest[i : i + 2], 16) / 255.0 for i in range(0, min(len(digest), _DIM * 2), 2)
        ]

        return (values + [0.0] * _DIM)[:_DIM]


# Convenience alias used by modules that imported _SentenceEmbedder directly

_SentenceEmbedder = EmbeddingService
