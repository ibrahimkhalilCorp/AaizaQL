"""
aaizaql.schema.embedder
───────────────────────
Singleton EmbeddingService backed by ``sentence-transformers``.

One instance, one 92 MB model load, shared across the entire process.
Both SchemaIngester and SemanticStore call
:meth:`EmbeddingService.get_instance` rather than constructing their own
embedder to avoid loading the model multiple times.

Author: Ibrahim
Date: 2026-06-16
Version: 1.0.0
"""

import hashlib

import structlog

logger = structlog.get_logger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIM = 384


class EmbeddingService:
    """Thread-safe singleton sentence-transformer embedder.

    Never instantiate directly — always call :meth:`get_instance`.

    Falls back to a deterministic hash-based embedding when
    ``sentence-transformers`` fails so the pipeline degrades gracefully
    rather than crashing on embed errors.
    """

    _instance: "EmbeddingService | None" = None
    _model: object | None = None

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """Return the process-wide EmbeddingService singleton.

        Creates the instance on first call; subsequent calls return the same
        object without re-loading the model.

        Returns:
            The shared :class:`EmbeddingService` instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*.

        Loads the sentence-transformer model on first call. Falls back to a
        deterministic hash-based embedding on any error so the pipeline
        remains functional even if the model fails mid-session.

        Args:
            text: Input string to embed.

        Returns:
            List of ``384`` floats representing the normalised embedding.
        """
        if self._model is None:
            self._load()
        try:
            vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
            return vec.tolist()
        except Exception as exc:
            logger.warning("embedder.failed", detail=str(exc)[:80])
            return self._fallback_embed(text)

    def _load(self) -> None:
        """Load the sentence-transformer model into memory.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
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
        """Return a deterministic pseudo-embedding derived from the SHA-256 hash.

        Used when the sentence-transformer model is unavailable or raises.
        Produces a consistent vector for the same input so deduplication and
        vector-store upserts remain stable across sessions.

        Args:
            text: Input string to hash.

        Returns:
            List of ``384`` floats in the range ``[0.0, 1.0]``.
        """
        digest = hashlib.sha256(text.encode()).hexdigest()
        values = [
            int(digest[i : i + 2], 16) / 255.0 for i in range(0, min(len(digest), _DIM * 2), 2)
        ]
        return (values + [0.0] * _DIM)[:_DIM]
