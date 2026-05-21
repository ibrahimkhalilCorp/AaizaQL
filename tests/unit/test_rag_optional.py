"""
tests/unit/test_rag_optional.py
────────────────────────────────
Unit tests for task 0.6 — chromadb and sentence-transformers are optional.

Verifies:
- chromadb and sentence-transformers are NOT in core dependencies
- VectorStoreAdapter raises VectorStoreError (not ImportError) with a clear
  message referencing aaizaql[rag] when chromadb is missing
- _SentenceEmbedder._load() raises ImportError with aaizaql[rag] hint
  when sentence-transformers is missing (no silent fallback)
- pyproject.toml declares a [rag] optional group containing both packages
No API keys or network calls required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── pyproject.toml structure ──────────────────────────────────────────────────


class TestPyprojectToml:
    def _load_toml(self) -> dict:
        from pathlib import Path

        import tomllib

        root = Path(__file__).parent.parent.parent
        toml_path = root / "pyproject.toml"
        with toml_path.open("rb") as f:
            return tomllib.load(f)

    def test_chromadb_not_in_core_dependencies(self) -> None:
        data = self._load_toml()
        core_deps = data["project"]["dependencies"]
        assert not any("chromadb" in dep for dep in core_deps), (
            "chromadb should not be in core dependencies"
        )

    def test_sentence_transformers_not_in_core_dependencies(self) -> None:
        data = self._load_toml()
        core_deps = data["project"]["dependencies"]
        assert not any("sentence-transformers" in dep for dep in core_deps), (
            "sentence-transformers should not be in core dependencies"
        )

    def test_rag_optional_group_exists(self) -> None:
        data = self._load_toml()
        optional = data["project"]["optional-dependencies"]
        assert "rag" in optional, "Missing [rag] optional dependency group"

    def test_rag_group_contains_chromadb(self) -> None:
        data = self._load_toml()
        rag_deps = data["project"]["optional-dependencies"]["rag"]
        assert any("chromadb" in dep for dep in rag_deps), (
            "chromadb missing from [rag] optional group"
        )

    def test_rag_group_contains_sentence_transformers(self) -> None:
        data = self._load_toml()
        rag_deps = data["project"]["optional-dependencies"]["rag"]
        assert any("sentence-transformers" in dep for dep in rag_deps), (
            "sentence-transformers missing from [rag] optional group"
        )

    def test_all_group_contains_chromadb(self) -> None:
        data = self._load_toml()
        all_deps = data["project"]["optional-dependencies"]["all"]
        assert any("chromadb" in dep for dep in all_deps), (
            "chromadb missing from [all] optional group"
        )

    def test_all_group_contains_sentence_transformers(self) -> None:
        data = self._load_toml()
        all_deps = data["project"]["optional-dependencies"]["all"]
        assert any("sentence-transformers" in dep for dep in all_deps), (
            "sentence-transformers missing from [all] optional group"
        )

    def test_dev_group_contains_chromadb(self) -> None:
        data = self._load_toml()
        dev_deps = data["project"]["optional-dependencies"]["dev"]
        assert any("chromadb" in dep for dep in dev_deps)

    def test_dev_group_contains_sentence_transformers(self) -> None:
        data = self._load_toml()
        dev_deps = data["project"]["optional-dependencies"]["dev"]
        assert any("sentence-transformers" in dep for dep in dev_deps)


# ── VectorStoreAdapter missing chromadb ───────────────────────────────────────


class TestVectorStoreAdapterMissingChromadb:
    def test_raises_vector_store_error_not_import_error(self) -> None:
        from aaizaql.core.config import Settings
        from aaizaql.core.exceptions import VectorStoreError
        from aaizaql.memory.vector_store import VectorStoreAdapter

        settings = Settings(vector_store="chroma")

        with patch.dict(sys.modules, {"chromadb": None}):
            with pytest.raises(VectorStoreError) as exc_info:
                VectorStoreAdapter(settings)

        assert not isinstance(exc_info.value.__cause__, type(None))

    def test_error_message_mentions_rag_extra(self) -> None:
        from aaizaql.core.config import Settings
        from aaizaql.core.exceptions import VectorStoreError
        from aaizaql.memory.vector_store import VectorStoreAdapter

        settings = Settings(vector_store="chroma")

        with patch.dict(sys.modules, {"chromadb": None}):
            with pytest.raises(VectorStoreError) as exc_info:
                VectorStoreAdapter(settings)

        msg = str(exc_info.value)
        assert "rag" in msg.lower(), f"Expected 'rag' in error message, got: {msg}"
        assert "pip install" in msg, f"Expected install instruction in error message, got: {msg}"

    def test_error_message_mentions_chromadb(self) -> None:
        from aaizaql.core.config import Settings
        from aaizaql.core.exceptions import VectorStoreError
        from aaizaql.memory.vector_store import VectorStoreAdapter

        settings = Settings(vector_store="chroma")

        with patch.dict(sys.modules, {"chromadb": None}):
            with pytest.raises(VectorStoreError) as exc_info:
                VectorStoreAdapter(settings)

        assert "chromadb" in str(exc_info.value)


# ── _SentenceEmbedder missing sentence-transformers ───────────────────────────


class TestSentenceEmbedderMissingSentenceTransformers:
    def _make_embedder(self) -> "_SentenceEmbedder":  # noqa: F821
        from aaizaql.schema.ingestion import _SentenceEmbedder

        return _SentenceEmbedder()

    def test_raises_import_error_when_missing(self) -> None:
        embedder = self._make_embedder()
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                embedder._load()

        msg = str(exc_info.value)
        assert "sentence-transformers" in msg

    def test_error_message_mentions_rag_extra(self) -> None:
        embedder = self._make_embedder()
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                embedder._load()

        msg = str(exc_info.value)
        assert "rag" in msg.lower(), f"Expected 'rag' in error message, got: {msg}"
        assert "pip install" in msg

    def test_no_silent_fallback_on_import_error(self) -> None:
        """embed() must propagate the ImportError — no silent hash fallback."""
        embedder = self._make_embedder()
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError):
                embedder.embed("SELECT 1")

    def test_fallback_embed_still_works_for_runtime_errors(self) -> None:
        """_fallback_embed() itself must still produce a vector (used for encode() failures)."""
        from aaizaql.schema.ingestion import _SentenceEmbedder

        vec = _SentenceEmbedder._fallback_embed("some text")
        assert isinstance(vec, list)
        assert len(vec) == _SentenceEmbedder._DIM
        assert all(isinstance(v, float) for v in vec)

    def test_embed_falls_back_when_encode_raises_at_runtime(self) -> None:
        """If the model is loaded but encode() fails (e.g. OOM), fallback is used."""
        embedder = self._make_embedder()
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("out of memory")
        embedder._model = mock_model

        result = embedder.embed("test text")
        assert isinstance(result, list)
        assert len(result) == embedder._DIM