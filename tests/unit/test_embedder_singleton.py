"""
tests/unit/test_embedder_singleton.py
──────────────────────────────────────
Tests for T2.4 — EmbeddingService must be a true singleton so the
92MB sentence-transformer model is loaded at most once per process,
regardless of how many SchemaIngester or SemanticStore instances exist.

Also verifies that the dead _SentenceEmbedder class has been removed
from ingestion.py and that no second model-load path survives.

No database or live LLM connection required.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from aaizaql.schema.embedder import EmbeddingService


# ── Singleton identity ────────────────────────────────────────────────────────


class TestSingletonIdentity:
    def setup_method(self) -> None:
        """Reset singleton state before each test."""
        EmbeddingService._instance = None
        EmbeddingService._model = None

    def test_get_instance_returns_same_object(self) -> None:
        """Two calls to get_instance() must return the exact same object."""
        a = EmbeddingService.get_instance()
        b = EmbeddingService.get_instance()
        assert a is b, "get_instance() returned two different objects — singleton broken"

    def test_get_instance_called_100_times_is_still_one_object(self) -> None:
        """Stress: 100 calls, all the same instance."""
        instances = {id(EmbeddingService.get_instance()) for _ in range(100)}
        assert len(instances) == 1, f"Expected 1 unique instance, got {len(instances)}"

    def test_model_loaded_exactly_once(self) -> None:
        """SentenceTransformer constructor must be called exactly once even when
        embed() is called many times."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384

        with patch("aaizaql.schema.embedder.EmbeddingService._load") as mock_load:

            def _side_effect(self_inner=None) -> None:
                EmbeddingService._model = mock_model

            mock_load.side_effect = _side_effect

            svc = EmbeddingService.get_instance()
            for _ in range(10):
                svc.embed("some text")

            assert (
                mock_load.call_count == 1
            ), f"_load() called {mock_load.call_count} times — model loaded more than once"


# ── No second embedder class survives in ingestion.py ────────────────────────


class TestDeadCodeRemoved:
    def test_SentenceEmbedder_not_in_ingestion_module(self) -> None:
        """The dead _SentenceEmbedder class must not exist in ingestion.py."""
        import aaizaql.schema.ingestion as ingestion_mod

        assert not hasattr(
            ingestion_mod, "_SentenceEmbedder"
        ), "_SentenceEmbedder still exists in ingestion.py — dead embedder class not removed"

    def test_no_SentenceTransformer_instantiation_in_ingestion_source(self) -> None:
        """ingestion.py source must not contain a SentenceTransformer() call.
        The only legitimate embedder lives in embedder.py."""
        import aaizaql.schema.ingestion as ingestion_mod

        source = inspect.getsource(ingestion_mod)
        assert "SentenceTransformer(" not in source, (
            "ingestion.py still contains a SentenceTransformer() call — "
            "dead embedder class may not have been fully removed"
        )

    def test_SentenceEmbedder_alias_not_in_embedder_module(self) -> None:
        """The _SentenceEmbedder alias must not exist in embedder.py either —
        it was a misleading migration shim that has been removed."""
        import aaizaql.schema.embedder as embedder_mod

        assert not hasattr(
            embedder_mod, "_SentenceEmbedder"
        ), "_SentenceEmbedder alias still present in embedder.py — remove it"

    def test_ingestion_uses_embedding_service_singleton(self) -> None:
        """SchemaIngester.__init__ must fetch EmbeddingService.get_instance(),
        not create any new embedder object."""
        import aaizaql.schema.ingestion as ingestion_mod

        source = inspect.getsource(ingestion_mod)
        assert "EmbeddingService.get_instance()" in source, (
            "SchemaIngester no longer calls EmbeddingService.get_instance() — "
            "check that the singleton wiring is still in place"
        )


# ── Cross-component: one load shared between Ingester and SemanticStore ───────


class TestSharedModelAcrossComponents:
    def setup_method(self) -> None:
        EmbeddingService._instance = None
        EmbeddingService._model = None

    def test_ingester_and_semantic_store_share_same_embedder(self) -> None:
        """SchemaIngester and SemanticStore must both reach the same
        EmbeddingService instance — confirming one model load for both."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = []
        mock_vs.list_ids.return_value = set()

        from aaizaql.schema.ingestion import SchemaIngester
        from aaizaql.schema.semantic_store import SemanticStore

        SchemaIngester(mock_vs)  # instantiation must not raise
        SemanticStore(mock_vs)  # instantiation must not raise

        # Both must reach the exact same EmbeddingService instance
        ingester_svc = EmbeddingService.get_instance()
        store_svc = EmbeddingService.get_instance()

        assert (
            ingester_svc is store_svc
        ), "SchemaIngester and SemanticStore are using different EmbeddingService instances"

    def test_model_load_count_across_ingester_and_semantic_store(self) -> None:
        """Even if both components call embed(), _load() fires only once."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.0] * 384

        load_call_count = 0

        def fake_load(self_inner=None) -> None:
            nonlocal load_call_count
            load_call_count += 1
            EmbeddingService._model = mock_model

        with patch.object(EmbeddingService, "_load", fake_load):
            svc = EmbeddingService.get_instance()
            svc.embed("ingester text")  # simulates SchemaIngester
            svc.embed("semantic text")  # simulates SemanticStore
            svc.embed("another query")

        assert (
            load_call_count == 1
        ), f"Model loaded {load_call_count} times across components — expected exactly 1"
