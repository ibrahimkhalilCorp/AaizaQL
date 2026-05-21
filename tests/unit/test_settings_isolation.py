"""
tests/unit/test_settings_isolation.py
──────────────────────────────────────
Unit tests for task 0.7 — per-engine Settings isolation.

Verifies:
- make_settings() produces independent Settings instances
- Two QueryEngine instances with different configs don't share state
- QueryEngine never mutates the module-level singleton
- The singleton is still importable for backwards-compat but is isolated
- Settings passed explicitly via settings= kwarg are used as-is
- make_settings is exported from the top-level package
No API keys or network calls required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── make_settings factory ─────────────────────────────────────────────────────


class TestMakeSettings:
    def test_returns_settings_instance(self) -> None:
        from aaizaql.core.config import Settings, make_settings

        s = make_settings()
        assert isinstance(s, Settings)

    def test_applies_overrides(self) -> None:
        from aaizaql.core.config import make_settings

        s = make_settings(llm_timeout_seconds=99)
        assert s.llm_timeout_seconds == 99

    def test_each_call_returns_independent_instance(self) -> None:
        from aaizaql.core.config import make_settings

        s1 = make_settings(llm_timeout_seconds=10)
        s2 = make_settings(llm_timeout_seconds=20)
        assert s1 is not s2
        assert s1.llm_timeout_seconds == 10
        assert s2.llm_timeout_seconds == 20

    def test_does_not_mutate_singleton(self) -> None:
        from aaizaql.core.config import make_settings
        from aaizaql.core.config import settings as singleton

        original_timeout = singleton.llm_timeout_seconds
        make_settings(llm_timeout_seconds=original_timeout + 100)
        # singleton must be unchanged
        assert singleton.llm_timeout_seconds == original_timeout

    def test_exported_from_top_level_package(self) -> None:
        from aaizaql import make_settings  # noqa: F401 — just assert importable

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AAIZAQL_LLM_TIMEOUT_SECONDS", "77")
        from aaizaql.core.config import make_settings

        s = make_settings()
        assert s.llm_timeout_seconds == 77

    def test_kwarg_wins_over_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AAIZAQL_LLM_TIMEOUT_SECONDS", "77")
        from aaizaql.core.config import make_settings

        s = make_settings(llm_timeout_seconds=55)
        assert s.llm_timeout_seconds == 55


# ── QueryEngine per-instance isolation ───────────────────────────────────────


def _make_engine(llm: str = "groq", timeout: int = 30, **extra) -> QueryEngine:  # noqa: F821
    """Build a QueryEngine with all heavy components mocked out."""
    from aaizaql.core.engine import QueryEngine

    with (
        patch("aaizaql.core.engine.build_llm_provider", return_value=MagicMock()),
        patch("aaizaql.core.engine.VectorStoreAdapter", return_value=MagicMock()),
        patch("aaizaql.core.engine.SchemaIngester", return_value=MagicMock()),
        patch("aaizaql.core.engine.SemanticStore", return_value=MagicMock()),
        patch("aaizaql.core.engine.SQLGenerator", return_value=MagicMock()),
        patch("aaizaql.core.engine.SQLValidator", return_value=MagicMock()),
        patch("aaizaql.core.engine.SelfCorrector", return_value=MagicMock()),
        patch("aaizaql.core.engine.NLSummarizer", return_value=MagicMock()),
        patch("aaizaql.core.engine.ResultRenderer", return_value=MagicMock()),
        patch("aaizaql.core.engine.ContextManager", return_value=MagicMock()),
        patch("aaizaql.connectors.REGISTRY", {"sqlite": MagicMock(return_value=MagicMock())}),
    ):
        return QueryEngine(
            llm=llm,
            database="sqlite",
            dsn="sqlite:///:memory:",
            llm_timeout_seconds=timeout,
            **extra,
        )


class TestQueryEngineSettingsIsolation:
    def test_two_engines_have_independent_settings(self) -> None:
        e1 = _make_engine(timeout=10)
        e2 = _make_engine(timeout=60)
        assert e1._settings is not e2._settings
        assert e1._settings.llm_timeout_seconds == 10
        assert e2._settings.llm_timeout_seconds == 60

    def test_engine_does_not_share_singleton(self) -> None:
        from aaizaql.core.config import settings as singleton

        engine = _make_engine(timeout=299)
        assert engine._settings is not singleton

    def test_engine_does_not_mutate_singleton(self) -> None:
        from aaizaql.core.config import settings as singleton

        original = singleton.llm_timeout_seconds
        # Use a value different from original but within valid range
        different = 299 if original != 299 else 298
        _make_engine(timeout=different)
        assert singleton.llm_timeout_seconds == original

    def test_explicit_settings_kwarg_used_as_is(self) -> None:
        from aaizaql.core.config import make_settings
        from aaizaql.core.engine import QueryEngine

        explicit = make_settings(llm_timeout_seconds=42)

        with (
            patch("aaizaql.core.engine.build_llm_provider", return_value=MagicMock()),
            patch("aaizaql.core.engine.VectorStoreAdapter", return_value=MagicMock()),
            patch("aaizaql.core.engine.SchemaIngester", return_value=MagicMock()),
            patch("aaizaql.core.engine.SemanticStore", return_value=MagicMock()),
            patch("aaizaql.core.engine.SQLGenerator", return_value=MagicMock()),
            patch("aaizaql.core.engine.SQLValidator", return_value=MagicMock()),
            patch("aaizaql.core.engine.SelfCorrector", return_value=MagicMock()),
            patch("aaizaql.core.engine.NLSummarizer", return_value=MagicMock()),
            patch("aaizaql.core.engine.ResultRenderer", return_value=MagicMock()),
            patch("aaizaql.core.engine.ContextManager", return_value=MagicMock()),
            patch("aaizaql.connectors.REGISTRY", {"sqlite": MagicMock(return_value=MagicMock())}),
        ):
            engine = QueryEngine(
                llm="groq",
                database="sqlite",
                dsn="sqlite:///:memory:",
                settings=explicit,
            )

        assert engine._settings is explicit
        assert engine._settings.llm_timeout_seconds == 42

    def test_make_settings_kwargs_reach_engine(self) -> None:
        engine = _make_engine(timeout=77)
        assert engine._settings.llm_timeout_seconds == 77

    def test_engines_in_parallel_have_independent_settings(self) -> None:
        """Simulate concurrent engine creation — each must get its own settings."""
        engines = [_make_engine(timeout=i * 10) for i in range(1, 6)]
        timeouts = [e._settings.llm_timeout_seconds for e in engines]
        assert timeouts == [10, 20, 30, 40, 50]


# ── Singleton backwards-compat ────────────────────────────────────────────────


class TestSingletonBackwardsCompat:
    def test_singleton_still_importable(self) -> None:
        from aaizaql.core.config import settings  # noqa: F401

    def test_singleton_is_settings_instance(self) -> None:
        from aaizaql.core.config import Settings, settings

        assert isinstance(settings, Settings)

    def test_singleton_is_stable_across_imports(self) -> None:
        from aaizaql.core.config import settings as s1
        from aaizaql.core.config import settings as s2

        assert s1 is s2