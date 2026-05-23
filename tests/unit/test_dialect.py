"""
tests/unit/test_dialect.py
──────────────────────────
Unit tests for T1.1 — SQL dialect must come from connector.name,
never from the LLM provider name.

No database or live LLM connection required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aaizaql.core.config import Settings
from aaizaql.nlp.generator import SQLGenerator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _settings() -> Settings:
    return Settings(llm_provider="groq")  # type: ignore[arg-type]


def _mock_connector(name: str = "sqlite") -> MagicMock:
    connector = MagicMock()
    connector.name = name
    return connector


def _mock_llm(response: str = "SELECT 1") -> MagicMock:
    llm = MagicMock()
    llm.name = "groq"
    llm.complete.return_value = response
    return llm


def _mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.search.return_value = []
    return vs


def _make_generator(connector_name: str = "sqlite") -> SQLGenerator:
    return SQLGenerator(
        llm=_mock_llm(),
        vector_store=_mock_vector_store(),
        settings=_settings(),
        semantic_store=None,
        connector=_mock_connector(connector_name),
    )


# ── Construction-time guards ──────────────────────────────────────────────────


class TestDialectConstructionGuards:
    def test_raises_if_connector_is_none(self) -> None:
        """SQLGenerator must not accept connector=None — dialect would be unknowable."""
        with pytest.raises(ValueError, match="requires a connector instance"):
            SQLGenerator(
                llm=_mock_llm(),
                vector_store=_mock_vector_store(),
                settings=_settings(),
                connector=None,
            )

    def test_raises_if_connector_name_is_empty(self) -> None:
        """A connector with name='' must be caught at construction time."""
        bad_connector = MagicMock()
        bad_connector.name = ""
        with pytest.raises(ValueError, match="empty or missing"):
            SQLGenerator(
                llm=_mock_llm(),
                vector_store=_mock_vector_store(),
                settings=_settings(),
                connector=bad_connector,
            )

    def test_raises_if_connector_has_no_name_attr(self) -> None:
        """A connector missing the name attribute entirely must be caught."""
        bad_connector = MagicMock(spec=[])  # spec=[] → no attributes at all
        with pytest.raises(ValueError, match="empty or missing"):
            SQLGenerator(
                llm=_mock_llm(),
                vector_store=_mock_vector_store(),
                settings=_settings(),
                connector=bad_connector,
            )

    def test_valid_connector_constructs_successfully(self) -> None:
        """A properly named connector should construct without error."""
        gen = _make_generator("postgresql")
        assert gen is not None


# ── Dialect reaches the prompt ────────────────────────────────────────────────


class TestDialectInPrompt:
    """
    Verify that connector.name (not llm_provider) is what the LLM receives
    as the dialect hint in the prompt.
    """

    @pytest.mark.parametrize(
        "connector_name, llm_provider",
        [
            ("sqlite", "groq"),  # the classic mismatch that caused the bug
            ("postgresql", "claude"),  # different provider, different dialect
            ("mysql", "openai"),
            ("snowflake", "gemini"),
            ("duckdb", "groq"),
        ],
    )
    def test_prompt_contains_connector_dialect_not_llm_provider(
        self, connector_name: str, llm_provider: str
    ) -> None:
        settings = Settings(llm_provider=llm_provider)  # type: ignore[arg-type]
        connector = _mock_connector(connector_name)
        llm = _mock_llm("SELECT 1")

        gen = SQLGenerator(
            llm=llm,
            vector_store=_mock_vector_store(),
            settings=settings,
            connector=connector,
        )
        gen.generate("show all rows", history=[])

        # The prompt passed to llm.complete must contain the connector dialect
        call_args = llm.complete.call_args  # type: ignore[attr-defined]
        prompt: str = call_args[0][0] if call_args[0] else call_args[1]["prompt"]

        assert f"dialect: {connector_name}" in prompt, (
            f"Expected 'dialect: {connector_name}' in prompt, "
            f"but got llm_provider '{llm_provider}' or something else.\n"
            f"Prompt snippet: {prompt[:300]}"
        )
        assert f"dialect: {llm_provider}" not in prompt, (
            f"LLM provider name '{llm_provider}' must never appear as the dialect. "
            f"Prompt snippet: {prompt[:300]}"
        )

    def test_sqlite_dialect_not_confused_with_groq(self) -> None:
        """Explicit regression test for the original bug report scenario."""
        gen = _make_generator("sqlite")
        llm = gen._llm

        gen.generate("how many users?", history=[])

        prompt = llm.complete.call_args[0][0]  # type: ignore[attr-defined]
        assert "dialect: sqlite" in prompt
        assert "dialect: groq" not in prompt

    def test_postgresql_dialect_not_confused_with_claude(self) -> None:
        """Regression: postgresql connector with claude LLM must show 'postgresql'."""
        settings = Settings(llm_provider="claude")  # type: ignore[arg-type]
        llm = _mock_llm("SELECT COUNT(*) FROM users")
        gen = SQLGenerator(
            llm=llm,
            vector_store=_mock_vector_store(),
            settings=settings,
            connector=_mock_connector("postgresql"),
        )
        gen.generate("count users", history=[])

        prompt = llm.complete.call_args[0][0]  # type: ignore[attr-defined]
        assert "dialect: postgresql" in prompt
        assert "dialect: claude" not in prompt
