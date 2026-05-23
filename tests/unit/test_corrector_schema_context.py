"""
tests/unit/test_corrector_schema_context.py
────────────────────────────────────────────
Unit tests for the schema-context fix in SelfCorrector:

    Correction attempts must include the database dialect, enum mappings,
    and relevant documentation in the prompt sent to the LLM — the same
    enrichment the initial SQLGenerator provides — so the LLM has full
    context when it rewrites a broken query.

No database or live LLM connection required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import DatabaseError
from aaizaql.nlp.corrector import SelfCorrector


# ── Helpers ───────────────────────────────────────────────────────────────────


def _settings(retries: int = 1) -> Settings:
    return Settings(
        llm_provider="groq",  # type: ignore[arg-type]
        max_self_correction_retries=retries,
        schema_top_k=3,
    )


def _make_vector_store(schema_text: str = "TABLE employees (id INT, status INT)") -> MagicMock:
    hit = MagicMock()
    hit.text = schema_text
    vs = MagicMock()
    vs.search.return_value = [hit]
    return vs


def _make_semantic_store(
    has_enums: bool = False,
    enum_block: str = "",
    docs: str = "",
) -> MagicMock:
    sem = MagicMock()
    sem.has_enums.return_value = has_enums
    sem.get_enum_block.return_value = enum_block
    sem.search_documentation.return_value = docs
    return sem


def _executor_fails_once(corrected_sql: str = "SELECT * FROM employees") -> MagicMock:
    executor = MagicMock()
    executor.execute.side_effect = [
        DatabaseError("syntax error near 'FORM'", sql="SELECT * FORM employees"),
        pd.DataFrame({"id": [1]}),
    ]
    return executor


# ── Tests: dialect in correction prompt ───────────────────────────────────────


class TestDialectInCorrectionPrompt:
    def test_dialect_label_forwarded_to_llm(self) -> None:
        """The dialect string must appear in the prompt sent to the LLM."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            dialect="postgresql",
        )
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        prompt_sent = llm.complete.call_args[0][0]
        assert (
            "postgresql" in prompt_sent
        ), "Dialect 'postgresql' must appear in the correction prompt"

    def test_sqlite_dialect_forwarded(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            dialect="sqlite",
        )
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        prompt_sent = llm.complete.call_args[0][0]
        assert "sqlite" in prompt_sent

    def test_no_dialect_does_not_crash(self) -> None:
        """Omitting dialect= is backward-compatible — corrector still runs."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            # dialect not passed — defaults to ""
        )
        executor = _executor_fails_once()

        data, was_corrected, _ = corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        assert was_corrected is True
        assert len(data) == 1


# ── Tests: schema chunks in correction prompt ─────────────────────────────────


class TestSchemaChunksInCorrectionPrompt:
    def test_schema_chunk_text_in_prompt(self) -> None:
        """Schema DDL retrieved from the vector store must appear in the prompt."""
        schema_ddl = "TABLE orders (order_id INT PRIMARY KEY, amount DECIMAL)"
        llm = MagicMock()
        llm.complete.return_value = "SELECT order_id FROM orders"

        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(schema_ddl),
            dialect="sqlite",
        )
        executor = _executor_fails_once("SELECT order_id FROM orders")

        corrector.execute_with_correction(
            sql="SELECT order_id FORM orders",
            executor=executor,
            question="List all orders",
        )

        prompt_sent = llm.complete.call_args[0][0]
        assert "orders" in prompt_sent
        assert "order_id" in prompt_sent

    def test_schema_unavailable_fallback_text(self) -> None:
        """When vector_store is None the prompt should note schema is unavailable."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=None,  # no vector store wired
        )
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show employees",
        )

        prompt_sent = llm.complete.call_args[0][0]
        assert "unavailable" in prompt_sent.lower()

    def test_vector_store_searched_with_question(self) -> None:
        """Vector store must be queried with the original question, not the SQL."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        vs = _make_vector_store()
        corrector = SelfCorrector(llm, _settings(), vector_store=vs, dialect="sqlite")
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        vs.search.assert_called_once()
        search_call_kwargs = vs.search.call_args
        # first positional arg or 'query' kwarg must be the question
        actual_query = search_call_kwargs[1].get("query") or search_call_kwargs[0][0]
        assert "employees" in actual_query.lower()


# ── Tests: enum block in correction prompt ────────────────────────────────────


class TestEnumBlockInCorrectionPrompt:
    def test_enum_mappings_injected_when_present(self) -> None:
        """Enum mappings must appear in the correction prompt when defined."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees WHERE status = 1"

        enum_text = "employees.status: 1=Active, 2=Resigned, 3=Terminated"
        sem = _make_semantic_store(has_enums=True, enum_block=enum_text)

        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            semantic_store=sem,
            dialect="sqlite",
        )
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FROM employees WHERE status = 'active'",
            executor=executor,
            question="Show active employees",
        )

        prompt_sent = llm.complete.call_args[0][0]
        assert enum_text in prompt_sent, "Enum block must be present in the correction prompt"

    def test_no_enum_block_when_no_enums_defined(self) -> None:
        """When no enums are registered the correction prompt must not include the section."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        sem = _make_semantic_store(has_enums=False)
        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            semantic_store=sem,
            dialect="sqlite",
        )
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        prompt_sent = llm.complete.call_args[0][0]
        assert "COLUMN CODE MAPPINGS" not in prompt_sent

    def test_enum_block_not_injected_when_no_semantic_store(self) -> None:
        """Without a semantic_store the prompt must still be valid (no KeyError)."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            semantic_store=None,
        )
        executor = _executor_fails_once()

        data, was_corrected, _ = corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        assert was_corrected is True


# ── Tests: documentation block in correction prompt ───────────────────────────


class TestDocBlockInCorrectionPrompt:
    def test_documentation_injected_when_available(self) -> None:
        """Relevant documentation from the semantic store must appear in the correction prompt."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        doc_text = "Use strftime('%Y-%m', hire_date) for SQLite month grouping."
        sem = _make_semantic_store(docs=doc_text)

        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            semantic_store=sem,
            dialect="sqlite",
        )
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Employees hired this month",
        )

        prompt_sent = llm.complete.call_args[0][0]
        assert doc_text in prompt_sent

    def test_no_doc_block_when_docs_empty(self) -> None:
        """When search_documentation returns empty string, the doc section is omitted."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        sem = _make_semantic_store(docs="")  # no relevant docs
        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            semantic_store=sem,
        )
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        prompt_sent = llm.complete.call_args[0][0]
        assert "BUSINESS CONTEXT" not in prompt_sent

    def test_documentation_search_uses_question(self) -> None:
        """search_documentation must be called with the original question."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        sem = _make_semantic_store(docs="some context")
        corrector = SelfCorrector(
            llm,
            _settings(),
            vector_store=_make_vector_store(),
            semantic_store=sem,
        )
        executor = _executor_fails_once()

        corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        sem.search_documentation.assert_called_once()
        call_args = sem.search_documentation.call_args
        question_arg = call_args[0][0] if call_args[0] else call_args[1].get("question", "")
        assert "employees" in question_arg.lower()


# ── Tests: backward-compatibility ────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_original_two_arg_construction_still_works(self) -> None:
        """SelfCorrector(llm, settings) with no extra kwargs must not crash."""
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"
        settings = _settings()

        corrector = SelfCorrector(llm, settings)
        executor = _executor_fails_once()

        data, was_corrected, _ = corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        assert was_corrected is True
        assert len(data) == 1

    def test_context_on_every_retry(self) -> None:
        """Each retry attempt (not just the first) should fetch fresh schema context."""
        llm = MagicMock()
        # First correction still produces bad SQL; second correction succeeds.
        llm.complete.side_effect = [
            "SELECT * FORM employees",  # still bad
            "SELECT * FROM employees",  # correct
        ]

        vs = _make_vector_store()
        corrector = SelfCorrector(
            llm,
            _settings(retries=2),
            vector_store=vs,
            dialect="sqlite",
        )

        executor = MagicMock()
        executor.execute.side_effect = [
            DatabaseError("syntax error", sql="SELECT * FORM employees"),
            DatabaseError("syntax error again", sql="SELECT * FORM employees"),
            pd.DataFrame({"id": [1]}),
        ]

        data, was_corrected, _ = corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        assert was_corrected is True
        # Vector store must have been searched once per retry (2 total).
        assert vs.search.call_count == 2
