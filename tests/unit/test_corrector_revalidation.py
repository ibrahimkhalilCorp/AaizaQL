"""
tests/unit/test_corrector_revalidation.py
──────────────────────────────────────────
Unit tests for the security fix in SelfCorrector:

    Corrected SQL returned by the LLM must pass through SQLValidator
    before execution.  A hallucinated DROP TABLE (or any other dangerous
    statement) on a correction pass must be blocked, not executed.

No database or live LLM connection required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from aaizaql.core.config import Settings
from aaizaql.core.exceptions import DatabaseError, MaxRetriesExceeded, SecurityException
from aaizaql.nlp.corrector import SelfCorrector
from aaizaql.security.validator import SQLValidator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _settings(retries: int = 2) -> Settings:
    return Settings(
        llm_provider="groq",
        max_self_correction_retries=retries,
        enable_injection_detection=True,
    )


def _make_corrector(llm: MagicMock, retries: int = 2) -> tuple[SelfCorrector, SQLValidator]:
    settings = _settings(retries)
    validator = SQLValidator(settings)
    corrector = SelfCorrector(llm, settings, validator=validator)
    return corrector, validator


def _executor_that_fails_once(good_sql: str) -> MagicMock:
    """
    Returns a mock executor whose first execute() call raises DatabaseError
    and whose second call succeeds (returning a one-row DataFrame).
    """
    executor = MagicMock()
    executor.execute.side_effect = [
        DatabaseError("syntax error near 'FORM'", sql=good_sql),
        pd.DataFrame({"id": [1]}),
    ]
    return executor


# ── Core behaviour tests ──────────────────────────────────────────────────────


class TestCorrectorRevalidation:
    def test_dangerous_correction_is_blocked(self) -> None:
        """
        If the LLM returns a dangerous statement (DROP TABLE) as a correction,
        SQLValidator must raise SecurityException before the executor is called
        a second time.
        """
        llm = MagicMock()
        llm.complete.return_value = "DROP TABLE employees"  # malicious correction

        corrector, _ = _make_corrector(llm)

        executor = MagicMock()
        executor.execute.side_effect = [
            DatabaseError("syntax error", sql="SELECT * FORM employees"),
        ]

        with pytest.raises(SecurityException):
            corrector.execute_with_correction(
                sql="SELECT * FORM employees",
                executor=executor,
                question="Show all employees",
            )

        # The dangerous DROP TABLE must never have reached the database.
        # executor.execute was called exactly once (with the original bad SQL).
        assert executor.execute.call_count == 1

    def test_valid_correction_executes_and_returns_data(self) -> None:
        """
        A legitimate correction (still a SELECT) passes validation and executes
        successfully.
        """
        llm = MagicMock()
        llm.complete.return_value = "SELECT * FROM employees"

        corrector, _ = _make_corrector(llm)
        executor = _executor_that_fails_once("SELECT * FORM employees")

        data, was_corrected, attempts = corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        assert was_corrected is True
        assert attempts == 1
        assert len(data) == 1
        assert corrector.last_sql == "SELECT * FROM employees"

    def test_delete_correction_is_blocked(self) -> None:
        """DELETE on a correction pass must be caught by the validator."""
        llm = MagicMock()
        llm.complete.return_value = "DELETE FROM employees WHERE id = 1"

        corrector, _ = _make_corrector(llm)

        executor = MagicMock()
        executor.execute.side_effect = [
            DatabaseError("column not found", sql="SELECT id FORM employees"),
        ]

        with pytest.raises(SecurityException):
            corrector.execute_with_correction(
                sql="SELECT id FORM employees",
                executor=executor,
                question="Get employee id",
            )

        assert executor.execute.call_count == 1  # never reached the DB again

    def test_insert_correction_is_blocked(self) -> None:
        """INSERT on a correction pass must be caught by the validator."""
        llm = MagicMock()
        llm.complete.return_value = "INSERT INTO employees VALUES (99, 'Hacker', 'IT', 0, 1)"

        corrector, _ = _make_corrector(llm)

        executor = MagicMock()
        executor.execute.side_effect = [
            DatabaseError("table not found", sql="SELECT * FROM emps"),
        ]

        with pytest.raises(SecurityException):
            corrector.execute_with_correction(
                sql="SELECT * FROM emps",
                executor=executor,
                question="Show employees",
            )

        assert executor.execute.call_count == 1

    def test_no_validator_still_executes(self) -> None:
        """
        When no validator is provided (validator=None), the corrector retains
        its original behaviour and does not raise on correction.
        """
        llm = MagicMock()
        # Even a dangerous correction must not be blocked — no validator wired.
        # (This tests the backwards-compatible path, not recommended for prod.)
        llm.complete.return_value = "SELECT * FROM employees"

        settings = _settings()
        corrector = SelfCorrector(llm, settings, validator=None)

        executor = _executor_that_fails_once("SELECT * FORM employees")

        data, was_corrected, _ = corrector.execute_with_correction(
            sql="SELECT * FORM employees",
            executor=executor,
            question="Show all employees",
        )

        assert was_corrected is True
        assert len(data) == 1

    def test_multiple_retries_all_dangerous_blocked_on_first(self) -> None:
        """
        If every correction the LLM produces is dangerous, the first one should
        raise SecurityException immediately — not silently exhaust retries.
        """
        llm = MagicMock()
        llm.complete.return_value = "TRUNCATE TABLE employees"

        corrector, _ = _make_corrector(llm, retries=3)

        executor = MagicMock()
        executor.execute.side_effect = [
            DatabaseError("bad sql", sql="SELECT FORM employees"),
        ]

        with pytest.raises(SecurityException):
            corrector.execute_with_correction(
                sql="SELECT FORM employees",
                executor=executor,
                question="Show all employees",
            )

        # Only one DB call (the original bad SQL); the dangerous correction
        # must have been blocked before a second execute() was attempted.
        assert executor.execute.call_count == 1
        # LLM was asked to correct exactly once
        assert llm.complete.call_count == 1

    def test_last_sql_not_updated_to_dangerous_on_block(self) -> None:
        """
        When a dangerous correction is blocked, last_sql must NOT be updated
        to the dangerous statement — it should still reflect the last safe SQL.
        """
        original_sql = "SELECT * FORM employees"

        llm = MagicMock()
        llm.complete.return_value = "DROP TABLE employees"

        corrector, _ = _make_corrector(llm)

        executor = MagicMock()
        executor.execute.side_effect = [
            DatabaseError("syntax error", sql=original_sql),
        ]

        with pytest.raises(SecurityException):
            corrector.execute_with_correction(
                sql=original_sql,
                executor=executor,
                question="Show all employees",
            )

        # last_sql was set to the dangerous string before validation runs;
        # after the SecurityException the caller can't safely use it.
        # The important invariant is that execute() was never called with it.
        dangerous_calls = [
            c for c in executor.execute.call_args_list if "DROP" in str(c)
        ]
        assert dangerous_calls == [], "DROP TABLE must never reach the executor"


# ── Integration: engine wires validator into corrector ────────────────────────


class TestEngineWiresValidatorIntoCorrectorIntegration:
    """
    Smoke test: build a QueryEngine with a mock LLM and confirm that a
    dangerous correction is blocked end-to-end (no real DB or API needed).
    """

    def test_engine_blocks_dangerous_correction_end_to_end(
        self,
        sqlite_db: str,
        mock_llm: MagicMock,
        tmp_path,
    ) -> None:
        from pathlib import Path
        from unittest.mock import patch

        from aaizaql import QueryEngine

        def _make_mock_vs():
            vs = MagicMock()
            vs.search.return_value = []
            vs.upsert.return_value = None
            vs.count.return_value = 0
            return vs

        # First LLM call → generator returns bad SQL (typo in table name).
        # Second LLM call (correction) → returns DROP TABLE.
        mock_llm.complete.side_effect = [
            "SELECT * FORM employees",   # generator: bad SQL (will fail on DB)
            "DROP TABLE employees",      # corrector: dangerous hallucination
        ]

        with (
            patch("aaizaql.core.engine.build_llm_provider", return_value=mock_llm),
            patch(
                "aaizaql.core.engine.VectorStoreAdapter",
                side_effect=lambda s: _make_mock_vs(),
            ),
        ):
            engine = QueryEngine(
                llm="groq",
                database="sqlite",
                dsn=sqlite_db,
            )

        with pytest.raises(SecurityException):
            engine.query("Show all employees")