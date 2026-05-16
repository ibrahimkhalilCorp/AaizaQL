"""
tests/test_nlp.py
──────────────────
Unit tests for SelfCorrector and SQL parsing logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from aqlix.core.config import Settings
from aqlix.core.exceptions import DatabaseError, MaxRetriesExceeded
from aqlix.nlp.corrector import SelfCorrector


@pytest.fixture()
def settings() -> Settings:
    return Settings(llm_provider="groq", max_self_correction_retries=2)


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.name = "mock/test"
    return llm


class TestSelfCorrector:
    def test_success_on_first_attempt(
        self, mock_llm: MagicMock, settings: Settings
    ) -> None:
        corrector = SelfCorrector(mock_llm, settings)
        mock_executor = MagicMock()
        mock_executor.execute.return_value = pd.DataFrame({"x": [1, 2]})

        df, was_corrected, attempts = corrector.execute_with_correction(
            sql="SELECT x FROM t",
            executor=mock_executor,
            question="show x",
        )
        assert len(df) == 2
        assert was_corrected is False
        assert attempts == 0

    def test_correction_on_first_failure(
        self, mock_llm: MagicMock, settings: Settings
    ) -> None:
        corrector = SelfCorrector(mock_llm, settings)
        mock_executor = MagicMock()
        good_df = pd.DataFrame({"x": [42]})

        # First call fails, second succeeds
        mock_executor.execute.side_effect = [
            DatabaseError("no such table", sql="SELECT x FROM t"),
            good_df,
        ]
        mock_llm.complete.return_value = "SELECT x FROM existing_table"

        df, was_corrected, attempts = corrector.execute_with_correction(
            sql="SELECT x FROM t",
            executor=mock_executor,
            question="show x",
        )
        assert len(df) == 1
        assert was_corrected is True
        assert attempts == 1

    def test_raises_after_max_retries(
        self, mock_llm: MagicMock, settings: Settings
    ) -> None:
        corrector = SelfCorrector(mock_llm, settings)
        mock_executor = MagicMock()
        mock_executor.execute.side_effect = DatabaseError("always fails")
        mock_llm.complete.return_value = "SELECT 1"  # LLM keeps trying

        with pytest.raises(MaxRetriesExceeded):
            corrector.execute_with_correction(
                sql="SELECT * FROM bad_table",
                executor=mock_executor,
                question="bad query",
            )

    def test_last_sql_updated_after_correction(
        self, mock_llm: MagicMock, settings: Settings
    ) -> None:
        corrector = SelfCorrector(mock_llm, settings)
        mock_executor = MagicMock()
        corrected_sql = "SELECT id FROM employees"

        mock_executor.execute.side_effect = [
            DatabaseError("syntax error"),
            pd.DataFrame({"id": [1]}),
        ]
        mock_llm.complete.return_value = corrected_sql

        corrector.execute_with_correction(
            sql="SELEC id FROM employees",  # typo
            executor=mock_executor,
            question="get ids",
        )
        assert corrector.last_sql == corrected_sql
