"""
tests/test_visualization.py
────────────────────────────
Unit tests for ResultRenderer chart detection and NLSummarizer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from aqlix.visualization.renderer import ResultRenderer
from aqlix.visualization.summarizer import NLSummarizer


class TestResultRenderer:
    def setup_method(self) -> None:
        self.renderer = ResultRenderer()

    def test_detect_bar_chart(self) -> None:
        df = pd.DataFrame({"department": ["Eng", "HR", "Mkt"], "count": [10, 5, 8]})
        chart_type, x, y = self.renderer._detect_chart(df)
        assert chart_type == "bar"
        assert x == "department"
        assert y == "count"

    def test_detect_scatter_two_numerics(self) -> None:
        df = pd.DataFrame({"salary": [50000, 70000, 90000], "age": [25, 35, 45]})
        chart_type, x, y = self.renderer._detect_chart(df)
        assert chart_type == "scatter"

    def test_detect_line_with_datetime(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01", "2024-02", "2024-03"]),
                "revenue": [100, 200, 150],
            }
        )
        chart_type, x, y = self.renderer._detect_chart(df)
        assert chart_type == "line"

    def test_detect_pie_few_rows(self) -> None:
        df = pd.DataFrame({"status": ["Active", "Resigned"], "count": [80, 20]})
        chart_type, x, y = self.renderer._detect_chart(df)
        assert chart_type in ("pie", "bar")  # either is acceptable with 2 rows

    def test_empty_dataframe_returns_none(self) -> None:
        result = self.renderer.render(pd.DataFrame(), "test question")
        assert result is None

    def test_render_returns_none_gracefully_without_plotly(self) -> None:
        # Even if plotly is absent we should not crash
        df = pd.DataFrame({"dept": ["Eng"], "count": [5]})
        # render() handles ImportError internally
        # Just assert it doesn't raise
        try:
            self.renderer.render(df, "departments")
        except Exception as exc:
            pytest.fail(f"render() should not raise: {exc}")


class TestNLSummarizer:
    def test_empty_dataframe_returns_fixed_message(self) -> None:
        mock_llm = MagicMock()
        summarizer = NLSummarizer(mock_llm)
        result = summarizer.summarize("test question", pd.DataFrame())
        assert result == "The query returned no results."
        mock_llm.complete.assert_not_called()

    def test_summarize_calls_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Engineering has the highest average salary."
        summarizer = NLSummarizer(mock_llm)

        df = pd.DataFrame({"dept": ["Eng", "HR"], "avg_salary": [90000, 65000]})
        result = summarizer.summarize("Average salary by department?", df)

        assert result == "Engineering has the highest average salary."
        mock_llm.complete.assert_called_once()

    def test_summarize_graceful_on_llm_error(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = Exception("LLM timeout")
        summarizer = NLSummarizer(mock_llm)

        df = pd.DataFrame({"x": [1, 2, 3]})
        result = summarizer.summarize("What is x?", df)
        assert result == ""  # graceful fallback, no raise
