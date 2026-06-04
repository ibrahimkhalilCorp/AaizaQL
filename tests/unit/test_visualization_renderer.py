"""
tests/unit/test_visualization_renderer.py
──────────────────────────────────────────
Unit tests for aaizaql.visualization.renderer.ResultRenderer.

The renderer does `import plotly.express as px` *inside* render(), so we
patch it via sys.modules at the point of import, and we also patch the
module-level reference used after import.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
from aaizaql.visualization.renderer import ResultRenderer, _is_datetime_col

# ── _is_datetime_col ──────────────────────────────────────────────────────────


class TestIsDatetimeCol:
    def test_native_datetime_dtype(self) -> None:
        s = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01"]))
        assert _is_datetime_col(s) is True

    def test_integer_col_is_not_datetime(self) -> None:
        s = pd.Series([1, 2, 3])
        assert _is_datetime_col(s) is False

    def test_float_col_is_not_datetime(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        assert _is_datetime_col(s) is False

    def test_plain_string_col_is_not_datetime(self) -> None:
        s = pd.Series(["Alice", "Bob", "Charlie"])
        assert _is_datetime_col(s) is False

    def test_object_col_not_parseable_returns_false(self) -> None:
        s = pd.Series(["foo-bar", "baz-qux", "not-a-date"])
        assert _is_datetime_col(s) is False


# ── ResultRenderer helpers ────────────────────────────────────────────────────


def _make_px_mock():
    """Return (fake_px, fake_fig) with bar/line/scatter/pie all wired up."""
    fake_fig = MagicMock()
    fake_px = MagicMock()
    fake_px.line.return_value = fake_fig
    fake_px.bar.return_value = fake_fig
    fake_px.scatter.return_value = fake_fig
    fake_px.pie.return_value = fake_fig
    return fake_px, fake_fig


def _patch_plotly(fake_px):
    """
    Patch plotly so that `import plotly.express as px` inside render() gets
    our mock, and the import itself does not raise ImportError.
    """
    fake_plotly = MagicMock()
    fake_plotly.express = fake_px
    return patch.dict(
        "sys.modules",
        {
            "plotly": fake_plotly,
            "plotly.express": fake_px,
        },
    )


# ── ResultRenderer.render ─────────────────────────────────────────────────────


class TestResultRenderer:
    def _renderer(self) -> ResultRenderer:
        return ResultRenderer()

    # ── guard paths ───────────────────────────────────────────────────────────

    def test_empty_dataframe_returns_none(self) -> None:
        r = self._renderer()
        assert r.render(pd.DataFrame()) is None

    def test_plotly_not_installed_returns_none(self) -> None:
        r = self._renderer()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with patch.dict("sys.modules", {"plotly": None, "plotly.express": None}):
            result = r.render(df)
        assert result is None

    # ── line chart (time-series) ──────────────────────────────────────────────

    def test_datetime_plus_numeric_produces_line_chart(self) -> None:
        fake_px, fake_fig = _make_px_mock()
        r = self._renderer()
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "revenue": [100, 200, 150],
            }
        )
        with _patch_plotly(fake_px):
            result = r.render(df, question="Monthly revenue")
        fake_px.line.assert_called_once()
        assert result is fake_fig

    # ── bar chart (categorical + numeric) ─────────────────────────────────────

    def test_categorical_plus_numeric_produces_bar_chart(self) -> None:
        fake_px, fake_fig = _make_px_mock()
        r = self._renderer()
        df = pd.DataFrame({"product": ["A", "B", "C"], "sales": [10, 20, 15]})
        with _patch_plotly(fake_px):
            result = r.render(df)
        fake_px.bar.assert_called_once()
        assert result is fake_fig

    # ── scatter (two numeric) ─────────────────────────────────────────────────

    def test_two_numeric_cols_produces_scatter(self) -> None:
        fake_px, fake_fig = _make_px_mock()
        r = self._renderer()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        with _patch_plotly(fake_px):
            result = r.render(df)
        fake_px.scatter.assert_called_once()
        assert result is fake_fig

    # ── no suitable chart ─────────────────────────────────────────────────────

    def test_single_numeric_col_returns_none(self) -> None:
        fake_px, _ = _make_px_mock()
        r = self._renderer()
        df = pd.DataFrame({"value": [1, 2, 3]})
        with _patch_plotly(fake_px):
            result = r.render(df)
        assert result is None

    def test_single_string_col_returns_none(self) -> None:
        fake_px, _ = _make_px_mock()
        r = self._renderer()
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        with _patch_plotly(fake_px):
            result = r.render(df)
        assert result is None

    # ── plotly exception ──────────────────────────────────────────────────────

    def test_plotly_error_returns_none(self) -> None:
        fake_px, _ = _make_px_mock()
        fake_px.bar.side_effect = Exception("render error")
        r = self._renderer()
        df = pd.DataFrame({"cat": ["A", "B"], "val": [1, 2]})
        with _patch_plotly(fake_px):
            result = r.render(df)
        assert result is None

    # ── question / title ──────────────────────────────────────────────────────

    def test_question_used_as_title(self) -> None:
        fake_px, _ = _make_px_mock()
        r = self._renderer()
        df = pd.DataFrame({"cat": ["X"], "num": [99]})
        with _patch_plotly(fake_px):
            r.render(df, question="How many items?")
        call_kwargs = fake_px.bar.call_args.kwargs
        assert call_kwargs["title"] == "How many items?"

    def test_default_title_when_no_question(self) -> None:
        fake_px, _ = _make_px_mock()
        r = self._renderer()
        df = pd.DataFrame({"cat": ["X"], "num": [99]})
        with _patch_plotly(fake_px):
            r.render(df)
        call_kwargs = fake_px.bar.call_args.kwargs
        assert call_kwargs["title"] == "Query Result"

    def test_long_question_truncated_to_80_chars(self) -> None:
        fake_px, _ = _make_px_mock()
        r = self._renderer()
        df = pd.DataFrame({"cat": ["X"], "num": [99]})
        long_q = "Q" * 120
        with _patch_plotly(fake_px):
            r.render(df, question=long_q)
        call_kwargs = fake_px.bar.call_args.kwargs
        assert len(call_kwargs["title"]) == 80

    # ── _detect_chart directly ────────────────────────────────────────────────

    def test_detect_chart_line(self) -> None:
        r = self._renderer()
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"]), "val": [1.0]})
        chart_type, x, y = r._detect_chart(df)
        assert chart_type == "line"
        assert x == "ts"
        assert y == "val"

    def test_detect_chart_bar(self) -> None:
        r = self._renderer()
        df = pd.DataFrame({"cat": ["A", "B"], "num": [1.0, 2.0]})
        chart_type, x, y = r._detect_chart(df)
        assert chart_type == "bar"
        assert x == "cat"
        assert y == "num"

    def test_detect_chart_scatter(self) -> None:
        r = self._renderer()
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        chart_type, x, y = r._detect_chart(df)
        assert chart_type == "scatter"

    def test_detect_chart_none_single_numeric(self) -> None:
        r = self._renderer()
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        chart_type, x, y = r._detect_chart(df)
        assert chart_type == "none"
        assert x is None
        assert y is None
