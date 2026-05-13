"""
aqlix.visualization.renderer
──────────────────────────────
ResultRenderer: auto-detects the best chart type and renders a Plotly figure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _has_plotly() -> bool:
    try:
        import plotly  # noqa: F401

        return True
    except ImportError:
        return False


class ResultRenderer:
    """
    Analyses a result DataFrame and produces the most appropriate Plotly chart.

    Chart selection heuristic:
    - 1 numeric column  →  histogram / single bar
    - 1 categorical + 1 numeric  →  bar chart
    - 1 date/time + 1 numeric  →  line chart
    - 2 numeric columns  →  scatter
    - Many columns / wide table  →  table only (no chart)
    """

    def render(self, data: pd.DataFrame, question: str = "") -> Any | None:
        """
        Parameters
        ----------
        data     : pd.DataFrame  Query result.
        question : str           The original question (used for title).

        Returns
        -------
        plotly.graph_objects.Figure | None
        """
        if data.empty or not _has_plotly():
            return None

        import plotly.express as px

        chart_type = self._detect_chart_type(data)

        try:
            if chart_type == "bar":
                return self._bar(data, question, px)
            elif chart_type == "line":
                return self._line(data, question, px)
            elif chart_type == "scatter":
                return self._scatter(data, question, px)
            elif chart_type == "pie":
                return self._pie(data, question, px)
        except Exception:
            pass  # Chart rendering is best-effort; don't break the query

        return None

    # ── Chart builders ────────────────────────────────────────────────────────

    def _bar(self, df: pd.DataFrame, title: str, px: Any) -> Any:
        cat_col = self._first_categorical(df)
        num_col = self._first_numeric(df)
        return px.bar(df, x=cat_col, y=num_col, title=title)

    def _line(self, df: pd.DataFrame, title: str, px: Any) -> Any:
        date_col = self._first_datetime(df) or df.columns[0]
        num_col = self._first_numeric(df)
        return px.line(df, x=date_col, y=num_col, title=title, markers=True)

    def _scatter(self, df: pd.DataFrame, title: str, px: Any) -> Any:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return px.scatter(df, x=num_cols[0], y=num_cols[1], title=title)

    def _pie(self, df: pd.DataFrame, title: str, px: Any) -> Any:
        cat_col = self._first_categorical(df)
        num_col = self._first_numeric(df)
        return px.pie(df, names=cat_col, values=num_col, title=title)

    # ── Heuristics ────────────────────────────────────────────────────────────

    def _detect_chart_type(self, df: pd.DataFrame) -> str:
        has_datetime = self._first_datetime(df) is not None
        num_count = sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
        cat_count = sum(
            1
            for c in df.columns
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])
        )

        if has_datetime and num_count >= 1:
            return "line"
        if cat_count == 1 and num_count == 1 and len(df) <= 5:
            return "pie"
        if cat_count >= 1 and num_count >= 1:
            return "bar"
        if num_count >= 2:
            return "scatter"
        return "bar"

    def _first_numeric(self, df: pd.DataFrame) -> str | None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        return df.columns[0] if len(df.columns) > 0 else None

    def _first_categorical(self, df: pd.DataFrame) -> str | None:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
                return c
        return df.columns[0] if len(df.columns) > 0 else None

    def _first_datetime(self, df: pd.DataFrame) -> str | None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
            if any(kw in c.lower() for kw in ("date", "time", "month", "year", "week", "day")):
                return c
        return None
