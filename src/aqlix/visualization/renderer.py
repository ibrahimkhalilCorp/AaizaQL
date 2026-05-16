"""
aqlix.visualization.renderer
─────────────────────────────
ResultRenderer: inspects a DataFrame and auto-selects the best Plotly chart.

Detection heuristic (in priority order):
1. Time-series  → line chart   (date/datetime column detected)
2. Categorical + single numeric → bar chart
3. Two numeric columns         → scatter plot
4. Single numeric, few rows    → pie chart  (≤ 8 distinct values)
5. Fallback                    → data table (no chart)

Plotly is an optional dependency.  If it is not installed the renderer
returns None gracefully so the rest of the pipeline still works.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Threshold for pie chart (too many slices = unreadable)
_PIE_MAX_ROWS = 8


def _is_datetime_col(series: pd.Series) -> bool:
    """True if the column looks like a date/time column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        try:
            pd.to_datetime(series.dropna().head(5), infer_datetime_format=True)
            return True
        except Exception:
            return False
    return False


class ResultRenderer:
    """
    Auto-detect the best chart for a DataFrame and render it with Plotly.

    result.chart is either a plotly.graph_objects.Figure or None.
    """

    def render(self, data: pd.DataFrame, question: str = "") -> Any:
        """
        Return a Plotly Figure for data, or None if a chart is not appropriate.

        Parameters
        ----------
        data     : pd.DataFrame   Query result.
        question : str            Original NL question (used for chart title).
        """
        if data.empty or len(data.columns) < 1:
            logger.debug("renderer.skip", reason="empty_dataframe")
            return None

        try:
            import plotly.express as px  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("renderer.skip", reason="plotly_not_installed")
            return None

        chart_type, x_col, y_col = self._detect_chart(data)
        title = question[:80] if question else "Query Result"

        try:
            if chart_type == "line" and x_col and y_col:
                fig = px.line(data, x=x_col, y=y_col, title=title)
            elif chart_type == "bar" and x_col and y_col:
                fig = px.bar(data, x=x_col, y=y_col, title=title)
            elif chart_type == "scatter" and x_col and y_col:
                fig = px.scatter(data, x=x_col, y=y_col, title=title)
            elif chart_type == "pie" and x_col and y_col:
                fig = px.pie(data, names=x_col, values=y_col, title=title)
            else:
                logger.debug("renderer.skip", reason="no_suitable_chart")
                return None

            fig.update_layout(margin={"l": 40, "r": 20, "t": 50, "b": 40})
            logger.info(
                "renderer.chart_ready",
                chart_type=chart_type,
                x=x_col,
                y=y_col,
            )
            return fig

        except Exception as exc:
            logger.warning("renderer.failed", detail=str(exc)[:80])
            return None

    # ── Detection logic ───────────────────────────────────────────────────────

    def _detect_chart(
        self, data: pd.DataFrame
    ) -> tuple[str, str | None, str | None]:
        """
        Returns (chart_type, x_column, y_column).
        chart_type is one of: 'line', 'bar', 'scatter', 'pie', or 'none'.
        """
        cols = list(data.columns)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]
        date_cols = [c for c in cols if _is_datetime_col(data[c])]
        cat_cols = [c for c in cols if c not in numeric_cols and c not in date_cols]

        # 1. Time series → line
        if date_cols and numeric_cols:
            return "line", date_cols[0], numeric_cols[0]

        # 2. Categorical + numeric → bar
        if cat_cols and numeric_cols:
            return "bar", cat_cols[0], numeric_cols[0]

        # 3. Two numeric → scatter
        if len(numeric_cols) >= 2:
            return "scatter", numeric_cols[0], numeric_cols[1]

        # 4. Single numeric, few rows → pie
        if cat_cols and numeric_cols and len(data) <= _PIE_MAX_ROWS:
            return "pie", cat_cols[0], numeric_cols[0]

        return "none", None, None
