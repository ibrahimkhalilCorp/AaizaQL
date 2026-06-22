"""
aaizaql.visualization.renderer
───────────────────────────────
ResultRenderer: inspects a DataFrame and auto-selects the best Plotly chart.

Detection heuristic (in priority order):
1. Time-series  → line chart   (date/datetime column detected)
2. Categorical + single numeric → bar chart
3. Two numeric columns         → scatter plot
4. Single numeric, few rows    → pie chart  (≤ 8 distinct values)
5. Fallback                    → None (no chart)

Plotly is an optional dependency.  If it is not installed the renderer
returns None gracefully so the rest of the pipeline still works.

Author : Ibrahim
Date   : 2024-01-01
Version: 1.0.0
"""

from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Threshold for pie chart: too many slices become unreadable.
_PIE_MAX_ROWS = 8

# Margin settings applied to every generated figure.
_FIGURE_MARGIN = {"l": 40, "r": 20, "t": 50, "b": 40}


def _is_datetime_col(series: pd.Series) -> bool:
    """
    Return True if *series* looks like a date or datetime column.

    Checks the pandas dtype first; falls back to attempting a parse on the
    first five non-null values for object-typed columns.

    Parameters
    ----------
    series : pd.Series
        A single DataFrame column to inspect.

    Returns
    -------
    bool
        True when the column contains date/datetime values.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        try:
            pd.to_datetime(series.dropna().head(5))
            return True
        except Exception:
            return False
    return False


class ResultRenderer:
    """
    Auto-detect the best chart type for a DataFrame and render it with Plotly.

    The returned value is either a ``plotly.graph_objects.Figure`` or ``None``
    when no chart is appropriate or Plotly is not installed.
    """

    def render(self, data: pd.DataFrame, question: str = "") -> Any:
        """
        Return a Plotly Figure for *data*, or None if no chart is appropriate.

        Parameters
        ----------
        data     : pd.DataFrame
            Query result to visualise.
        question : str
            Original natural-language question used as the chart title.

        Returns
        -------
        plotly.graph_objects.Figure | None
            A rendered figure, or None when a chart cannot be produced.
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

        return self._build_figure(px, chart_type, data, x_col, y_col, title)

    # ── Detection logic ───────────────────────────────────────────────────────

    def _detect_chart(self, data: pd.DataFrame) -> tuple[str, str | None, str | None]:
        """
        Choose the most appropriate chart type for *data*.

        Parameters
        ----------
        data : pd.DataFrame
            The query result to inspect.

        Returns
        -------
        tuple[str, str | None, str | None]
            ``(chart_type, x_column, y_column)`` where *chart_type* is one
            of ``'line'``, ``'bar'``, ``'scatter'``, ``'pie'``, or ``'none'``.
        """
        cols = list(data.columns)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]
        date_cols = [c for c in cols if _is_datetime_col(data[c])]
        cat_cols = [c for c in cols if c not in numeric_cols and c not in date_cols]

        if date_cols and numeric_cols:
            return "line", date_cols[0], numeric_cols[0]

        if cat_cols and numeric_cols:
            return "bar", cat_cols[0], numeric_cols[0]

        if len(numeric_cols) >= 2:
            return "scatter", numeric_cols[0], numeric_cols[1]

        if cat_cols and numeric_cols and len(data) <= _PIE_MAX_ROWS:
            return "pie", cat_cols[0], numeric_cols[0]

        return "none", None, None

    def _build_figure(
        self,
        px: Any,
        chart_type: str,
        data: pd.DataFrame,
        x_col: str | None,
        y_col: str | None,
        title: str,
    ) -> Any:
        """
        Construct and return a Plotly figure for the given chart configuration.

        Parameters
        ----------
        px         : module         Plotly Express module (injected to avoid re-import).
        chart_type : str            One of 'line', 'bar', 'scatter', 'pie', or 'none'.
        data       : pd.DataFrame   Source data.
        x_col      : str | None     Column name for the x-axis (or pie names).
        y_col      : str | None     Column name for the y-axis (or pie values).
        title      : str            Chart title.

        Returns
        -------
        plotly.graph_objects.Figure | None
            Rendered figure, or None when the chart type is 'none' or an
            error occurs during rendering.
        """
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

            fig.update_layout(margin=_FIGURE_MARGIN)
            logger.info("renderer.chart_ready", chart_type=chart_type, x=x_col, y=y_col)
            return fig

        except Exception as exc:
            logger.warning("renderer.failed", detail=str(exc)[:80])
            return None
