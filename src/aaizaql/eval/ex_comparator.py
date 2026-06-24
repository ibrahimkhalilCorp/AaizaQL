"""
aaizaql.eval.ex_comparator
──────────────────────────
B2 — Execution-Accuracy (EX) comparator.

Executes a generated SQL and a gold SQL against the same connector, then
compares their result sets using the rules below.

Comparison rules:
- Column order is ignored (columns sorted by name before comparison).
- Row order is ignored (rows sorted after normalisation).
- All cell values are cast to str for type-agnostic comparison.
- NULL / NaN cells are normalised to the string ``"NULL"``.
- Two result sets are equal when their sorted row-tuple multisets match.

Usage::

    comparator = ExComparator(connector)
    result = comparator.compare(
        generated_sql="SELECT name FROM artist ORDER BY name",
        gold_sql="SELECT name FROM artist",
    )
    if result.passed:
        print("PASS")
    else:
        print("FAIL", result.row_diff)

Author: Ibrahim
Date: 2026-06-24
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import structlog

from aaizaql.connectors.base import DatabaseConnector

logger = structlog.get_logger(__name__)

_DIFF_SAMPLE = 10  # max rows per side in row_diff summaries


@dataclass(frozen=True)
class ExResult:
    """Outcome of one EX comparison.

    Attributes:
        passed:        True when generated result set matches gold.
        generated_sql: The SQL that was evaluated.
        gold_sql:      The gold-standard SQL.
        error:         Non-None when either SQL failed to execute.
        row_diff:      Dict with keys ``"missing"`` (rows in gold but not
                       generated) and ``"extra"`` (rows in generated but not
                       gold), each capped at ``_DIFF_SAMPLE``.  Empty dict
                       when ``passed`` is True or on execution error.
    """

    passed: bool
    generated_sql: str
    gold_sql: str
    error: str | None = None
    row_diff: dict[str, list[tuple[str, ...]]] = field(
        default_factory=dict, compare=False, hash=False
    )


class ExComparator:
    """Compares generated SQL against gold SQL using execution accuracy.

    Args:
        connector: A *connected* :class:`~aaizaql.connectors.base.DatabaseConnector`.
            ``connect()`` must have been called before passing it here.
    """

    def __init__(self, connector: DatabaseConnector) -> None:
        self._connector = connector

    def compare(self, generated_sql: str, gold_sql: str) -> ExResult:
        """Execute both SQL strings and compare their result sets.

        Args:
            generated_sql: SQL produced by the NL→SQL pipeline.
            gold_sql:      Gold-standard reference SQL.

        Returns:
            :class:`ExResult` with ``passed=True`` when result sets are equal.
        """
        gen_df, gen_err = self._execute_safe(generated_sql, label="generated")
        gold_df, gold_err = self._execute_safe(gold_sql, label="gold")

        if gen_err or gold_err:
            error_msg = "; ".join(filter(None, [gen_err, gold_err]))
            logger.warning(
                "ex.execution_error",
                generated_sql=generated_sql[:120],
                error=error_msg,
            )
            return ExResult(
                passed=False,
                generated_sql=generated_sql,
                gold_sql=gold_sql,
                error=error_msg,
            )

        # gen_df and gold_df are both valid DataFrames here
        gen_rows = _normalise(gen_df)   # type: ignore[arg-type]
        gold_rows = _normalise(gold_df)  # type: ignore[arg-type]

        passed = gen_rows == gold_rows

        row_diff: dict[str, list[tuple[str, ...]]] = {}
        if not passed:
            gen_counter = Counter(gen_rows)
            gold_counter = Counter(gold_rows)
            missing = list((gold_counter - gen_counter).elements())[:_DIFF_SAMPLE]
            extra = list((gen_counter - gold_counter).elements())[:_DIFF_SAMPLE]
            row_diff = {"missing": missing, "extra": extra}

        logger.info(
            "ex.compare",
            passed=passed,
            generated_rows=len(gen_rows),
            gold_rows=len(gold_rows),
        )
        return ExResult(
            passed=passed,
            generated_sql=generated_sql,
            gold_sql=gold_sql,
            row_diff=row_diff,
        )

    def _execute_safe(
        self,
        sql: str,
        *,
        label: str = "sql",
    ) -> tuple[pd.DataFrame | None, str | None]:
        """Execute *sql* and return ``(dataframe, None)`` or ``(None, error_str)``."""
        try:
            return self._connector.execute(sql), None
        except Exception as exc:  # noqa: BLE001
            return None, f"[{label}] {type(exc).__name__}: {exc}"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cell_str(v: object) -> str:
    """Normalise a single cell to a comparison string.

    NULL-like values (Python ``None``, ``float('nan')``, ``pd.NA``,
    ``pd.NaT``) all become the canonical string ``"NULL"`` so that they
    compare equal between generated and gold when both return NULL.
    """
    if v is None:
        return "NULL"
    if isinstance(v, float) and math.isnan(v):
        return "NULL"
    # Handle pandas NA / NaT without importing the full pandas namespace
    try:
        if pd.isna(v):  # type: ignore[arg-type]
            return "NULL"
    except (TypeError, ValueError):
        pass
    return str(v)


def _normalise(df: pd.DataFrame) -> list[tuple[str, ...]]:
    """Return a sorted list of normalised row-tuples, column-order-independent.

    Steps:
    1. Sort columns alphabetically — column order is irrelevant for EX.
    2. Normalise every cell via :func:`_cell_str`.
    3. Sort rows — row order is irrelevant for EX.
    """
    if df.empty:
        return []
    sorted_cols = sorted(df.columns.tolist())
    df = df[sorted_cols]
    rows = [
        tuple(_cell_str(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    return sorted(rows)
