"""Unit tests for aaizaql.eval.ex_comparator (B2 — EX comparator)."""

from __future__ import annotations

import pandas as pd
import pytest

from aaizaql.connectors.sqlite import SQLiteConnector
from aaizaql.eval.ex_comparator import ExComparator, ExResult, _cell_str, _normalise


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mem_connector() -> SQLiteConnector:
    """In-memory SQLite connector pre-loaded with a tiny table."""
    conn = SQLiteConnector()
    conn.connect("sqlite:///:memory:")
    conn._conn.execute(
        "CREATE TABLE items (id INTEGER, name TEXT, score REAL)"
    )
    conn._conn.executemany(
        "INSERT INTO items VALUES (?, ?, ?)",
        [(1, "alpha", 9.5), (2, "beta", 7.0), (3, "gamma", 8.0)],
    )
    conn._conn.commit()
    return conn


@pytest.fixture()
def comparator(mem_connector: SQLiteConnector) -> ExComparator:
    return ExComparator(mem_connector)


# ── _cell_str ─────────────────────────────────────────────────────────────────


def test_cell_str_none_is_null() -> None:
    assert _cell_str(None) == "NULL"


def test_cell_str_nan_is_null() -> None:
    assert _cell_str(float("nan")) == "NULL"


def test_cell_str_int() -> None:
    assert _cell_str(42) == "42"


def test_cell_str_str() -> None:
    assert _cell_str("hello") == "hello"


# ── _normalise ────────────────────────────────────────────────────────────────


def test_normalise_empty_df() -> None:
    assert _normalise(pd.DataFrame()) == []


def test_normalise_casts_to_str() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    rows = _normalise(df)
    assert rows == [("1", "3.0"), ("2", "4.0")]


def test_normalise_column_order_independent() -> None:
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"b": [2], "a": [1]})
    assert _normalise(df1) == _normalise(df2)


def test_normalise_row_order_independent() -> None:
    df1 = pd.DataFrame({"x": [1, 2]})
    df2 = pd.DataFrame({"x": [2, 1]})
    assert _normalise(df1) == _normalise(df2)


def test_normalise_null_cells_unified() -> None:
    # pandas promotes [None, 1] to float64, so 1 becomes 1.0 → "1.0"
    df = pd.DataFrame({"a": [None, 1]})
    rows = _normalise(df)
    assert ("NULL",) in rows
    assert any(r == ("1",) or r == ("1.0",) for r in rows)


# ── ExComparator.compare() — passing cases ────────────────────────────────────


def test_compare_identical_sql_passes(comparator: ExComparator) -> None:
    result = comparator.compare("SELECT * FROM items", "SELECT * FROM items")
    assert result.passed is True
    assert result.error is None
    assert result.row_diff == {}


def test_compare_column_order_independent(comparator: ExComparator) -> None:
    result = comparator.compare(
        "SELECT name, id FROM items WHERE id = 1",
        "SELECT id, name FROM items WHERE id = 1",
    )
    assert result.passed is True


def test_compare_row_order_independent(comparator: ExComparator) -> None:
    result = comparator.compare(
        "SELECT id FROM items ORDER BY id ASC",
        "SELECT id FROM items ORDER BY id DESC",
    )
    assert result.passed is True


def test_compare_both_empty_passes(comparator: ExComparator) -> None:
    result = comparator.compare(
        "SELECT * FROM items WHERE id = 999",
        "SELECT * FROM items WHERE id = 999",
    )
    assert result.passed is True


# ── ExComparator.compare() — failing cases ────────────────────────────────────


def test_compare_different_rows_fails(comparator: ExComparator) -> None:
    result = comparator.compare(
        "SELECT * FROM items WHERE id = 1",
        "SELECT * FROM items WHERE id = 2",
    )
    assert result.passed is False
    assert result.error is None
    assert "missing" in result.row_diff
    assert "extra" in result.row_diff


def test_compare_generated_superset_fails(comparator: ExComparator) -> None:
    result = comparator.compare(
        "SELECT * FROM items",
        "SELECT * FROM items WHERE id = 1",
    )
    assert result.passed is False
    assert len(result.row_diff["extra"]) > 0
    assert result.row_diff["missing"] == []


def test_compare_generated_subset_fails(comparator: ExComparator) -> None:
    result = comparator.compare(
        "SELECT * FROM items WHERE id = 1",
        "SELECT * FROM items",
    )
    assert result.passed is False
    assert len(result.row_diff["missing"]) > 0
    assert result.row_diff["extra"] == []


# ── ExComparator.compare() — error cases ─────────────────────────────────────


def test_compare_generated_error_returns_failure(comparator: ExComparator) -> None:
    result = comparator.compare("SELECT * FROM nonexistent_table", "SELECT * FROM items")
    assert result.passed is False
    assert result.error is not None
    assert "[generated]" in result.error
    assert "no such table" in result.error.lower()


def test_compare_gold_error_returns_failure(comparator: ExComparator) -> None:
    result = comparator.compare("SELECT * FROM items", "SELECT * FROM nonexistent_table")
    assert result.passed is False
    assert result.error is not None
    assert "[gold]" in result.error


def test_compare_both_error_combines_messages(comparator: ExComparator) -> None:
    result = comparator.compare("SELECT * FROM bad1", "SELECT * FROM bad2")
    assert result.passed is False
    assert result.error is not None
    assert "[generated]" in result.error
    assert "[gold]" in result.error
    assert result.row_diff == {}


# ── ExResult dataclass ────────────────────────────────────────────────────────


def test_result_is_frozen(comparator: ExComparator) -> None:
    result = comparator.compare("SELECT * FROM items", "SELECT * FROM items")
    with pytest.raises((AttributeError, TypeError)):
        result.passed = False  # type: ignore[misc]


def test_result_fields_populated(comparator: ExComparator) -> None:
    gen_sql = "SELECT id FROM items WHERE id = 1"
    gold_sql = "SELECT id FROM items WHERE id = 2"
    result = comparator.compare(gen_sql, gold_sql)
    assert isinstance(result, ExResult)
    assert result.generated_sql == gen_sql
    assert result.gold_sql == gold_sql


# ── row_diff cap (_DIFF_SAMPLE) ───────────────────────────────────────────────


def test_row_diff_capped(mem_connector: SQLiteConnector) -> None:
    """row_diff entries should be at most _DIFF_SAMPLE rows each."""
    from aaizaql.eval.ex_comparator import _DIFF_SAMPLE

    # Insert enough rows to exceed the cap
    for i in range(4, 4 + _DIFF_SAMPLE + 5):
        mem_connector._conn.execute(
            "INSERT INTO items VALUES (?, ?, ?)", (i, f"item{i}", float(i))
        )
    mem_connector._conn.commit()

    cmp = ExComparator(mem_connector)
    result = cmp.compare(
        "SELECT * FROM items WHERE id = 1",  # only 1 row
        "SELECT * FROM items",               # many rows
    )
    assert result.passed is False
    assert len(result.row_diff["missing"]) <= _DIFF_SAMPLE
