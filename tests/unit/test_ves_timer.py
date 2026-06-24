"""Unit tests for aaizaql.eval.ves_timer (B3 — VES timing)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aaizaql.connectors.sqlite import SQLiteConnector
from aaizaql.eval.ex_comparator import ExResult
from aaizaql.eval.ves_timer import VesResult, VesTimer, _compute_ves


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mem_connector() -> SQLiteConnector:
    """In-memory SQLite connector with a simple table."""
    conn = SQLiteConnector()
    conn.connect("sqlite:///:memory:")
    conn._conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
    conn._conn.executemany(
        "INSERT INTO t VALUES (?, ?)",
        [(1, "a"), (2, "b"), (3, "c")],
    )
    conn._conn.commit()
    return conn


@pytest.fixture()
def timer(mem_connector: SQLiteConnector) -> VesTimer:
    return VesTimer(mem_connector)


# ── _compute_ves unit tests ───────────────────────────────────────────────────


def test_compute_ves_ex_fails_returns_zero() -> None:
    assert _compute_ves(ex_passed=False, generated_time_s=0.1, gold_time_s=0.05) == 0.0


def test_compute_ves_ex_fails_zero_times_returns_zero() -> None:
    assert _compute_ves(ex_passed=False, generated_time_s=0.0, gold_time_s=0.0) == 0.0


def test_compute_ves_both_zero_returns_one() -> None:
    """Sub-microsecond tie — both times 0.0 → VES = 1.0."""
    assert _compute_ves(ex_passed=True, generated_time_s=0.0, gold_time_s=0.0) == 1.0


def test_compute_ves_generated_faster_than_gold() -> None:
    """Generated runs in 0.05 s, gold in 0.1 s → VES = 1.0 (capped)."""
    ves = _compute_ves(ex_passed=True, generated_time_s=0.05, gold_time_s=0.1)
    assert ves == 1.0


def test_compute_ves_generated_same_speed_as_gold() -> None:
    ves = _compute_ves(ex_passed=True, generated_time_s=0.1, gold_time_s=0.1)
    assert ves == pytest.approx(1.0)


def test_compute_ves_generated_twice_as_slow() -> None:
    """Generated: 0.2 s, gold: 0.1 s → VES = 0.1/0.2 = 0.5."""
    ves = _compute_ves(ex_passed=True, generated_time_s=0.2, gold_time_s=0.1)
    assert ves == pytest.approx(0.5)


def test_compute_ves_generated_much_slower() -> None:
    """Generated: 1.0 s, gold: 0.1 s → VES = 0.1/1.0 = 0.1."""
    ves = _compute_ves(ex_passed=True, generated_time_s=1.0, gold_time_s=0.1)
    assert ves == pytest.approx(0.1)


def test_compute_ves_score_bounded_between_zero_and_one() -> None:
    for gen, gld in [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (10.0, 0.001)]:
        ves = _compute_ves(ex_passed=True, generated_time_s=gen, gold_time_s=gld)
        assert 0.0 <= ves <= 1.0, f"VES={ves} out of bounds for gen={gen}, gld={gld}"


# ── VesTimer.score() — result structure ──────────────────────────────────────


def test_score_returns_ves_result(timer: VesTimer) -> None:
    result = timer.score("SELECT * FROM t", "SELECT * FROM t")
    assert isinstance(result, VesResult)


def test_score_result_is_frozen(timer: VesTimer) -> None:
    result = timer.score("SELECT * FROM t", "SELECT * FROM t")
    with pytest.raises((AttributeError, TypeError)):
        result.ves_score = 0.0  # type: ignore[misc]


def test_score_timing_fields_are_non_negative(timer: VesTimer) -> None:
    result = timer.score("SELECT * FROM t", "SELECT * FROM t")
    assert result.generated_time_s >= 0.0
    assert result.gold_time_s >= 0.0


def test_score_contains_ex_result(timer: VesTimer) -> None:
    result = timer.score("SELECT * FROM t", "SELECT * FROM t")
    assert isinstance(result.ex_result, ExResult)


# ── VesTimer.score() — VES values ────────────────────────────────────────────


def test_score_identical_sql_passes_with_nonzero_ves(timer: VesTimer) -> None:
    result = timer.score("SELECT * FROM t", "SELECT * FROM t")
    assert result.ex_result.passed is True
    assert result.ves_score > 0.0


def test_score_mismatch_gives_zero_ves(timer: VesTimer) -> None:
    result = timer.score(
        "SELECT * FROM t WHERE id = 1",
        "SELECT * FROM t WHERE id = 2",
    )
    assert result.ex_result.passed is False
    assert result.ves_score == 0.0


def test_score_generated_error_gives_zero_ves(timer: VesTimer) -> None:
    result = timer.score("SELECT * FROM no_such_table", "SELECT * FROM t")
    assert result.ex_result.passed is False
    assert result.ves_score == 0.0
    assert result.ex_result.error is not None


def test_score_gold_error_gives_zero_ves(timer: VesTimer) -> None:
    result = timer.score("SELECT * FROM t", "SELECT * FROM no_such_table")
    assert result.ex_result.passed is False
    assert result.ves_score == 0.0


# ── VesTimer.score() — timing injection via mock ─────────────────────────────


def test_score_uses_wall_clock_for_ves_computation(timer: VesTimer) -> None:
    """Patch perf_counter to control timing and verify VES formula."""
    # Sequence: gen_start, gen_end, gold_start, gold_end
    # generated_time = 0.2 s, gold_time = 0.1 s → VES = 0.1/0.2 = 0.5
    side_effects = [0.0, 0.2, 0.3, 0.4]

    with patch("aaizaql.eval.ves_timer.time.perf_counter", side_effect=side_effects):
        result = timer.score("SELECT * FROM t", "SELECT * FROM t")

    assert result.generated_time_s == pytest.approx(0.2)
    assert result.gold_time_s == pytest.approx(0.1)
    assert result.ves_score == pytest.approx(0.5)


def test_score_ves_one_when_generated_faster(timer: VesTimer) -> None:
    """Generated = 0.05 s, gold = 0.1 s → VES = 1.0."""
    side_effects = [0.0, 0.05, 0.2, 0.3]

    with patch("aaizaql.eval.ves_timer.time.perf_counter", side_effect=side_effects):
        result = timer.score("SELECT * FROM t", "SELECT * FROM t")

    assert result.ves_score == pytest.approx(1.0)


def test_score_ves_zero_when_ex_fails_regardless_of_timing(timer: VesTimer) -> None:
    """Even if generated is faster, VES must be 0 when EX fails."""
    side_effects = [0.0, 0.01, 0.1, 0.5]  # generated much faster than gold

    with patch("aaizaql.eval.ves_timer.time.perf_counter", side_effect=side_effects):
        result = timer.score(
            "SELECT * FROM t WHERE id = 1",
            "SELECT * FROM t WHERE id = 2",
        )

    assert result.ves_score == 0.0


# ── VesResult fields ──────────────────────────────────────────────────────────


def test_ves_result_stores_sql_via_ex_result(timer: VesTimer) -> None:
    gen = "SELECT id FROM t"
    gold = "SELECT id FROM t ORDER BY id"
    result = timer.score(gen, gold)
    assert result.ex_result.generated_sql == gen
    assert result.ex_result.gold_sql == gold
