"""
aaizaql.eval.ves_timer
──────────────────────
B3 — Valid Efficiency Score (VES) timing.

Wraps ExComparator with wall-clock timing and computes:

    VES = EX × (gold_time / max(generated_time, gold_time))

where EX ∈ {0, 1}:
- EX = 0 (result mismatch or execution error) → VES = 0.0.
- EX = 1, both times = 0 (sub-microsecond) → VES = 1.0 (treated as a tie).
- EX = 1, generated slower than gold → VES = gold_time / generated_time ∈ (0, 1).
- EX = 1, generated at least as fast as gold → VES = 1.0.

Usage::

    timer = VesTimer(connector)
    result = timer.score(
        generated_sql="SELECT name FROM artist ORDER BY name",
        gold_sql="SELECT name FROM artist",
    )
    print(result.ves_score)       # float in [0.0, 1.0]
    print(result.ex_result.passed)
    print(result.generated_time_s, result.gold_time_s)

Author: Ibrahim
Date: 2026-06-24
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from aaizaql.connectors.base import DatabaseConnector
from aaizaql.eval.ex_comparator import ExComparator, ExResult

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class VesResult:
    """Outcome of one VES evaluation.

    Attributes:
        ex_result:        Full :class:`~aaizaql.eval.ex_comparator.ExResult` from
                          the underlying EX comparison (contains passed / error /
                          row_diff).
        generated_time_s: Wall-clock seconds for the *generated* SQL execution.
        gold_time_s:      Wall-clock seconds for the *gold* SQL execution.
        ves_score:        VES ∈ [0.0, 1.0].  0.0 when EX fails; capped at 1.0
                          when generated SQL is at least as fast as gold.
    """

    ex_result: ExResult
    generated_time_s: float
    gold_time_s: float
    ves_score: float


class VesTimer:
    """Times SQL execution and computes VES on top of EX accuracy.

    Args:
        connector: A *connected* :class:`~aaizaql.connectors.base.DatabaseConnector`.
            ``connect()`` must have been called before passing it here.
    """

    def __init__(self, connector: DatabaseConnector) -> None:
        self._connector = connector
        self._comparator = ExComparator(connector)

    def score(self, generated_sql: str, gold_sql: str) -> VesResult:
        """Execute both SQL strings, compare results, and compute VES.

        Both SQLs are executed independently with wall-clock timing.  The
        :class:`ExComparator` is then used for result-set comparison so that
        all EX normalisation rules (column order, NULL handling, etc.) apply
        unchanged.

        Args:
            generated_sql: SQL produced by the NL→SQL pipeline.
            gold_sql:      Gold-standard reference SQL.

        Returns:
            :class:`VesResult` containing the EX outcome, timing, and VES score.
        """
        generated_time_s = self._time_execution(generated_sql)
        gold_time_s = self._time_execution(gold_sql)

        ex_result = self._comparator.compare(generated_sql, gold_sql)

        ves = _compute_ves(
            ex_passed=ex_result.passed,
            generated_time_s=generated_time_s,
            gold_time_s=gold_time_s,
        )

        logger.info(
            "ves.score",
            passed=ex_result.passed,
            ves=round(ves, 4),
            generated_time_s=round(generated_time_s, 6),
            gold_time_s=round(gold_time_s, 6),
        )
        return VesResult(
            ex_result=ex_result,
            generated_time_s=generated_time_s,
            gold_time_s=gold_time_s,
            ves_score=ves,
        )

    def _time_execution(self, sql: str) -> float:
        """Return wall-clock seconds for one SQL execution, or 0.0 on error.

        Errors are intentionally swallowed here — EX comparison (run after
        timing) will surface them via its own error field.
        """
        start = time.perf_counter()
        try:
            self._connector.execute(sql)
        except Exception:  # noqa: BLE001
            pass
        return time.perf_counter() - start


# ── Formula ───────────────────────────────────────────────────────────────────


def _compute_ves(
    *,
    ex_passed: bool,
    generated_time_s: float,
    gold_time_s: float,
) -> float:
    """Compute VES = EX × (gold_time / max(generated_time, gold_time)).

    Args:
        ex_passed:        True when EX comparison passed.
        generated_time_s: Wall-clock seconds for generated SQL.
        gold_time_s:      Wall-clock seconds for gold SQL.

    Returns:
        VES score ∈ [0.0, 1.0].
    """
    if not ex_passed:
        return 0.0

    denominator = max(generated_time_s, gold_time_s)
    if denominator == 0.0:
        # Both queries are sub-microsecond — treat as a perfect tie.
        return 1.0

    return gold_time_s / denominator
