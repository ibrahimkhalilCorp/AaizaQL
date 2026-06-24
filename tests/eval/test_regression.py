"""
tests/eval/test_regression.py
──────────────────────────────
B6 — CI regression gate for the NL→SQL pipeline.

Runs ExComparator on a pinned 50-question Spider dev subset and asserts that
EX accuracy does not drop below EX_BASELINE.

Prerequisites (test skips when either is absent):
  AAIZAQL_SPIDER_DIR — local Spider dataset root (must contain dev.json and
                       the database/ sub-directory with per-db SQLite files)
  At least one key  — GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY

Run manually::

    AAIZAQL_SPIDER_DIR=/data/spider \\
    GROQ_API_KEY=gsk_... \\
    pytest tests/eval/test_regression.py -m eval -v

To recalibrate the baseline: run once, observe the printed EX accuracy,
then update EX_BASELINE and commit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aaizaql.core.engine import QueryEngine
from aaizaql.eval.ex_comparator import ExComparator
from aaizaql.eval.spider_loader import EvalItem

# ── Pinned configuration ───────────────────────────────────────────────────────

REGRESSION_SUBSET_PATH = Path(__file__).parent / "fixtures" / "regression_subset.json"

# Fail CI if measured EX accuracy drops below this.
# Start conservative; bump after the first successful calibration run.
EX_BASELINE: float = 0.40

_LLM_KEY_VARS = ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


# ── Internal helpers ───────────────────────────────────────────────────────────


def _spider_dir() -> Path | None:
    raw = os.getenv("AAIZAQL_SPIDER_DIR", "").strip()
    return Path(raw) if raw else None


def _any_llm_key_set() -> bool:
    return any(os.getenv(k) for k in _LLM_KEY_VARS)


def _preferred_llm() -> str:
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    return "openai"


def _make_eval_item(entry: dict, spider_dir: Path) -> EvalItem | None:
    db_id = entry["db_id"]
    db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        return None
    return EvalItem(
        db_id=db_id,
        db_path=db_path,
        question=entry["question"],
        gold_sql=entry["gold_sql"],
    )


# ── Regression gate ────────────────────────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.slow
def test_ex_regression_gate() -> None:
    """EX accuracy on a pinned 50-question Spider subset must not drop below EX_BASELINE."""
    spider_dir = _spider_dir()
    if spider_dir is None:
        pytest.skip("AAIZAQL_SPIDER_DIR not set — skipping regression gate")
    if not _any_llm_key_set():
        pytest.skip("No LLM API key set — skipping regression gate")

    llm = _preferred_llm()
    subset: list[dict] = json.loads(REGRESSION_SUBSET_PATH.read_text(encoding="utf-8"))

    engine_cache: dict[str, QueryEngine] = {}
    passed = 0
    total = 0
    failures: list[str] = []

    for entry in subset:
        item = _make_eval_item(entry, spider_dir)
        if item is None:
            continue

        total += 1

        # Reuse one engine per db_id to amortise schema ingestion cost.
        if item.db_id not in engine_cache:
            engine = QueryEngine(
                llm=llm,
                database="sqlite",
                dsn=item.dsn,
                vector_store_namespace=item.db_id,
            )
            try:
                engine.ingest_schema()
            except Exception:  # noqa: BLE001
                pass
            engine_cache[item.db_id] = engine
        else:
            engine = engine_cache[item.db_id]

        generated_sql = ""
        try:
            result = engine.query(item.question)
            generated_sql = result.sql
        except Exception as exc:  # noqa: BLE001
            failures.append(f"[query error] {item.db_id}: {exc}")

        item_passed = False
        if generated_sql:
            cmp_conn = item.build_connector()
            try:
                cmp_conn.connect(item.dsn)
                ex = ExComparator(cmp_conn).compare(generated_sql, item.gold_sql)
                item_passed = ex.passed
                if not item_passed:
                    failures.append(
                        f"[EX FAIL] {item.db_id} | Q: {item.question[:60]}"
                        f" | gen: {generated_sql[:80]}"
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"[compare error] {item.db_id}: {exc}")
            finally:
                try:
                    cmp_conn.close()
                except Exception:  # noqa: BLE001
                    pass

        if item_passed:
            passed += 1

    if total == 0:
        pytest.skip(
            "No SQLite files found under AAIZAQL_SPIDER_DIR/database/ — "
            "check that the Spider dataset is fully extracted"
        )

    ex_accuracy = passed / total
    failure_summary = "\n  ".join(failures[:20])  # cap for readability
    assert ex_accuracy >= EX_BASELINE, (
        f"EX regression: {ex_accuracy:.1%} ({passed}/{total}) < baseline {EX_BASELINE:.1%}\n"
        f"First failures:\n  {failure_summary}"
    )
