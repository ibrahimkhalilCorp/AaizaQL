"""
python -m aaizaql.eval
──────────────────────
B5 — CLI evaluation runner for BIRD + Spider benchmarks.

Usage::

    python -m aaizaql.eval \\
        --dataset spider \\
        --dataset-dir /data/spider \\
        --llm groq \\
        --output results.json \\
        [--split dev] \\
        [--max-items 50]

Output (results.json)::

    {
      "summary": {
        "total": 100,
        "ex_passed": 75,
        "ex_accuracy": 0.75,
        "mean_ves": 0.82,
        "mean_latency_s": 2.1
      },
      "results": [
        {
          "db_id": "...",
          "question": "...",
          "gold_sql": "...",
          "generated_sql": "...",
          "ex_passed": true,
          "ves_score": 0.95,
          "latency_s": 1.23,
          "generated_time_s": 0.012,
          "gold_time_s": 0.009,
          "error": null,
          "dataset": "spider",
          "split": "dev"
        },
        ...
      ]
    }

Author: Ibrahim
Date: 2026-06-24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import structlog

from aaizaql.core.engine import QueryEngine
from aaizaql.eval.bird_loader import BirdLoader
from aaizaql.eval.ex_comparator import ExComparator
from aaizaql.eval.spider_loader import EvalItem, SpiderLoader
from aaizaql.eval.ves_timer import VesTimer

logger = structlog.get_logger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m aaizaql.eval",
        description="AaizaQL evaluation harness — BIRD + Spider benchmarks",
    )
    parser.add_argument(
        "--dataset",
        choices=["spider", "bird"],
        required=True,
        help="Benchmark dataset to evaluate.",
    )
    parser.add_argument(
        "--dataset-dir",
        metavar="DIR",
        required=True,
        help="Root directory of the dataset (contains dev.json and the databases sub-dir).",
    )
    parser.add_argument(
        "--llm",
        default="groq",
        help="LLM provider name (default: groq).  Any provider supported by QueryEngine.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to write JSON results (default: results.json).",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "train", "test"],
        help="Dataset split to evaluate (default: dev).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N items.  Useful for smoke runs.",
    )
    return parser.parse_args(argv)


# ── Loader ────────────────────────────────────────────────────────────────────


def _iter_items(
    dataset: str,
    dataset_dir: str,
    split: str,
    max_items: int | None,
) -> Iterator[EvalItem]:
    root = Path(dataset_dir)
    if dataset == "spider":
        loader: SpiderLoader | BirdLoader = SpiderLoader(root, strict=False)
    else:
        loader = BirdLoader(root, strict=False)
    yield from loader.items(split=split, max_items=max_items)


# ── Engine cache ──────────────────────────────────────────────────────────────


def _get_or_create_engine(
    cache: dict[str, QueryEngine],
    item: EvalItem,
    llm: str,
) -> QueryEngine:
    """Return a cached QueryEngine for *item.db_id*, creating it on first call.

    Each database gets its own vector-store namespace so ChromaDB collections
    do not collide across the 200+ Spider / BIRD databases.
    """
    if item.db_id in cache:
        return cache[item.db_id]

    engine = QueryEngine(
        llm=llm,
        database="sqlite",
        dsn=item.dsn,
        # Isolate each database's RAG collection so schema chunks don't bleed
        # between databases when multiple engines share one ChromaDB directory.
        vector_store_namespace=item.db_id,
    )
    try:
        engine.ingest_schema()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "eval.schema_ingest_failed",
            db_id=item.db_id,
            error=str(exc),
        )

    cache[item.db_id] = engine
    return engine


# ── Per-item evaluation ───────────────────────────────────────────────────────


def _evaluate_item(
    item: EvalItem,
    engine: QueryEngine,
    *,
    dataset: str,
    split: str,
) -> dict:
    """Run one question through the engine and return a result record."""
    t0 = time.perf_counter()
    error: str | None = None
    generated_sql = ""
    ex_passed = False
    ves_score = 0.0
    generated_time_s = 0.0
    gold_time_s = 0.0

    try:
        query_result = engine.query(item.question)
        generated_sql = query_result.sql
    except Exception as exc:  # noqa: BLE001
        error = f"query: {type(exc).__name__}: {exc}"
        logger.warning("eval.query_failed", db_id=item.db_id, error=error)

    latency_s = time.perf_counter() - t0

    if generated_sql and not error:
        # Open a fresh connector for EX/VES comparison — independent of the
        # engine's internal connector so the comparison does not interfere with
        # the engine's transaction state.
        cmp_connector = item.build_connector()
        try:
            cmp_connector.connect(item.dsn)
            timer = VesTimer(cmp_connector)
            ves_result = timer.score(generated_sql, item.gold_sql)
            ex_passed = ves_result.ex_result.passed
            ves_score = ves_result.ves_score
            generated_time_s = ves_result.generated_time_s
            gold_time_s = ves_result.gold_time_s
        except Exception as exc:  # noqa: BLE001
            error = f"comparator: {type(exc).__name__}: {exc}"
            logger.warning("eval.compare_failed", db_id=item.db_id, error=error)

    return {
        "db_id": item.db_id,
        "question": item.question,
        "gold_sql": item.gold_sql,
        "generated_sql": generated_sql,
        "ex_passed": ex_passed,
        "ves_score": round(ves_score, 6),
        "latency_s": round(latency_s, 3),
        "generated_time_s": round(generated_time_s, 6),
        "gold_time_s": round(gold_time_s, 6),
        "error": error,
        "dataset": dataset,
        "split": split,
    }


# ── Summary ───────────────────────────────────────────────────────────────────


def _compute_summary(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "ex_passed": 0,
            "ex_accuracy": 0.0,
            "mean_ves": 0.0,
            "mean_latency_s": 0.0,
        }
    ex_passed_count = sum(1 for r in records if r["ex_passed"])
    return {
        "total": total,
        "ex_passed": ex_passed_count,
        "ex_accuracy": round(ex_passed_count / total, 4),
        "mean_ves": round(sum(r["ves_score"] for r in records) / total, 4),
        "mean_latency_s": round(sum(r["latency_s"] for r in records) / total, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    engine_cache: dict[str, QueryEngine] = {}
    records: list[dict] = []

    for item in _iter_items(args.dataset, args.dataset_dir, args.split, args.max_items):
        engine = _get_or_create_engine(engine_cache, item, llm=args.llm)

        # BIRD evidence: inject domain hints into the engine's semantic store
        # so the retriever can surface them during RAG context assembly.
        if args.dataset == "bird" and item.evidence:
            try:
                BirdLoader.inject_evidence(item, engine._semantic)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "eval.evidence_inject_failed",
                    db_id=item.db_id,
                    error=str(exc),
                )

        record = _evaluate_item(item, engine, dataset=args.dataset, split=args.split)
        records.append(record)

        status = "PASS" if record["ex_passed"] else "FAIL"
        print(  # noqa: T201
            f"[{len(records):>4}] {status}  "
            f"{item.db_id:<30}  "
            f"EX={record['ex_passed']}  "
            f"VES={record['ves_score']:.3f}  "
            f"{item.question[:55]}"
        )

    summary = _compute_summary(records)
    output = {"summary": summary, "results": records}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(  # noqa: T201
        f"\nResults: {summary['ex_passed']}/{summary['total']} EX "
        f"({summary['ex_accuracy']:.1%})  mean VES={summary['mean_ves']:.3f}  "
        f"mean latency={summary['mean_latency_s']:.2f}s"
    )
    print(f"Written: {out_path.resolve()}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
