"""
aaizaql.eval
────────────
Evaluation harness for BIRD + Spider benchmarks.

Usage::

    from aaizaql.eval import SpiderLoader, BirdLoader, EvalItem, ExComparator, ExResult

    # Spider
    loader = SpiderLoader("/data/spider")
    for item in loader.items():
        connector = item.build_connector()
        connector.connect(item.dsn)
        comparator = ExComparator(connector)
        result = comparator.compare(generated_sql, item.gold_sql)
        ...

    # BIRD (with evidence injection)
    from aaizaql.schema.semantic_store import SemanticStore
    bird = BirdLoader("/data/bird")
    for item in bird.items():
        BirdLoader.inject_evidence(item, store)
        connector = item.build_connector()
        connector.connect(item.dsn)
        ...
"""

from aaizaql.eval.bird_loader import BirdLoader
from aaizaql.eval.ex_comparator import ExComparator, ExResult
from aaizaql.eval.spider_loader import EvalItem, SpiderLoader
from aaizaql.eval.ves_timer import VesResult, VesTimer

__all__ = [
    "BirdLoader",
    "EvalItem",
    "ExComparator",
    "ExResult",
    "SpiderLoader",
    "VesResult",
    "VesTimer",
]
