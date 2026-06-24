"""
aaizaql.eval.bird_loader
────────────────────────
BIRD benchmark dataset loader (B4).

Expected layout on disk::

    <dataset_dir>/
        dev.json                  # evaluation split (array of items)
        dev_databases/
            <db_id>/
                <db_id>.sqlite
                database_description/   # optional CSV column docs
                    <table_name>.csv

BIRD JSON item schema (relevant fields)::

    {
        "question_id": 0,
        "db_id":       "debit_card_specializing",
        "question":    "How many customers ...",
        "evidence":    "External knowledge: ...",
        "SQL":         "SELECT COUNT(*) FROM ...",
        "difficulty":  "simple"
    }

Differences vs Spider:
- Gold SQL field: ``SQL`` (not ``query``).
- Database sub-dir per split: ``dev_databases/``, ``train_databases/``, etc.
- ``evidence`` is the critical BIRD feature — domain hints that improve SQL.
- ``difficulty`` is a quality label (simple / moderate / challenging).

Evidence injection::

    store = SemanticStore(vector_store)
    for item in loader.items():
        BirdLoader.inject_evidence(item, store)   # class-level helper

Author: Ibrahim
Date: 2026-06-24
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import structlog

from aaizaql.connectors.sqlite import SQLiteConnector
from aaizaql.eval.spider_loader import EvalItem

if TYPE_CHECKING:
    from aaizaql.schema.semantic_store import SemanticStore

logger = structlog.get_logger(__name__)

# Maps split name → databases sub-directory name
_SPLIT_DB_DIRS: dict[str, str] = {
    "dev": "dev_databases",
    "train": "train_databases",
    "test": "test_databases",
}

# Maps split name → JSON filename
_SPLIT_FILES: dict[str, str] = {
    "dev": "dev.json",
    "train": "train.json",
    "test": "test.json",
}


class BirdLoader:
    """Loads BIRD dev/train/test splits from a local dataset directory.

    Args:
        dataset_dir: Root directory of the BIRD dataset.  Must contain the
            split JSON file (e.g. ``dev.json``) and the matching databases
            sub-directory (e.g. ``dev_databases/``).
        strict: When *True* (default), raise ``FileNotFoundError`` if the
            ``.sqlite`` file for a given ``db_id`` is missing.  When *False*,
            log a warning and skip the item.

    Raises:
        FileNotFoundError: If ``dataset_dir`` does not exist.
    """

    def __init__(self, dataset_dir: str | Path, *, strict: bool = True) -> None:
        self._root = Path(dataset_dir).resolve()
        if not self._root.exists():
            raise FileNotFoundError(f"BIRD dataset directory not found: {self._root}")
        self._strict = strict

    # ── Public API ────────────────────────────────────────────────────────────

    def items(
        self,
        split: str = "dev",
        max_items: int | None = None,
    ) -> Iterator[EvalItem]:
        """Iterate over evaluation items for the requested split.

        Args:
            split:     Dataset split — ``"dev"``, ``"train"``, or ``"test"``.
            max_items: If set, stop after this many items (useful for smoke tests).

        Yields:
            :class:`~aaizaql.eval.spider_loader.EvalItem` for each valid row.

        Raises:
            FileNotFoundError: If the split JSON file does not exist.
            ValueError:        If *split* is not a recognised name.
        """
        json_path, db_dir = self._resolve_split(split)
        raw_items = self._load_json(json_path)

        count = 0
        for raw in raw_items:
            if max_items is not None and count >= max_items:
                break

            item = self._parse_item(raw, db_dir)
            if item is None:
                continue

            count += 1
            yield item

        logger.info("bird.loaded", split=split, count=count, root=str(self._root))

    def db_ids(self, split: str = "dev") -> list[str]:
        """Return the unique database IDs present in a split (in order of first appearance)."""
        seen: dict[str, None] = {}
        json_path, _ = self._resolve_split(split)
        for raw in self._load_json(json_path):
            db_id = raw.get("db_id", "")
            if db_id:
                seen[db_id] = None
        return list(seen)

    @staticmethod
    def inject_evidence(item: EvalItem, store: SemanticStore) -> None:
        """Inject an item's BIRD evidence string into a SemanticStore.

        Evidence is stored as free-text documentation so it is retrieved
        alongside schema context during RAG-augmented generation.  Items
        with no evidence are silently skipped.

        Args:
            item:  :class:`~aaizaql.eval.spider_loader.EvalItem` whose
                   ``evidence`` field will be injected.
            store: Target :class:`~aaizaql.schema.semantic_store.SemanticStore`
                   instance (must be initialised before injection).
        """
        if not item.evidence:
            return
        store.train_documentation(item.evidence)
        logger.debug("bird.evidence_injected", db_id=item.db_id, chars=len(item.evidence))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _resolve_split(self, split: str) -> tuple[Path, Path]:
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split {split!r}. Valid options: {sorted(_SPLIT_FILES)}"
            )
        json_path = self._root / _SPLIT_FILES[split]
        if not json_path.exists():
            raise FileNotFoundError(f"Split file not found: {json_path}")
        db_dir = self._root / _SPLIT_DB_DIRS[split]
        return json_path, db_dir

    @staticmethod
    def _load_json(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
        return data

    def _parse_item(self, raw: dict, db_dir: Path) -> EvalItem | None:
        db_id = raw.get("db_id", "").strip()
        question = raw.get("question", "").strip()
        # BIRD uses "SQL" (uppercase), Spider uses "query"
        gold_sql = raw.get("SQL", raw.get("query", "")).strip()
        evidence = raw.get("evidence", "").strip()

        if not db_id or not question or not gold_sql:
            logger.warning("bird.skip_incomplete", raw_keys=list(raw.keys()))
            return None

        db_path = db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            msg = f"SQLite file missing for db_id={db_id!r}: {db_path}"
            if self._strict:
                raise FileNotFoundError(msg)
            logger.warning("bird.missing_db", db_id=db_id, path=str(db_path))
            return None

        known_keys = {"db_id", "question", "SQL", "query", "evidence", "question_id"}
        extra = {k: v for k, v in raw.items() if k not in known_keys}

        return EvalItem(
            db_id=db_id,
            db_path=db_path,
            question=question,
            gold_sql=gold_sql,
            evidence=evidence,
            extra=extra,
        )
