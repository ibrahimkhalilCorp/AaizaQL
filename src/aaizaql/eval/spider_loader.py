"""
aaizaql.eval.spider_loader
──────────────────────────
Spider benchmark dataset loader (B1).

Expected layout on disk::

    <dataset_dir>/
        dev.json          # evaluation split  (array of items)
        train_spider.json # training split     (optional)
        database/
            <db_id>/
                <db_id>.sqlite

Each JSON item has at minimum:
    db_id    : str   — database identifier matching the folder name
    question : str   — natural-language question
    query    : str   — gold SQL

Usage::

    loader = SpiderLoader("/data/spider")
    for item in loader.items(split="dev"):
        connector = item.build_connector()
        connector.connect(item.dsn)
        schema_ddl = connector.get_schema()
        ...  # ingest schema, generate SQL, compare with item.gold_sql

Author: Ibrahim
Date: 2026-06-24
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import structlog

from aaizaql.connectors.sqlite import SQLiteConnector

logger = structlog.get_logger(__name__)

# Filename for each evaluation split inside dataset_dir
_SPLIT_FILES: dict[str, str] = {
    "dev": "dev.json",
    "train": "train_spider.json",
    "test": "test.json",
}


@dataclass(frozen=True)
class EvalItem:
    """One evaluation sample from the Spider (or compatible) dataset.

    Attributes:
        db_id:     Database identifier — matches the folder name under ``database/``.
        db_path:   Absolute path to the ``.sqlite`` file.
        question:  Natural-language question.
        gold_sql:  Gold-standard SQL query.
        evidence:  Optional external-knowledge text (used by BIRD; empty for Spider).
        extra:     Any remaining fields from the raw JSON item (preserved verbatim).
    """

    db_id: str
    db_path: Path
    question: str
    gold_sql: str
    evidence: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def dsn(self) -> str:
        """SQLite DSN ready to pass to ``SQLiteConnector.connect()``."""
        return f"sqlite:///{self.db_path}"

    def build_connector(self) -> SQLiteConnector:
        """Return a *disconnected* SQLiteConnector for this item's database.

        Call ``.connect(item.dsn)`` on the returned object before use.  A new
        connector is created on every call so callers own the lifecycle.
        """
        return SQLiteConnector()


class SpiderLoader:
    """Loads Spider dev/train/test splits from a local dataset directory.

    Args:
        dataset_dir: Root directory of the Spider dataset (contains ``dev.json``
            and the ``database/`` sub-directory).
        strict: When *True* (default), raise ``FileNotFoundError`` if the
            ``.sqlite`` file for a given ``db_id`` is missing.  When *False*,
            log a warning and skip the item.

    Raises:
        FileNotFoundError: If ``dataset_dir`` does not exist.
    """

    def __init__(self, dataset_dir: str | Path, *, strict: bool = True) -> None:
        self._root = Path(dataset_dir).resolve()
        if not self._root.exists():
            raise FileNotFoundError(f"Spider dataset directory not found: {self._root}")
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
            :class:`EvalItem` for each valid row in the split JSON.

        Raises:
            FileNotFoundError: If the split JSON file does not exist.
            ValueError:        If *split* is not a recognised name.
        """
        json_path = self._resolve_split(split)
        raw_items = self._load_json(json_path)

        count = 0
        for raw in raw_items:
            if max_items is not None and count >= max_items:
                break

            item = self._parse_item(raw)
            if item is None:
                continue

            count += 1
            yield item

        logger.info("spider.loaded", split=split, count=count, root=str(self._root))

    def db_ids(self, split: str = "dev") -> list[str]:
        """Return the unique database IDs present in a split (in order of first appearance)."""
        seen: dict[str, None] = {}
        json_path = self._resolve_split(split)
        for raw in self._load_json(json_path):
            db_id = raw.get("db_id", "")
            if db_id:
                seen[db_id] = None
        return list(seen)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _resolve_split(self, split: str) -> Path:
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split {split!r}. Valid options: {sorted(_SPLIT_FILES)}"
            )
        filename = _SPLIT_FILES[split]
        path = self._root / filename
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        return path

    @staticmethod
    def _load_json(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
        return data

    def _parse_item(self, raw: dict) -> EvalItem | None:
        db_id = raw.get("db_id", "").strip()
        question = raw.get("question", "").strip()
        gold_sql = raw.get("query", "").strip()
        evidence = raw.get("evidence", "").strip()  # BIRD field; absent in Spider

        if not db_id or not question or not gold_sql:
            logger.warning("spider.skip_incomplete", raw_keys=list(raw.keys()))
            return None

        db_path = self._root / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            msg = f"SQLite file missing for db_id={db_id!r}: {db_path}"
            if self._strict:
                raise FileNotFoundError(msg)
            logger.warning("spider.missing_db", db_id=db_id, path=str(db_path))
            return None

        known_keys = {"db_id", "question", "query", "evidence"}
        extra = {k: v for k, v in raw.items() if k not in known_keys}

        return EvalItem(
            db_id=db_id,
            db_path=db_path,
            question=question,
            gold_sql=gold_sql,
            evidence=evidence,
            extra=extra,
        )
